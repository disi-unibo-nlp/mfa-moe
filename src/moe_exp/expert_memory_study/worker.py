"""Explicit GPU worker; hooks install before each rank loads its model."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gc
import os

from .contracts import digest, load, require


def storage_inventory(module):
    """Unique backing storage includes quantization buffers and avoids aliases."""
    seen, rows = set(), []
    tensors = list(module.named_parameters()) + list(module.named_buffers())
    for name, t in tensors:
        if t.numel() == 0:
            continue
        s = t.untyped_storage()
        key = (str(t.device), s.data_ptr(), s.nbytes())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "bytes": s.nbytes()})
    return {"unique_bytes": sum(r["bytes"] for r in rows), "storages": rows}


def memory_snapshot():
    import torch
    torch.cuda.synchronize()
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        # Process residence, not whole-device memory used by neighboring jobs.
        residence = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            device = pynvml.nvmlDeviceGetHandleByIndex(index)
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(device):
                if process.pid == os.getpid():
                    residence.append({"device_uuid": pynvml.nvmlDeviceGetUUID(device),
                                      "bytes": int(process.usedGpuMemory)})
        nvml = residence
    finally:
        if nvml is not None:
            pynvml.nvmlShutdown()
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
            "nvml_process_residence": nvml}


def cache_value(value):
    if is_dataclass(value):
        return cache_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): cache_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cache_value(v) for v in value]
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def backend_identity(experts):
    method = experts.quant_method
    kernel = getattr(method, "moe_kernel", None)
    def kind(value):
        return None if value is None else type(value).__module__ + "." + type(value).__qualname__
    return {"quant_method": kind(method), "kernel": kind(kernel),
            "fused_experts": kind(getattr(kernel, "fused_experts", None)),
            "prepare_finalize": kind(getattr(kernel, "prepare_finalize", None)),
            "tensor_dtypes": sorted({str(t.dtype) for _, t in list(experts.named_parameters()) + list(experts.named_buffers())})}


def numerical_error(a, b, *, atol, rtol):
    import torch
    require(a.shape == b.shape, "numerical fixture shape differs")
    delta = (a.float() - b.float()).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"max_abs": float(delta.max().item()),
            "max_relative": float((delta / b.float().abs().clamp_min(1e-12)).max().item()),
            "pass": finite and bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)),
            "atol": atol, "rtol": rtol}


def get_worker_class():
    """Delay CUDA/vLLM imports until an allocated launcher explicitly requests them."""
    from vllm.v1.worker.gpu_worker import Worker
    import torch

    class ExpertMemoryWorker(Worker):
        def load_model(self, *, load_dummy_weights=False):
            from .loader import activate, deactivate
            config = load(self.vllm_config.load_config.model_loader_extra_config["binding_path"])
            require(not load_dummy_weights, "dummy weights cannot qualify expert loading")
            self.memory_hooks = None
            try:
                from vllm.config import set_current_vllm_config
                with set_current_vllm_config(self.vllm_config):
                    self.memory_hooks = activate(config)
                super().load_model(load_dummy_weights=False)
                require(set(self.memory_hooks.layers) == {int(l) for l in config["masks"]},
                        "construction hooks missed routed layers")
                require(all(not x["runner"].is_monolithic for x in self.memory_hooks.layers.values()),
                        "native monolithic kernel bypasses private router; unqualified")
                self.memory_load_snapshot = memory_snapshot()
            finally:
                deactivate()

        def initialize_from_config(self, kv_cache_config):
            self.memory_cache_spec = cache_value(kv_cache_config)
            super().initialize_from_config(kv_cache_config)

        def memory_status(self, *, reset=False):
            import torch
            hooks = self.memory_hooks
            expert = {str(l): storage_inventory(x["runner"].routed_experts) for l, x in hooks.layers.items()}
            routes = {str(l): {"calls": x["routing"].calls, "rows": x["routing"].rows,
                               "removed_dispatch": x["routing"].removed_dispatch,
                               "original_to_compact": {str(e): i for i, e in enumerate(x["routing"].keep)}}
                      for l, x in hooks.layers.items()}
            result = {"expert_storage": expert, "total_expert_bytes": sum(v["unique_bytes"] for v in expert.values()),
                      "total_model_storage": storage_inventory(self.model_runner.get_model()),
                      "load_memory": self.memory_load_snapshot, "memory": memory_snapshot(),
                      "cache_spec": self.memory_cache_spec, "cache_sha256": digest(self.memory_cache_spec),
                      "routes": routes, "load_ledger_sha256": digest(hooks.config["load_ledger"]),
                      "load_ledger_items": len(hooks.config["load_ledger"]),
                      "backend": {str(l): backend_identity(x["runner"].routed_experts) for l, x in hooks.layers.items()}}
            if reset:
                torch.cuda.reset_peak_memory_stats()
                for x in hooks.layers.values():
                    x["routing"].calls = x["routing"].rows = x["routing"].removed_dispatch = 0
            return result

        def memory_phase(self, *, reset=False):
            snapshot = memory_snapshot()
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return snapshot

        @torch.inference_mode()
        def memory_qualify(self):
            """One full resident reference layer at a time; original checkpoint loader."""
            import torch
            from vllm.config import set_current_vllm_config
            from vllm.forward_context import set_forward_context
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            from .loader import RouterMask, weight_iterator
            model, hooks = self.model_runner.get_model(), self.memory_hooks
            require(hooks.config["compact"], "qualification requires compact allocation")
            rows, reference_bytes = [], 0
            for lid, item in sorted(hooks.layers.items()):
                runner, route = item["runner"], item["routing"]
                path = next(name for name, module in model.named_modules() if module is runner)
                parent_name, leaf = path.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                args = dict(item["native_args"])
                static = self.vllm_config.compilation_config
                prefix = args["prefix"]
                old_context = static.static_forward_context[prefix]
                old_layers = list(static.static_all_moe_layers)
                reference, reference_route = None, None
                try:
                    del static.static_forward_context[prefix]
                    with set_current_vllm_config(self.vllm_config), torch.device(next(runner.parameters()).device):
                        reference = item["factory"](**args)
                    static.static_forward_context[prefix] = old_context
                    static.static_all_moe_layers[:] = old_layers
                    setattr(parent, leaf, reference)
                    ledger = {}
                    loaded = set(model.load_weights(weight_iterator(hooks.config["checkpoint"], hooks.config["masks"],
                                 hooks.config["experts"], compact=False, only_layer=lid, ledger=ledger)))
                    # Native mappings use the nested routed_experts prefix.
                    expected = {path + ".routed_experts." + n for n, _ in reference.routed_experts.named_parameters()}
                    require(expected <= loaded, "reference expert/packed-scale load coverage incomplete")
                    with set_current_vllm_config(self.vllm_config):
                        process_weights_after_loading(reference.routed_experts, self.vllm_config.model_config,
                                                      next(reference.parameters()).device)
                    setattr(parent, leaf, runner)
                    reference_route = RouterMask(reference.router, route.keep, route.experts, compact=False)
                    full_bytes = storage_inventory(reference.routed_experts)["unique_bytes"]
                    compact_bytes = storage_inventory(runner.routed_experts)["unique_bytes"]
                    require(compact_bytes < full_bytes, "no realized unique expert storage reduction")
                    reference_bytes += full_bytes
                    for phase, count in (("prefill", 17), ("decode", 1)):
                        device = next(runner.parameters()).device
                        generator = torch.Generator(device=device).manual_seed(20261008 + lid + count)
                        input_width = getattr(runner.routed_input_transform, "input_size", args["hidden_size"])
                        x = torch.randn(count, input_width, device=device, dtype=torch.bfloat16, generator=generator)
                        logits = torch.randn(count, route.experts, device=device, dtype=torch.float32, generator=generator)
                        # Native/all-keep routing fidelity at exactly the same state.
                        bound = reference_route.signature.bind(x, logits, torch.int32)
                        nw, ni = reference_route.original(*bound.args, **bound.kwargs)
                        reference_route.mode = "all_keep"
                        aw, ai = reference_route.compute(x, logits, torch.int32)
                        all_keep = numerical_error(aw, nw, atol=1e-6, rtol=1e-5)
                        require(torch.equal(ai, ni) and all_keep["pass"], "all-keep native routing mismatch")
                        reference_route.mode = "masked"
                        rw, ri = reference_route.compute(x, logits, torch.int32)
                        cw, ci = route.compute(x, logits, torch.int32)
                        original_ids = torch.tensor(route.keep, device=ci.device)[ci.long()]
                        routing = numerical_error(cw, rw, atol=1e-6, rtol=1e-5)
                        require(torch.equal(original_ids, ri) and routing["pass"], "compact mapping/routing mismatch")
                        with set_forward_context(None, self.vllm_config):
                            static.static_forward_context[prefix] = runner
                            compact_calls = route.calls
                            compact_out = runner(x.clone(), logits.clone())
                            require(route.calls > compact_calls, "fixture bypassed expected compact layer router")
                            static.static_forward_context[prefix] = reference
                            reference_calls = reference_route.calls
                            resident_out = reference(x.clone(), logits.clone())
                            require(reference_route.calls > reference_calls, "fixture bypassed expected resident layer router")
                        static.static_forward_context[prefix] = runner
                        require(isinstance(compact_out, torch.Tensor) and isinstance(resident_out, torch.Tensor),
                                "unrecognized native expert output")
                        output = numerical_error(compact_out, resident_out, atol=1e-2, rtol=1e-2)
                        require(output["pass"], "full-resident masked/compact output mismatch")
                        rows.append({"layer": lid, "phase": phase, "routing": routing,
                                     "all_keep": all_keep, "output": output,
                                     "resident_expert_bytes": full_bytes, "compact_expert_bytes": compact_bytes,
                                     "full_reference_loaded_tensors": len(ledger)})
                finally:
                    setattr(parent, leaf, runner)
                    static.static_forward_context[prefix] = old_context
                    static.static_all_moe_layers[:] = old_layers
                    if reference_route is not None:
                        reference_route.restore()
                    del reference
                    gc.collect()
                    torch.cuda.empty_cache()
            return {"status": "PASS", "cases": rows, "full_reference_expert_bytes": reference_bytes,
                    "within_execution": True, "cleanup": "native model and static context restored"}

    return ExpertMemoryWorker


# vLLM resolves this explicit class path only inside the allocated child process.
def __getattr__(name):
    if name == "ExpertMemoryWorker":
        return get_worker_class()
    raise AttributeError(name)
