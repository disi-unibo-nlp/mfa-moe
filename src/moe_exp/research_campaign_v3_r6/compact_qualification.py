"""Same native compact fixture, typed RPC diagnostics, unchanged tolerances."""
import gc
from .contracts import require
from .lifecycle import ShapeLimitation, exception_record, failure_scope
from moe_exp.expert_memory_study.worker import storage_inventory, numerical_error

def qualify(self):
    """One full resident reference layer at a time; original checkpoint loader."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from moe_exp.expert_memory_study.loader import RouterMask, weight_iterator
    model, hooks = self.model_runner.get_model(), self.memory_hooks
    require(hooks.config["compact"], "qualification requires compact allocation")
    rows, reference_bytes = [], 0
    diagnostics = []
    def qualified_require(condition, message):
        if not condition:
            error = ShapeLimitation(message)
            error.campaign_numerical_diagnostics = list(diagnostics)
            raise error
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
            qualified_require(expected <= loaded, "reference expert/packed-scale load coverage incomplete")
            with set_current_vllm_config(self.vllm_config):
                process_weights_after_loading(reference.routed_experts, self.vllm_config.model_config,
                                              next(reference.parameters()).device)
            setattr(parent, leaf, runner)
            reference_route = RouterMask(reference.router, route.keep, route.experts, compact=False)
            full_bytes = storage_inventory(reference.routed_experts)["unique_bytes"]
            compact_bytes = storage_inventory(runner.routed_experts)["unique_bytes"]
            qualified_require(compact_bytes < full_bytes, "no realized unique expert storage reduction")
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
                diagnostics.append({"layer":lid,"phase":phase,"all_keep":all_keep})
                qualified_require(torch.equal(ai, ni) and all_keep["pass"], "all-keep native routing mismatch")
                reference_route.mode = "masked"
                rw, ri = reference_route.compute(x, logits, torch.int32)
                cw, ci = route.compute(x, logits, torch.int32)
                original_ids = torch.tensor(route.keep, device=ci.device)[ci.long()]
                routing = numerical_error(cw, rw, atol=1e-6, rtol=1e-5)
                diagnostics.append({"layer":lid,"phase":phase,"routing":routing})
                qualified_require(torch.equal(original_ids, ri) and routing["pass"], "compact mapping/routing mismatch")
                with set_forward_context(None, self.vllm_config):
                    static.static_forward_context[prefix] = runner
                    compact_calls = route.calls
                    compact_out = runner(x.clone(), logits.clone())
                    qualified_require(route.calls > compact_calls, "fixture bypassed expected compact layer router")
                    static.static_forward_context[prefix] = reference
                    reference_calls = reference_route.calls
                    resident_out = reference(x.clone(), logits.clone())
                    qualified_require(reference_route.calls > reference_calls, "fixture bypassed expected resident layer router")
                static.static_forward_context[prefix] = runner
                qualified_require(isinstance(compact_out, torch.Tensor) and isinstance(resident_out, torch.Tensor),
                        "unrecognized native expert output")
                output = numerical_error(compact_out, resident_out, atol=1e-2, rtol=1e-2)
                diagnostics.append({"layer":lid,"phase":phase,"output":output})
                qualified_require(output["pass"], "full-resident masked/compact output mismatch")
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

def outcome(worker):
    try:
        return qualify(worker)
    except Exception as exc:
        return {"status":"FAIL","failure_scope":failure_scope(exc),"failure":exception_record(exc,"compact_fixture"),
            "numerical_diagnostics":getattr(exc,"campaign_numerical_diagnostics",[]),"within_execution":True}
