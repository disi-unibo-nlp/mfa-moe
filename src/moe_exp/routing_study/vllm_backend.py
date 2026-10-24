"""Version-pinned, process-local vLLM integration. Never edits site-packages."""
from __future__ import annotations
import functools
import re
import time
from .actions import logits_action, weights_action
from .common import CHECKPOINT, file_hash, require_compute
from .policy import validate
from .runtime import RequestHistory

BACKEND_FILES = ("model_executor/layers/fused_moe/runner/moe_runner.py",
    "model_executor/layers/fused_moe/router/fused_topk_router.py",
    "model_executor/layers/fused_moe/router/fused_moe_router.py",
    "model_executor/layers/fused_moe/routed_experts.py", "v1/worker/gpu_model_runner.py",
    "model_executor/models/qwen3_next.py", "model_executor/models/qwen3_5.py",
    "v1/engine/input_processor.py", "v1/engine/llm_engine.py",
    "v1/worker/gpu_worker.py", "config/vllm.py", "envs.py", "utils/network_utils.py",
    "model_executor/layers/quantization/fp8.py",
    "model_executor/layers/fused_moe/oracle/fp8.py",
    "model_executor/layers/fused_moe/experts/marlin_moe.py",
    "model_executor/layers/fused_moe/modular_kernel.py",
    "model_executor/layers/quantization/utils/marlin_utils_fp8.py")


def backend_fingerprint():
    from importlib.metadata import distribution
    from pathlib import Path
    dist = distribution("vllm")
    root = Path(dist.locate_file("vllm"))
    if dist.version != "0.29.0+cu129":
        raise RuntimeError("Only inspected vLLM 0.29.0+cu129 is supported")
    checkpoint = checkpoint_fingerprint()
    return {"version": dist.version, "files": {p: file_hash(root / p) for p in BACKEND_FILES},
            "checkpoint_files_sha256": checkpoint, "model_runner": "v1",
            "moe_backend": "marlin", "request_ids": "unique_external_randomization_disabled", "expert_arithmetic": "FP8 weights, BF16 activations (W8A16)"}


def checkpoint_fingerprint(*, verify_weights=False):
    from pathlib import Path
    import json
    manifest = Path(__file__).with_name("checkpoint_files.json")
    data = json.loads(manifest.read_text())
    if verify_weights:
        require_compute()
    for name, expected in data["files"].items():
        path = Path(data["cache"]) / name
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Frozen checkpoint cache changed: " + name)
        if name.endswith(".safetensors") and path.resolve().name != expected["sha256"]:
            raise ValueError("Checkpoint blob identity changed")
        if (verify_weights or expected["verified_content"]) and file_hash(path) != expected["sha256"]:
            raise ValueError("Checkpoint content hash mismatch: " + name)
    return file_hash(manifest)


def validate_config(config):
    model, parallel, cache = config.model_config, config.parallel_config, config.cache_config
    from .common import MODEL, REVISION
    from pathlib import Path
    import json
    snapshot = json.loads(Path(__file__).with_name("checkpoint_files.json").read_text())["cache"]
    # HF_HUB_OFFLINE rewrites the repo ID to this exact revision's cached path.
    # Content/metadata verification remains mandatory in backend_fingerprint.
    if (model.model not in (MODEL, snapshot) or model.revision != REVISION or model.quantization != "fp8"):
        raise RuntimeError("Loaded model/revision/quantization differs from the FP8 contract")
    text = model.hf_text_config
    if (text.model_type != "qwen3_5_moe_text" or text.num_experts != 256
            or text.num_experts_per_tok != 8 or text.num_hidden_layers != 40
            or text.shared_expert_intermediate_size <= 0):
        raise RuntimeError("Unsupported checkpoint routing architecture")
    if (config.use_v2_model_runner or config.kernel_config.moe_backend != "marlin"
            or not model.enforce_eager or parallel.tensor_parallel_size != 2
            or parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or getattr(parallel, "enable_dbo", False)
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or config.speculative_config is not None or cache.enable_prefix_caching
            or config.scheduler_config.max_num_seqs != 8
            or getattr(config.scheduler_config, "async_scheduling", False)):
        raise RuntimeError("Study requires V1 runner, modular Marlin, eager TP2, concurrency 8, synchronous scheduling, no speculative/prefix/EPLB/EP/SP/DBO")


def validate_runner(module):
    runner = module.experts
    if (type(runner).__name__ != "MoERunner"
            or runner.routed_experts.quant_method.is_monolithic
            or getattr(getattr(runner.routed_experts.quant_method, "fp8_backend", None), "name", None) != "MARLIN"
            or getattr(runner, "_fse_fuse_gate", False)
            or getattr(module, "is_fused_shared_expert_enabled", False)
            or getattr(module, "is_sequence_parallel", False)
            or getattr(module, "shared_expert", None) is None
            or type(runner.router).__name__ != "FusedTopKRouter"
            or runner.router.scoring_func != "softmax" or not runner.router.renormalize
            or getattr(runner.router, "eplb_state", None) is not None
            or not callable(getattr(runner.routed_experts, "forward_modular", None))):
        raise RuntimeError("Unsupported fused/monolithic/redistributed routing path")
    return runner


def make_frame(owner, request_ids, starts, tokens, positions):
    if (len(starts) != len(request_ids)+1 or starts[0] != 0
            or any(b < a for a,b in zip(starts,starts[1:]))
            or starts[-1] > len(tokens) or len(tokens) != len(positions)):
        raise RuntimeError("Invalid query/padding boundaries")
    frame = [None] * len(tokens)
    for i,key in enumerate(request_ids):
        if key not in owner.requests:
            raise RuntimeError("Unregistered/cancelled request reached model")
        state = owner.requests[key]
        for j in range(int(starts[i]),int(starts[i+1])):
            frame[j] = (state,state.tokens.observe(positions[j],tokens[j]))
    return frame


class DispatchHook:
    def __init__(self, layer, owner, runner):
        self.layer, self.owner, self.runner = layer, owner, runner
        self.select_original = runner.router.select_experts
        self.dispatch_original = runner.routed_experts.forward_modular
        self.pending = None
        self.calls = self.dispatches = 0
        self.seconds = 0.
        runner.router.select_experts = self.select
        runner.routed_experts.forward_modular = self.dispatch

    def select(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        import torch
        started = time.perf_counter()
        if self.pending is not None or self.owner.frame is None:
            raise RuntimeError("Missing request frame or unmatched selection")
        native_weights, native_ids = self.select_original(hidden_states, router_logits,
            topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
        native_ids_cpu = native_ids.detach().cpu().numpy().copy()
        native_weights_cpu = native_weights.detach().float().cpu().numpy().copy()
        scores_cpu = router_logits.detach().float().cpu().numpy()
        if len(native_ids) != len(self.owner.frame):
            raise RuntimeError("Routing batch/absolute position alignment differs")
        active = []
        for i, entry in enumerate(self.owner.frame):
            if entry is None:
                active.append(False)
                continue
            state, layout = entry
            active.append(state.layers[self.layer].plan(layout, native_ids_cpu[i],
                                                        native_weights_cpu[i], scores_cpu[i]))
        mask = torch.tensor(active, device=router_logits.device, dtype=torch.bool)
        actual_weights, actual_ids = native_weights, native_ids
        if any(active):
            candidate = logits_action(router_logits[mask], self.owner.policy, self.layer)
            if self.owner.policy["action"] in ("identity", "margin", "steer_promote", "steer_suppress"):
                guided = router_logits.clone()
                guided[mask] = candidate
                selected_weights, selected_ids = self.select_original(hidden_states, guided,
                    topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
                actual_weights = torch.where(mask[:, None], selected_weights, native_weights)
                actual_ids = torch.where(mask[:, None], selected_ids, native_ids)
            if self.owner.policy["action"] in ("rice", "temperature"):
                actual_weights = native_weights.clone()
                actual_weights[mask] = weights_action(native_weights[mask], native_ids[mask],
                                                       self.owner.policy, self.layer)
        self.pending = (native_ids_cpu, native_weights_cpu, actual_ids, actual_weights, active)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return actual_weights, actual_ids

    def dispatch(self, *args, **kwargs):
        import torch
        started = time.perf_counter()
        if args or self.pending is None:
            raise RuntimeError("Unverified expert dispatch call signature")
        native_ids, native_weights, ids, weights, active = self.pending
        if kwargs.get("topk_ids") is not ids or kwargs.get("topk_weights") is not weights:
            raise RuntimeError("Kernel did not receive the captured dispatch tensors")
        ids_cpu = ids.detach().cpu().numpy()
        weights_cpu = weights.detach().float().cpu().numpy()
        for i, entry in enumerate(self.owner.frame):
            if entry is not None:
                state, layout = entry
                state.layers[self.layer].dispatch(layout, native_ids[i], native_weights[i],
                                                 ids_cpu[i], weights_cpu[i], active[i])
        self.dispatches += 1
        self.pending = None
        self.seconds += time.perf_counter() - started
        # Preserve every other argument, including the original shared expert object.
        return self.dispatch_original(**kwargs)


class StudyWorkerExtension:
    def study_configure(self, policy, backend, checkpoint, validation=False):
        require_compute()
        validate(policy, current_code=True)
        if checkpoint != CHECKPOINT or backend_fingerprint() != backend:
            raise ValueError("Checkpoint/backend binding changed")
        validate_config(self.vllm_config)
        if type(self.model_runner).__name__ != "GPUModelRunner":
            raise RuntimeError("Unverified model runner implementation")
        if hasattr(self, "study"):
            raise RuntimeError("Start a fresh process for each study condition")
        from types import SimpleNamespace
        from vllm.distributed import get_tensor_model_parallel_rank
        from transformers import AutoTokenizer
        from .common import MODEL, REVISION
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        decode_piece = functools.lru_cache(maxsize=262144)(
            lambda token: tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        self.study = SimpleNamespace(policy=policy, frame=None, requests={}, retired=set(),
                                     rank=get_tensor_model_parallel_rank(), decode_piece=decode_piece, validation=validation, backend=backend)
        model = self.model_runner.get_model()
        found = {}
        for name, module in model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", name)
            if match:
                layer = int(match[1])
                if layer in found:
                    raise RuntimeError("Ambiguous MoE layer identity")
                found[layer] = validate_runner(module)
        if set(found) != set(range(40)):
            raise RuntimeError("All 40 routed layers must use the verified dispatch path")
        self.study.hooks = {l: DispatchHook(l, self.study, runner) for l, runner in found.items()}
        original = self.model_runner._model_forward
        def forward(*args, **kwargs):
            if args or self.study.frame is not None:
                raise RuntimeError("Unsupported nested/positional model forward")
            ids, positions = kwargs.get("input_ids"), kwargs.get("positions")
            if ids is None or positions is None or positions.ndim not in (1, 2) or ids.ndim != 1:
                raise RuntimeError("Study requires text input IDs and verified text position axes")
            batch = self.model_runner.input_batch
            starts = self.model_runner.query_start_loc.np[:batch.num_reqs + 1]
            if starts[0] != 0 or any(b < a for a, b in zip(starts, starts[1:])) or starts[-1] > len(ids):
                raise RuntimeError("Invalid query/padding boundaries")
            tokens = ids.detach().cpu().tolist()
            axes = positions.detach().cpu().numpy()
            if axes.ndim == 1:
                axes = axes[None, :]
            absolute = [0] * len(tokens)
            for i in range(batch.num_reqs):
                start, stop = int(starts[i]), int(starts[i+1])
                expected = list(range(int(batch.num_computed_tokens_cpu[i]),
                                      int(batch.num_computed_tokens_cpu[i]) + stop - start))
                if axes.shape[-1] != len(tokens) or not (axes[:, start:stop] == expected).all():
                    raise RuntimeError("Text rotary axes differ from absolute request token positions")
                absolute[start:stop] = expected
            frame = make_frame(self.study, batch.req_ids, starts, tokens, absolute)
            self.study.frame = frame
            before = {l: h.dispatches for l, h in self.study.hooks.items()}
            try:
                result = original(**kwargs)
                if any(h.pending is not None or h.dispatches != before[l] + 1 for l, h in self.study.hooks.items()):
                    raise RuntimeError("Backend bypassed a routing or dispatch hook")
                return result
            finally:
                self.study.frame = None
        self.model_runner._model_forward = forward
        return {"rank": self.study.rank, "backend": backend, "layers": sorted(found)}

    def study_set_validation_policy(self, policy):
        if not self.study.validation or self.study.requests or self.study.frame is not None:
            raise RuntimeError("Policy switching is allowed only between isolated smoke conditions")
        validate(policy, current_code=True)
        self.study.policy = policy
        self.study.retired.clear()
        for hook in self.study.hooks.values():
            hook.calls = hook.dispatches = 0
            hook.seconds = 0.

    def study_register(self, requests):
        for key, spec in requests.items():
            if key in self.study.requests or key in self.study.retired:
                raise ValueError("Duplicate/cancelled request ID")
            self.study.requests[key] = RequestHistory(spec, self.study.policy, self.study.decode_piece)

    def study_snapshot(self, key, full=False, study_sha256=None):
        state = self.study.requests[key]
        value = {"rank": self.study.rank, "layers": state.summary(),
                 "hook_seconds": sum(h.seconds for h in self.study.hooks.values()),
                 "observed_input_tokens": len(state.tokens.tokens),
                 "reasoning_tokens": state.tokens.reasoning_count}
        if full and self.study.rank == 0:
            value["sample"] = state.calibration_sample(study_sha256, self.study.backend)
        return value

    def study_release(self, keys):
        for key in keys:
            self.study.requests.pop(key, None)
            self.study.retired.add(key)


def verify_reports(reports, *, minimum_tokens=1):
    if len(reports) != 2 or {r["rank"] for r in reports} != {0, 1}:
        raise RuntimeError("Require both TP replicas")
    left, right = sorted(reports, key=lambda r: r["rank"])
    for layer in map(str, range(40)):
        a, b = left["layers"][layer], right["layers"][layer]
        if a != b:
            raise RuntimeError("TP replicas disagree; do not pool duplicated observations")
        if a["diagnostics"]["tokens"] < minimum_tokens or a["diagnostics"]["dispatch_calls"] < minimum_tokens:
            raise RuntimeError("Missing actual dispatch observations")
    return left
