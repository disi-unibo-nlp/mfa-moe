from __future__ import annotations

import math
import re

import torch

from .calibration import validate_policy


class IdentityBias:
    """Hook vLLM's unquantized gate output before its native expert selection."""

    def __init__(self, scores, strength: float, top_k: int, diagnostics="full"):
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        self.diagnostics = diagnostics
        self.scores = torch.tensor(scores, dtype=torch.float32)
        if self.scores.ndim != 1 or not 1 <= top_k <= len(scores):
            raise ValueError("Invalid score vector or top_k")
        if not torch.isfinite(self.scores).all() or (self.scores < 0).any():
            raise ValueError("Scores must be finite and nonnegative")
        self.strength, self.top_k = strength, top_k
        self.reset()

    def reset(self):
        self.calls = self.tokens = 0
        self.changed = self.before = self.after = None

    def __call__(self, module, args, output):
        logits = output[0] if isinstance(output, tuple) else output
        if logits.ndim != 2 or logits.shape[-1] != len(self.scores):
            raise ValueError("Gate output does not match policy expert count")
        self.scores = self.scores.to(logits.device)
        # Retain the gate dtype expected by the native fused kernel.
        guided = (logits.float() + self.strength * self.scores).to(logits.dtype)
        self.calls += 1
        self.tokens += logits.shape[0]
        if self.diagnostics == "full":
            old_ids = logits.float().topk(self.top_k, dim=-1).indices
            new_ids = guided.float().topk(self.top_k, dim=-1).indices
            changes = (old_ids.sort(-1).values != new_ids.sort(-1).values).any(-1).sum()
            before = torch.bincount(old_ids.flatten(), minlength=len(self.scores))
            after = torch.bincount(new_ids.flatten(), minlength=len(self.scores))
            if self.changed is None:
                self.changed, self.before, self.after = changes.detach(), before, after
            else:
                self.changed += changes.detach()
                self.before += before
                self.after += after
        if self.strength == 0:
            return output
        return (guided, *output[1:]) if isinstance(output, tuple) else guided

    def snapshot(self):
        if self.diagnostics != "full":
            return {"calls": self.calls, "token_evaluations": self.tokens,
                    "diagnostics": "minimal"}
        return {"calls": self.calls, "token_evaluations": self.tokens,
                "changed_topk_sets": 0 if self.changed is None else self.changed.item(),
                "expert_counts_before": [] if self.before is None else self.before.tolist(),
                "expert_counts_after": [] if self.after is None else self.after.tolist()}


def routing_spec(text):
    """Native vLLM router locations and config dimensions (not HF hook layouts)."""
    family = text.model_type
    if family in ("qwen3_5_moe", "qwen3_5_moe_text"):
        return "qwen", text.num_experts, text.num_experts_per_tok
    if family == "gpt_oss":
        return "oss", text.num_local_experts, text.num_experts_per_tok
    if family in ("gemma4", "gemma4_text"):
        return "gemma", text.num_experts, text.top_k_experts
    raise ValueError("Identity guiding supports Qwen3.5 MoE, GPT-OSS and Gemma4 MoE only")


def install(model, policy: dict, strength: float, family: str = "qwen", diagnostics="full"):
    if family not in ("qwen", "oss", "gemma"):
        raise ValueError(f"Unsupported routing family: {family}")
    validate_policy(policy)
    found = {}
    for name, module in model.named_modules():
        suffix = "" if family == "gemma" else r"\.mlp"
        match = re.search(r"(?:^|\.)layers\.(\d+)" + suffix + "$", name)
        if match and match[1] in policy["layers"]:
            layer = match[1]
            gate = getattr(module, "gate" if family == "qwen" else "router", None)
            if gate is None or layer in found:
                raise RuntimeError(f"Missing/ambiguous native gate for layer {layer}")
            if getattr(getattr(module, "experts", None), "_fse_fuse_gate", False):
                raise RuntimeError("Fused shared gate bypasses hooks; disable shared gate fusion")
            found[layer] = gate
    if set(found) != set(policy["layers"]):
        raise RuntimeError("Model is missing selected policy layers")
    hooks, handles = {}, []
    try:
        for layer, gate in found.items():
            hook = IdentityBias(policy["layers"][layer]["scores"], strength, policy["top_k"], diagnostics=diagnostics)
            handles.append(gate.register_forward_hook(hook))
            hooks[layer] = hook
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return hooks, handles


class IdentityWorkerExtension:
    def identity_configure(self, policy: dict, strength: float, condition: str, diagnostics="minimal"):
        config = self.vllm_config
        text = getattr(config.model_config, "hf_text_config", None)
        if text is None:
            text = config.model_config.hf_config
        family, num_experts, top_k = routing_spec(text)
        validate_policy(policy)
        if (num_experts, top_k) != (policy["num_experts"], policy["top_k"]):
            raise ValueError("Policy and checkpoint expert counts differ")
        parallel = config.parallel_config
        if (not config.model_config.enforce_eager or parallel.enable_expert_parallel
                or parallel.enable_eplb or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1 or getattr(parallel, "enable_dbo", False)
                or config.speculative_config is not None):
            raise ValueError("Use eager execution, TP only, without speculative decoding/DBO")
        if family == "oss":
            # The ROCm implementation calls GEMM directly, bypassing mlp.router.
            from vllm.platforms import current_platform
            if current_platform.is_rocm():
                raise ValueError("GPT-OSS identity guiding requires the native router call (CUDA)")
        if condition not in ("baseline", "guided"):
            raise ValueError("Unknown condition")
        if hasattr(self, "identity_hooks"):
            raise RuntimeError("Identity routing already configured; start a fresh process")
        self.identity_condition = condition
        self.identity_hooks, self.identity_handles = {}, []
        if condition == "guided":
            self.identity_hooks, self.identity_handles = install(
                self.model_runner.get_model(), policy, strength, family, diagnostics=diagnostics)
        return self.identity_diagnostics()

    def identity_diagnostics(self):
        return {"condition": self.identity_condition,
                "layers": {layer: hook.snapshot() for layer, hook in self.identity_hooks.items()}}
