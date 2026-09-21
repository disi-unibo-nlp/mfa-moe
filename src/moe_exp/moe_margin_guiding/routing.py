from __future__ import annotations

import math
import re

import torch

from .calibration import validate_policy
from moe_exp.moe_identity_guiding.routing import routing_spec


class MarginGuide:
    """Move a softmax probability gap toward a calibrated interval.

    Convex mixing with uniform probabilities decreases the gap. Mixing with
    uniform mass on the top-r experts increases it. Both preserve rank in exact
    arithmetic away from endpoints, without assigning preferred expert IDs.
    """

    def __init__(self, target, strength, top_k, num_experts,
                 metric="router_boundary_margin", diagnostics="full"):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be finite and in [0, 1]")
        if not 1 <= top_k < num_experts or metric not in ("router_margin", "router_boundary_margin"):
            raise ValueError("Invalid expert counts or margin metric")
        self.diagnostics = diagnostics
        self.rank = top_k if metric == "router_boundary_margin" else 1
        self.lower, self.upper = target["lower"], target["upper"]
        if (not all(math.isfinite(v) for v in (self.lower, self.upper))
                or not 0 <= self.lower <= self.upper <= 1 / self.rank):
            raise ValueError("Invalid target margin range")
        self.strength, self.top_k, self.num_experts = strength, top_k, num_experts
        self.reset()

    def reset(self):
        self.calls = self.tokens = 0
        self.totals = None

    def __call__(self, module, args, output):
        logits = output[0] if isinstance(output, tuple) else output
        if logits.ndim != 2 or logits.shape[-1] != self.num_experts:
            raise ValueError("Gate output does not match policy expert count")
        probabilities = logits.float().softmax(-1)
        top = probabilities.topk(self.rank + 1, dim=-1)
        margin = top.values[:, self.rank - 1] - top.values[:, self.rank]
        target = margin.clamp(self.lower, self.upper)
        desired = margin + self.strength * (target - margin)
        decrease = (margin - desired).clamp_min(0) / margin.clamp_min(1e-30)
        increase = (desired - margin).clamp_min(0) / (1 / self.rank - margin).clamp_min(1e-30)
        # Avoid the singular all-uniform / top-r-only endpoint, retaining tiny
        # positive mass for every expert. Native dtype rounding can still tie.
        decrease = decrease.clamp(0, 1 - 1e-6)
        increase = increase.clamp(0, 1 - 1e-6)
        anchor = torch.zeros_like(probabilities).scatter_(
            -1, top.indices[:, :self.rank], 1 / self.rank)
        adjusted = (probabilities * (1 - decrease[:, None] - increase[:, None])
                    + decrease[:, None] / self.num_experts + increase[:, None] * anchor)
        changed = (desired != margin)
        candidate = adjusted.clamp_min(torch.finfo(torch.float32).tiny).log().to(logits.dtype)
        # Keep unmodified rows bit-identical, including strength=0 controls.
        guided = torch.where(changed[:, None], candidate, logits)
        self.calls += 1
        self.tokens += logits.shape[0]
        if self.diagnostics == "full":
            post = guided.float().softmax(-1).topk(self.rank + 1, dim=-1).values
            after = post[:, self.rank - 1] - post[:, self.rank]
            old_ids = logits.topk(self.top_k, dim=-1).indices.sort(-1).values
            new_ids = guided.topk(self.top_k, dim=-1).indices.sort(-1).values
            before_distance = (margin - target).abs()
            after_distance = (after - after.clamp(self.lower, self.upper)).abs()
            totals = torch.stack((changed.sum(), (old_ids != new_ids).any(-1).sum(),
                                  margin.double().sum(), after.double().sum(),
                                  before_distance.double().sum(), after_distance.double().sum(),
                                  (guided != logits).any(-1).sum())).detach()
            self.totals = totals if self.totals is None else self.totals + totals
        if self.strength == 0:
            return output
        return (guided, *output[1:]) if isinstance(output, tuple) else guided

    def snapshot(self):
        if self.diagnostics != "full":
            return dict(calls=self.calls, token_evaluations=self.tokens,
                        diagnostics="minimal", target_range=[self.lower, self.upper])
        values = [0] * 7 if self.totals is None else self.totals.tolist()
        return dict(calls=self.calls, token_evaluations=self.tokens,
                    interventions=int(values[0]), changed_topk_sets=int(values[1]),
                    mean_margin_before=values[2] / self.tokens if self.tokens else None,
                    mean_margin_after=values[3] / self.tokens if self.tokens else None,
                    mean_distance_before=values[4] / self.tokens if self.tokens else None,
                    mean_distance_after=values[5] / self.tokens if self.tokens else None,
                    changed_logit_rows=int(values[6]),
                    target_range=[self.lower, self.upper])

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
            hook = MarginGuide(policy["layers"][layer], strength, policy["top_k"],
                               policy["num_experts"], policy["metric"], diagnostics=diagnostics)
            handles.append(gate.register_forward_hook(hook))
            hooks[layer] = hook
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return hooks, handles


class MarginWorkerExtension:
    def margin_configure(self, policy: dict, strength: float, condition: str, diagnostics="minimal"):
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
                raise ValueError("GPT-OSS margin guiding requires the native router call (CUDA)")
        if condition not in ("baseline", "guided"):
            raise ValueError("Unknown condition")
        if hasattr(self, "margin_hooks"):
            raise RuntimeError("Margin routing already configured; start a fresh process")
        self.margin_condition = condition
        self.margin_hooks, self.margin_handles = {}, []
        if condition == "guided":
            self.margin_hooks, self.margin_handles = install(
                self.model_runner.get_model(), policy, strength, family, diagnostics=diagnostics)
        return self.margin_diagnostics()

    def margin_diagnostics(self):
        return {"condition": self.margin_condition,
                "layers": {layer: hook.snapshot() for layer, hook in self.margin_hooks.items()}}
