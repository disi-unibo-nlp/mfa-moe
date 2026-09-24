from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def problem_key(row: dict) -> str:
    identity = row.get("source_problem_id") or row.get("problem_id") or row.get("id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Each row needs source_problem_id, problem_id, or id")
    return json.dumps([row.get("dataset", ""), identity], separators=(",", ":"))


def fit(rows, *, model: str, num_experts: int, top_k: int, tensor_base_dir=Path("."),
        min_support: int = 4, max_experts: int | None = None, expert_polarity: str = "positive",
        guiding_method: str = "fixed", paper_epsilon: float = 0.01) -> dict:
    """Estimate accuracy weighted by per-trace any-top-k frequency, per layer.

    Repeated attempts receive equal weight within each problem. Long traces do
    not dominate simply because they contain more routing decisions.
    """
    import torch

    validate_options(expert_polarity, guiding_method, paper_epsilon)
    direction = 1 if expert_polarity == "positive" else -1
    rows = list(rows)
    if not rows or not model or not 1 <= top_k <= num_experts:
        raise ValueError("Need calibration rows, model, and valid expert counts")
    if max_experts is None:
        max_experts = top_k
    if min_support < 1 or not 1 <= max_experts <= top_k:
        raise ValueError("min_support must be positive; max_experts must be in [1, top_k]")
    if any(type(row.get("is_correct")) is not bool for row in rows):
        raise ValueError("Every calibration trace must have boolean is_correct; score it first")
    if {row["is_correct"] for row in rows} != {False, True}:
        raise ValueError("Calibration needs both correct and incorrect answers to estimate lift")
    sources = {row.get("model_id") for row in rows if row.get("model_id")}
    if len(sources) > 1:
        raise ValueError("Do not mix generation models in calibration")
    attempts = Counter(problem_key(row) for row in rows)
    totals, correct, support, layers_seen = {}, {}, {}, None
    baseline = sum(row["is_correct"] / attempts[problem_key(row)] for row in rows) / len(attempts)
    digests, extraction_configs = [], []
    for row in rows:
        logs = row.get("model_logs") or {}
        selected = row.get("selected_experts", logs.get("selected_experts"))
        if isinstance(selected, (str, Path)):
            path = Path(selected)
            if not path.is_absolute():
                path = Path(tensor_base_dir) / path
            sidecar = path.with_name(path.name.replace("_experts.pt", "_extraction.json"))
            if sidecar != path and sidecar.is_file():
                config = json.loads(sidecar.read_text()).get("config", {})
                extraction_configs.append({key: config.get(key) for key in
                                           ("model_id", "quantization", "router_capture_version")})
            selected = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(selected, torch.Tensor) or selected.ndim != 3 or not selected.numel():
            raise ValueError("Need nonempty [layers, tokens, top_k] selected_experts tensor")
        if selected.dtype == torch.bool or selected.is_complex():
            raise ValueError("Expert IDs must be nonnegative integers")
        if not torch.isfinite(selected).all() or (selected < 0).any():
            raise ValueError("Expert IDs must be finite and nonnegative")
        if selected.is_floating_point() and not torch.equal(selected, selected.round()):
            raise ValueError("Expert IDs must be integers")
        selected = selected.cpu().long()
        if selected.shape[-1] != top_k or selected.max().item() >= num_experts:
            raise ValueError("Calibration tensor does not match num_experts/top_k")
        if (selected.sort(-1).values.diff(dim=-1) == 0).any():
            raise ValueError("Repeated expert IDs within a top-k pool")
        layer_ids = row.get("layer_indices") or logs.get("layer_indices")
        layers = tuple(range(selected.shape[0])) if layer_ids is None else tuple(layer_ids)
        if len(layers) != selected.shape[0] or any(type(i) is not int for i in layers):
            raise ValueError("layer_indices must match tensor layers")
        if len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
            raise ValueError("Invalid layer_indices")
        if layers_seen is not None and layers != layers_seen:
            raise ValueError("All calibration traces must cover the same ordered layers")
        layers_seen = layers
        key = problem_key(row)
        digests.append(hashlib.sha256(selected.contiguous().numpy().tobytes()).hexdigest())
        for index, layer in enumerate(layers):
            rates = torch.bincount(selected[index].flatten(), minlength=num_experts).double()
            rates /= selected.shape[1]
            if layer not in totals:
                totals[layer] = torch.zeros(num_experts, dtype=torch.float64)
                correct[layer] = torch.zeros(num_experts, dtype=torch.float64)
                support[layer] = [set() for _ in range(num_experts)]
            totals[layer] += rates / attempts[key]
            correct[layer] += rates * row["is_correct"] / attempts[key]
            for expert in rates.nonzero().flatten().tolist():
                support[layer][expert].add(key)
    result = {}
    for layer in sorted(totals):
        stats = []
        for expert in range(num_experts):
            mass = totals[layer][expert].item()
            accuracy = correct[layer][expert].item() / mass if mass else None
            stats.append({"expert": expert, "problem_support": len(support[layer][expert]),
                          "frequency_mass": mass, "accuracy": accuracy,
                          "lift": accuracy - baseline if accuracy is not None else None})
        eligible = [s for s in stats if s["problem_support"] >= min_support
                    and s["lift"] is not None and direction * s["lift"] > 0]
        eligible.sort(key=lambda s: (-direction * s["lift"], -s["problem_support"], s["expert"]))
        chosen = eligible[:max_experts]
        peak = chosen[0]["lift"] if chosen else 1.0
        scores = [0.0] * num_experts
        for s in chosen:
            scores[s["expert"]] = s["lift"] / peak
        result[str(layer)] = {"scores": scores, "experts": stats}
    if not any(any(info["scores"]) for info in result.values()):
        raise ValueError(f"No supported {expert_polarity}-accuracy identities; no guided policy was created")
    return {"schema_version": 1, "model": model, "num_experts": num_experts, "top_k": top_k,
            "expert_polarity": expert_polarity, "guiding_method": guiding_method,
            "paper_epsilon": paper_epsilon, "generation_models": sorted(sources), "baseline_accuracy": baseline,
            "estimator": "problem_balanced_trace_frequency_weighted_accuracy_lift",
            "min_support": min_support, "max_experts": max_experts,
            "calibration_problems": sorted(attempts), "num_traces": len(rows),
            "tensor_sha256": digests,
            "extraction_configs": [json.loads(s) for s in sorted({
                json.dumps(config, sort_keys=True) for config in extraction_configs})],
            "layers": result}


def validate_policy(policy: dict, model: str | None = None) -> None:
    if policy.get("schema_version") != 1 or not policy.get("model"):
        raise ValueError("Unsupported or incomplete identity policy")
    if model is not None and policy["model"] != model:
        raise ValueError("Policy model must exactly match --model; expert IDs are checkpoint-specific")
    validate_options(policy.get("expert_polarity", "positive"),
                     policy.get("guiding_method", "fixed"), policy.get("paper_epsilon", 0.01))
    n, k = policy.get("num_experts"), policy.get("top_k")
    if type(n) is not int or type(k) is not int or not 1 <= k <= n:
        raise ValueError("Invalid policy expert counts")
    if not policy.get("layers"):
        raise ValueError("Policy must contain layers")
    for layer, info in policy["layers"].items():
        if str(int(layer)) != layer or int(layer) < 0:
            raise ValueError("Policy layer IDs must be canonical nonnegative integers")
        scores = info["scores"]
        if len(scores) != n or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Policy scores must have E finite values in [0, 1]")
        if sum(s > 0 for s in scores) > k:
            raise ValueError(
                f"Layer {layer} targets more experts than model top_k={k}; "
                "refit the policy in a new output directory before generating")


def validate_options(expert_polarity, guiding_method, paper_epsilon):
    if expert_polarity not in ("positive", "negative"):
        raise ValueError("expert_polarity must be positive or negative")
    if guiding_method not in ("fixed", "paper"):
        raise ValueError("guiding_method must be fixed or paper")
    if not isinstance(paper_epsilon, (int, float)) or not math.isfinite(paper_epsilon) or paper_epsilon <= 0:
        raise ValueError("paper_epsilon must be finite and positive")
