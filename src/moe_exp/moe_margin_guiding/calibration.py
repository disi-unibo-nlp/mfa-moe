from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from moe_exp.moe_identity_guiding.calibration import problem_key

METRICS = ("router_boundary_margin", "router_margin")


def read_margins(row, metric, num_experts, top_k, tensor_base_dir):
    """Read completion-average, per-layer margins from cached features or logits."""
    metadata = row.get("metadata") or {}
    cached = metadata.get("correlation_features")
    logs = row.get("model_logs") or {}
    if cached:
        if (cached.get("num_experts"), cached.get("top_k")) != (num_experts, top_k):
            raise ValueError("Cached feature expert counts differ from policy")
        layers = cached.get("layer_indices")
        if layers is None:
            layers = list(range(cached["num_layers"]))
        values = cached["values"]
        margins = [values[f"{metric}_l{layer:02d}"] for layer in layers]
        provenance = {"source": "correlation_features", "config": cached.get("config", {}),
                      "sha256": hashlib.sha256(json.dumps(cached, sort_keys=True).encode()).hexdigest()}
    else:
        import torch

        source = row.get("router_logits", logs.get("router_logits"))
        if isinstance(source, (str, Path)):
            path = Path(tensor_base_dir) / source
            source = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(source, torch.Tensor) or source.ndim != 3 or not source.numel():
            raise ValueError("Need cached correlation_features or [layers,tokens,experts] logits")
        if source.shape[-1] != num_experts or not torch.isfinite(source).all():
            raise ValueError("Invalid router logits or expert count")
        source = source.cpu().float()
        rank = top_k if metric == "router_boundary_margin" else 1
        probs = source.softmax(-1).topk(rank + 1, dim=-1).values
        margins = (probs[..., rank - 1] - probs[..., rank]).mean(-1).tolist()
        layers = row.get("layer_indices") or logs.get("layer_indices")
        if layers is None:
            layers = list(range(source.shape[0]))
        provenance = {"source": "router_logits", "sha256": hashlib.sha256(
            source.contiguous().numpy().tobytes()).hexdigest()}
    if (len(layers) != len(margins) or not layers or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)):
        raise ValueError("Invalid calibration layer indices")
    rank = top_k if metric == "router_boundary_margin" else 1
    if any(not math.isfinite(m) or not 0 <= m <= 1 / rank + 1e-6 for m in margins):
        raise ValueError("Margins must be finite, nonnegative softmax probability gaps")
    return dict(zip(map(str, layers), (min(m, 1 / rank) for m in margins))), provenance


def fit(rows, *, model, num_experts, top_k, tensor_base_dir=Path("."),
        min_support=4, bins=5, metric="router_boundary_margin"):
    """Fit supported high-accuracy ranges on problem-balanced trace averages."""
    rows = list(rows)
    if not rows or not model or not 1 <= top_k < num_experts:
        raise ValueError("Need rows, model, and 1 <= top_k < num_experts")
    if metric not in METRICS or bins < 2 or min_support < 1:
        raise ValueError("Need a supported metric, bins >= 2 and min_support >= 1")
    if any(type(row.get("is_correct")) is not bool for row in rows):
        raise ValueError("Every calibration trace needs boolean is_correct")
    if {row["is_correct"] for row in rows} != {False, True}:
        raise ValueError("Calibration needs both correct and incorrect answers")
    models = {row["model_id"] for row in rows if row.get("model_id")}
    if len(models) > 1:
        raise ValueError("Do not mix generation models in calibration")
    attempts = Counter(problem_key(row) for row in rows)
    weights = [1 / attempts[problem_key(row)] for row in rows]
    baseline = sum(w * row["is_correct"] for w, row in zip(weights, rows)) / len(attempts)
    observations, provenance, expected = [], [], None
    for row in rows:
        margins, source = read_margins(row, metric, num_experts, top_k, tensor_base_dir)
        if expected is not None and set(margins) != expected:
            raise ValueError("All calibration traces must cover the same layers")
        expected = set(margins)
        observations.append(margins)
        provenance.append(source)
    layers, analysis = {}, {}
    for layer in sorted(expected, key=int):
        values = [item[layer] for item in observations]
        ordered = sorted(zip(values, weights))
        # Weighted quantile cuts give each problem total weight one, regardless
        # of its number of attempts. Repeated values never get split across bins.
        cuts, cumulative, quantile = [], 0.0, 1
        for value, weight in ordered:
            cumulative += weight
            while quantile < bins and cumulative >= len(attempts) * quantile / bins:
                cuts.append(value)
                quantile += 1
        cuts = sorted(set(cuts) - {max(values)})
        stats = []
        for bucket in range(len(cuts) + 1):
            indices = [i for i, value in enumerate(values)
                       if bisect.bisect_left(cuts, value) == bucket]
            if not indices:
                continue
            mass = sum(weights[i] for i in indices)
            accuracy = sum(weights[i] * rows[i]["is_correct"] for i in indices) / mass
            stats.append(dict(lower=min(values[i] for i in indices),
                              upper=max(values[i] for i in indices),
                              problem_support=len({problem_key(rows[i]) for i in indices}),
                              traces=len(indices), weight=mass, accuracy=accuracy,
                              lift=accuracy - baseline))
        eligible = [s for s in stats if s["problem_support"] >= min_support and s["lift"] > 1e-12]
        analysis[layer] = {"quantile_cuts": cuts, "bins": stats}
        if eligible:
            best = max(eligible, key=lambda s: (s["accuracy"], s["problem_support"], -s["lower"]))
            layers[layer] = dict(best)
    if not layers:
        raise ValueError("No supported positive-accuracy margin ranges; no policy created")
    return dict(schema_version=1, experiment="moe_margin_guiding", model=model,
                num_experts=num_experts, top_k=top_k, metric=metric,
                estimator="problem_balanced_trace_mean_quantile_accuracy",
                intervention="probability_mixture_to_nearest_range_endpoint",
                baseline_accuracy=baseline, min_support=min_support, requested_bins=bins,
                generation_models=sorted(models), calibration_problems=sorted(attempts),
                num_traces=len(rows), feature_provenance=provenance,
                layers=layers, analysis=analysis)


def validate_policy(policy, model=None):
    if (policy.get("schema_version") != 1 or policy.get("experiment") != "moe_margin_guiding"
            or not policy.get("model") or policy.get("metric") not in METRICS):
        raise ValueError("Unsupported or incomplete margin policy")
    if model is not None and model != policy["model"]:
        raise ValueError("Policy model must exactly match --model")
    n, k = policy.get("num_experts"), policy.get("top_k")
    if type(n) is not int or type(k) is not int or not 1 <= k < n:
        raise ValueError("Invalid policy expert counts")
    if not policy.get("layers") or not isinstance(policy.get("calibration_problems"), list):
        raise ValueError("Policy needs target layers and calibration problem IDs")
    rank = k if policy["metric"] == "router_boundary_margin" else 1
    for layer, info in policy["layers"].items():
        if str(int(layer)) != layer or int(layer) < 0:
            raise ValueError("Policy layers must be canonical nonnegative integers")
        lo, hi = info["lower"], info["upper"]
        if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi <= 1 / rank:
            raise ValueError("Invalid target margin range")
