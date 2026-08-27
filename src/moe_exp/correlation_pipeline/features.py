from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_SCHEMA_VERSION = 2


def _mean_or_nan(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() else float("nan")


def _geometry_correlation(
    hidden: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    max_tokens: int,
) -> float:
    token_count = hidden.shape[0]
    if token_count < 3:
        return float("nan")
    if token_count > max_tokens:
        indices = torch.linspace(0, token_count - 1, max_tokens).round().long().unique()
        hidden = hidden.index_select(0, indices)
        probabilities = probabilities.index_select(0, indices)
    hidden = F.normalize(hidden.to(torch.float32), p=2, dim=-1)
    probabilities = F.normalize(probabilities.to(torch.float32), p=2, dim=-1)
    hidden_similarity = hidden @ hidden.transpose(0, 1)
    router_similarity = probabilities @ probabilities.transpose(0, 1)
    upper = torch.triu_indices(hidden.shape[0], hidden.shape[0], offset=1)
    hidden_values = hidden_similarity[upper[0], upper[1]].cpu().numpy()
    router_values = router_similarity[upper[0], upper[1]].cpu().numpy()
    if np.std(hidden_values) == 0 or np.std(router_values) == 0:
        return float("nan")
    return float(np.corrcoef(hidden_values, router_values)[0, 1])


def compute_layer_features(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor | None,
    selected_experts: torch.Tensor | None,
    *,
    max_geometry_tokens: int,
    layer_indices: list[int] | None = None,
) -> dict[str, float]:
    """Reduce per-token router/hidden tensors to per-trace scalar features.

    The reduction is deliberately performed before serialization in the
    correlation forward stage. This keeps the statistics used by the analysis
    without retaining the large ``layers x tokens x width`` tensors.
    """
    if router_logits.ndim != 3 or router_logits.numel() == 0:
        raise ValueError(f"Expected non-empty 3D router logits, got {tuple(router_logits.shape)}")
    if max_geometry_tokens < 3:
        raise ValueError("max_geometry_tokens must be at least 3")

    features: dict[str, float] = {}
    num_layers, token_count, num_experts = router_logits.shape
    if hidden_states is not None and hidden_states.shape[:2] != router_logits.shape[:2]:
        raise ValueError(
            f"Hidden/router shape mismatch: {tuple(hidden_states.shape)} vs "
            f"{tuple(router_logits.shape)}"
        )
    if selected_experts is not None and selected_experts.shape[:2] != router_logits.shape[:2]:
        raise ValueError("Selected-expert and router tensor shapes do not align")

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if len(layer_indices) != num_layers:
        raise ValueError("Saved layer_indices do not match tensor dimension 0")

    top_k = int(selected_experts.shape[-1]) if selected_experts is not None else 1
    if top_k < 1 or top_k > num_experts:
        raise ValueError(f"Invalid top_k={top_k} for {num_experts} experts")

    aggregate: dict[str, list[float]] = {}
    for layer, original_layer in enumerate(layer_indices):
        probabilities = F.softmax(router_logits[layer].to(torch.float32), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = (
            1.0 - entropy / math.log(num_experts)
            if num_experts > 1
            else torch.ones_like(entropy)
        )

        sorted_probabilities = torch.topk(
            probabilities,
            k=min(top_k + 1, num_experts),
            dim=-1,
        ).values
        selected_values = sorted_probabilities[:, :top_k]
        selected_mass = selected_values.sum(dim=-1)
        if top_k < num_experts:
            boundary_margin = selected_values[:, -1] - sorted_probabilities[:, top_k]
            non_selected_mean = (1.0 - selected_mass) / (num_experts - top_k)
            topk_non_topk_gap = selected_values.mean(dim=-1) - non_selected_mean
        else:
            boundary_margin = selected_values[:, -1]
            topk_non_topk_gap = selected_values.mean(dim=-1)

        top_two = torch.topk(probabilities, k=min(2, num_experts), dim=-1).values
        margin = top_two[:, 0] - top_two[:, 1] if num_experts > 1 else top_two[:, 0]
        top_one = probabilities.argmax(dim=-1)
        switch_rate = (
            _mean_or_nan((top_one[1:] != top_one[:-1]).to(torch.float32))
            if token_count > 1
            else float("nan")
        )
        if selected_experts is not None and token_count > 1:
            previous = selected_experts[layer, :-1].to(torch.long)
            current = selected_experts[layer, 1:].to(torch.long)
            intersection = (
                previous.unsqueeze(-1) == current.unsqueeze(-2)
            ).any(dim=-1).sum(dim=-1)
            topk_overlap = _mean_or_nan(intersection.to(torch.float32) / previous.shape[-1])
        else:
            topk_overlap = float("nan")

        layer_values = {
            "router_entropy": _mean_or_nan(entropy),
            "router_confidence": _mean_or_nan(confidence),
            # Kept for continuity with the pilot table; the three top-k-aware
            # features below are the prespecified confidence measures.
            "router_margin": _mean_or_nan(margin),
            "router_selected_mass": _mean_or_nan(selected_mass),
            "router_boundary_margin": _mean_or_nan(boundary_margin),
            "router_topk_non_topk_gap": _mean_or_nan(topk_non_topk_gap),
            "router_switch_rate": switch_rate,
            "router_topk_overlap": topk_overlap,
        }
        if hidden_states is not None:
            hidden = hidden_states[layer].to(torch.float32)
            hidden_norm = hidden.norm(dim=-1)
            if token_count > 1:
                trajectory_distance = 1.0 - F.cosine_similarity(
                    hidden[1:], hidden[:-1], dim=-1
                )
                hidden_step_distance = _mean_or_nan(trajectory_distance)
            else:
                hidden_step_distance = float("nan")
            layer_values.update(
                {
                    "hidden_norm": _mean_or_nan(hidden_norm),
                    "hidden_step_distance": hidden_step_distance,
                    "hidden_router_geometry": _geometry_correlation(
                        hidden,
                        probabilities,
                        max_tokens=max_geometry_tokens,
                    ),
                }
            )

        for name, value in layer_values.items():
            features[f"{name}_l{original_layer:02d}"] = value
            aggregate.setdefault(name, []).append(value)

    for name, values in aggregate.items():
        array = np.asarray(values, dtype=np.float64)
        features[f"{name}_mean_layers"] = (
            float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")
        )
        features[f"{name}_std_layers"] = (
            float(np.nanstd(array)) if np.isfinite(array).any() else float("nan")
        )
    return features


def json_safe_features(features: dict[str, float]) -> dict[str, float | None]:
    """Convert NaN/inf feature values to JSON null for portable checkpoints."""
    return {
        name: float(value) if math.isfinite(float(value)) else None
        for name, value in features.items()
    }


def restore_features(features: dict[str, Any]) -> dict[str, float]:
    """Restore JSON feature values, mapping null back to NaN for pandas."""
    restored: dict[str, float] = {}
    for name, value in features.items():
        if value is None:
            restored[name] = float("nan")
        elif isinstance(value, (int, float)):
            restored[name] = float(value)
    return restored
