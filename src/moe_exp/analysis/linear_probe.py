"""Layerwise linear probes over trace-pooled hidden and/or router states.

Each trace contributes exactly one sample per layer.  Cross-validation is
therefore grouped by construction: no token from a held-out trace can appear in
the training set.  This module is deliberately a decoding analysis, not a
causal test and not an online failure predictor (the full trace is pooled).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moe_exp.schemas import TraceRecord


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FeatureSource = Literal["router", "hidden", "combined"]
TARGETS = ("correctness", "first_error", "contradiction", "backtracking", "self_correction")


def router_features(router_logits: torch.Tensor) -> np.ndarray:
    """Return one feature vector per layer from an ``(L,T,E)`` tensor.

    Features are mean/std router probability (2E), top-1 expert frequency (E),
    entropy mean/std, margin mean/std, and top-1 switch rate: 3E+5 values.
    Sequence length itself is intentionally excluded.
    """
    logits = router_logits.to(torch.float32)
    probs = F.softmax(logits, dim=-1)
    mean_probs = probs.mean(dim=1)
    std_probs = probs.std(dim=1, unbiased=False)

    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    entropy_stats = torch.stack(
        (entropy.mean(dim=1), entropy.std(dim=1, unbiased=False)), dim=-1
    )
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    margin_stats = torch.stack(
        (margin.mean(dim=1), margin.std(dim=1, unbiased=False)), dim=-1
    )

    top1 = probs.argmax(dim=-1)
    top1_frequency = F.one_hot(top1, num_classes=probs.shape[-1]).float().mean(dim=1)
    if top1.shape[1] > 1:
        switch_rate = (top1[:, 1:] != top1[:, :-1]).float().mean(dim=1, keepdim=True)
    else:
        switch_rate = torch.zeros((top1.shape[0], 1), dtype=torch.float32)

    features = torch.cat(
        (
            mean_probs,
            std_probs,
            top1_frequency,
            entropy_stats,
            margin_stats,
            switch_rate,
        ),
        dim=-1,
    )
    return features.cpu().numpy().astype(np.float32, copy=False)


def hidden_features(hidden_states: torch.Tensor) -> np.ndarray:
    """Mean-pool an ``(L,T,H)`` hidden-state tensor into one sample per layer."""
    return hidden_states.to(torch.float32).mean(dim=1).cpu().numpy().astype(np.float32, copy=False)


def _target_value(trace: TraceRecord, target: str) -> int | None:
    labels = trace.step_labels
    if target == "correctness":
        return None if trace.is_correct is None else int(trace.is_correct)
    if target == "first_error":
        return int(labels.first_error_step is not None)
    if target == "contradiction":
        return int(bool(labels.contradiction_steps))
    if target == "backtracking":
        return int(bool(labels.backtracking_steps))
    if target == "self_correction":
        return int(bool(labels.self_correction_steps))
    raise ValueError(f"Unknown target: {target}")


def _load_feature_tensor(trace: TraceRecord, source: FeatureSource) -> np.ndarray | None:
    router: np.ndarray | None = None
    hidden: np.ndarray | None = None
    if source in ("router", "combined"):
        path = trace.model_logs.router_logits
        if path is None or not Path(path).exists():
            return None
        router = router_features(torch.load(path, map_location="cpu", weights_only=True))
    if source in ("hidden", "combined"):
        path = trace.model_logs.hidden_states
        if path is None or not Path(path).exists():
            return None
        hidden = hidden_features(torch.load(path, map_location="cpu", weights_only=True))
    if router is None:
        return hidden
    if hidden is None:
        return router
    if router.shape[0] != hidden.shape[0]:
        raise ValueError(
            f"Layer mismatch for {trace.dataset}/{trace.problem_id}: "
            f"router={router.shape[0]}, hidden={hidden.shape[0]}"
        )
    return np.concatenate((hidden, router), axis=-1)


def _bootstrap_intervals(
    y: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float]]:
    """Stratified trace-level bootstrap intervals over out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    draws: dict[str, list[float]] = {"auroc": [], "balanced_accuracy": [], "f1": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        draws["auroc"].append(float(roc_auc_score(y[indices], probabilities[indices])))
        draws["balanced_accuracy"].append(
            float(balanced_accuracy_score(y[indices], predictions[indices]))
        )
        draws["f1"].append(float(f1_score(y[indices], predictions[indices], zero_division=0)))
    return {
        name: [float(x) for x in np.quantile(values, (0.025, 0.975))]
        for name, values in draws.items()
    }


def evaluate_layer(
    x: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Generate out-of-fold predictions and trace-bootstrap their metrics."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(y), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=seed,
            ),
        )
        classifier.fit(x[train], y[train])
        fold_probabilities = classifier.predict_proba(x[test])[:, 1]
        probabilities[test] = fold_probabilities
        fold_predictions = (fold_probabilities >= 0.5).astype(np.int8)
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(test)),
                "auroc": float(roc_auc_score(y[test], fold_probabilities)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], fold_predictions)),
                "f1": float(f1_score(y[test], fold_predictions, zero_division=0)),
            }
        )

    predictions = (probabilities >= 0.5).astype(np.int8)
    metrics = {
        "auroc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }
    return {
        "metrics": metrics,
        "bootstrap_ci_95": _bootstrap_intervals(
            y, probabilities, predictions, bootstrap_samples, seed
        ),
        "folds": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-source", choices=("router", "hidden", "combined"), default="router"
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=["correctness"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--minimum-class-count",
        type=int,
        default=10,
        help="Skip targets with fewer examples in either class (default: 10)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    traces: list[TraceRecord] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(TraceRecord.model_validate_json(line))
    if args.limit is not None:
        traces = traces[: args.limit]

    features: list[np.ndarray] = []
    kept_traces: list[TraceRecord] = []
    for index, trace in enumerate(traces, start=1):
        feature = _load_feature_tensor(trace, args.feature_source)
        if feature is not None:
            features.append(feature)
            kept_traces.append(trace)
        if index % 500 == 0:
            logger.info("Loaded features for %d/%d traces", len(features), index)
    if not features:
        raise RuntimeError(
            f"No {args.feature_source} features were found. Hidden/combined probes require "
            "model_logs.hidden_states paths produced by a GPU extraction."
        )
    layer_counts = {feature.shape[0] for feature in features}
    feature_counts = {feature.shape[1] for feature in features}
    if len(layer_counts) != 1 or len(feature_counts) != 1:
        raise ValueError(
            f"Inconsistent feature shapes: layers={sorted(layer_counts)}, "
            f"dimensions={sorted(feature_counts)}"
        )
    x_all = np.stack(features, axis=0)  # (N,L,D)

    target_results: dict[str, Any] = {}
    for target in args.targets:
        raw_labels = [_target_value(trace, target) for trace in kept_traces]
        mask = np.asarray([label is not None for label in raw_labels])
        y = np.asarray([label for label in raw_labels if label is not None], dtype=np.int8)
        counts = np.bincount(y, minlength=2)
        if counts.min() < args.minimum_class_count:
            target_results[target] = {
                "status": "skipped",
                "reason": "insufficient examples in at least one class",
                "class_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
            }
            continue
        folds = min(args.folds, int(counts.min()))
        x = x_all[mask]
        layer_rows = []
        for layer in range(x.shape[1]):
            logger.info("Probing target=%s layer=%d", target, layer)
            row = evaluate_layer(x[:, layer, :], y, folds, args.seed, args.bootstrap_samples)
            row["layer"] = layer
            layer_rows.append(row)
        target_results[target] = {
            "status": "complete",
            "n_samples": int(len(y)),
            "class_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
            "n_folds": folds,
            "layers": layer_rows,
        }

    output = {
        "config": {
            "input": args.input.as_posix(),
            "feature_source": args.feature_source,
            "unit": "one mean-pooled sample per trace per layer",
            "uses_full_trace": True,
            "causal_or_online": False,
            "n_input_traces": len(traces),
            "n_feature_traces": len(kept_traces),
            "num_layers": int(x_all.shape[1]),
            "features_per_layer": int(x_all.shape[2]),
            "folds_requested": args.folds,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "targets": target_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    logger.info("Probe results saved to %s", args.output)


if __name__ == "__main__":
    main()
