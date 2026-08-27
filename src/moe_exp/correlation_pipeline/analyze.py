from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr, spearmanr
from tqdm import tqdm

from moe_exp.correlation_pipeline.benchmarks import BENCHMARKS, DEFAULT_BENCHMARKS
from moe_exp.correlation_pipeline.defaults import DEFAULT_FORWARD_MODEL
from moe_exp.correlation_pipeline.features import compute_layer_features, restore_features
from moe_exp.schemas import TraceRecord

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_TARGETS = ("is_correct", "has_backtracking", "has_contradiction", "has_self_correction")
_PRIMARY_FEATURES = {
    "token_count",
    "step_count",
    "character_count",
    "router_confidence_mean_layers",
    "router_margin_mean_layers",
    "router_selected_mass_mean_layers",
    "router_boundary_margin_mean_layers",
    "router_topk_non_topk_gap_mean_layers",
    "router_switch_rate_mean_layers",
    "router_topk_overlap_mean_layers",
    "hidden_norm_mean_layers",
    "hidden_step_distance_mean_layers",
    "hidden_router_geometry_mean_layers",
}
_IDENTIFIERS = {
    "dataset",
    "problem_id",
    "source_problem_id",
    "sample_id",
    "model_id",
    "scoring_method",
    *_TARGETS,
}


def _model_slug(model: str) -> str:
    return _SAFE_FILENAME.sub("--", model).strip("-")


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _resolve_tensor(path_value: str | None, input_path: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    candidates = (path, input_path.parent / path, input_path.parent / "tensors" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Tensor referenced by {input_path} does not exist: {path_value}")


def extract_trace_features(
    trace: TraceRecord,
    *,
    input_path: Path,
    max_geometry_tokens: int,
) -> dict[str, Any]:
    compact = trace.metadata.get("correlation_features")
    compact_values = compact.get("values") if isinstance(compact, dict) else None
    compact_token_count = compact.get("token_count") if isinstance(compact, dict) else None
    row: dict[str, Any] = {
        "dataset": trace.dataset,
        "problem_id": trace.problem_id,
        "source_problem_id": trace.source_problem_id or trace.problem_id,
        "sample_id": trace.sample_id,
        "model_id": trace.model_id,
        "scoring_method": trace.scoring_method,
        "is_correct": np.nan if trace.is_correct is None else int(trace.is_correct),
        "has_backtracking": int(bool(trace.step_labels.backtracking_steps)),
        "has_contradiction": int(bool(trace.step_labels.contradiction_steps)),
        "has_self_correction": int(bool(trace.step_labels.self_correction_steps)),
        "token_count": int(compact_token_count) if compact_token_count is not None else 0,
        "step_count": len(trace.steps),
        "character_count": len(trace.cot_text),
    }
    if isinstance(compact_values, dict):
        row.update(restore_features(compact_values))
    else:
        router_path = _resolve_tensor(trace.model_logs.router_logits, input_path)
        if router_path is None:
            raise ValueError(
                f"{trace.dataset}/{trace.problem_id} has neither compact features nor a router tensor"
            )
        hidden_path = _resolve_tensor(trace.model_logs.hidden_states, input_path)
        experts_path = _resolve_tensor(trace.model_logs.selected_experts, input_path)
        router_logits = torch.load(router_path, map_location="cpu", weights_only=True)
        hidden_states = (
            torch.load(hidden_path, map_location="cpu", weights_only=True)
            if hidden_path is not None
            else None
        )
        selected_experts = (
            torch.load(experts_path, map_location="cpu", weights_only=True)
            if experts_path is not None
            else None
        )
        if router_logits.ndim != 3 or router_logits.numel() == 0:
            raise ValueError(f"Invalid router tensor at {router_path}: {tuple(router_logits.shape)}")
        row["token_count"] = int(router_logits.shape[1])
        row.update(
            compute_layer_features(
                router_logits,
                hidden_states,
                selected_experts,
                max_geometry_tokens=max_geometry_tokens,
                layer_indices=trace.model_logs.layer_indices,
            )
        )
    episode_features = trace.metadata.get("episode_features")
    if isinstance(episode_features, dict):
        for name, value in episode_features.items():
            if isinstance(value, (int, float)):
                row[f"episode_{name}"] = float(value)
    return row


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in _IDENTIFIERS or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        # Confidence is 1 - normalized entropy, so reporting correlations for
        # both would duplicate the same evidence with the opposite sign.
        if column.startswith("router_entropy_"):
            confidence = column.replace("router_entropy_", "router_confidence_", 1)
            if confidence in frame.columns:
                continue
        columns.append(column)
    return columns


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 4 or np.unique(y).size < 2 or np.std(x) == 0:
        return float("nan"), float("nan")
    result = pointbiserialr(y, x)
    return float(result.statistic), float(result.pvalue)


def _bootstrap_ci(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    grouped_indices = [group.index.to_numpy() for _, group in frame.groupby("source_problem_id")]
    if len(grouped_indices) < 2 or samples < 1:
        return None, None
    values: list[float] = []
    x_all = pd.to_numeric(frame[feature], errors="coerce")
    y_all = pd.to_numeric(frame[target], errors="coerce")
    for _ in range(samples):
        selected_groups = rng.integers(0, len(grouped_indices), len(grouped_indices))
        indices = np.concatenate([grouped_indices[index] for index in selected_groups])
        x = x_all.loc[indices].to_numpy(dtype=np.float64)
        y = y_all.loc[indices].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        correlation, _ = _correlation(x[valid], y[valid])
        if np.isfinite(correlation):
            values.append(correlation)
    if len(values) < max(20, samples // 2):
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _binary_correlations(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scopes = [("all", frame)] + [
        (dataset, subset) for dataset, subset in frame.groupby("dataset", sort=True)
    ]
    for scope, subset in scopes:
        for target in _TARGETS:
            target_results: list[dict[str, Any]] = []
            for feature in feature_columns:
                x = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=np.float64)
                y = pd.to_numeric(subset[target], errors="coerce").to_numpy(dtype=np.float64)
                valid = np.isfinite(x) & np.isfinite(y)
                correlation, naive_p_value = _correlation(x[valid], y[valid])
                if not np.isfinite(correlation):
                    continue
                target_results.append(
                    {
                        "scope": scope,
                        "target": target,
                        "feature": feature,
                        "n_traces": int(valid.sum()),
                        "n_problem_clusters": int(subset.loc[valid, "source_problem_id"].nunique()),
                        "point_biserial_r": correlation,
                        "naive_independent_p_value": naive_p_value,
                        "cluster_bootstrap_ci": None,
                    }
                )
            for item in target_results:
                if item["feature"] not in _PRIMARY_FEATURES:
                    continue
                ci_low, ci_high = _bootstrap_ci(
                    subset,
                    feature=item["feature"],
                    target=target,
                    samples=bootstrap_samples,
                    rng=rng,
                )
                if ci_low is not None:
                    item["cluster_bootstrap_ci"] = [ci_low, ci_high]
            target_results.sort(key=lambda item: abs(item["point_biserial_r"]), reverse=True)
            results.extend(target_results)
    return results


def _repeated_problem_analysis(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, list[dict[str, Any]]]:
    problem_level: list[dict[str, Any]] = []
    within_problem: list[dict[str, Any]] = []
    scored = frame[np.isfinite(pd.to_numeric(frame["is_correct"], errors="coerce"))].copy()
    for dataset, dataset_frame in scored.groupby("dataset", sort=True):
        group_sizes = dataset_frame.groupby("source_problem_id").size()
        repeated_ids = group_sizes[group_sizes > 1].index
        repeated = dataset_frame[dataset_frame["source_problem_id"].isin(repeated_ids)]
        if repeated.empty:
            continue
        for feature in feature_columns:
            summaries: list[dict[str, float]] = []
            contrasts: list[float] = []
            for _, group in repeated.groupby("source_problem_id"):
                values = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=np.float64)
                outcomes = group["is_correct"].to_numpy(dtype=np.float64)
                valid = np.isfinite(values) & np.isfinite(outcomes)
                values = values[valid]
                outcomes = outcomes[valid]
                if len(values) < 2:
                    continue
                summaries.append(
                    {
                        "correct_rate": float(outcomes.mean()),
                        "feature_mean": float(values.mean()),
                        "feature_std": float(values.std()),
                    }
                )
                if np.unique(outcomes).size == 2:
                    contrasts.append(float(values[outcomes == 1].mean() - values[outcomes == 0].mean()))

            if len(summaries) >= 4:
                correct_rates = np.asarray([summary["correct_rate"] for summary in summaries])
                for statistic in ("feature_mean", "feature_std"):
                    feature_values = np.asarray([summary[statistic] for summary in summaries])
                    if np.std(correct_rates) > 0 and np.std(feature_values) > 0:
                        result = spearmanr(correct_rates, feature_values)
                        problem_level.append(
                            {
                                "dataset": dataset,
                                "feature": feature,
                                "problem_feature": statistic,
                                "n_problems": len(summaries),
                                "spearman_rho": float(result.statistic),
                                "naive_independent_p_value": float(result.pvalue),
                            }
                        )
            if contrasts and dataset != "gpqa_diamond":
                contrast_array = np.asarray(contrasts)
                bootstrap_means = [
                    float(rng.choice(contrast_array, size=len(contrast_array), replace=True).mean())
                    for _ in range(bootstrap_samples)
                ]
                within_problem.append(
                    {
                        "dataset": dataset,
                        "feature": feature,
                        "n_mixed_outcome_problems": len(contrasts),
                        "mean_correct_minus_incorrect": float(contrast_array.mean()),
                        "problem_bootstrap_ci": [
                            float(np.quantile(bootstrap_means, 0.025)),
                            float(np.quantile(bootstrap_means, 0.975)),
                        ],
                    }
                )
    problem_level.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)
    within_problem.sort(key=lambda item: abs(item["mean_correct_minus_incorrect"]), reverse=True)
    return {
        "problem_level": problem_level,
        "within_problem": within_problem,
        "within_problem_exclusions": {
            "gpqa_diamond": (
                "Excluded because each repeated attempt permutes answer positions; "
                "correct-minus-incorrect contrasts would conflate reasoning with position bias."
            )
        },
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    forward_root = args.forward_dir / _model_slug(args.model_id)
    rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    for dataset in args.datasets:
        input_path = forward_root / dataset / "traces_with_routing.jsonl"
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing forward traces for {dataset}: {input_path}")
        traces = [
            TraceRecord(**json.loads(line))
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit is not None:
            traces = traces[: args.limit]
        dataset_counts[dataset] = len(traces)
        for trace in tqdm(traces, desc=f"Features {dataset}"):
            rows.append(
                extract_trace_features(
                    trace,
                    input_path=input_path,
                    max_geometry_tokens=args.max_geometry_tokens,
                )
            )
    if not rows:
        raise RuntimeError("No traces were available for correlation analysis")

    frame = pd.DataFrame(rows)
    feature_columns = _feature_columns(frame)
    output_root = args.output_dir / _model_slug(args.model_id)
    output_root.mkdir(parents=True, exist_ok=True)
    features_path = output_root / "trace_features.csv"
    frame.to_csv(features_path, index=False)
    rng = np.random.default_rng(args.seed)
    binary = _binary_correlations(
        frame,
        feature_columns=feature_columns,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    repeated = _repeated_problem_analysis(
        frame,
        feature_columns=feature_columns,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    result = {
        "status": "complete",
        "model_id": args.model_id,
        "datasets": dataset_counts,
        "n_traces": len(frame),
        "n_problem_clusters": int(frame["source_problem_id"].nunique()),
        "n_features": len(feature_columns),
        "feature_table": features_path.as_posix(),
        "methodology": {
            "binary_correlation": "point-biserial correlation",
            "uncertainty": "problem-cluster bootstrap on prespecified aggregate features",
            "repeated_attempts": (
                "problem-level Spearman correlation and within-problem correct-minus-incorrect contrasts"
            ),
            "warning": (
                "Naive p-values are retained for diagnostics only; repeated attempts are not independent. "
                "Layer-specific features are exploratory and do not receive confidence intervals."
            ),
        },
        "binary_correlations": binary,
        "repeated_problem_analysis": repeated,
    }
    _write_json_atomic(result, output_root / "correlations.json")
    logger.info("Correlation analysis complete: %s", output_root)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze correctness, reasoning-event, routing, and hidden-state correlations"
    )
    parser.add_argument("--model-id", default=DEFAULT_FORWARD_MODEL)
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--forward-dir",
        type=Path,
        default=Path("results/correlation_pipeline/forward"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/correlation_pipeline/analysis"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-geometry-tokens", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.max_geometry_tokens < 3:
        raise ValueError("--max-geometry-tokens must be at least 3")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    analyze(args)


if __name__ == "__main__":
    main()
