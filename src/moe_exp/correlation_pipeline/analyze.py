from __future__ import annotations

import argparse
import hashlib
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
from moe_exp.correlation_pipeline.expert_analysis import (
    ExpertAnalysisConfig,
    analyze_expert_feature_table,
    build_expert_feature_table,
)
from moe_exp.correlation_pipeline.features import compute_layer_features, restore_features
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_EVALUATION_METRIC = re.compile(r"^(?:avg|pass)@(\d+)$", re.IGNORECASE)
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
    "evaluation_metric",
    "generation_sha256",
    "generation_finish_reason",
    "generation_completion_tokens",
    "generation_max_tokens",
    "generation_hit_token_limit",
    "generation_has_final_content",
    "invalid_answer_counted_incorrect",
    "router_num_layers",
    "router_num_experts",
    "router_top_k",
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
    usage = trace.metadata.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    generation_config = trace.metadata.get("generation_config")
    generation_config = generation_config if isinstance(generation_config, dict) else {}
    finish_reason = trace.metadata.get("finish_reason")
    completion_tokens = usage.get("completion_tokens")
    max_tokens = generation_config.get("max_tokens")
    has_budget_values = isinstance(completion_tokens, (int, float)) and isinstance(
        max_tokens, (int, float)
    )
    hit_token_limit: float | int = np.nan
    if finish_reason is not None or has_budget_values:
        hit_token_limit = int(
            finish_reason == "length"
            or (has_budget_values and completion_tokens >= max_tokens)
        )
    has_final_content: float | int = np.nan
    if "assistant_content" in trace.metadata:
        has_final_content = int(bool(str(trace.metadata.get("assistant_content") or "").strip()))
    has_scoring_target = bool(trace.gold_answer.strip()) and (
        trace.scoring_method != "unscored_no_gold_answer"
    )
    invalid_answer_counted_incorrect = bool(
        trace.is_correct is None and has_scoring_target
    )
    analysis_correctness: float | int = (
        0
        if invalid_answer_counted_incorrect
        else np.nan
        if trace.is_correct is None
        else int(trace.is_correct)
    )
    row: dict[str, Any] = {
        "dataset": trace.dataset,
        "problem_id": trace.problem_id,
        "source_problem_id": trace.source_problem_id or trace.problem_id,
        "sample_id": trace.sample_id,
        "model_id": trace.model_id,
        "scoring_method": trace.scoring_method,
        "evaluation_metric": trace.metadata.get("evaluation_metric"),
        "generation_sha256": hashlib.sha256(trace.cot_text.encode("utf-8")).hexdigest(),
        "generation_finish_reason": finish_reason,
        "generation_completion_tokens": completion_tokens,
        "generation_max_tokens": max_tokens,
        "generation_hit_token_limit": hit_token_limit,
        "generation_has_final_content": has_final_content,
        "invalid_answer_counted_incorrect": int(invalid_answer_counted_incorrect),
        "router_num_layers": compact.get("num_layers") if isinstance(compact, dict) else None,
        "router_num_experts": compact.get("num_experts") if isinstance(compact, dict) else None,
        "router_top_k": compact.get("top_k") if isinstance(compact, dict) else None,
        "is_correct": analysis_correctness,
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
        row["router_num_layers"] = int(router_logits.shape[0])
        row["router_num_experts"] = int(router_logits.shape[2])
        if selected_experts is not None:
            row["router_top_k"] = int(selected_experts.shape[-1])
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


def _generation_budget_audit(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "dataset",
        "is_correct",
        "generation_finish_reason",
        "generation_completion_tokens",
        "generation_max_tokens",
        "generation_hit_token_limit",
        "generation_has_final_content",
    }
    if not required.issubset(frame.columns):
        return {
            "status": "unavailable",
            "reason": "generation metadata is absent from the feature table",
            "scopes": [],
        }

    scopes = [("all", frame)] + [
        (dataset, subset) for dataset, subset in frame.groupby("dataset", sort=True)
    ]
    rows: list[dict[str, Any]] = []
    for scope, subset in scopes:
        hit_limit = pd.to_numeric(
            subset["generation_hit_token_limit"], errors="coerce"
        )
        known_limit = hit_limit.notna()
        limited = known_limit & hit_limit.astype("Int64").eq(1)
        completed = known_limit & hit_limit.astype("Int64").eq(0)
        correct = pd.to_numeric(subset["is_correct"], errors="coerce")
        final_content = pd.to_numeric(
            subset["generation_has_final_content"], errors="coerce"
        )
        invalid_answers = pd.to_numeric(
            subset.get(
                "invalid_answer_counted_incorrect",
                pd.Series(index=subset.index, dtype=float),
            ),
            errors="coerce",
        )
        completion_tokens = pd.to_numeric(
            subset["generation_completion_tokens"], errors="coerce"
        )
        max_tokens = pd.to_numeric(subset["generation_max_tokens"], errors="coerce")
        comparable_budget = completion_tokens.notna() & max_tokens.notna()
        reasons = subset["generation_finish_reason"].dropna().astype(str).value_counts()
        rate = float(limited.sum() / known_limit.sum()) if known_limit.any() else None
        rows.append(
            {
                "scope": scope,
                "n_traces": len(subset),
                "n_with_limit_status": int(known_limit.sum()),
                "finish_reason_counts": {
                    name: int(count) for name, count in reasons.items()
                },
                "max_tokens": sorted(
                    int(value) for value in max_tokens.dropna().unique().tolist()
                ),
                "token_limit_hits": int(limited.sum()),
                "token_limit_hit_rate": rate,
                "completion_tokens_at_or_above_budget": int(
                    (comparable_budget & completion_tokens.ge(max_tokens)).sum()
                ),
                "traces_without_final_content": int(final_content.eq(0).sum()),
                "invalid_answers_counted_incorrect": int(invalid_answers.eq(1).sum()),
                "limited_traces_without_final_content": int(
                    (limited & final_content.eq(0)).sum()
                ),
                "limited_correct_without_final_content": int(
                    (limited & final_content.eq(0) & correct.eq(1)).sum()
                ),
                "accuracy_all": (
                    float(correct.dropna().mean()) if correct.notna().any() else None
                ),
                "accuracy_limited": (
                    float(correct[limited].dropna().mean())
                    if correct[limited].notna().any()
                    else None
                ),
                "accuracy_completed": (
                    float(correct[completed].dropna().mean())
                    if correct[completed].notna().any()
                    else None
                ),
                "severity": (
                    "unknown"
                    if rate is None
                    else "frequent_limit_hits"
                    if rate > 0.05
                    else "rare_limit_hits"
                    if rate > 0
                    else "none"
                ),
            }
        )
    overall = rows[0]
    return {
        "status": overall["severity"],
        "threshold_for_frequent_limit_hits": 0.05,
        "interpretation": (
            "A token-limit hit is finish_reason=length or completion_tokens >= max_tokens. "
            "Frequent hits censor response length and can induce associations with correctness; "
            "such runs require a larger feasible budget and regeneration before natural-length "
            "claims are made."
        ),
        "scopes": rows,
    }


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


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _benjamini_hochberg(p_values: list[float]) -> list[float | None]:
    """Return Benjamini-Hochberg adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return [None] * len(values)
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=np.float64)
    ranked_adjusted = ranked * len(ranked) / ranks
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    restored = np.empty_like(ranked_adjusted)
    restored[order] = np.clip(ranked_adjusted, 0.0, 1.0)
    adjusted[finite_indices] = restored
    return [float(value) if np.isfinite(value) else None for value in adjusted]


def _expected_attempts(group: pd.DataFrame, dataset: str) -> tuple[int | None, str | None]:
    metrics: set[str] = set()
    if "evaluation_metric" in group.columns:
        metrics = {
            str(value).strip()
            for value in group["evaluation_metric"].dropna().tolist()
            if str(value).strip()
        }
    parsed = {
        int(match.group(1))
        for metric in metrics
        if (match := _EVALUATION_METRIC.fullmatch(metric)) is not None
    }
    if len(metrics) == 1 and len(parsed) == 1:
        expected = next(iter(parsed))
        return (expected, next(iter(metrics))) if expected > 0 else (None, next(iter(metrics)))
    if metrics:
        return None, ",".join(sorted(metrics))
    spec = BENCHMARKS.get(dataset)
    if spec is None:
        return None, None
    expected = int(spec.default_samples)
    return expected, f"avg@{expected}" if expected > 1 else "pass@1"


def _problem_level_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build one auditable row per problem without feature-specific target leakage."""
    rows: list[dict[str, Any]] = []
    for (dataset, source_problem_id), group in frame.groupby(
        ["dataset", "source_problem_id"], sort=True, dropna=False
    ):
        dataset = str(dataset)
        expected, evaluation_metric = _expected_attempts(group, dataset)
        sample_values = pd.to_numeric(group["sample_id"], errors="coerce")
        finite_samples = sample_values[np.isfinite(sample_values)].astype(int)
        sample_counts = finite_samples.value_counts()
        unique_samples = {int(value) for value in finite_samples.unique()}
        expected_samples = set(range(expected)) if expected is not None else set()
        missing_samples = sorted(expected_samples - unique_samples)
        unexpected_samples = (
            sorted(unique_samples - expected_samples) if expected is not None else []
        )
        duplicate_samples = sorted(int(value) for value in sample_counts[sample_counts > 1].index)
        generation_hashes = (
            group["generation_sha256"].dropna().astype(str)
            if "generation_sha256" in group.columns
            else pd.Series(dtype=str)
        )
        n_generation_hashes = len(generation_hashes)
        duplicate_generation_rows = int(
            n_generation_hashes - generation_hashes.nunique()
        )
        generation_duplicate_check_available = n_generation_hashes == len(group)
        independent_generation_group = bool(
            not generation_duplicate_check_available or duplicate_generation_rows == 0
        )
        n_rows = len(group)
        n_unique_samples = len(unique_samples)
        complete_attempt_group = bool(
            expected is not None
            and n_rows == expected
            and n_unique_samples == expected
            and not duplicate_samples
            and not missing_samples
            and not unexpected_samples
        )
        correctness = pd.to_numeric(group["is_correct"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        n_binary_outcomes = int(np.isfinite(correctness).sum())
        complete_binary_outcomes = bool(
            complete_attempt_group and n_binary_outcomes == expected
        )
        row: dict[str, Any] = {
            "dataset": dataset,
            "source_problem_id": source_problem_id,
            "evaluation_metric": evaluation_metric,
            "expected_attempts": expected,
            "observed_rows": n_rows,
            "unique_sample_ids": n_unique_samples,
            "duplicate_rows": n_rows - n_unique_samples,
            "duplicate_sample_ids": ";".join(map(str, duplicate_samples)),
            "missing_sample_ids": ";".join(map(str, missing_samples)),
            "unexpected_sample_ids": ";".join(map(str, unexpected_samples)),
            "binary_outcomes": n_binary_outcomes,
            "generation_duplicate_check_available": generation_duplicate_check_available,
            "unique_generation_hashes": int(generation_hashes.nunique()),
            "duplicate_generation_rows": duplicate_generation_rows,
            "independent_generation_group": independent_generation_group,
            "complete_attempt_group": complete_attempt_group,
            "complete_binary_outcomes": complete_binary_outcomes,
            "eligible_for_repeated_analysis": bool(
                complete_binary_outcomes
                and independent_generation_group
                and expected is not None
                and expected > 1
            ),
            "mean_correctness": (
                float(correctness.mean()) if complete_binary_outcomes else np.nan
            ),
        }
        for feature in feature_columns:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            n_valid = int(finite.sum())
            full_feature_coverage = bool(
                complete_binary_outcomes and expected is not None and n_valid == expected
            )
            row[f"{feature}__n_valid"] = n_valid
            row[f"{feature}__mean"] = float(values.mean()) if full_feature_coverage else np.nan
            row[f"{feature}__std"] = float(values.std()) if full_feature_coverage else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _repeated_group_audit(problem_frame: pd.DataFrame) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for dataset, subset in problem_frame.groupby("dataset", sort=True):
        repeated = subset[
            pd.to_numeric(subset["expected_attempts"], errors="coerce").gt(1)
            | pd.to_numeric(subset["observed_rows"], errors="coerce").gt(1)
        ]
        if repeated.empty:
            continue
        structurally_complete = repeated["complete_attempt_group"].astype(bool)
        complete_outcomes = repeated["complete_binary_outcomes"].astype(bool)
        independent_generations = repeated["independent_generation_group"].astype(bool)
        eligible = repeated["eligible_for_repeated_analysis"].astype(bool)
        duplicates = pd.to_numeric(repeated["duplicate_rows"], errors="coerce").fillna(0)
        metrics = sorted(str(value) for value in repeated["evaluation_metric"].dropna().unique())
        excluded = repeated[~eligible]
        audits.append(
            {
                "dataset": dataset,
                "evaluation_metrics": metrics,
                "expected_attempts": sorted(
                    int(value)
                    for value in pd.to_numeric(
                        repeated["expected_attempts"], errors="coerce"
                    ).dropna().unique()
                ),
                "problems": len(repeated),
                "complete_attempt_groups": int(structurally_complete.sum()),
                "complete_binary_outcome_groups": int(complete_outcomes.sum()),
                "eligible_repeated_analysis_groups": int(eligible.sum()),
                "incomplete_or_invalid_groups": int((~eligible).sum()),
                "groups_with_duplicate_sample_ids": int(duplicates.gt(0).sum()),
                "duplicate_rows": int(duplicates.sum()),
                "groups_missing_binary_outcomes": int(
                    (structurally_complete & ~complete_outcomes).sum()
                ),
                "groups_with_exact_duplicate_generations": int(
                    (~independent_generations).sum()
                ),
                "duplicate_generation_rows": int(
                    pd.to_numeric(
                        repeated["duplicate_generation_rows"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "generation_duplicate_check_available_for_all_groups": bool(
                    repeated["generation_duplicate_check_available"].astype(bool).all()
                ),
                "excluded_problem_examples": [
                    str(value) for value in excluded["source_problem_id"].head(20).tolist()
                ],
            }
        )
    return audits


def _bootstrap_ci(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    strata: list[list[np.ndarray]] = []
    for _, dataset_frame in frame.groupby("dataset", sort=False):
        groups = [
            group.index.to_numpy()
            for _, group in dataset_frame.groupby("source_problem_id", sort=False)
        ]
        if groups:
            strata.append(groups)
    n_clusters = sum(len(groups) for groups in strata)
    if n_clusters < 2 or samples < 1:
        return None, None
    values: list[float] = []
    x_all = pd.to_numeric(frame[feature], errors="coerce")
    y_all = pd.to_numeric(frame[target], errors="coerce")
    for _ in range(samples):
        indices = np.concatenate(
            [
                groups[index]
                for groups in strata
                for index in rng.integers(0, len(groups), len(groups))
            ]
        )
        x = x_all.loc[indices].to_numpy(dtype=np.float64)
        y = y_all.loc[indices].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        correlation, _ = _correlation(x[valid], y[valid])
        if np.isfinite(correlation):
            values.append(correlation)
    if len(values) < max(20, samples // 2):
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 4 or samples < 1:
        return None, None
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        correlation, _ = _spearman_correlation(x[indices], y[indices])
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
                        "n_problem_clusters": int(
                            subset.loc[valid, ["dataset", "source_problem_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
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
            adjusted = _benjamini_hochberg(
                [item["naive_independent_p_value"] for item in target_results]
            )
            for item, q_value in zip(target_results, adjusted, strict=True):
                item["benjamini_hochberg_q_value"] = q_value
            target_results.sort(key=lambda item: abs(item["point_biserial_r"]), reverse=True)
            results.extend(target_results)
    return results


def _repeated_problem_analysis(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
    problem_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if problem_frame is None:
        problem_frame = _problem_level_table(frame, feature_columns=feature_columns)
    problem_level: list[dict[str, Any]] = []
    within_problem: list[dict[str, Any]] = []
    for dataset, dataset_problems in problem_frame.groupby("dataset", sort=True):
        repeated_problems = dataset_problems[
            pd.to_numeric(dataset_problems["expected_attempts"], errors="coerce").gt(1)
            | pd.to_numeric(dataset_problems["observed_rows"], errors="coerce").gt(1)
        ]
        if repeated_problems.empty:
            continue
        eligible_problems = repeated_problems[
            repeated_problems["eligible_for_repeated_analysis"].astype(bool)
        ]
        expected_values = sorted(
            int(value)
            for value in pd.to_numeric(
                repeated_problems["expected_attempts"], errors="coerce"
            ).dropna().unique()
        )
        evaluation_metrics = sorted(
            str(value)
            for value in repeated_problems["evaluation_metric"].dropna().unique()
        )
        for feature in feature_columns:
            target = pd.to_numeric(
                eligible_problems["mean_correctness"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            for statistic, suffix in (("feature_mean", "mean"), ("feature_std", "std")):
                predictor = pd.to_numeric(
                    eligible_problems[f"{feature}__{suffix}"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                valid = np.isfinite(target) & np.isfinite(predictor)
                correlation, p_value = _spearman_correlation(
                    predictor[valid], target[valid]
                )
                if not np.isfinite(correlation):
                    continue
                ci: list[float] | None = None
                if feature in _PRIMARY_FEATURES and statistic == "feature_mean":
                    ci_low, ci_high = _bootstrap_spearman_ci(
                        predictor[valid],
                        target[valid],
                        samples=bootstrap_samples,
                        rng=rng,
                    )
                    if ci_low is not None:
                        ci = [ci_low, ci_high]
                problem_level.append(
                    {
                        "dataset": dataset,
                        "evaluation_metrics": evaluation_metrics,
                        "expected_attempts": expected_values,
                        "target": "mean_correctness",
                        "feature": feature,
                        "problem_feature": statistic,
                        "analysis_role": (
                            "primary" if statistic == "feature_mean" else "exploratory"
                        ),
                        "n_total_repeated_problems": len(repeated_problems),
                        "n_complete_outcome_problems": len(eligible_problems),
                        "n_problems": int(valid.sum()),
                        "spearman_rho": correlation,
                        "p_value": p_value,
                        "problem_bootstrap_ci": ci,
                    }
                )

            if dataset == "gpqa_diamond":
                continue
            eligible_ids = set(eligible_problems["source_problem_id"].tolist())
            dataset_frame = frame[
                frame["dataset"].eq(dataset)
                & frame["source_problem_id"].isin(eligible_ids)
            ]
            contrasts: list[float] = []
            for _, group in dataset_frame.groupby("source_problem_id", sort=False):
                values = pd.to_numeric(group[feature], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                outcomes = pd.to_numeric(group["is_correct"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                if not (np.isfinite(values).all() and np.isfinite(outcomes).all()):
                    continue
                if np.unique(outcomes).size == 2:
                    contrasts.append(
                        float(values[outcomes == 1].mean() - values[outcomes == 0].mean())
                    )
            if contrasts:
                contrast_array = np.asarray(contrasts)
                bootstrap_means = [
                    float(rng.choice(contrast_array, size=len(contrast_array), replace=True).mean())
                    for _ in range(bootstrap_samples)
                ]
                within_problem.append(
                    {
                        "dataset": dataset,
                        "evaluation_metrics": evaluation_metrics,
                        "expected_attempts": expected_values,
                        "feature": feature,
                        "n_mixed_outcome_problems": len(contrasts),
                        "mean_correct_minus_incorrect": float(contrast_array.mean()),
                        "problem_bootstrap_ci": [
                            float(np.quantile(bootstrap_means, 0.025)),
                            float(np.quantile(bootstrap_means, 0.975)),
                        ],
                    }
                )

    for problem_feature in ("feature_mean", "feature_std"):
        for dataset in sorted({item["dataset"] for item in problem_level}):
            family = [
                item
                for item in problem_level
                if item["dataset"] == dataset
                and item["problem_feature"] == problem_feature
            ]
            adjusted = _benjamini_hochberg([item["p_value"] for item in family])
            for item, q_value in zip(family, adjusted, strict=True):
                item["benjamini_hochberg_q_value"] = q_value
    problem_level.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)
    within_problem.sort(key=lambda item: abs(item["mean_correct_minus_incorrect"]), reverse=True)
    return {
        "group_completeness": _repeated_group_audit(problem_frame),
        "problem_level": problem_level,
        "within_problem": within_problem,
        "within_problem_exclusions": {
            "gpqa_diamond": (
                "Excluded because each repeated attempt permutes answer positions; "
                "correct-minus-incorrect contrasts would conflate reasoning with position bias."
            )
        },
    }


def _cross_feature_correlations(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    problem_frame: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Spearman cross-correlations for prespecified across-layer features."""
    primary = sorted(set(feature_columns) & _PRIMARY_FEATURES)
    trace_level: list[dict[str, Any]] = []
    scopes = [("all", frame)] + [
        (dataset, subset) for dataset, subset in frame.groupby("dataset", sort=True)
    ]
    for scope, subset in scopes:
        family: list[dict[str, Any]] = []
        for left_index, left in enumerate(primary):
            for right in primary[left_index + 1 :]:
                x = pd.to_numeric(subset[left], errors="coerce").to_numpy(dtype=np.float64)
                y = pd.to_numeric(subset[right], errors="coerce").to_numpy(dtype=np.float64)
                valid = np.isfinite(x) & np.isfinite(y)
                correlation, p_value = _spearman_correlation(x[valid], y[valid])
                if not np.isfinite(correlation):
                    continue
                family.append(
                    {
                        "scope": scope,
                        "analysis_unit": "trace",
                        "feature_x": left,
                        "feature_y": right,
                        "n_traces": int(valid.sum()),
                        "n_problem_clusters": int(
                            subset.loc[valid, ["dataset", "source_problem_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "spearman_rho": correlation,
                        "naive_independent_p_value": p_value,
                    }
                )
        adjusted = _benjamini_hochberg(
            [item["naive_independent_p_value"] for item in family]
        )
        for item, q_value in zip(family, adjusted, strict=True):
            item["benjamini_hochberg_q_value"] = q_value
        family.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)
        trace_level.extend(family)

    problem_level: list[dict[str, Any]] = []
    for dataset, subset in problem_frame.groupby("dataset", sort=True):
        repeated = subset[subset["eligible_for_repeated_analysis"].astype(bool)]
        if repeated.empty:
            continue
        family = []
        for left_index, left in enumerate(primary):
            for right in primary[left_index + 1 :]:
                x = pd.to_numeric(repeated[f"{left}__mean"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                y = pd.to_numeric(repeated[f"{right}__mean"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                valid = np.isfinite(x) & np.isfinite(y)
                correlation, p_value = _spearman_correlation(x[valid], y[valid])
                if not np.isfinite(correlation):
                    continue
                family.append(
                    {
                        "dataset": dataset,
                        "analysis_unit": "problem",
                        "evaluation_metrics": sorted(
                            str(value)
                            for value in repeated["evaluation_metric"].dropna().unique()
                        ),
                        "feature_x": left,
                        "feature_y": right,
                        "n_problems": int(valid.sum()),
                        "spearman_rho": correlation,
                        "p_value": p_value,
                    }
                )
        adjusted = _benjamini_hochberg([item["p_value"] for item in family])
        for item, q_value in zip(family, adjusted, strict=True):
            item["benjamini_hochberg_q_value"] = q_value
        family.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)
        problem_level.extend(family)
    return {"trace_level": trace_level, "repeated_problem_level": problem_level}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    from moe_exp.correlation_pipeline.view_analysis import analyze_views, collect_view_rows

    forward_root = args.forward_dir / _model_slug(args.model_id)
    rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    excluded_truncated: dict[str, int] = {}
    expert_rows: list[dict[str, Any]] = []
    traces_without_expert_tensors = 0
    dataset_counts: dict[str, int] = {}
    view_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    view_contract = None
    for dataset in args.datasets:
        input_path = forward_root / dataset / "traces_with_routing.jsonl"
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing forward traces for {dataset}: {input_path}")
        traces = [TraceRecord(**record) for record in iter_jsonl(input_path)]
        if args.limit is not None:
            traces = traces[: args.limit]
        dataset_counts[dataset] = 0
        excluded_truncated[dataset] = 0
        for trace in tqdm(traces, desc=f"Features {dataset}"):
            row = extract_trace_features(
                trace,
                input_path=input_path,
                max_geometry_tokens=args.max_geometry_tokens,
            )
            budget_rows.append(row)
            if row["generation_hit_token_limit"] == 1:
                excluded_truncated[dataset] += 1
                continue
            dataset_counts[dataset] += 1
            rows.append(row)
            if getattr(args, "views", None):
                scoped_rows, contract = collect_view_rows(trace, row, args.views)
                if view_contract is not None and contract != view_contract:
                    raise ValueError("Cannot mix different classifier/position-reference contracts")
                view_contract = contract
                for key, scoped_row in scoped_rows:
                    view_rows.setdefault(key, []).append(scoped_row)
            selected_experts = _resolve_tensor(trace.model_logs.selected_experts, input_path)
            if selected_experts is None:
                traces_without_expert_tensors += 1
                continue
            expert_rows.append(
                {
                    "dataset": row["dataset"],
                    "source_problem_id": row["source_problem_id"],
                    "sample_id": row["sample_id"],
                    "selected_experts": selected_experts,
                    "layer_indices": trace.model_logs.layer_indices,
                    "num_experts": row["router_num_experts"],
                    "targets": {target: row[target] for target in _TARGETS},
                    "trace_features": {
                        feature: row.get(feature, np.nan) for feature in _PRIMARY_FEATURES
                    },
                }
            )
    if not rows:
        raise RuntimeError("No traces remain for correlation analysis after excluding token-limit generations")

    frame = pd.DataFrame(rows)
    feature_columns = _feature_columns(frame)
    output_root = args.output_dir / _model_slug(args.model_id)
    output_root.mkdir(parents=True, exist_ok=True)
    features_path = output_root / "trace_features.csv"
    frame.to_csv(features_path, index=False)
    problem_frame = _problem_level_table(frame, feature_columns=feature_columns)
    problem_features_path = output_root / "problem_features.csv"
    problem_frame.to_csv(problem_features_path, index=False)
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
        problem_frame=problem_frame,
    )
    cross_feature = _cross_feature_correlations(
        frame,
        feature_columns=feature_columns,
        problem_frame=problem_frame,
    )
    if getattr(args, "skip_expert_identity", False):
        expert_identity: dict[str, Any] = {"status": "skipped"}
    elif traces_without_expert_tensors:
        expert_identity = {
            "status": "unavailable",
            "reason": (
                "Expert identity requires a selected_experts tensor for every trace; "
                f"{traces_without_expert_tensors} of {len(frame)} traces are missing one."
            ),
        }
    else:
        expert_config = ExpertAnalysisConfig(
            min_trace_support=getattr(args, "expert_min_trace_support", 4),
            max_pairs=getattr(args, "max_expert_pairs", 128),
        )
        expert_continuous_targets = sorted(set(feature_columns) & _PRIMARY_FEATURES)
        expert_table = build_expert_feature_table(
            expert_rows,
            target_names=(*_TARGETS, *expert_continuous_targets),
            config=expert_config,
        )
        expert_features_path = output_root / "expert_trace_features.csv"
        expert_table.frame.to_csv(expert_features_path, index=False)
        expert_identity = analyze_expert_feature_table(
            expert_table,
            binary_targets=_TARGETS,
            continuous_targets=expert_continuous_targets,
            problem_table=problem_frame,
            problem_target="mean_correctness",
            config=expert_config,
        )
        expert_identity["status"] = "complete"
        expert_identity["feature_table"] = expert_features_path.as_posix()
    generation_budget_audit = _generation_budget_audit(pd.DataFrame(budget_rows))
    generation_budget_audit["population"] = "all input traces before truncation exclusion"
    result = {
        "status": "complete",
        "model_id": args.model_id,
        "datasets": dataset_counts,
        "n_traces": len(frame),
        "excluded_truncated_generations": excluded_truncated,
        "n_problem_clusters": int(
            frame[["dataset", "source_problem_id"]].drop_duplicates().shape[0]
        ),
        "n_features": len(feature_columns),
        "feature_table": features_path.as_posix(),
        "problem_feature_table": problem_features_path.as_posix(),
        "analysis_config": {
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "expert_min_trace_support": getattr(args, "expert_min_trace_support", 4),
            "max_expert_pairs": getattr(args, "max_expert_pairs", 128),
        },
        "methodology": {
            "truncation": (
                "Exclude attempts with finish_reason=length or completion_tokens >= max_tokens "
                "from all correlation, expert, and view analyses. Unknown limit status is retained. "
                "Problem-level avg@n still requires all declared n attempts after exclusion. "
                "The generation-budget audit includes excluded attempts."
            ),
            "binary_correlation": "point-biserial correlation",
            "binary_target_encoding": (
                "Targets are encoded false=0 and true=1. Positive r means the feature is "
                "higher in the true class; negative r means it is lower. scipy.pointbiserialr "
                "is called as pointbiserialr(target, feature), exactly matching Pearson r for "
                "this binary encoding. An attempt with a known gold answer but no parseable "
                "answer is counted as incorrect; genuinely unscored tasks remain missing."
            ),
            "uncertainty": (
                "problem-cluster bootstrap for prespecified trace-level aggregates and "
                "whole-problem bootstrap for prespecified avg@n Spearman coefficients"
            ),
            "repeated_attempts": (
                "Canonical correctness and every predictor are aggregated once per problem. "
                "Only complete groups with exactly the declared avg@n unique sample IDs and "
                "complete binary outcomes enter problem-level Spearman correlations."
            ),
            "cross_features": (
                "Spearman rank correlation among prespecified across-layer features. "
                "Trace-level repeated-attempt p-values are diagnostic; a separate problem-level "
                "table averages complete avg@n groups."
            ),
            "multiplicity": (
                "Benjamini-Hochberg q-values are computed separately within each scope, target, "
                "aggregation statistic, or cross-feature family."
            ),
            "literature_basis": {
                "binary": (
                    "Liu et al. (NAACL 2022) directly use point-biserial correlation for "
                    "continuous quantities against binary model correctness."
                ),
                "graded_or_aggregated": (
                    "Hong et al. (Findings ACL 2025), Xu et al. (PNAS 2025), and Hewitt and "
                    "Manning (NAACL 2019) provide analogous precedent for Spearman correlation "
                    "between representation-derived quantities and graded or ordinal targets; "
                    "none uniquely establishes Spearman for avg@32."
                ),
                "uncertainty": (
                    "Xu et al. use 10,000 bootstrap samples for one category-feature analysis; "
                    "the number of resamples here remains an explicit configurable parameter."
                ),
            },
            "warning": (
                "Naive trace-level p-values are retained for diagnostics only; repeated attempts "
                "are not independent. Layer-specific features are exploratory and do not receive "
                "confidence intervals."
            ),
        },
        "binary_correlations": binary,
        "repeated_problem_analysis": repeated,
        "cross_feature_correlations": cross_feature,
        "expert_identity_analysis": expert_identity,
        "generation_budget_audit": generation_budget_audit,
    }
    if view_rows:
        result["reasoning_view_analysis"] = analyze_views(
            view_rows, output_root, bootstrap_samples=args.bootstrap_samples,
            seed=args.seed, contract=view_contract,
        )
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
    parser.add_argument("--views", nargs="+", choices=("full", "class", "position"), default=None,
                        help="Analyze versioned reasoning views saved by forward --views")
    parser.add_argument("--max-geometry-tokens", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--expert-min-trace-support",
        type=int,
        default=4,
        help="Minimum number of traces selecting an expert or top1->top2 pair.",
    )
    parser.add_argument(
        "--max-expert-pairs",
        type=int,
        default=128,
        help="Outcome-independent cap on ordered top1->top2 pair features.",
    )
    parser.add_argument(
        "--skip-expert-identity",
        action="store_true",
        help="Skip expert identity/pair analysis for legacy runs without expert tensors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.max_geometry_tokens < 3:
        raise ValueError("--max-geometry-tokens must be at least 3")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    if args.expert_min_trace_support < 1:
        raise ValueError("--expert-min-trace-support must be at least 1")
    if args.max_expert_pairs < 0:
        raise ValueError("--max-expert-pairs cannot be negative")
    analyze(args)


if __name__ == "__main__":
    main()
