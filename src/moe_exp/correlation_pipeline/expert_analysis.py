"""Outcome-independent expert-identity features and association tests.

The ordinary correlation pipeline reduces routing tensors to entropy/margin-style
trace features.  This module provides a complementary analysis that preserves
expert identity.  Candidate experts and expert pairs are selected only from
routing support; correctness and other targets are never consulted during
feature selection.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr, spearmanr

EXPERT_ANALYSIS_SCHEMA_VERSION = 1
_TOP1 = "top1"
_ANY_TOPK = "any_topk"
_PAIR = "top1_top2_pair"
_FeatureKey = tuple[str, int, int | None]


@dataclass(frozen=True)
class ExpertAnalysisConfig:
    """Controls support filtering and the size of expert-association outputs."""

    min_trace_support: int = 4
    max_pairs: int = 128
    min_correlation_n: int = 4
    include_pooled_scope: bool = True

    def __post_init__(self) -> None:
        if self.min_trace_support < 1:
            raise ValueError("min_trace_support must be at least 1")
        if self.max_pairs < 0:
            raise ValueError("max_pairs cannot be negative")
        if self.min_correlation_n < 3:
            raise ValueError("min_correlation_n must be at least 3")


@dataclass
class ExpertFeatureTable:
    """Dense selected feature table plus auditable selection metadata."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_metadata: tuple[dict[str, Any], ...]
    selection_summary: dict[str, Any]


def _as_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    model_dump = getattr(row, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("Each expert-analysis row must be a mapping or a Pydantic model")


def _nested_value(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for container_name in ("targets", "trace_features", "features"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    return np.nan


def _selected_experts_value(row: Mapping[str, Any]) -> Any:
    value = row.get("selected_experts")
    if value is not None:
        return value
    model_logs = row.get("model_logs")
    if isinstance(model_logs, Mapping):
        return model_logs.get("selected_experts")
    return None


def _layer_indices_value(row: Mapping[str, Any], num_layers: int) -> tuple[int, ...]:
    value = row.get("layer_indices")
    if value is None:
        model_logs = row.get("model_logs")
        if isinstance(model_logs, Mapping):
            value = model_logs.get("layer_indices")
    if value is None:
        return tuple(range(num_layers))
    indices = tuple(int(index) for index in value)
    if len(indices) != num_layers:
        raise ValueError("layer_indices length does not match selected_experts")
    return indices


def _declared_num_experts(row: Mapping[str, Any], tensor: torch.Tensor) -> int | None:
    value = row.get("num_experts")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    num_experts = int(value)
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if int(tensor.max()) >= num_experts:
        raise ValueError("selected_experts contains an ID outside declared num_experts")
    return num_experts


def _load_selected_experts(value: Any, *, tensor_base_dir: Path | None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and tensor_base_dir is not None:
            path = tensor_base_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Selected-experts tensor does not exist: {path}")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, torch.Tensor):
            raise TypeError(f"Selected-experts checkpoint is not a tensor: {path}")
        tensor = loaded
    else:
        raise TypeError("A selected_experts tensor or tensor path is required for every row")

    if tensor.ndim != 3 or tensor.numel() == 0:
        raise ValueError(
            "selected_experts must be a non-empty [layers, tokens, top_k] tensor; "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype == torch.bool:
        raise ValueError("selected_experts must contain non-negative integer expert IDs")
    if torch.is_floating_point(tensor):
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
            raise ValueError("selected_experts contains non-integer or non-finite values")
        tensor = tensor.to(dtype=torch.long)
    if bool((tensor < 0).any()):
        raise ValueError("selected_experts contains a negative expert ID")
    return tensor


def _sparse_trace_counts(
    tensor: torch.Tensor,
    *,
    include_pairs: bool = True,
) -> tuple[dict[_FeatureKey, int], int, int]:
    """Reduce one routing tensor without allocating an experts-wide dense tensor."""

    flat = tensor.reshape(-1, tensor.shape[-1])
    decisions = int(flat.shape[0])
    top_k = int(flat.shape[1])
    counts: dict[_FeatureKey, int] = {}

    top1_ids, top1_counts = torch.unique(flat[:, 0], return_counts=True)
    for expert, count in zip(top1_ids.tolist(), top1_counts.tolist(), strict=True):
        counts[(_TOP1, int(expert), None)] = int(count)

    # Sorting each routing decision lets us count an expert at most once even if
    # malformed input repeats an ID within its top-k selection.
    ordered = torch.sort(flat, dim=1).values
    first_occurrence = torch.ones_like(ordered, dtype=torch.bool)
    if top_k > 1:
        first_occurrence[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    any_ids, any_counts = torch.unique(ordered[first_occurrence], return_counts=True)
    for expert, count in zip(any_ids.tolist(), any_counts.tolist(), strict=True):
        counts[(_ANY_TOPK, int(expert), None)] = int(count)

    if include_pairs and top_k >= 2:
        pairs, pair_counts = torch.unique(flat[:, :2], dim=0, return_counts=True)
        for pair, count in zip(pairs.tolist(), pair_counts.tolist(), strict=True):
            counts[(_PAIR, int(pair[0]), int(pair[1]))] = int(count)

    return counts, decisions, top_k


def _feature_name(key: _FeatureKey) -> str:
    kind, first, second = key
    if kind == _TOP1:
        return f"expert_top1_rate_e{first}"
    if kind == _ANY_TOPK:
        return f"expert_any_topk_rate_e{first}"
    if kind == _PAIR and second is not None:
        return f"expert_pair_top1_e{first}_top2_e{second}_rate"
    raise ValueError(f"Unknown expert feature key: {key}")


def build_expert_feature_table(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    target_names: Sequence[str] = (),
    config: ExpertAnalysisConfig | None = None,
    tensor_base_dir: Path | None = None,
) -> ExpertFeatureTable:
    """Derive selected per-trace expert rates while preserving input row order.

    Experts are retained when they occur anywhere in top-k in at least
    ``min_trace_support`` traces.  Both top-1 and any-top-k rates are then emitted
    for each retained expert.  Ordered top1-to-top2 pairs use the same support
    threshold and are capped by ``max_pairs``.  Ties are resolved by event count
    and expert IDs, never by an analysis target.
    """

    config = config or ExpertAnalysisConfig()
    materialized = [_as_mapping(row) for row in rows]
    if not materialized:
        raise ValueError("At least one row is required for expert-identity analysis")

    decision_counts: list[int] = []
    top_ks: list[int] = []
    layer_index_sets: list[tuple[int, ...]] = []
    declared_expert_counts: list[int | None] = []
    support: Counter[_FeatureKey] = Counter()
    event_counts: Counter[_FeatureKey] = Counter()
    observed_experts: set[int] = set()

    for row in materialized:
        tensor = _load_selected_experts(
            _selected_experts_value(row), tensor_base_dir=tensor_base_dir
        )
        counts, decisions, top_k = _sparse_trace_counts(tensor)
        decision_counts.append(decisions)
        top_ks.append(top_k)
        layer_index_sets.append(_layer_indices_value(row, int(tensor.shape[0])))
        declared_expert_counts.append(_declared_num_experts(row, tensor))
        for key, count in counts.items():
            support[key] += 1
            event_counts[key] += count
            observed_experts.add(key[1])
            if key[2] is not None:
                observed_experts.add(key[2])

    retained_experts = sorted(
        key[1]
        for key, trace_support in support.items()
        if key[0] == _ANY_TOPK and trace_support >= config.min_trace_support
    )
    retained_pairs = [
        key
        for key, trace_support in support.items()
        if key[0] == _PAIR and trace_support >= config.min_trace_support
    ]
    retained_pairs.sort(
        key=lambda key: (-support[key], -event_counts[key], key[1], int(key[2] or 0))
    )
    pair_candidates = len(retained_pairs)
    retained_pairs = retained_pairs[: config.max_pairs]

    retained_keys: list[_FeatureKey] = []
    for expert in retained_experts:
        retained_keys.extend(((_TOP1, expert, None), (_ANY_TOPK, expert, None)))
    retained_keys.extend(retained_pairs)

    target_names = tuple(dict.fromkeys(target_names))
    records: list[dict[str, Any]] = []
    # Scan again after support selection.  This keeps memory proportional to one
    # tensor plus the bounded retained feature set instead of retaining every
    # observed pair for every trace.
    for index, (row, expected_decisions, expected_top_k, layer_indices) in enumerate(
        zip(materialized, decision_counts, top_ks, layer_index_sets, strict=True)
    ):
        tensor = _load_selected_experts(
            _selected_experts_value(row), tensor_base_dir=tensor_base_dir
        )
        counts, decisions, top_k = _sparse_trace_counts(
            tensor, include_pairs=bool(retained_pairs)
        )
        if decisions != expected_decisions or top_k != expected_top_k:
            raise RuntimeError("selected_experts changed between analysis passes")
        dataset = row.get("dataset")
        source_problem_id = row.get("source_problem_id") or row.get("problem_id")
        if dataset is None or source_problem_id is None:
            raise ValueError("Every row needs dataset and source_problem_id (or problem_id)")
        record: dict[str, Any] = {
            "dataset": str(dataset),
            "source_problem_id": str(source_problem_id),
            "sample_id": row.get("sample_id", index),
            "expert_routing_decisions": decisions,
            "expert_top_k": top_k,
            "expert_layer_indices": ";".join(map(str, layer_indices)),
        }
        for target in target_names:
            record[target] = _nested_value(row, target)
        for key in retained_keys:
            # A top1->top2 pair has no defined opportunity when this trace has k=1.
            record[_feature_name(key)] = (
                np.nan
                if key[0] == _PAIR and top_k < 2
                else counts.get(key, 0) / decisions
            )
        records.append(record)

    metadata: list[dict[str, Any]] = []
    for key in retained_keys:
        kind, first, second = key
        item: dict[str, Any] = {
            "feature": _feature_name(key),
            "kind": kind,
            "expert_id": first,
            "trace_support": int(support[key]),
            "event_count": int(event_counts[key]),
        }
        if kind in {_TOP1, _ANY_TOPK}:
            item["selection_trace_support"] = int(support[(_ANY_TOPK, first, None)])
        if second is not None:
            item["top2_expert_id"] = second
        metadata.append(item)

    top_k_counts = Counter(top_ks)
    layer_index_counts = Counter(layer_index_sets)
    declared_counts = Counter(
        value for value in declared_expert_counts if value is not None
    )
    selection_summary = {
        "selection_basis": (
            "Routing support only; outcome and continuous target values are not used."
        ),
        "n_traces": len(materialized),
        "min_trace_support": config.min_trace_support,
        "max_pairs": config.max_pairs,
        "observed_expert_ids": sorted(observed_experts),
        "n_observed_experts": len(observed_experts),
        "declared_num_experts_trace_counts": {
            str(key): value for key, value in sorted(declared_counts.items())
        },
        "retained_expert_ids": retained_experts,
        "n_retained_experts": len(retained_experts),
        "pair_candidates_meeting_support": pair_candidates,
        "n_retained_pairs": len(retained_pairs),
        "pairs_truncated_by_cap": max(0, pair_candidates - len(retained_pairs)),
        "top_k_trace_counts": {str(key): value for key, value in sorted(top_k_counts.items())},
        "layer_index_set_trace_counts": {
            ";".join(map(str, key)): value
            for key, value in sorted(layer_index_counts.items())
        },
    }
    return ExpertFeatureTable(
        frame=pd.DataFrame.from_records(records),
        feature_columns=tuple(_feature_name(key) for key in retained_keys),
        feature_metadata=tuple(metadata),
        selection_summary=selection_summary,
    )


def _scopes(
    frame: pd.DataFrame, *, include_pooled: bool
) -> list[tuple[str, pd.DataFrame]]:
    scopes: list[tuple[str, pd.DataFrame]] = []
    if include_pooled:
        scopes.append(("all", frame))
    scopes.extend(
        (str(dataset), subset)
        for dataset, subset in frame.groupby("dataset", sort=True, dropna=False)
    )
    return scopes


def _bh_fdr(rows: list[dict[str, Any]], *, p_key: str = "p_value") -> None:
    """Attach Benjamini-Hochberg adjusted p-values to one test family."""

    indexed = [
        (index, float(row[p_key]))
        for index, row in enumerate(rows)
        if np.isfinite(float(row[p_key]))
    ]
    if not indexed:
        return
    indexed.sort(key=lambda item: item[1])
    count = len(indexed)
    adjusted = [0.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        _, p_value = indexed[rank_index]
        rank = rank_index + 1
        running = min(running, p_value * count / rank)
        adjusted[rank_index] = min(1.0, running)
    for (row_index, _), q_value in zip(indexed, adjusted, strict=True):
        rows[row_index]["benjamini_hochberg_q_value"] = float(q_value)


def _trace_binary_associations(
    table: ExpertFeatureTable,
    *,
    targets: Sequence[str],
    config: ExpertAnalysisConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in targets:
        all_values = pd.to_numeric(table.frame[target], errors="coerce").to_numpy(float)
        finite_values = set(np.unique(all_values[np.isfinite(all_values)]).tolist())
        if not finite_values.issubset({0.0, 1.0}):
            raise ValueError(f"Binary target {target!r} contains values outside 0/1")
        for scope, subset in _scopes(
            table.frame, include_pooled=config.include_pooled_scope
        ):
            family: list[dict[str, Any]] = []
            y_all = pd.to_numeric(subset[target], errors="coerce").to_numpy(float)
            for feature in table.feature_columns:
                x_all = pd.to_numeric(subset[feature], errors="coerce").to_numpy(float)
                valid = np.isfinite(x_all) & np.isfinite(y_all)
                x = x_all[valid]
                y = y_all[valid]
                if (
                    len(x) < config.min_correlation_n
                    or np.unique(x).size < 2
                    or np.unique(y).size != 2
                ):
                    continue
                result = pointbiserialr(y, x)
                if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
                    continue
                family.append(
                    {
                        "analysis_family": f"trace/point_biserial/{scope}/{target}",
                        "scope": scope,
                        "target": target,
                        "feature": feature,
                        "n_traces": int(valid.sum()),
                        "n_problem_clusters": int(
                            subset.loc[valid, ["dataset", "source_problem_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "n_positive": int((y == 1).sum()),
                        "n_negative": int((y == 0).sum()),
                        "point_biserial_r": float(result.statistic),
                        "naive_independent_p_value": float(result.pvalue),
                    }
                )
            _bh_fdr(family, p_key="naive_independent_p_value")
            family.sort(key=lambda item: (-abs(item["point_biserial_r"]), item["feature"]))
            output.extend(family)
    return output


def _trace_spearman_associations(
    table: ExpertFeatureTable,
    *,
    targets: Sequence[str],
    config: ExpertAnalysisConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in targets:
        for scope, subset in _scopes(
            table.frame, include_pooled=config.include_pooled_scope
        ):
            family: list[dict[str, Any]] = []
            y_all = pd.to_numeric(subset[target], errors="coerce").to_numpy(float)
            for feature in table.feature_columns:
                if feature == target:
                    continue
                x_all = pd.to_numeric(subset[feature], errors="coerce").to_numpy(float)
                valid = np.isfinite(x_all) & np.isfinite(y_all)
                x = x_all[valid]
                y = y_all[valid]
                if (
                    len(x) < config.min_correlation_n
                    or np.unique(x).size < 2
                    or np.unique(y).size < 2
                ):
                    continue
                result = spearmanr(x, y)
                if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
                    continue
                family.append(
                    {
                        "analysis_family": f"trace/spearman/{scope}/{target}",
                        "scope": scope,
                        "target": target,
                        "feature": feature,
                        "n_traces": int(valid.sum()),
                        "n_problem_clusters": int(
                            subset.loc[valid, ["dataset", "source_problem_id"]]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "spearman_rho": float(result.statistic),
                        "naive_independent_p_value": float(result.pvalue),
                    }
                )
            _bh_fdr(family, p_key="naive_independent_p_value")
            family.sort(key=lambda item: (-abs(item["spearman_rho"]), item["feature"]))
            output.extend(family)
    return output


def _problem_frame(
    table: ExpertFeatureTable,
    *,
    problem_table: Sequence[Mapping[str, Any] | Any] | pd.DataFrame | None,
    problem_target: str,
    correctness_target: str,
) -> tuple[pd.DataFrame, str]:
    keys = ["dataset", "source_problem_id"]
    feature_means = table.frame.groupby(keys, sort=True)[list(table.feature_columns)].mean()

    if problem_table is None:
        correctness = pd.to_numeric(table.frame[correctness_target], errors="coerce")
        canonical = table.frame[keys].copy()
        canonical[correctness_target] = correctness
        canonical = canonical.groupby(keys, sort=True)[correctness_target].agg(
            ["mean", "count"]
        )
        canonical = canonical.rename(
            columns={"mean": problem_target, "count": "n_scored_attempts"}
        )
        # A problem-level average is useful here only for repeated generations.
        canonical = canonical[canonical["n_scored_attempts"] > 1]
        source = f"derived canonical mean of {correctness_target} over supplied attempts"
    else:
        if isinstance(problem_table, pd.DataFrame):
            canonical = problem_table.copy()
        else:
            canonical = pd.DataFrame.from_records(
                [_as_mapping(row) for row in problem_table]
            )
        if "source_problem_id" not in canonical and "problem_id" in canonical:
            canonical["source_problem_id"] = canonical["problem_id"]
        missing = {*keys, problem_target}.difference(canonical.columns)
        if missing:
            raise ValueError(
                "Problem table is missing required columns: " + ", ".join(sorted(missing))
            )
        if canonical.duplicated(keys).any():
            raise ValueError("Problem table must contain one canonical row per dataset/problem")
        if "eligible_for_repeated_analysis" in canonical.columns:
            canonical = canonical[
                canonical["eligible_for_repeated_analysis"].fillna(False).astype(bool)
            ]
        canonical = canonical.set_index(keys)
        source = "caller-supplied canonical problem table"

    canonical[problem_target] = pd.to_numeric(canonical[problem_target], errors="coerce")
    # Wide groupby results can retain one block per feature. Consolidate before
    # reset_index inserts the problem identifiers into the joined frame.
    joined = canonical.join(feature_means, how="inner").copy()
    return joined.reset_index(), source


def _problem_spearman_associations(
    table: ExpertFeatureTable,
    *,
    problem_table: Sequence[Mapping[str, Any] | Any] | pd.DataFrame | None,
    problem_target: str,
    correctness_target: str,
    config: ExpertAnalysisConfig,
) -> tuple[list[dict[str, Any]], str]:
    frame, source = _problem_frame(
        table,
        problem_table=problem_table,
        problem_target=problem_target,
        correctness_target=correctness_target,
    )
    output: list[dict[str, Any]] = []
    for scope, subset in _scopes(frame, include_pooled=config.include_pooled_scope):
        family: list[dict[str, Any]] = []
        y_all = pd.to_numeric(subset[problem_target], errors="coerce").to_numpy(float)
        for feature in table.feature_columns:
            x_all = pd.to_numeric(subset[feature], errors="coerce").to_numpy(float)
            valid = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[valid]
            y = y_all[valid]
            if (
                len(x) < config.min_correlation_n
                or np.unique(x).size < 2
                or np.unique(y).size < 2
            ):
                continue
            result = spearmanr(x, y)
            if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
                continue
            family.append(
                {
                    "analysis_family": f"problem/spearman/{scope}/{problem_target}",
                    "scope": scope,
                    "target": problem_target,
                    "feature": feature,
                    "n_problems": int(valid.sum()),
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
        _bh_fdr(family)
        family.sort(key=lambda item: (-abs(item["spearman_rho"]), item["feature"]))
        output.extend(family)
    return output, source


def analyze_expert_feature_table(
    table: ExpertFeatureTable,
    *,
    binary_targets: Sequence[str] = ("is_correct",),
    continuous_targets: Sequence[str] = (),
    problem_table: Sequence[Mapping[str, Any] | Any] | pd.DataFrame | None = None,
    problem_target: str = "avg_correctness",
    correctness_target: str = "is_correct",
    config: ExpertAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Run expert-identity associations on an already selected feature table.

    Binary targets use point-biserial correlation and continuous targets use
    Spearman correlation. A supplied problem table must contain one canonical
    row per dataset/problem and a column named by ``problem_target``.
    """

    config = config or ExpertAnalysisConfig()
    binary_targets = tuple(dict.fromkeys(binary_targets))
    continuous_targets = tuple(dict.fromkeys(continuous_targets))
    binary = _trace_binary_associations(
        table, targets=binary_targets, config=config
    )
    continuous = _trace_spearman_associations(
        table, targets=continuous_targets, config=config
    )
    problem, problem_source = _problem_spearman_associations(
        table,
        problem_table=problem_table,
        problem_target=problem_target,
        correctness_target=correctness_target,
        config=config,
    )
    return {
        "schema_version": EXPERT_ANALYSIS_SCHEMA_VERSION,
        "config": asdict(config),
        "selection": table.selection_summary,
        "feature_metadata": list(table.feature_metadata),
        "trace_point_biserial": binary,
        "trace_spearman": continuous,
        "problem_spearman": problem,
        "problem_target_source": problem_source,
        "methodology": {
            "expert_selection": (
                "Experts and ordered top1-to-top2 pairs are selected only by routing support, "
                "before consulting outcomes. Pair candidates are capped to bound output size."
            ),
            "trace_binary": "Point-biserial correlation (Pearson with a 0/1 target).",
            "trace_continuous": "Spearman rank correlation.",
            "problem_correctness": (
                "Spearman rank correlation after averaging expert rates once per problem and "
                "joining to the caller-supplied canonical avg@n target."
            ),
            "warning": (
                "Trace-level p-values assume independent attempts and are exploratory. "
                "Benjamini-Hochberg q-values control each stated scan family but do not repair "
                "within-problem dependence."
            ),
        },
    }


def analyze_expert_identity(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    binary_targets: Sequence[str] = ("is_correct",),
    continuous_targets: Sequence[str] = (),
    problem_table: Sequence[Mapping[str, Any] | Any] | pd.DataFrame | None = None,
    problem_target: str = "avg_correctness",
    correctness_target: str = "is_correct",
    config: ExpertAnalysisConfig | None = None,
    tensor_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build support-selected expert features and run trace/problem associations."""

    config = config or ExpertAnalysisConfig()
    binary_targets = tuple(dict.fromkeys(binary_targets))
    continuous_targets = tuple(dict.fromkeys(continuous_targets))
    copied_targets = tuple(
        dict.fromkeys((*binary_targets, *continuous_targets, correctness_target))
    )
    table = build_expert_feature_table(
        rows,
        target_names=copied_targets,
        config=config,
        tensor_base_dir=tensor_base_dir,
    )
    return analyze_expert_feature_table(
        table,
        binary_targets=binary_targets,
        continuous_targets=continuous_targets,
        problem_table=problem_table,
        problem_target=problem_target,
        correctness_target=correctness_target,
        config=config,
    )
