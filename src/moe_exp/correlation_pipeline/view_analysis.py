"""Apply the common correlation statistics to one row per attempt and view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from moe_exp.correlation_pipeline.features import restore_features
from moe_exp.correlation_pipeline.spans import (
    SENTENCE_LABELS,
    SPAN_SCHEMA_VERSION,
    digest,
    trace_digest,
)


def collect_view_rows(trace: Any, base_row: dict[str, Any], modes: list[str]):
    from moe_exp.correlation_pipeline.analyze import _IDENTIFIERS

    payload = trace.metadata.get("correlation_views")
    if not isinstance(payload, dict) or payload.get("schema_version") != SPAN_SCHEMA_VERSION:
        raise ValueError("Missing/version-mismatched reasoning views; rerun forward with --views")
    if payload.get("trace_sha256") != trace_digest(trace):
        raise ValueError("Reasoning view features belong to a different generation")
    if payload.get("sentence_selection") != trace.metadata.get("sentence_selection"):
        raise ValueError("Reasoning view features belong to a different sentence selection")
    reference = payload.get("position_reference")
    annotation = trace.metadata.get("reasoning_annotation")
    if "class" in modes and (
        annotation is None or payload.get("annotation_sha256") != digest(annotation)
    ):
        raise ValueError("Reasoning views do not match the sentence annotations")
    expected = set()
    if "full" in modes:
        expected.add(("full", "reasoning"))
    if "class" in modes:
        expected.update(("class", label) for label in SENTENCE_LABELS)
    if "position" in modes:
        if not isinstance(reference, dict):
            raise ValueError("Position reference is missing")
        expected.update(("position", f"bin_{i:02d}") for i in range(reference["bins"]))
        expected.add(("position", "overflow"))
    found = set()
    rows = []
    for scope in payload["scopes"]:
        key = (scope["view"], scope["name"])
        if key[0] not in modes:
            continue
        if key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicate reasoning scope {key}")
        found.add(key)
        row = {name: value for name, value in base_row.items() if name in _IDENTIFIERS}
        row.update(restore_features(scope["values"]))
        row["selected_token_count"] = scope["token_count"]
        row["transition_count"] = scope["transition_count"]
        # Absence of a class/window is missing feature data, not a zero-valued
        # routing statistic. Keep the attempt row for avg@n completeness audits.
        row["token_count"] = scope["token_count"] if scope["token_count"] else np.nan
        rows.append((key, row))
    if found != expected:
        raise ValueError(f"Missing reasoning scopes: {sorted(expected - found)}")
    contract = {
        "schema_version": SPAN_SCHEMA_VERSION,
        "classifier": annotation.get("classifier") if annotation and "class" in modes else None,
        "position_reference": reference if "position" in modes else None,
        "sentence_sampling": (
            {key: value for key, value in payload["sentence_selection"].items() if key != "indices"}
            if payload.get("sentence_selection") else None
        ),
    }
    return rows, contract


def analyze_views(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    output_root: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    from moe_exp.correlation_pipeline.analyze import (
        _binary_correlations,
        _cross_feature_correlations,
        _feature_columns,
        _problem_level_table,
        _repeated_problem_analysis,
        _write_json_atomic,
    )

    summaries = []
    for (view, name), rows in sorted(grouped.items()):
        frame = pd.DataFrame(rows)
        features = [
            column
            for column in _feature_columns(frame)
            if column not in {"selected_token_count", "transition_count"}
            and not (view == "position" and column == "token_count")
        ]
        problems = _problem_level_table(frame, feature_columns=features)
        rng = np.random.default_rng(seed)
        result = {
            "schema_version": SPAN_SCHEMA_VERSION,
            "view": view,
            "name": name,
            "contract": contract,
            "coverage": {
                "traces": len(frame),
                "traces_with_tokens": int(frame.selected_token_count.gt(0).sum()),
                "missing_by_feature": {
                    feature: int(frame[feature].isna().sum()) for feature in features
                },
                "datasets": {
                    dataset: {
                        "traces": len(group),
                        "traces_with_tokens": int(group.selected_token_count.gt(0).sum()),
                    }
                    for dataset, group in frame.groupby("dataset")
                },
            },
            "methodology": {
                "analysis_unit": "one row per original attempt per scope; sentences are not independent trials",
                "absent_scope": "missing metrics; original attempt and correctness retained",
                "repeated_problem_features": "require complete avg@n outcomes and all n finite feature values",
                "position_token_count": "excluded from correlations",
                "class_transitions": "only adjacent original tokens in the same sentence",
                "multiplicity": "BH per scope/dataset/target or metric-pair family; cross-scope comparisons exploratory",
                "targets": "whole-attempt correctness and legacy lexical event flags",
            },
            "binary_correlations": _binary_correlations(
                frame, feature_columns=features, bootstrap_samples=bootstrap_samples, rng=rng
            ),
            "repeated_problem_analysis": _repeated_problem_analysis(
                frame,
                feature_columns=features,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
                problem_frame=problems,
            ),
            "cross_feature_correlations": _cross_feature_correlations(
                frame, feature_columns=features, problem_frame=problems
            ),
        }
        directory = output_root / f"views-v{SPAN_SCHEMA_VERSION}" / view / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "trace_features.csv", index=False)
        problems.to_csv(directory / "problem_features.csv", index=False)
        pd.DataFrame(result["binary_correlations"]).to_csv(
            directory / "binary_correlations.csv", index=False
        )
        pd.DataFrame(result["cross_feature_correlations"]["trace_level"]).to_csv(
            directory / "cross_feature_correlations.csv", index=False
        )
        _write_json_atomic(result, directory / "correlations.json")
        summaries.append(
            {
                "view": view,
                "name": name,
                "path": directory.as_posix(),
                "coverage": result["coverage"],
            }
        )
    summary = {"status": "complete", "contract": contract, "views": summaries}
    _write_json_atomic(summary, output_root / f"views-v{SPAN_SCHEMA_VERSION}" / "summary.json")
    return summary
