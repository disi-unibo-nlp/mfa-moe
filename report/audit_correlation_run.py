"""Read-only checks and descriptive summaries for the September correlation report.

Run with the project's Python environment. This script prints JSON; it does not
change experiment artifacts, fit models, or regenerate traces/tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/correlation_pipeline/reasoning-vllm-untagged-v1"
FEATURES = [
    "token_count", "router_confidence_mean_layers", "router_margin_mean_layers",
    "router_selected_mass_mean_layers", "router_boundary_margin_mean_layers",
    "hidden_norm_mean_layers", "hidden_step_distance_mean_layers",
    "hidden_router_geometry_mean_layers",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    analysis_dir = run / "analysis/unsloth--Qwen3.5-35B-A3B"
    analysis_path = analysis_dir / "correlations.json"
    analysis = json.loads(analysis_path.read_text())
    forward = json.loads((run / "forward/unsloth--Qwen3.5-35B-A3B/summary.json").read_text())
    frame = pd.read_csv(analysis_dir / "trace_features.csv")
    assert analysis["status"] == forward["status"] == "complete"
    assert len(frame) == analysis["n_traces"]
    assert not frame.duplicated(["dataset", "problem_id"]).any()
    assert frame["is_correct"].isin([0, 1]).all()
    assert set(frame["generation_max_tokens"]) == {32768}
    lookup = frame.set_index(["dataset", "problem_id"])
    checked = {"generation": 0, "forward": 0}

    for dataset in forward["datasets"]:
        name = dataset["dataset"]
        for stage, key in [("generation", "input"), ("forward", "output")]:
            seen = set()
            with (ROOT / dataset[key]).open() as source:
                for line in source:
                    trace = json.loads(line)
                    assert trace["problem_id"] not in seen
                    seen.add(trace["problem_id"])
                    row = lookup.loc[(name, trace["problem_id"])]
                    assert row["generation_sha256"] == hashlib.sha256(
                        trace["cot_text"].encode()
                    ).hexdigest()
                    assert int(bool(trace["is_correct"])) == row["is_correct"]
                    assert trace["metadata"]["generation_config"]["max_tokens"] == 32768
                    if stage == "forward":
                        compact = trace["metadata"]["correlation_features"]
                        assert compact["token_count"] == row["token_count"]
                        assert trace["model_logs"]["layer_indices"] == forward["layer_indices"]
                        for feature in FEATURES[1:]:
                            assert np.isclose(compact["values"][feature], row[feature])
                    checked[stage] += 1
            assert len(seen) == dataset["traces"] == analysis["datasets"][name]

    # Recompute the saved binary coefficients from the CSV, independently of
    # the pipeline's correlation helper. Bootstrap intervals remain as saved.
    max_error = 0.0
    for row in analysis["binary_correlations"]:
        group = frame if row["scope"] == "all" else frame[frame.dataset.eq(row["scope"])]
        pair = group[[row["target"], row["feature"]]].dropna()
        value = float(pointbiserialr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
        max_error = max(max_error, abs(value - row["point_biserial_r"]))
    assert max_error < 1e-10
    max_rank_error = 0.0
    for row in analysis["repeated_problem_analysis"]["problem_level"]:
        if row["problem_feature"] != "feature_mean":
            continue
        group = frame[frame.dataset.eq(row["dataset"])].groupby("source_problem_id")
        means = group[["is_correct", row["feature"]]].mean().dropna()
        value = float(spearmanr(means.is_correct, means[row["feature"]]).statistic)
        max_rank_error = max(max_rank_error, abs(value - row["spearman_rho"]))
    assert max_rank_error < 1e-10

    summary = {
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "matched_trace_rows": checked,
        "max_binary_coefficient_error": max_error,
        "max_problem_spearman_error": max_rank_error,
        "datasets": {},
    }
    for name in [*analysis["datasets"], "all"]:
        group = frame if name == "all" else frame[frame.dataset.eq(name)]
        completed = group[group.generation_hit_token_limit.eq(0)]
        failed = group[group.is_correct.eq(0)]
        data = {
            "problems": len(group[["dataset", "source_problem_id"]].drop_duplicates()),
            "traces": len(group), "correct": int(group.is_correct.sum()),
            "events": {k: int(group[k].sum()) for k in [
                "has_backtracking", "has_contradiction", "has_self_correction",
            ]},
            "accuracy": float(group.is_correct.mean()),
            "limit_hits": int(group.generation_hit_token_limit.sum()),
            "failures": len(failed),
            "limited_failures": int(failed.generation_hit_token_limit.sum()),
            "nonlimited_traces": len(completed),
            "nonlimited_failures": int(completed.is_correct.eq(0).sum()),
            "nonlimited_accuracy": float(completed.is_correct.mean()),
            "nonlimited_point_biserial": {},
        }
        for feature in FEATURES:
            data["nonlimited_point_biserial"][feature] = float(
                pointbiserialr(completed.is_correct, completed[feature]).statistic
            )
        summary["datasets"][name] = data

    primary = [r for r in analysis["repeated_problem_analysis"]["problem_level"]
               if r["problem_feature"] == "feature_mean" and r["feature"] in FEATURES]
    summary["primary_problem_associations"] = primary
    summary["within_problem"] = [r for r in analysis["repeated_problem_analysis"]["within_problem"]
                                  if r["feature"] in FEATURES[:3]]
    summary["cross_features"] = [
        r for r in analysis["cross_feature_correlations"]["trace_level"]
        if {r["feature_x"], r["feature_y"]}
        == {"token_count", "router_confidence_mean_layers"}
    ]
    experts = analysis["expert_identity_analysis"]
    summary["expert_selection"] = {k: v for k, v in experts["selection"].items()
                                   if not k.endswith("expert_ids")}
    summary["expert_problem_maxima"] = []
    for name in [r["dataset"] for r in analysis["repeated_problem_analysis"]["group_completeness"]]:
        rows = [r for r in experts["problem_spearman"] if r["scope"] == name]
        summary["expert_problem_maxima"].append(max(rows, key=lambda r: abs(r["spearman_rho"])))
    expert_frame = pd.read_csv(analysis_dir / "expert_trace_features.csv")
    expert_error = 0.0
    for row in summary["expert_problem_maxima"]:
        means = (expert_frame[expert_frame.dataset.eq(row["scope"])]
                 .groupby("source_problem_id")[["is_correct", row["feature"]]].mean())
        assert len(means) == row["n_problems"]
        value = float(spearmanr(means.is_correct, means[row["feature"]]).statistic)
        expert_error = max(expert_error, abs(value - row["spearman_rho"]))
    assert expert_error < 1e-10
    summary["max_expert_maximum_coefficient_error"] = expert_error
    pair_error = 0.0
    for row in analysis["cross_feature_correlations"]["trace_level"]:
        group = frame if row["scope"] == "all" else frame[frame.dataset.eq(row["scope"])]
        pair = group[[row["feature_x"], row["feature_y"]]].dropna()
        value = float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
        pair_error = max(pair_error, abs(value - row["spearman_rho"]))
    assert pair_error < 1e-10
    summary["max_metric_pair_coefficient_error"] = pair_error
    summary["storage"] = {key: sum(d["storage"][key] for d in forward["datasets"])
                          for key in forward["datasets"][0]["storage"]}
    # Check each view's saved coefficients against its own finite observations;
    # late windows retain outcomes even when their routing features are absent.
    summary["views"] = []
    for view in analysis.get("reasoning_view_analysis", {}).get("views", []):
        path = ROOT / view["path"]
        view_analysis = json.loads((path / "correlations.json").read_text())
        view_frame = pd.read_csv(path / "trace_features.csv")
        assert len(view_frame) == len(frame)
        pd.testing.assert_series_equal(
            view_frame.set_index(["dataset", "problem_id"])["is_correct"].sort_index(),
            lookup["is_correct"].sort_index(),
        )
        view_error = 0.0
        for row in view_analysis["binary_correlations"]:
            group = (view_frame if row["scope"] == "all"
                     else view_frame[view_frame.dataset.eq(row["scope"])])
            pair = group[[row["target"], row["feature"]]].dropna()
            assert len(pair) == row["n_traces"]
            value = float(pointbiserialr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
            view_error = max(view_error, abs(value - row["point_biserial_r"]))
        assert view_error < 1e-10
        summary["views"].append({
            "view": view["view"], "name": view["name"],
            "traces_with_tokens": view["coverage"]["traces_with_tokens"],
            "max_binary_coefficient_error": view_error,
        })
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
