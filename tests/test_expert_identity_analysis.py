from __future__ import annotations

import warnings

import pandas as pd
import pytest
import torch

from moe_exp.correlation_pipeline.expert_analysis import (
    ExpertAnalysisConfig,
    ExpertFeatureTable,
    _problem_frame,
    analyze_expert_identity,
    build_expert_feature_table,
)


@pytest.mark.parametrize("supplied_canonical", [False, True])
def test_wide_problem_frame_preserves_means_without_fragmentation_warning(
    supplied_canonical,
) -> None:
    features = tuple(f"expert_top1_rate_e{index}" for index in range(128))
    # Separate blocks reproduce the wide groupby layout that triggers warnings
    # when reset_index inserts the two identifier columns.
    frame = pd.concat(
        [
            pd.DataFrame({
                "dataset": ["synthetic"] * 4,
                "source_problem_id": ["p1", "p0", "p1", "p0"],
                "is_correct": [True, False, True, True],
            }),
            *[pd.Series([0.0, 0.2, 1.0, 0.4], name=name) for name in features],
        ],
        axis=1,
    )
    table = ExpertFeatureTable(frame, features, (), {})
    canonical = pd.DataFrame({
        "dataset": ["synthetic", "synthetic"],
        "source_problem_id": ["p1", "p0"],
        "avg_correctness": [0.75, 0.25],
    }) if supplied_canonical else None

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        result, _ = _problem_frame(
            table,
            problem_table=canonical,
            problem_target="avg_correctness",
            correctness_target="is_correct",
        )

    result = result.set_index("source_problem_id").sort_index()
    assert result.index.tolist() == ["p0", "p1"]
    assert result["dataset"].tolist() == ["synthetic", "synthetic"]
    for feature in features:
        assert result[feature].tolist() == pytest.approx([0.3, 0.5])
    assert result["avg_correctness"].tolist() == (
        [0.25, 0.75] if supplied_canonical else [0.5, 1.0]
    )


def test_expert_rates_keep_large_ids_and_cap_ordered_pairs(tmp_path) -> None:
    first = torch.tensor(
        [[[255, 7], [255, 7], [3, 7], [3, 255]]], dtype=torch.int16
    )
    second = torch.tensor(
        [[[255, 7], [3, 7], [3, 7], [3, 255]]], dtype=torch.int16
    )
    torch.save(first, tmp_path / "first.pt")
    rows = [
        {
            "dataset": "synthetic",
            "source_problem_id": "p0",
            "sample_id": 0,
            "selected_experts": "first.pt",
            "is_correct": False,
        },
        {
            "dataset": "synthetic",
            "source_problem_id": "p1",
            "sample_id": 0,
            "selected_experts": second,
            "is_correct": True,
        },
    ]

    table = build_expert_feature_table(
        rows,
        target_names=("is_correct",),
        config=ExpertAnalysisConfig(min_trace_support=1, max_pairs=2),
        tensor_base_dir=tmp_path,
    )

    assert table.selection_summary["observed_expert_ids"] == [3, 7, 255]
    assert table.selection_summary["n_retained_experts"] == 3
    assert table.selection_summary["pair_candidates_meeting_support"] == 3
    assert table.selection_summary["n_retained_pairs"] == 2
    assert table.selection_summary["pairs_truncated_by_cap"] == 1
    assert "expert_pair_top1_e3_top2_e7_rate" in table.feature_columns
    assert "expert_pair_top1_e255_top2_e7_rate" in table.feature_columns
    assert "expert_pair_top1_e3_top2_e255_rate" not in table.feature_columns
    assert table.frame.loc[0, "expert_top1_rate_e255"] == pytest.approx(0.5)
    assert table.frame.loc[0, "expert_any_topk_rate_e255"] == pytest.approx(0.75)
    assert table.frame.loc[0, "expert_any_topk_rate_e7"] == pytest.approx(0.75)
    assert table.frame.loc[0, "expert_pair_top1_e3_top2_e7_rate"] == pytest.approx(0.25)
    assert table.selection_summary["selection_basis"].startswith("Routing support only")

    flipped_rows = [dict(row, is_correct=not row["is_correct"]) for row in rows]
    flipped = build_expert_feature_table(
        flipped_rows,
        target_names=("is_correct",),
        config=ExpertAnalysisConfig(min_trace_support=1, max_pairs=2),
        tensor_base_dir=tmp_path,
    )
    assert flipped.feature_columns == table.feature_columns
    assert flipped.feature_metadata == table.feature_metadata


def test_trace_associations_use_point_biserial_spearman_and_family_fdr() -> None:
    rows = []
    for index in range(8):
        experts = [9] * index + [0] * (7 - index)
        rows.append(
            {
                "dataset": "synthetic",
                "source_problem_id": f"p{index}",
                "sample_id": 0,
                "selected_experts": torch.tensor([experts], dtype=torch.int16).unsqueeze(-1),
                "targets": {"is_correct": index >= 4},
                "trace_features": {"router_signal": float(index)},
            }
        )

    result = analyze_expert_identity(
        rows,
        binary_targets=("is_correct",),
        continuous_targets=("router_signal",),
        config=ExpertAnalysisConfig(
            min_trace_support=1,
            max_pairs=0,
            min_correlation_n=4,
            include_pooled_scope=False,
        ),
    )

    binary = next(
        row
        for row in result["trace_point_biserial"]
        if row["feature"] == "expert_top1_rate_e9"
    )
    continuous = next(
        row
        for row in result["trace_spearman"]
        if row["feature"] == "expert_top1_rate_e9"
    )
    assert binary["point_biserial_r"] > 0
    assert binary["n_positive"] == 4
    assert binary["n_negative"] == 4
    assert binary["benjamini_hochberg_q_value"] >= binary["naive_independent_p_value"]
    assert continuous["spearman_rho"] == pytest.approx(1.0)
    assert (
        continuous["benjamini_hochberg_q_value"]
        >= continuous["naive_independent_p_value"]
    )
    assert binary["analysis_family"] == "trace/point_biserial/synthetic/is_correct"
    assert continuous["analysis_family"] == "trace/spearman/synthetic/router_signal"


def test_problem_spearman_accepts_canonical_average_table() -> None:
    rows = []
    expert_nines = [0, 1, 3, 4, 0]
    for problem_index, count in enumerate(expert_nines):
        for sample_id in range(2):
            experts = [9] * count + [0] * (4 - count)
            rows.append(
                {
                    "dataset": "synthetic",
                    "source_problem_id": f"p{problem_index}",
                    "sample_id": sample_id,
                    "selected_experts": torch.tensor([experts], dtype=torch.int16).unsqueeze(-1),
                    "is_correct": bool(problem_index >= 2),
                }
            )
    canonical = pd.DataFrame(
        {
            "dataset": ["synthetic"] * 5,
            "source_problem_id": ["p0", "p1", "p2", "p3", "p4"],
            "avg_correctness": [0.0, 0.25, 0.75, 1.0, 1.0],
            "eligible_for_repeated_analysis": [True, True, True, True, False],
        }
    )

    result = analyze_expert_identity(
        rows,
        problem_table=canonical,
        config=ExpertAnalysisConfig(
            min_trace_support=1,
            max_pairs=0,
            min_correlation_n=4,
            include_pooled_scope=False,
        ),
    )

    association = next(
        row
        for row in result["problem_spearman"]
        if row["feature"] == "expert_top1_rate_e9"
    )
    assert association["scope"] == "synthetic"
    assert association["n_problems"] == 4
    assert association["spearman_rho"] == pytest.approx(1.0)
    assert association["benjamini_hochberg_q_value"] >= association["p_value"]
    assert result["problem_target_source"] == "caller-supplied canonical problem table"


def test_problem_spearman_derives_average_only_from_repeated_attempts() -> None:
    rows = []
    for problem_index, correct_count in enumerate((0, 1, 2, 3)):
        for sample_id in range(3):
            count = problem_index + 1
            experts = [11] * count + [0] * (4 - count)
            rows.append(
                {
                    "dataset": "synthetic",
                    "source_problem_id": f"p{problem_index}",
                    "sample_id": sample_id,
                    "selected_experts": torch.tensor([experts], dtype=torch.int16).unsqueeze(-1),
                    "is_correct": sample_id < correct_count,
                }
            )

    result = analyze_expert_identity(
        rows,
        config=ExpertAnalysisConfig(
            min_trace_support=1,
            max_pairs=0,
            min_correlation_n=4,
            include_pooled_scope=False,
        ),
    )

    association = next(
        row
        for row in result["problem_spearman"]
        if row["feature"] == "expert_top1_rate_e11"
    )
    assert association["n_problems"] == 4
    assert association["spearman_rho"] == pytest.approx(1.0)
    assert result["problem_target_source"].startswith("derived canonical mean")
