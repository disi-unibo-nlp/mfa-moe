from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_exp.correlation_pipeline.batch_benchmark import (
    SELECTION_RULE,
    build_cells,
    latency_percentiles,
    select_winner,
    summarize_cell,
)


def test_cells_cover_the_authorized_matrix() -> None:
    cells = build_cells(batch_sizes=[1, 4, 8, 16, 32], concurrencies=[1, 2], mtp_modes=[False, True])
    assert len(cells) == 20
    assert {c["batch_size"] for c in cells} == {1, 4, 8, 16, 32}
    assert {c["concurrency"] for c in cells} == {1, 2}
    assert {c["mtp"] for c in cells} == {False, True}


def test_only_the_batch32_cell_raises_max_num_seqs() -> None:
    cells = build_cells(batch_sizes=[1, 4, 8, 16, 32], concurrencies=[1], mtp_modes=[False])
    by_batch = {c["batch_size"]: c for c in cells}
    assert by_batch[32]["max_num_seqs"] == 32
    for batch in (1, 4, 8, 16):
        assert by_batch[batch]["max_num_seqs"] == 16


def test_cells_are_grouped_so_one_server_serves_many_cells() -> None:
    """Server restarts are expensive; cells must be ordered by (mtp, max_num_seqs)."""
    cells = build_cells(batch_sizes=[1, 16, 32], concurrencies=[1, 2], mtp_modes=[False, True])
    keys = [(c["mtp"], c["max_num_seqs"]) for c in cells]
    assert keys == sorted(keys, key=lambda k: (k[0], k[1]))
    assert len({k for k in keys}) == 4  # 2 mtp modes x 2 max_num_seqs values


def test_latency_percentiles() -> None:
    p50, p95 = latency_percentiles([10.0, 20.0, 30.0, 40.0, 100.0])
    assert p50 == 30.0
    assert p95 == 100.0


def test_summarize_cell_computes_throughput_and_failures() -> None:
    summary = summarize_cell(
        cell={"batch_size": 16, "concurrency": 1, "mtp": True, "max_num_seqs": 16},
        latencies_ms=[1000.0, 2000.0],
        sentences=32,
        completion_tokens=640,
        elapsed_seconds=4.0,
        failures=1,
        http_errors=2,
        gpu_memory_used_mb=40000,
        gpu_utilization_pct=88.0,
    )
    assert summary["sentences_per_second"] == 8.0
    assert summary["aggregate_tokens_per_second"] == 160.0
    assert summary["batch_generations_per_second"] == 0.5
    assert summary["failures"] == 1
    assert summary["http_errors"] == 2
    assert summary["max_num_seqs"] == 16
    assert summary["mtp"] is True


def test_selection_rule_rejects_cells_with_failures() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    bad = _row(batch=16, tok_s=500.0, p95=1200.0, failures=1)
    assert select_winner([baseline, bad]) is None


def test_selection_rule_rejects_p95_regression_beyond_2x() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    slow = _row(batch=16, tok_s=500.0, p95=2001.0)
    assert select_winner([baseline, slow]) is None


def test_selection_rule_requires_ten_percent_gain_over_baseline() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    marginal = _row(batch=16, tok_s=109.0, p95=1000.0)
    assert select_winner([baseline, marginal]) is None


def test_selection_rule_picks_highest_tokens_per_second() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    good = _row(batch=16, tok_s=300.0, p95=1500.0)
    better = _row(batch=32, tok_s=400.0, p95=1800.0)
    winner = select_winner([baseline, good, better])
    assert winner is not None and winner["batch_size"] == 32


def test_ties_within_five_percent_prefer_the_smaller_batch() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    small = _row(batch=8, tok_s=390.0, p95=1100.0)
    big = _row(batch=32, tok_s=400.0, p95=1900.0)
    winner = select_winner([baseline, small, big])
    assert winner is not None and winner["batch_size"] == 8


def test_selection_rule_rejects_excess_gpu_memory() -> None:
    baseline = _row(batch=1, tok_s=100.0, p95=1000.0)
    hungry = _row(batch=32, tok_s=900.0, p95=1100.0, gpu_mb=62000)
    assert select_winner([baseline, hungry]) is None


def test_selection_rule_is_documented() -> None:
    assert "10" in SELECTION_RULE and "p95" in SELECTION_RULE


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_ROOT = REPO_ROOT / "results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b"
MATH500 = GEN_ROOT / "math500/traces.jsonl"
AMC23 = GEN_ROOT / "amc23/traces.jsonl"

# The slice is asserted against the real trace files the sweep actually reads, so a
# schema drift in TraceRecord surfaces here instead of inside a GPU job.
requires_traces = pytest.mark.skipif(
    not (MATH500.is_file() and AMC23.is_file()),
    reason="generation traces for math500/amc23 are not present",
)


@requires_traces
def test_balanced_slice_is_thirty_two_items_split_sixteen_each() -> None:
    from moe_exp.correlation_pipeline.batch_probe import build_balanced_slice

    items = build_balanced_slice([MATH500, AMC23], 32)

    assert len(items) == 32
    counts: dict[str, int] = {}
    for item in items:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    assert counts == {"math500": 16, "amc23": 16}


@requires_traces
def test_balanced_slice_rejects_an_uneven_split() -> None:
    from moe_exp.correlation_pipeline.batch_probe import build_balanced_slice

    with pytest.raises(ValueError):
        build_balanced_slice([MATH500, AMC23], 33)


@requires_traces
def test_balanced_slice_is_deterministic() -> None:
    from moe_exp.correlation_pipeline.batch_probe import build_balanced_slice

    first = build_balanced_slice([MATH500, AMC23], 32)
    second = build_balanced_slice([MATH500, AMC23], 32)
    assert [item["sentence"] for item in first] == [item["sentence"] for item in second]


@requires_traces
def test_balanced_slice_items_carry_every_judge_input_field() -> None:
    from moe_exp.correlation_pipeline.batch_predictor import JUDGE_INPUT_FIELDS
    from moe_exp.correlation_pipeline.batch_probe import build_balanced_slice

    for item in build_balanced_slice([MATH500, AMC23], 32):
        assert set(JUDGE_INPUT_FIELDS) <= set(item)


def _row(
    *,
    batch: int,
    tok_s: float,
    p95: float,
    failures: int = 0,
    gpu_mb: int = 40000,
) -> dict:
    return {
        "batch_size": batch,
        "concurrency": 1,
        "mtp": False,
        "max_num_seqs": 16 if batch <= 16 else 32,
        "aggregate_tokens_per_second": tok_s,
        "latency_p95_ms": p95,
        "failures": failures,
        "http_errors": 0,
        "gpu_memory_used_mb": gpu_mb,
    }
