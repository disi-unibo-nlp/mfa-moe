"""Tests for the stratified 100k sentence sampler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.annotation_batch import enumerate_items
from labeling.sample_stratified import (
    proportional_quotas,
    quartile_groups,
    reasoning_tokens,
    replay_reasoning_tokens,
    sample,
    server_reasoning_tokens,
    waterfill,
)
from schemas import TraceRecord

DATASETS = ("math500", "aime24")


def _reasoning(sentences: int) -> str:
    return " ".join(f"Reasoning{index} continues here." for index in range(sentences))


def _trace_row(
    dataset: str,
    problem: str,
    sample_id: int,
    correct: bool | None,
    reasoning_tokens: int,
    sentences: int,
) -> dict:
    reasoning = _reasoning(sentences)
    trace = TraceRecord(
        dataset=dataset,
        problem_id=f"{problem}__sample_{sample_id:02d}",
        source_problem_id=problem,
        sample_id=sample_id,
        prompt="Question?",
        gold_answer="9",
        model_id="test",
        model_answer="9",
        is_correct=correct,
        cot_text=reasoning + " Final answer: 9.",
        metadata={
            "reasoning_content": reasoning,
            "usage": {
                "completion_tokens": reasoning_tokens + 50,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        },
    )
    return trace.model_dump()


def _write_dataset(root: Path, dataset: str, rows: list[dict], samples_per_problem: int) -> None:
    directory = root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "samples_per_problem": samples_per_problem,
                "problems": len({row["source_problem_id"] for row in rows}),
                "traces": len(rows),
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, unscored: bool = False, status: str = "complete") -> Path:
    generation = tmp_path / "generation"
    root = generation / "test"
    math500 = [
        _trace_row("math500", f"m{index}", 0, index % 2 == 0, 100 * (index + 1), 10)
        for index in range(8)
    ]
    if unscored:
        math500.append(
            _trace_row("math500", "m-unscored", 0, None, 50_000, 200)
        )
    aime24 = [
        _trace_row("aime24", f"a{problem}", sample_id, (problem + sample_id) % 3 != 0,
                   100 + 70 * problem + 30 * sample_id, 12)
        for problem in range(4)
        for sample_id in range(4)
    ]
    _write_dataset(root, "math500", math500, 1)
    _write_dataset(root, "aime24", aime24, 4)
    if status != "complete":
        manifest_path = root / "math500" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return generation


def _args(generation: Path, output: Path, *, seed: int = 42,
          parts: int = 2, part_size: int = 32, dry_run: bool = False,
          datasets: list[str] | None = None) -> dict:
    """The keyword arguments of `sample` for the small fixture."""
    return dict(
        generation_dir=generation,
        generation_model="test",
        datasets=list(datasets if datasets is not None else DATASETS),
        output_dir=output,
        parts=parts,
        part_size=part_size,
        seed=seed,
        dry_run=dry_run,
    )


def _identity(dataset: str, row: dict, sentence_index: int) -> tuple:
    return (dataset, row["problem_id"], row["sample_id"], sentence_index)


def _read_identities(path: Path, dataset: str) -> list[tuple]:
    identities = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        indices = row["metadata"]["sentence_selection"]["indices"]
        identities.extend(_identity(dataset, row, index) for index in indices)
    return identities


def test_quartile_groups_split_equal_counts() -> None:
    groups = quartile_groups([f"t{index}" for index in range(8)])
    assert groups == {
        "t0": 0, "t1": 0, "t2": 1, "t3": 1, "t4": 2, "t5": 2, "t6": 3, "t7": 3,
    }
    with pytest.raises(ValueError):
        quartile_groups([])


def test_proportional_quotas_split_by_supply_and_never_exceed_it() -> None:
    assert proportional_quotas({"a": 1, "b": 10, "c": 10}, 8) == {"a": 0, "b": 4, "c": 4}
    # largest remainders get the extra unit, ties go to the earlier cell name
    assert proportional_quotas({"a": 3, "b": 3, "c": 3}, 4) == {"a": 2, "b": 1, "c": 1}
    # taking everything gives every cell its whole supply
    assert proportional_quotas({"a": 2, "b": 0, "c": 5}, 7) == {"a": 2, "b": 0, "c": 5}
    for supply, total in [({"a": 1}, 4), ({"a": 10}, 0)]:
        with pytest.raises(ValueError):
            proportional_quotas(supply, total)


def test_waterfill_is_max_min_fair() -> None:
    assert waterfill({"a": 1, "b": 10, "c": 10}, 8) == {"a": 1, "b": 4, "c": 3}
    assert waterfill({"a": 3, "b": 4}, 0) == {"a": 0, "b": 0}
    assert waterfill({"a": 3, "b": 4}, 7) == {"a": 3, "b": 4}
    with pytest.raises(ValueError):
        waterfill({"a": 3}, 4)


def test_sample_writes_exact_stratified_parts(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    output = tmp_path / "sampled"
    result = sample(**_args(generation, output))
    assert result["selected_sentences"] == 64
    assert result["parts"] == [{"part": 0, "identities": 32}, {"part": 1, "identities": 32}]
    assert result["eligible"]["math500"]["traces"] == 8
    assert result["eligible"]["aime24"]["traces"] == 16

    combined_path = output / "generation" / "test"
    combined = []
    for dataset in DATASETS:
        combined.extend(_read_identities(combined_path / dataset / "traces.jsonl", dataset))
    assert len(combined) == 64
    assert len(set(combined)) == 64

    problems = [
        (dataset, row["source_problem_id"])
        for dataset in DATASETS
        for row in (
            json.loads(line)
            for line in (combined_path / dataset / "traces.jsonl").read_text().splitlines()
        )
    ]
    assert len(problems) == len(set(problems)), "one completion per source problem"

    part_identities = []
    for part in range(2):
        part_root = output / "parts" / f"part-{part:02d}" / "generation" / "test"
        identities = []
        for dataset in DATASETS:
            identities.extend(_read_identities(part_root / dataset / "traces.jsonl", dataset))
        assert len(identities) == 32
        part_identities.append(set(identities))
    assert part_identities[0].isdisjoint(part_identities[1])
    assert part_identities[0] | part_identities[1] == set(combined)

    for part in range(2):
        manifest_path = output / "parts" / f"part-{part:02d}" / "part_manifest.json"
        part_manifest = json.loads(manifest_path.read_text())
        assert part_manifest["identities"] == 32
        assert sum(part_manifest["cells"].values()) == 32
        assert set(part_manifest["datasets"]) == set(DATASETS)

    sampling = json.loads((output / "sampling_manifest.json").read_text())
    chosen_total = sum(cell["chosen_sentences"] for cell in sampling["cells"].values())
    assert chosen_total >= 64
    for cell in sampling["cells"].values():
        exact = sampling["selected_sentences"] * cell["chosen_sentences"] / chosen_total
        assert abs(cell["quota"] - exact) <= 1


def test_sample_output_is_consumable_by_annotation_partition(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    output = tmp_path / "sampled"
    sample(**_args(generation, output))
    root = output / "parts" / "part-00" / "generation" / "test"
    items = enumerate_items(root, DATASETS)
    assert len(items) == 32
    identities = [
        (item["identity"]["dataset"], item["identity"]["problem_id"], item["identity"]["sample_id"],
         item["identity"]["sentence_index"])
        for item in items
    ]
    assert len(set(identities)) == 32


def test_sample_is_deterministic_and_seed_sensitive(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    first = sample(**_args(generation, tmp_path / "first"))
    second = sample(**_args(generation, tmp_path / "second"))
    third = sample(**_args(generation, tmp_path / "third", seed=7))
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest_sha256"] != third["manifest_sha256"]
    first_identities = _read_identities(
        tmp_path / "first" / "generation" / "test" / "math500" / "traces.jsonl", "math500"
    )
    third_identities = _read_identities(
        tmp_path / "third" / "generation" / "test" / "math500" / "traces.jsonl", "math500"
    )
    assert first_identities != third_identities


def test_unscored_traces_are_excluded(tmp_path: Path) -> None:
    generation = _fixture(tmp_path, unscored=True)
    result = sample(**_args(generation, tmp_path / "sampled"))
    assert result["eligible"]["math500"]["dropped_unscored"] == 1
    combined = _read_identities(
        tmp_path / "sampled" / "generation" / "test" / "math500" / "traces.jsonl", "math500"
    )
    assert all(problem != "m-unscored__sample_00" for _, problem, _, _ in combined)


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    generation = _fixture(tmp_path, status="running")
    with pytest.raises(ValueError, match="requires completed generations"):
        sample(**_args(generation, tmp_path / "sampled"))


def test_still_generating_dataset_is_rejected(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    (generation / "test" / "aime24" / "manifest.json").unlink()
    (generation / "test" / "aime24" / "generation_shards").mkdir()
    with pytest.raises(ValueError, match="still generating"):
        sample(**_args(generation, tmp_path / "sampled"))


def test_different_plan_in_existing_output_is_rejected(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    output = tmp_path / "sampled"
    sample(**_args(generation, output))
    with pytest.raises(ValueError, match="different sampling plan"):
        sample(**_args(generation, output, seed=7))
    assert sample(**_args(generation, output))["selected_sentences"] == 64


def test_insufficient_supply_is_rejected(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    with pytest.raises(ValueError, match="fewer than the 4096 to select"):
        sample(**_args(generation, tmp_path / "sampled", parts=2, part_size=2048))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    generation = _fixture(tmp_path)
    output = tmp_path / "sampled"
    result = sample(**_args(generation, output, dry_run=True))
    assert result["selected_sentences"] == 64
    assert not output.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# --- reasoning-length metric resolution -------------------------------------
# vLLM's gpt-oss harmony path reports a constant reasoning_tokens=0 while still
# parsing reasoning_content, so length strata built on that field would collapse
# into one. The resolver falls back to the saved token replay.


def _replay_trace(
    reasoning: str,
    tail: str,
    *,
    server_tokens: int | None,
    reasoning_token_count: int,
    tail_token_count: int = 3,
) -> TraceRecord:
    """Build a trace whose offsets really index into its cot_text."""
    cot = reasoning + tail
    step = max(1, len(reasoning) // reasoning_token_count)
    offsets: list[list[int]] = []
    cursor = 0
    for index in range(reasoning_token_count):
        end = len(reasoning) if index == reasoning_token_count - 1 else min(len(reasoning), cursor + step)
        offsets.append([cursor, end])
        cursor = end
    for index in range(tail_token_count):
        start = len(reasoning) + index
        offsets.append([start, min(len(cot), start + 1)])
    usage: dict = {"completion_tokens": len(offsets)}
    if server_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": server_tokens}
    return TraceRecord(
        dataset="math500", problem_id="p__sample_00", source_problem_id="p", sample_id=0,
        prompt="Q?", gold_answer="9", model_id="test", model_answer="9", is_correct=True,
        cot_text=cot,
        metadata={
            "reasoning_content": reasoning,
            "usage": usage,
            "token_replay": {"schema_version": 1, "completion_offsets": offsets},
        },
    )


def test_positive_server_value_wins_over_the_replay() -> None:
    trace = _replay_trace("Think hard about it.", " Final: 9.", server_tokens=4242, reasoning_token_count=7)
    assert server_reasoning_tokens(trace) == 4242
    assert reasoning_tokens(trace) == (4242, "server")


def test_zero_server_value_falls_back_to_the_token_replay() -> None:
    trace = _replay_trace("Think hard about it.", " Final: 9.", server_tokens=0, reasoning_token_count=7)
    assert server_reasoning_tokens(trace) == 0
    assert replay_reasoning_tokens(trace) == 7
    assert reasoning_tokens(trace) == (7, "token_replay")


def test_missing_server_value_falls_back_to_the_token_replay() -> None:
    trace = _replay_trace("Think hard.", " Final: 9.", server_tokens=None, reasoning_token_count=5)
    assert server_reasoning_tokens(trace) is None
    assert reasoning_tokens(trace) == (5, "token_replay")


def test_unusable_replay_keeps_the_zero_server_value() -> None:
    trace = _replay_trace("Think hard.", " Final: 9.", server_tokens=0, reasoning_token_count=5)
    trace.metadata["reasoning_content"] = "text that is absent from cot_text"
    assert replay_reasoning_tokens(trace) is None
    assert reasoning_tokens(trace) == (0, "server")


def test_no_metric_at_all_is_unavailable() -> None:
    trace = _replay_trace("Think hard.", " Final: 9.", server_tokens=None, reasoning_token_count=5)
    trace.metadata.pop("token_replay")
    assert reasoning_tokens(trace) is None


def test_constant_reasoning_length_is_rejected(tmp_path: Path) -> None:
    """A corpus whose traces all report the same reasoning length cannot be stratified."""
    generation = tmp_path / "generation"
    root = generation / "test"
    rows = []
    for index in range(8):
        row = _trace_row("math500", f"m{index}", 0, index % 2 == 0, 0, 10)
        row["metadata"].pop("reasoning_content")
        rows.append(row)
    _write_dataset(root, "math500", rows, 1)
    with pytest.raises(ValueError, match="same reasoning length"):
        sample(**_args(generation, tmp_path / "out", datasets=["math500"], parts=2, part_size=4))
