"""Tests for the balanced-label collector/exporter."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from moe_exp.correlation_pipeline.annotation_batch import digest
from moe_exp.correlation_pipeline.labels_export import SOURCES, collect_source, export, main

PARTS = 2
PART_SIZE = 3


def _record(dataset: str, problem: str, sample_id: int, index: int, label: str) -> dict:
    text = f"Sentence {index} of {problem}."
    return {
        "identity": {
            "dataset": dataset,
            "problem_id": f"{problem}__sample_{sample_id:02d}",
            "source_problem_id": problem,
            "sample_id": sample_id,
            "sentence_index": index,
            "start": index * 10,
            "end": index * 10 + 9,
            "trace_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        },
        "unit": {"index": index, "start": index * 10, "end": index * 10 + 9, "text": text},
        "inputs": {"problem_statement": "Q?", "previous_sentence": "", "sentence": text, "next_sentence": ""},
        "label": label,
    }


def _part_records(part: int) -> list[dict]:
    problem = f"p{part}"
    return [
        _record("math500", problem, 0, 0, "Read"),
        _record("math500", problem, 0, 1, "Plan"),
        _record("aime24", f"a{part}", 0, 0, "Verify"),
    ]


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(root: Path, name: str = "gpt") -> tuple[Path, Path]:
    sampling_root = root / "sampling" / name
    label_root = root / "labels" / name
    plan_hash = "plan-" + name
    _write(
        sampling_root / "sampling_manifest.json",
        {
            "manifest_sha256": plan_hash,
            "parts": PARTS,
            "part_size": PART_SIZE,
            "selected_sentences": PARTS * PART_SIZE,
            "generation_model": "test/model",
            "selection": "stratified-accuracy-reasoning-length",
            "seed": 42,
        },
    )
    for part in range(PARTS):
        records = _part_records(part)
        _write(
            sampling_root / "parts" / f"part-{part:02d}" / "part_manifest.json",
            {
                "part": part,
                "identities": PART_SIZE,
                "sampling_manifest_sha256": plan_hash,
                "datasets": {"aime24": 1, "math500": 1},
                "cells": {"math500|correct|Q1": 2, "aime24|incorrect|Q4": 1},
                "trace_root": str(sampling_root / "parts" / f"part-{part:02d}" / "generation" / "slug"),
            },
        )
        _write(label_root / f"part-{part:02d}" / "annotations.json", records)
        _write(
            label_root / f"part-{part:02d}" / "summary.json",
            {
                "status": "complete",
                "completed": PART_SIZE,
                "expected": PART_SIZE,
                "annotations_sha256": digest(records),
                "binding_sha256": "b" * 64,
            },
        )
    return label_root, sampling_root


def test_collect_source_merges_parts_in_order(tmp_path: Path) -> None:
    label_root, sampling_root = _fixture(tmp_path)
    records, summary = collect_source("gpt", label_root, sampling_root, parts=PARTS, part_size=PART_SIZE)
    assert len(records) == PARTS * PART_SIZE
    assert [r["identity"]["problem_id"] for r in records[:3]] == ["p0__sample_00", "p0__sample_00", "a0__sample_00"]
    assert summary["rows"] == 6 and summary["unknown"] == 0
    assert summary["labels"]["Read"] == 2 and summary["labels"]["Plan"] == 2 and summary["labels"]["Verify"] == 2
    assert summary["datasets"] == {"aime24": 2, "math500": 4}
    assert summary["cells"] == {"aime24|incorrect|Q4": 2, "math500|correct|Q1": 4}
    assert summary["traces"] == 4 and summary["problems"] == 4
    assert summary["annotations_sha256"] == digest(records)
    assert [part["part"] for part in summary["parts"]] == [0, 1]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda label_root, _s: _write(label_root / "part-01" / "summary.json", {"status": "running", "completed": 1, "expected": PART_SIZE}), "not a complete"),
        (lambda label_root, _s: _write(label_root / "part-01" / "annotations.json", _part_records(1)[::-1]), "annotations_sha256"),
        (lambda label_root, _s: (label_root / "part-01" / "annotations.json").unlink(), "annotations.json"),
        (lambda _l, sampling_root: _write(sampling_root / "parts" / "part-00" / "part_manifest.json", {"part": 0, "identities": PART_SIZE, "sampling_manifest_sha256": "other", "datasets": {}, "cells": {}}), "different sampling plan"),
    ],
)
def test_collect_source_rejects_broken_parts(tmp_path: Path, mutate, message) -> None:
    label_root, sampling_root = _fixture(tmp_path)
    mutate(label_root, sampling_root)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        collect_source("gpt", label_root, sampling_root, parts=PARTS, part_size=PART_SIZE)


def test_collect_source_rejects_duplicate_identities_across_parts(tmp_path: Path) -> None:
    label_root, sampling_root = _fixture(tmp_path)
    records = _part_records(0)
    _write(label_root / "part-01" / "annotations.json", records)
    _write(
        label_root / "part-01" / "summary.json",
        {"status": "complete", "completed": PART_SIZE, "expected": PART_SIZE, "annotations_sha256": digest(records)},
    )
    with pytest.raises(ValueError, match="duplicate identity"):
        collect_source("gpt", label_root, sampling_root, parts=PARTS, part_size=PART_SIZE)


def test_collect_source_rejects_trace_counts_that_differ_from_the_part_manifest(tmp_path: Path) -> None:
    label_root, sampling_root = _fixture(tmp_path)
    manifest_path = sampling_root / "parts" / "part-00" / "part_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"] = {"aime24": 1, "math500": 2}
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="differ from"):
        collect_source("gpt", label_root, sampling_root, parts=PARTS, part_size=PART_SIZE)


def test_verify_only_writes_nothing(tmp_path: Path) -> None:
    label_root, sampling_root = _fixture(tmp_path)
    merged = tmp_path / "merged"
    manifest = export(
        {"gpt": {"label_root": label_root, "sampling_root": sampling_root}},
        merged_root=merged, zip_path=tmp_path / "out.zip", launch_record=None,
        verify_only=True, parts=PARTS, part_size=PART_SIZE,
    )
    assert manifest["verify_only"] is True
    assert manifest["totals"] == {"rows": 6, "unknown": 0, "per_source": {"gpt": 6}}
    assert not merged.exists() and not (tmp_path / "out.zip").exists()


def test_export_publishes_merged_tree_manifest_sums_and_zip(tmp_path: Path) -> None:
    gpt = _fixture(tmp_path, "gpt")
    gemma = _fixture(tmp_path, "gemma")
    merged = tmp_path / "merged"
    launch = tmp_path / "launch_record.json"
    _write(launch, {"jobs": {"gpt/part-00": "1"}})
    manifest = export(
        {
            "gpt": {"label_root": gpt[0], "sampling_root": gpt[1]},
            "gemma": {"label_root": gemma[0], "sampling_root": gemma[1]},
        },
        merged_root=merged, zip_path=tmp_path / "out" / "labels.zip", launch_record=launch,
        verify_only=False, parts=PARTS, part_size=PART_SIZE,
    )
    assert json.loads((merged / "gpt" / "annotations.json").read_text()) == _part_records(0) + _part_records(1)
    assert json.loads((merged / "gemma" / "summary.json").read_text())["rows"] == 6
    assert json.loads((merged / "summary.json").read_text())["rows"] == 12
    written = json.loads((merged / "manifest.json").read_text())
    assert written["launch_record"] == {"jobs": {"gpt/part-00": "1"}}
    assert written["totals"]["per_source"] == {"gpt": 6, "gemma": 6}
    assert set(written["sources"]) == {"gpt", "gemma"}
    assert (merged / "sampling" / "gpt" / "sampling_manifest.json").is_file()
    sums = (merged / "SHA256SUMS.txt").read_text().splitlines()
    assert {line.split("  ", 1)[1] for line in sums} == {
        "merged/gpt/annotations.json", "merged/gpt/summary.json", "sampling/gpt/sampling_manifest.json",
        "merged/gemma/annotations.json", "merged/gemma/summary.json", "sampling/gemma/sampling_manifest.json",
        "merged/summary.json", "manifest.json", "README.txt",
    }
    with zipfile.ZipFile(tmp_path / "out" / "labels.zip") as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert names == {line.split("  ", 1)[1] for line in sums} | {"SHA256SUMS.txt"}
        for line in sums:
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert "Supersedes" in archive.read("README.txt").decode()
    assert manifest["zip"]["sha256"] == hashlib.sha256((tmp_path / "out" / "labels.zip").read_bytes()).hexdigest()
    # a second export against unchanged inputs republishes the same rows
    again = export(
        {"gpt": {"label_root": gpt[0], "sampling_root": gpt[1]}, "gemma": {"label_root": gemma[0], "sampling_root": gemma[1]}},
        merged_root=merged, zip_path=None, launch_record=launch, verify_only=False, parts=PARTS, part_size=PART_SIZE,
    )
    assert again["totals"] == manifest["totals"]


def test_cli_accepts_root_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    label_root, sampling_root = _fixture(tmp_path, "qwen36")
    main(
        [
            "--sources", "qwen36",
            "--label-root", f"qwen36={label_root}",
            "--sampling-root", f"qwen36={sampling_root}",
            "--merged-root", str(tmp_path / "merged"),
            "--no-zip", "--parts", str(PARTS), "--part-size", str(PART_SIZE),
        ]
    )
    out = capsys.readouterr().out
    assert '"per_source": {"qwen36": 6}' in out
    assert (tmp_path / "merged" / "qwen36" / "annotations.json").is_file()
    with pytest.raises(ValueError, match="SOURCE=PATH"):
        main(["--sources", "gpt", "--label-root", "gemma=/x", "--verify-only"])


def test_every_source_key_is_registered() -> None:
    assert set(SOURCES) == {"gpt", "gemma", "qwen36", "nemotron", "glm", "qwen330b"}
    for name, spec in SOURCES.items():
        assert spec["generation_model"]
        assert str(spec["label_root"]).endswith(name)
        assert "sampling-100k-stratified" in str(spec["sampling_root"])
    # gpt reads the re-sampled root: its first sample had degenerate length quartiles
    assert str(SOURCES["gpt"]["sampling_root"]).endswith("sampling-100k-stratified-v2")


def test_readme_counts_sources_instead_of_hardcoding_four(tmp_path: Path) -> None:
    label_root, sampling_root = _fixture(tmp_path, "glm")
    merged = tmp_path / "merged"
    export(
        {"glm": {"label_root": label_root, "sampling_root": sampling_root}},
        merged_root=merged, zip_path=None, launch_record=None,
        verify_only=False, parts=PARTS, part_size=PART_SIZE,
    )
    readme = (merged / "README.txt").read_text()
    assert "All 1 sources were labelled" in readme
    assert "All four sources" not in readme


def _forward_adapter_fixture(tmp_path):
    from moe_exp.correlation_pipeline.annotation_batch import run_part
    from moe_exp.correlation_pipeline.annotation_partition import enumerate_items
    from moe_exp.schemas import TraceRecord
    sampling = tmp_path / "sample"
    labels = tmp_path / "labels"
    model = SOURCES["gpt"]["generation_model"]
    slug = "openai--gpt-oss-20b"
    plan = {"schema_version": 1, "parts": 2, "part_size": 3,
            "selected_sentences": 6, "generation_model": model,
            "generation_slug": slug, "datasets": ["math500"], "seed": 42,
            "selection": "stratified-accuracy-reasoning-length"}
    trace_hash = digest({"prompt": "Question", "messages": None,
                         "system_prompt": None, "cot_text": "One. Two. Three."})
    plan["completions"] = [
        {"dataset": "math500", "problem_id": "same-local-id", "source_problem_id": "p",
         "sample_id": part, "trace_sha256": trace_hash, "selected_sentences": 3,
         "is_correct": bool(part), "quartile": part+1, "reasoning_tokens": 3}
        for part in range(2)
    ]
    plan["completions"].append({**plan["completions"][0], "problem_id": "zero-allocation", "selected_sentences": 0})
    plan["manifest_sha256"] = digest(plan)
    _write(sampling / "sampling_manifest.json", plan)
    config = dict(program_sha256="a" * 64, judge_model="frozen", batch_size=64,
                  max_tokens=100, temperature=1.0, top_p=.95, top_k=20, min_p=0,
                  presence_penalty=0, repetition_penalty=1, reasoning_effort="low",
                  enable_thinking=True, preserve_thinking=False)
    traces = []
    unknown = {"status": "unknown", "raw_completion": "bad choice",
               "failure": {"kind": "parse", "error_type": "ValueError",
                           "message": "invalid label", "finish_reason": "stop"}}
    for part in range(2):
        trace = TraceRecord(
            dataset="math500", problem_id="same-local-id", source_problem_id="p",
            sample_id=part, prompt="Question", gold_answer="1", model_id=model,
            model_answer="1", is_correct=bool(part), cot_text="One. Two. Three.",
            metadata={"sentence_selection": {
                "schema_version": 1, "manifest_sha256": plan["manifest_sha256"],
                "selection": plan["selection"], "indices": [0, 1, 2],
                "stratum": {"dataset": "math500", "correct": bool(part), "quartile": part+1, "reasoning_tokens": 3},
            }},
        )
        traces.append(trace)
        part_root = sampling / "parts" / f"part-{part:02d}"
        root = part_root / "generation" / slug
        path = root / "math500/traces.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(trace.model_dump_json() + "\n")
        _write(part_root / "part_manifest.json", {
            "part": part, "identities": 3, "datasets": {"math500": 1},
            "sampling_manifest_sha256": plan["manifest_sha256"],
            "cells": {f"math500|correct|Q{part+1}": 3}, "trace_root": str(root),
        })
        run_part(enumerate_items(root, ("math500",)), config,
                 output_dir=labels / f"part-{part:02d}",
                 classify=lambda rows: ["Plan", unknown, "Verify"], expected_count=3)
    combined = sampling / "generation" / slug / "math500/traces.jsonl"
    combined.parent.mkdir(parents=True)
    combined.write_text("".join(t.model_dump_json() + "\n" for t in traces))
    return labels, sampling, combined


def test_forward_adapter_roundtrip_preserves_unknown_and_composite_identity(tmp_path):
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    from moe_exp.correlation_pipeline.spans import validate_annotation
    from moe_exp.schemas import TraceRecord
    labels, sampling, combined = _forward_adapter_fixture(tmp_path)
    output = tmp_path / "adapted"
    report = adapt_source("gpt", labels, sampling, output, parts=2, part_size=3)
    rows = [json.loads(line) for line in
            (output / "openai--gpt-oss-20b/math500/annotations.jsonl").read_text().splitlines()]
    traces = [TraceRecord(**json.loads(line)) for line in combined.read_text().splitlines()]
    assert report["unknown"] == 2 and report["traces"] == 2
    assert [row["sample_id"] for row in rows] == [0, 1]
    for trace, row in zip(traces, rows):
        validate_annotation(trace, row)
        assert row["coverage"] == {"selected": 3, "labeled": 2, "unknown": 1}
        assert row["units"][1]["failure"]["message"] == "invalid label"
        assert "label" not in row["units"][1]
    assert adapt_source("gpt", labels, sampling, output, parts=2, part_size=3) == report


@pytest.mark.parametrize("damage", [
    "manifest_hash", "missing", "duplicate", "offset", "trace_digest",
    "source_model", "classifier", "checkpoint", "unknown_failure", "stratum",
])
def test_forward_adapter_rejects_stale_or_incomplete_bindings(tmp_path, damage):
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    labels, sampling, combined = _forward_adapter_fixture(tmp_path)
    if damage == "manifest_hash":
        path = sampling / "sampling_manifest.json"
        value = json.loads(path.read_text()); value["seed"] = 999; _write(path, value)
    elif damage in {"missing", "duplicate", "source_model", "stratum"}:
        rows = [json.loads(line) for line in combined.read_text().splitlines()]
        if damage == "missing":
            rows.pop()
        elif damage == "duplicate":
            rows.append(rows[0])
        elif damage == "stratum":
            rows[0]["metadata"]["sentence_selection"]["stratum"]["quartile"] = 4
        else:
            rows[0]["model_id"] = "different/source"
        combined.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif damage in {"offset", "trace_digest", "unknown_failure"}:
        path = labels / "part-00/annotations.json"
        records = json.loads(path.read_text())
        if damage == "offset":
            records[0]["unit"]["start"] += 1
            records[0]["identity"]["start"] += 1
        elif damage == "trace_digest":
            records[0]["identity"]["trace_sha256"] = "stale"
        else:
            records[1]["failure"] = {}
        _write(path, records)
        summary_path = path.parent / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["annotations_sha256"] = digest(records); _write(summary_path, summary)
    else:
        path = labels / "part-01/checkpoints/batch-000000.json"
        value = json.loads(path.read_text())
        if damage == "classifier":
            value["binding"]["config"]["temperature"] = .2
        else:
            value["binding"]["items_sha256"] = "stale"
        _write(path, value)
        summary_path = labels / "part-01/summary.json"
        summary = json.loads(summary_path.read_text())
        summary["binding_sha256"] = digest(value["binding"]); _write(summary_path, summary)
    with pytest.raises(ValueError):
        adapt_source("gpt", labels, sampling, tmp_path / "out", parts=2, part_size=3)
    assert not (tmp_path / "out").exists()
