"""Merge tool: v1 rows stay authoritative, affected traces come from the v2 run."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from moe_exp.correlation_pipeline import annotation_partition as partition  # noqa: E402
from moe_exp.correlation_pipeline import spans  # noqa: E402
from moe_exp.correlation_pipeline.annotation_batch import run_part  # noqa: E402
from moe_exp.correlation_pipeline.annotation_merge import main  # noqa: E402
from moe_exp.correlation_pipeline.annotation_partition import DATASETS  # noqa: E402

MERGED_TEXT = (
    "Intro. Value is 5$ First sentence here. Second sentence here. "
    "And $x^{2}$ is the formula. Done."
)
PLAIN_TEXT = "Alpha sentence. Beta sentence."
SOURCES = ("gpt", "gemma")


def trace_row(dataset: str, problem_id: str, text: str) -> dict:
    return {
        "dataset": dataset,
        "problem_id": problem_id,
        "source_problem_id": problem_id,
        "sample_id": 0,
        "prompt": "Question?",
        "gold_answer": "1",
        "model_id": "test",
        "model_answer": "1",
        "is_correct": True,
        "cot_text": text,
        "metadata": {},
    }


def merged_id(source: str) -> str:
    """Each corpus has its own affected trace: the same id may exist in both corpora."""
    return f"merged_{source}"


def write_corpus(root: Path, source: str) -> None:
    for dataset in DATASETS:
        directory = root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        if dataset == "math500":
            rows = [
                trace_row("math500", merged_id(source), MERGED_TEXT),
                trace_row("math500", "plain", PLAIN_TEXT),
            ]
        (directory / "traces.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def label(text: str) -> str:
    return "Analyze"


def build_run(root: Path, output: Path, *, legacy: bool, include=None, expected=None) -> int:
    """Run one annotation part over the synthetic corpus and return its row count."""
    original = partition.sentence_spans
    if legacy:
        partition.sentence_spans = spans.sentence_spans_v1
    try:
        items = partition.enumerate_items(root, DATASETS, include=include)
    finally:
        partition.sentence_spans = original
    result = run_part(
        items,
        {"contract": "test"},
        output_dir=output,
        classify=lambda batch: [label("")] * len(batch),
        expected_count=len(items) if expected is None else expected,
    )
    return result["completed"]


def write_affected(path: Path, keys, units, verified) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_keys": [
                    {
                        "dataset": dataset,
                        "problem_id": problem_id,
                        "sample_id": sample_id,
                        "source": source,
                    }
                    for dataset, problem_id, sample_id, source in keys
                ],
                "expected_units": units,
                "verified_units": verified,
            }
        ),
        encoding="utf-8",
    )


def build_workspace(directory: Path):
    roots = {}
    keys = {}
    v1_root = directory / "v1"
    v2_root = directory / "v2"
    for source in SOURCES:
        corpus = directory / f"corpus-{source}"
        write_corpus(corpus, source)
        roots[source] = corpus
        keys[source] = ("math500", merged_id(source), 0)
        v1_rows = build_run(corpus, v1_root / source / "part-00", legacy=True)
        v2_rows = build_run(
            corpus, v2_root / source / "part-00", legacy=False, include={keys[source]}
        )
        assert v1_rows == 5, v1_rows
        assert v2_rows == 5, v2_rows
    affected = directory / "affected.json"
    write_affected(
        affected,
        [(key[0], key[1], key[2], source) for source, key in keys.items()],
        {"gpt": 5, "gemma": 5},
        {"gpt": {"v2": 7}, "gemma": {"v2": 7}},
    )
    return roots, v1_root, v2_root, affected


def merge_argv(roots, v1_root, v2_root, affected, merged_root, *extra):
    argv = [
        "--v1-root", str(v1_root),
        "--v2-root", str(v2_root),
        "--merged-root", str(merged_root),
        "--affected-traces", str(affected),
        "--parts", "1", "--part-size", "5", "--total", "5",
    ]
    for source, root in roots.items():
        argv.extend(["--trace-root", f"{source}={root}"])
    argv.extend(extra)
    return argv


def run_merge(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(argv)
    return [json.loads(line) for line in buffer.getvalue().strip().splitlines() if line.startswith("{")]


class MergeTests(unittest.TestCase):
    def test_dry_run_reports_the_v2_sequence_and_superseded_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            merged_root = directory / "merged"
            printed = run_merge(
                merge_argv(roots, v1_root, v2_root, affected, merged_root, "--dry-run", "--verify-units")
            )
            summary = printed[0]
            reports = {line["source"]: line for line in printed[1:] if "source" in line}
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["rows"], 14)
            # Only the plain traces are reused; the affected trace is replaced by its v2 units.
            self.assertEqual(summary["reused_from_v1"], 4)
            self.assertEqual(summary["relabeled_from_v2"], 10)
            self.assertEqual(summary["superseded_v1_records"], 2)
            for source in SOURCES:
                report = reports[source]
                self.assertEqual(report["rows"], 7)
                self.assertEqual(report["traces"], 2)
                self.assertEqual(report["unknown"], 0)
                self.assertEqual(len(report["superseded_details"]), 1)
                self.assertEqual(report["affected_traces"][0]["v1_rows"], 3)
                self.assertEqual(report["affected_traces"][0]["v2_rows"], 5)
            self.assertFalse(merged_root.exists())

    def test_sources_and_verify_datasets_selectors(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            argv = merge_argv(
                roots, v1_root, v2_root, affected, directory / "merged",
                "--dry-run", "--verify-units", "--sources", "gpt", "--verify-datasets", "math500",
            )
            printed = run_merge(argv)
            summary = printed[0]
            reports = {line["source"]: line for line in printed[1:] if "source" in line}
            self.assertEqual(sorted(reports), ["gpt"])
            self.assertEqual(summary["rows"], 7)
            self.assertEqual(reports["gpt"]["verification"]["datasets"], ["math500"])
            self.assertIn("traces_verified", reports["gpt"]["verification"])

    def test_publish_writes_both_sources_and_a_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            merged_root = directory / "merged"
            run_merge(merge_argv(roots, v1_root, v2_root, affected, merged_root))
            for source in SOURCES:
                rows = json.loads((merged_root / source / "annotations.json").read_text())
                self.assertEqual(len(rows), 7)
                self.assertEqual(
                    [record["unit"]["text"] for record in rows if record["unit"]["text"].startswith("Value is 5$")],
                    ["Value is 5$ First sentence here."],
                )
                self.assertTrue((merged_root / source / "summary.json").is_file())
            summary = json.loads((merged_root / "summary.json").read_text())
            self.assertEqual(summary["rows"], 14)

    def test_missing_v2_sub_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            path = v2_root / "gpt/part-00/annotations.json"
            rows = json.loads(path.read_text())
            rows = [row for row in rows if row["unit"]["text"] != "Second sentence here."]
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises((ValueError, SystemExit)):
                run_merge(merge_argv(roots, v1_root, v2_root, affected, directory / "merged", "--dry-run"))

    def test_truncated_v2_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            path = v2_root / "gemma/part-00/annotations.json"
            rows = json.loads(path.read_text())
            for row in rows:
                if row["unit"]["text"] == "Second sentence here.":
                    row["unit"]["end"] -= 5
                    row["unit"]["text"] = row["unit"]["text"][:-5]
                    row["identity"]["end"] = row["unit"]["end"]
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_merge(merge_argv(roots, v1_root, v2_root, affected, directory / "merged", "--dry-run"))

    def test_drifted_v1_label_is_rejected_with_verify_units(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots, v1_root, v2_root, affected = build_workspace(directory)
            path = v1_root / "gpt/part-00/annotations.json"
            rows = json.loads(path.read_text())
            drifted = [row for row in rows if row["identity"]["problem_id"] == "plain"]
            self.assertTrue(drifted)
            drifted[0]["unit"]["text"] = drifted[0]["unit"]["text"] + " drift"
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_merge(
                    merge_argv(roots, v1_root, v2_root, affected, directory / "merged", "--dry-run", "--verify-units")
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
