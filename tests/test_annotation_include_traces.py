"""Runner: targeted --include-traces labeling, exact small parts and resume."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from moe_exp.correlation_pipeline.annotation_batch import main, run_part  # noqa: E402
from moe_exp.correlation_pipeline.annotation_partition import (  # noqa: E402
    DATASETS,
    enumerate_items,
)
from moe_exp.schemas import TraceRecord  # noqa: E402

MERGED_TEXT = (
    "Intro. Value is 5$ First sentence here. Second sentence here. "
    "And $x^{2}$ is the formula. Done."
)
PLAIN_TEXT = "Alpha sentence. Beta sentence."


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


def write_corpus(root: Path) -> None:
    for dataset in DATASETS:
        directory = root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        if dataset == "math500":
            rows = [
                trace_row("math500", "merged", MERGED_TEXT),
                trace_row("math500", "plain", PLAIN_TEXT),
            ]
        (directory / "traces.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def write_affected(path: Path, keys: list[tuple[str, str, int]], units: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_keys": [
                    {"dataset": dataset, "problem_id": problem_id, "sample_id": sample_id}
                    for dataset, problem_id, sample_id in keys
                ],
                "expected_units": units,
            }
        ),
        encoding="utf-8",
    )


class IncludeTracesTests(unittest.TestCase):
    def test_include_traces_selects_only_the_requested_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_corpus(root)
            items = enumerate_items(root, DATASETS, include={("math500", "merged", 0)})
            self.assertEqual([item[0]["problem_id"] for item in items], ["merged"] * len(items))
            self.assertEqual(
                [item[2]["text"] for item in items],
                [
                    "Intro.",
                    "Value is 5$ First sentence here.",
                    "Second sentence here.",
                    "And $x^{2}$ is the formula.",
                    "Done.",
                ],
            )
            with self.assertRaises(ValueError):
                enumerate_items(root, DATASETS, include={("math500", "missing", 0)})

    def test_small_part_runs_with_its_own_expected_count(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            root = directory / "corpus"
            write_corpus(root)
            items = enumerate_items(root, DATASETS, include={("math500", "merged", 0)})
            calls = []

            def classify(batch):
                calls.append(len(batch))
                return ["Analyze"] * len(batch)

            with self.assertRaises(ValueError):
                run_part(items, {"contract": 1}, output_dir=directory / "bad", classify=classify)
            result = run_part(
                items,
                {"contract": 1},
                output_dir=directory / "part",
                classify=classify,
                expected_count=len(items),
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed"], len(items))
            self.assertEqual(calls, [len(items)])
            records = json.loads((directory / "part/annotations.json").read_text())
            self.assertEqual(len(records), len(items))
            self.assertTrue(all(record["label"] == "Analyze" for record in records))

    def test_plan_only_reports_the_include_count_without_the_judge(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            root = directory / "corpus"
            write_corpus(root)
            affected = directory / "affected.json"
            write_affected(affected, [("math500", "merged", 0)], {"gpt": 5})
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main(
                    [
                        "--trace-root", str(root),
                        "--output-dir", str(directory / "out"),
                        "--part", "0", "--parts", "1",
                        "--part-size", "5", "--total", "5",
                        "--judge-program", str(directory / "unused-program.json"),
                        "--model", "Qwen/Qwen3.8-27B",
                        "--base-url", "http://127.0.0.1:41820/v1",
                        "--include-traces", str(affected),
                        "--plan-only",
                    ]
                )
            payload = json.loads(buffer.getvalue().strip().splitlines()[-1])
            self.assertEqual(payload["part_count"], 5)
            self.assertEqual(payload["source_plan"]["selected_identities"], 5)
            self.assertFalse((directory / "out").exists())

    def test_wrong_total_is_rejected_before_the_judge_loads(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            root = directory / "corpus"
            write_corpus(root)
            affected = directory / "affected.json"
            write_affected(affected, [("math500", "merged", 0)], {"gpt": 5})
            with self.assertRaises(ValueError):
                main(
                    [
                        "--trace-root", str(root),
                        "--output-dir", str(directory / "out"),
                        "--part", "0", "--parts", "1",
                        "--part-size", "7", "--total", "7",
                        "--judge-program", str(directory / "unused-program.json"),
                        "--model", "Qwen/Qwen3.8-27B",
                        "--base-url", "http://127.0.0.1:41820/v1",
                        "--include-traces", str(affected),
                    ]
                )

    def test_resume_after_a_kill_does_not_re_request_finished_batches(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            root = directory / "corpus"
            write_corpus(root)
            items = enumerate_items(root, DATASETS, include={("math500", "merged", 0)})
            oversized = [
                (*item[:2], item[2], item[3]) for item in items
            ] * 13  # 65 rows force a second batch of 64
            calls = []

            def dying_classify(batch):
                calls.append(len(batch))
                if len(calls) > 1:
                    raise RuntimeError("client killed by the stalled-write watchdog")
                return ["Analyze"] * len(batch)

            with self.assertRaises(RuntimeError):
                run_part(
                    oversized,
                    {"contract": 2},
                    output_dir=directory / "part",
                    classify=dying_classify,
                    expected_count=len(oversized),
                )
            self.assertEqual(calls, [64, 1])

            resumed_calls = []

            def recording_classify(batch):
                resumed_calls.append(len(batch))
                return ["Verify"] * len(batch)

            result = run_part(
                oversized,
                {"contract": 2},
                output_dir=directory / "part",
                classify=recording_classify,
                expected_count=len(oversized),
            )
            self.assertEqual(resumed_calls, [1])
            self.assertEqual(result["completed"], len(oversized))
            records = json.loads((directory / "part/annotations.json").read_text())
            self.assertEqual([record["label"] for record in records[:64]], ["Analyze"] * 64)
            self.assertEqual(records[64]["label"], "Verify")


if __name__ == "__main__":
    unittest.main(verbosity=2)
