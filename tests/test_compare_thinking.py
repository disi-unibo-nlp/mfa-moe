import json
import sys
from types import SimpleNamespace

import pytest

from moe_exp.gepaLLMAsJudge import compare_thinking as benchmark


def test_failed_predictions_count_as_wrong():
    case = {"question_id": "q", "unit_id": 0, "gold_label": "Read", "inputs": {}}

    def fail(**kwargs):
        raise TimeoutError("server timed out")

    failed = benchmark.evaluate_case(fail, case)
    assert failed["label"] is None
    assert "TimeoutError" in failed["error"]
    rows = [
        {**failed, "level": "off", "repeat": 0},
        {**failed, "level": "high", "repeat": 0, "label": "Read", "error": None},
    ]
    report = benchmark.summarize(rows, ["off", "high"])
    assert report["off"]["accuracy"] == 0
    assert report["off"]["errors"] == 1
    assert report["high"]["accuracy_delta_vs_baseline"] == 1
    assert report["high"]["disagreement_vs_baseline"] == 1


def test_comparison_reuses_context_and_records_all_levels(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    labels = dataset / "responses_labeled"
    labels.mkdir(parents=True)
    for question_id in ["validation", "locked"]:
        (labels / f"{question_id}.json").write_text(json.dumps({
            "Question ID": question_id,
            "data": [{"text": "Read the question", "gt-class-2": "Read"},
                     {"text": "Check the result", "gt-class-2": "Verify"}],
        }))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"split_question_ids": {"validation": ["validation"]}}))
    program = tmp_path / "program.json"
    program.write_text("{}")
    output = tmp_path / "output"
    settings = []
    inputs = []

    def make_predictor(args):
        settings.append((args.enable_thinking, args.reasoning_effort))

        def predict(**kwargs):
            inputs.append(kwargs)
            return SimpleNamespace(label="Read" if kwargs["sentence"].startswith("Read")
                                   else "Verify")

        return predict

    monkeypatch.setitem(sys.modules, "moe_exp.correlation_pipeline.annotate",
                        SimpleNamespace(make_predictor=make_predictor))
    benchmark.main([
        "--dataset-dir", str(dataset), "--split-file", str(split),
        "--judge-program", str(program), "--output-dir", str(output), "--repeats", "2",
        "--api-key", "secret-test-key",
    ])
    assert settings == [(False, "low"), (True, "low"), (True, "medium"), (True, "high")]
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert {row["question_id"] for row in rows} == {"validation"}
    assert all(set(item) == {"problem_statement", "previous_sentence", "sentence",
                              "next_sentence"} for item in inputs)
    assert len({json.dumps(item, sort_keys=True) for item in inputs}) == 2
    report = json.loads((output / "summary.json").read_text())
    assert all(metrics["accuracy"] == 1 for metrics in report.values())
    assert "secret-test-key" not in (output / "config.json").read_text()
    assert (output / "summary.csv").is_file()
    with pytest.raises(FileExistsError):
        benchmark.main(["--dataset-dir", str(dataset), "--split-file", str(split),
                        "--judge-program", str(program), "--output-dir", str(output)])
