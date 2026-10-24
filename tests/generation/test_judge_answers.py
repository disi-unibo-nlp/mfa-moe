"""The list of rejected answers that the judge checks: its model comes from the traces, and its ids stay the same."""
import argparse
import json

import pytest

from generation import judge_answers
from generation.judge_answers import build_item

DATASETS = ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva")


def trace(problem_id, is_correct, finish="stop", profile="gpt"):
    return {"problem_id": problem_id, "sample_id": 0, "prompt": "Q", "gold_answer": "4", "is_correct": is_correct,
            "cot_text": "reasoning", "metadata": {"assistant_content": "\\boxed{5}", "finish_reason": finish,
                                                  "generation_config": {"profile": profile}}}


def write_traces(root, rows_by_dataset):
    for dataset in DATASETS:
        path = root / dataset / "traces.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows_by_dataset.get(dataset, [])))


def list_rejected(traces_dir, out):
    judge_answers.main(["build-corpus", "--traces-dir", str(traces_dir), "--out", str(out)])
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_the_items_are_the_finished_rejected_answers_of_the_model_in_the_traces(tmp_path):
    write_traces(tmp_path / "traces", {"aime24": [trace("a", False), trace("b", True), trace("c", None),
                                                 trace("d", False, finish="length")]})
    items = list_rejected(tmp_path / "traces", tmp_path / "made" / "items.jsonl")  # the output folder is created
    assert [(item["model"], item["corpus"], item["problem_id"]) for item in items] == [("gpt", "card-v3", "a")]
    # the id hashes the short key of the model and the corpus name, exactly as before the model became an input
    assert items[0]["item_id"] == build_item(corpus="card-v3", model="gpt", dataset="aime24", problem_id="a",
                                             sample_id=0, problem="Q", gold="4", answer_text="\\boxed{5}")["item_id"]


def test_traces_of_two_models_are_refused(tmp_path):
    write_traces(tmp_path / "traces", {"aime24": [trace("a", False)], "aime25": [trace("a", False, profile="glm")]})
    with pytest.raises(ValueError, match="one model"):
        list_rejected(tmp_path / "traces", tmp_path / "items.jsonl")


def test_judging_creates_the_folder_of_the_verdict_file(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("")
    out = tmp_path / "new" / "folder" / "verdicts.jsonl"
    judge_answers.run(argparse.Namespace(items=str(items), out=str(out), base_url="http://unused/v1",
                                         workers=1, timeout=1))
    assert out.exists()
