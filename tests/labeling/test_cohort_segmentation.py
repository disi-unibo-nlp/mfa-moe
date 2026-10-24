import json

import pytest

from labeling import sample_cohorts
from labeling.spans import reasoning_ranges, sentence_spans
from schemas import TraceRecord

REASONING = "First we set up the equation. Then we carve at each corner. So the answer is 4."


def gpt_trace(reasoning_content, cot_reasoning=REASONING, problem_id="p1", dataset="aime24"):
    cot = (f"<|channel|>analysis<|message|>{cot_reasoning}<|end|><|start|>assistant"
           "<|channel|>final<|message|>\\boxed{4}<|return|>")
    return dict(dataset=dataset, problem_id=problem_id, source_problem_id=problem_id, sample_id=0,
                prompt="Q", gold_answer="4", model_id="openai/gpt-oss-20b", model_answer="4",
                is_correct=True, cot_text=cot,
                metadata={"reasoning_content": reasoning_content, "assistant_content": "\\boxed{4}",
                          "token_replay": {"completion_token_ids": [1, 2, 3]}, "termination": "stop"})


def test_reserved_special_token_in_parsed_reasoning_is_located_in_replay_text():
    # vLLM keeps the reserved token as text; the exact replay decodes it to nothing
    parsed = REASONING.replace("carve at", "carve<|reserved_200871|> at")
    trace = TraceRecord(**gpt_trace(parsed))
    (start, end), = reasoning_ranges(trace)
    assert trace.cot_text[start:end] == REASONING
    assert sentence_spans(trace)


def test_unchanged_inputs_keep_their_spans():
    trace = TraceRecord(**gpt_trace(REASONING))
    (start, end), = reasoning_ranges(trace)
    assert trace.cot_text[start:end] == REASONING


def test_genuine_mismatch_still_raises():
    with pytest.raises(ValueError, match="does not occur"):
        reasoning_ranges(TraceRecord(**gpt_trace("completely different reasoning")))


def write_corpus(root, rows_by_dataset):
    for dataset in sample_cohorts.DATASETS:
        path = root / dataset / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows_by_dataset.get(dataset, [])))


def test_scan_excludes_unsegmentable_traces_and_binds_every_line(tmp_path, monkeypatch):
    good = [gpt_trace(REASONING, problem_id=f"g{i}") for i in range(299)]
    bad = gpt_trace("completely different reasoning", problem_id="bad")
    write_corpus(tmp_path, {"aime24": good + [bad]})
    rows, excluded, line_hashes = sample_cohorts.scan(tmp_path)
    assert len(rows) == 299 and len(line_hashes) == 300
    assert [e["problem_id"] for e in excluded] == ["bad"] and "does not occur" in excluded[0]["reason"]
    # more than 0.5% unsegmentable traces fails the whole model instead of silently shrinking
    write_corpus(tmp_path, {"aime24": good[:100] + [bad]})
    with pytest.raises(ValueError, match="cannot be segmented"):
        sample_cohorts.scan(tmp_path)
