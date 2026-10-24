import json

import pytest

from labeling import choose_sentences
from labeling.spans import reasoning_ranges, sentence_spans
from schemas import TraceRecord

REASONING = "First we set up the equation. Then we carve at each corner. So the answer is 4."


def gpt_trace(reasoning_content, cot_reasoning=REASONING, problem_id="p1", dataset="aime24", profile="gpt"):
    cot = (f"<|channel|>analysis<|message|>{cot_reasoning}<|end|><|start|>assistant"
           "<|channel|>final<|message|>\\boxed{4}<|return|>")
    return dict(dataset=dataset, problem_id=problem_id, source_problem_id=problem_id, sample_id=0,
                prompt="Q", gold_answer="4", model_id="openai/gpt-oss-20b", model_answer="4",
                is_correct=True, cot_text=cot,
                metadata={"reasoning_content": reasoning_content, "assistant_content": "\\boxed{4}",
                          "token_replay": {"completion_token_ids": [1, 2, 3]}, "termination": "stop",
                          "generation_config": {"profile": profile}})


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
    for dataset in choose_sentences.DATASETS:
        path = root / dataset / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows_by_dataset.get(dataset, [])))


def test_scan_excludes_unsegmentable_traces_and_binds_every_line(tmp_path, monkeypatch):
    good = [gpt_trace(REASONING, problem_id=f"g{i}") for i in range(299)]
    bad = gpt_trace("completely different reasoning", problem_id="bad")
    write_corpus(tmp_path, {"aime24": good + [bad]})
    rows, excluded, line_hashes, model = choose_sentences.scan(tmp_path)
    assert len(rows) == 299 and len(line_hashes) == 300 and model == "gpt"
    assert [e["problem_id"] for e in excluded] == ["bad"] and "does not occur" in excluded[0]["reason"]
    # more than 0.5% unsegmentable traces fails the whole model instead of silently shrinking
    write_corpus(tmp_path, {"aime24": good[:100] + [bad]})
    with pytest.raises(ValueError, match="cannot be segmented"):
        choose_sentences.scan(tmp_path)


def test_the_model_is_read_from_the_traces_and_all_traces_must_agree(tmp_path):
    good = [gpt_trace(REASONING, problem_id=f"g{i}") for i in range(3)]
    write_corpus(tmp_path, {"aime24": good, "aime25": [gpt_trace(REASONING, dataset="aime25")]})
    assert choose_sentences.scan(tmp_path)[3] == "gpt"
    write_corpus(tmp_path, {"aime24": good, "aime25": [gpt_trace(REASONING, dataset="aime25", profile="glm")]})
    with pytest.raises(ValueError, match="one model"):
        choose_sentences.scan(tmp_path)
    nameless = gpt_trace(REASONING, problem_id="nameless")
    del nameless["metadata"]["generation_config"]
    write_corpus(tmp_path, {"aime24": [nameless]})
    with pytest.raises(ValueError, match="does not name its model"):
        choose_sentences.scan(tmp_path)


def test_cohorts_carry_the_model_of_the_traces_and_never_overwrite_a_folder(tmp_path):
    traces = tmp_path / "generation" / "openai--gpt-oss-20b"
    write_corpus(traces, {"aime24": [gpt_trace(REASONING, problem_id=f"g{i}") for i in range(3)]})
    out = tmp_path / "cohorts"
    choose_sentences.main(["--traces-dir", str(traces), "--out", str(out)])
    cohorts = json.loads((out / "cohorts.json").read_text())
    assert cohorts["model"] == "gpt" and cohorts["counts"]["A"] == 3
    assert (out / "labels" / "part-00" / "generation" / "openai--gpt-oss-20b").is_dir()
    with pytest.raises(FileExistsError, match="never overwritten"):
        choose_sentences.main(["--traces-dir", str(traces), "--out", str(out)])
