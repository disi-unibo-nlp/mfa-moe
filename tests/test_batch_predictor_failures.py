"""Tolerant per-choice outcomes for the batched judge, without DSPy or a live server.

The fake adapter mirrors what DSPy's ChatAdapter does for the judge signature: `parse`
returns the raw assistant text in the shape the project's label normaliser expects, so
`parse_sentence_label` stays the only label validator under test. Transport is replaced
by a monkeypatched `post_batch`, which keeps the tests offline and deterministic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from moe_exp.correlation_pipeline import batch_predictor
from moe_exp.correlation_pipeline.batch_predictor import (
    classify_batch,
    classify_batch_outcomes,
)

ITEM = {
    "problem_statement": "problem",
    "previous_sentence": "<START OF RESPONSE>",
    "sentence": "target",
    "next_sentence": "<END OF RESPONSE>",
}

PREDICT = SimpleNamespace(signature=object(), demos=[])


class FakeAdapter:
    """Minimal stand-in for `dspy.adapters.chat_adapter.ChatAdapter`."""

    def format(self, signature, demos, inputs):
        return [{"role": "user", "content": inputs["sentence"]}]

    def parse(self, signature, completion):
        return {"label": completion}


ADAPTER = FakeAdapter()


def choice(index, content, finish_reason="stop"):
    entry = {"index": index, "message": {} if content is _MISSING else {"content": content}}
    if finish_reason is not _MISSING:
        entry["finish_reason"] = finish_reason
    return entry


_MISSING = object()


def _patch_transport(monkeypatch, choices):
    payloads = []

    def fake_post(payload, *, base_url, api_key, timeout=600.0):
        payloads.append(payload)
        return {"choices": choices}

    monkeypatch.setattr(batch_predictor, "post_batch", fake_post)
    return payloads


def _kwargs():
    return dict(
        adapter=ADAPTER,
        predict=PREDICT,
        model="judge-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="test-key",
        max_tokens=64,
        temperature=0.0,
        reasoning_effort="medium",
    )


def outcomes_for(monkeypatch, choices, items=None):
    _patch_transport(monkeypatch, choices)
    return classify_batch_outcomes(items if items is not None else [ITEM], **_kwargs())


def test_stopped_choice_returns_only_the_parsed_label(monkeypatch):
    outcomes = outcomes_for(monkeypatch, [choice(0, "Analyze")])
    assert outcomes == [{"label": "Analyze"}]


def test_non_stop_choice_is_unknown_even_when_its_text_is_a_valid_label(monkeypatch):
    outcomes = outcomes_for(monkeypatch, [choice(0, "Analyze", finish_reason="length")])
    (outcome,) = outcomes
    assert outcome["status"] == "unknown"
    assert outcome["raw_completion"] == "Analyze"
    assert outcome["failure"]["kind"] == "non_stop"
    assert outcome["failure"]["finish_reason"] == "length"
    assert outcome["failure"]["error_type"] == "ValueError"
    assert "length" in outcome["failure"]["message"]


def test_unparseable_stopped_choice_preserves_the_exact_text(monkeypatch):
    text = "not one of the labels"
    outcomes = outcomes_for(monkeypatch, [choice(0, text)])
    (outcome,) = outcomes
    assert outcome["status"] == "unknown"
    assert outcome["raw_completion"] == text
    assert outcome["failure"]["kind"] == "label_parse"
    assert outcome["failure"]["error_type"] == "LabelParseError"
    assert text in outcome["failure"]["message"]
    assert outcome["failure"]["finish_reason"] == "stop"


def test_stopped_choice_without_string_content_is_unknown(monkeypatch):
    outcomes = outcomes_for(monkeypatch, [choice(0, _MISSING)])
    (outcome,) = outcomes
    assert outcome["status"] == "unknown"
    assert outcome["raw_completion"] is None
    assert outcome["failure"]["kind"] == "missing_content"
    assert outcome["failure"]["finish_reason"] == "stop"


def test_out_of_order_choices_come_back_in_input_order(monkeypatch):
    items = [dict(ITEM, sentence=f"target {i}") for i in range(3)]
    scrambled = [choice(2, "Plan"), choice(0, "Read"), choice(1, "Analyze")]
    outcomes = outcomes_for(monkeypatch, scrambled, items=items)
    assert outcomes == [{"label": "Read"}, {"label": "Analyze"}, {"label": "Plan"}]


def test_strict_classify_batch_still_returns_label_strings(monkeypatch):
    _patch_transport(monkeypatch, [choice(0, "Analyze"), choice(1, "Plan")])
    labels = classify_batch([ITEM, dict(ITEM, sentence="other")], **_kwargs())
    assert labels == ["Analyze", "Plan"]
    assert all(isinstance(label, str) for label in labels)


def test_strict_classify_batch_still_fails_closed_on_a_bad_choice(monkeypatch):
    _patch_transport(monkeypatch, [choice(0, "Analyze", finish_reason="length")])
    with pytest.raises(ValueError):
        classify_batch([ITEM], **_kwargs())


def test_both_apis_send_the_same_batch_payload(monkeypatch):
    items = [ITEM, dict(ITEM, sentence="other")]
    choices = [choice(0, "Analyze"), choice(1, "Plan")]
    tolerant_payloads = _patch_transport(monkeypatch, choices)
    assert classify_batch_outcomes(items, **_kwargs()) == [
        {"label": "Analyze"},
        {"label": "Plan"},
    ]
    strict_payloads = _patch_transport(monkeypatch, choices)
    assert classify_batch(items, **_kwargs()) == ["Analyze", "Plan"]
    assert tolerant_payloads == strict_payloads


def _patch_context_limit(monkeypatch, too_long="HUGE", code=400):
    """Server stand-in: a batch containing an over-long conversation is rejected whole."""
    import io
    import urllib.error

    requests = []

    def fake_post(payload, *, base_url, api_key, timeout=600.0):
        texts = [conversation[0]["content"] for conversation in payload["messages"]]
        requests.append(texts)
        if any(text == too_long for text in texts):
            raise urllib.error.HTTPError(base_url, code, "Bad Request", {},
                                         io.BytesIO(b'{"message": "maximum context length exceeded"}'))
        return {"choices": [choice(i, "Plan") for i in range(len(texts))]}

    monkeypatch.setattr(batch_predictor, "post_batch", fake_post)
    return requests


def test_a_context_rejection_marks_only_the_offending_item_unknown(monkeypatch):
    items = [dict(ITEM, sentence=f"s{i}") for i in range(8)]
    items[5] = dict(ITEM, sentence="HUGE")
    requests = _patch_context_limit(monkeypatch)
    outcomes = classify_batch_outcomes(items, **_kwargs())
    assert [o.get("label") for o in outcomes] == ["Plan"] * 5 + [None] + ["Plan"] * 2
    assert outcomes[5]["status"] == "unknown"
    assert outcomes[5]["failure"]["kind"] == "request_rejected"
    assert "maximum context length" in outcomes[5]["failure"]["message"]
    assert len(requests) <= 1 + 2 * 3  # bisection, not one request per item


def test_other_http_errors_still_raise(monkeypatch):
    import urllib.error

    _patch_context_limit(monkeypatch, code=500)
    with pytest.raises(urllib.error.HTTPError):
        classify_batch_outcomes([dict(ITEM, sentence="HUGE")], **_kwargs())
