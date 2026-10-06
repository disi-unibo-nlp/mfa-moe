from __future__ import annotations

import pytest

from moe_exp.routing_control.live_gate_v1 import LivePrefixGate, QueryCostLedger


def decode(ids):
    return "".join(chr(token) for token in ids)


def encoded(text):
    return tuple(map(ord, text))


def test_only_complete_past_prefix_can_request_side_query():
    gate = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
    assert gate.observe_tokens("Find x", encoded("We get $x=5"), decode) == []
    assert gate.observe_tokens("Find x", encoded("We get $x=5$."), decode) == []
    current = "We get $x=5$. "
    queries = gate.observe_tokens("Find x", encoded(current), decode)
    assert len(queries) == 1
    assert queries[0].reader_input() == {
        "problem": "Find x", "emitted_prefix": current,
        "triggering_sentence": current.strip(),
    }
    assert gate.observe_tokens("Find x", encoded(current), decode) == []
    assert gate.observe_tokens("Find x", encoded(current + "</think>"), decode) == []
    assert gate.observe_tokens("Find x", encoded(current + "</think> More $x=6$. "), decode) == []


def test_future_fields_cannot_change_live_decision_or_cost_count():
    prefix = "We get $x=5$. "
    base = {"problem": "Find x", "emitted_token_ids": encoded(prefix)}
    changed = {**base, "emitted_text": "future counterfeit", "gold_answer": "5",
               "correctness": False, "next_sentence": "We check it.",
               "future_completion": "The answer is 6.", "future_labels": ["verify"]}
    a = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
    b = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
    assert LivePrefixGate(["candidate_to_verify"]).observe_record(base, decode) == []
    first, second = a.observe_record(base, decode), b.observe_record(changed, decode)
    assert first == second
    assert len(first) == 1
    ledger = QueryCostLedger()
    ledger.record(first[0].request_id, "vetoed", 412, 27, 1.5)
    assert ledger.totals()["prefill_tokens"] == 412
    assert ledger.totals()["generated_tokens"] == 27
    assert ledger.totals()["statuses"]["vetoed"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record(first[0].request_id, "accepted", 412, 27, 1.5)
