"""Prefix-only regressions for the prospective v2 live gate."""
from __future__ import annotations

import unittest

from moe_exp.routing_control.live_gate_v2 import LivePrefixGate
from moe_exp.routing_control.transitions_v24 import (
    StreamingTransitionDetectorV2,
    VERSION,
    classify_sentence_v2,
)


def encoded(text: str) -> tuple[int, ...]:
    return tuple(map(ord, text))


def decode(ids) -> str:
    return "".join(map(chr, ids))


def record(text: str, **extra) -> dict:
    return {"problem": "Find x", "emitted_token_ids": encoded(text),
            "emitted_text": text, **extra}


class LiveGateV2Tests(unittest.TestCase):
    def test_closed_display_math_and_truncated_stream(self):
        self.assertEqual(VERSION, "prefix-transition-candidates-v2.4-discovery")
        self.assertIn("candidate_to_verify",
                      classify_sentence_v2(r"We obtain \[x=2\]."))
        self.assertEqual(classify_sentence_v2(r"We obtain \[x=2"), ())

        gate = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
        first = "<think>We obtain \\[x=2\n"
        self.assertEqual(gate.observe_tokens("Find x", encoded(first), decode), [])
        # The newline above must remain part of the open math span, so the
        # completed sentence is still available when its close arrives.
        completed = first + "3\\]. "
        queries = gate.observe_tokens("Find x", encoded(completed), decode)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].transition, "candidate_to_verify")
        self.assertEqual(queries[0].triggering_sentence,
                         completed.strip())
        self.assertEqual(gate.observe_tokens("Find x", encoded(completed), decode), [])

    def test_candidate_must_start_and_end_in_same_sentence(self):
        detector = StreamingTransitionDetectorV2()
        first = "The answer is 3. "
        self.assertEqual([e.transition for e in detector.observe(record(first))["events"]],
                         ["candidate_to_verify"])
        # The numeric parser's full lookahead now reaches the next sentence;
        # it must not attach that earlier candidate to the later sentence.
        full = first + "abcdefghij\n"
        self.assertEqual(detector.observe(record(full))["events"], [])

    def test_future_fields_and_fake_emitted_text_cannot_change_decision(self):
        prefix = "We obtain \\[x=2\\]. "
        base = {"problem": "Find x", "emitted_token_ids": encoded(prefix)}
        contaminated = {**base, "emitted_text": "future counterfeit",
                        "gold_answer": "99", "correctness": False,
                        "future_completion": "The answer is 99.",
                        "future_labels": ["verify"],
                        "next_sentence": "We check it."}
        a = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
        b = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
        self.assertEqual(a.observe_record(base, decode),
                         b.observe_record(contaminated, decode))
        self.assertEqual(len(a.seen), 1)
        self.assertEqual(LivePrefixGate(["candidate_to_verify"])
                         .observe_record(base, decode), [])

    def test_next_sentence_and_reasoning_closure_abstain(self):
        sentence = "We obtain \\[x=2\\]. "
        gate = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
        self.assertEqual(gate.observe_tokens(
            "Find x", encoded(sentence + "Now check it. "), decode), [])
        closed = LivePrefixGate(["candidate_to_verify"], screen=lambda _: True)
        self.assertEqual(closed.observe_tokens(
            "Find x", encoded(sentence + "</think>"), decode), [])
        self.assertTrue(closed.closed)
        self.assertEqual(closed.observe_tokens(
            "Find x", encoded(sentence + "</think> More work."), decode), [])


if __name__ == "__main__":
    unittest.main()
