from moe_exp.routing_control.transitions_v2 import (
    StreamingTransitionDetectorV2,
    classify_sentence_v2,
)


def prefix(text, **extra):
    return {"problem": "Find x.", "emitted_token_ids": list(range(len(text))),
            "emitted_text": text, **extra}


def test_complete_math_candidate_and_incomplete_expression():
    assert "candidate_to_verify" in classify_sentence_v2("$400/40 = 10$.")
    assert classify_sentence_v2("$400/40 = 10") == ()
    assert "candidate_to_verify" in classify_sentence_v2("$$T-3 = 43-3 = 40.$$") 
    assert classify_sentence_v2("$$T-3 = 43-3") == ()
    assert "candidate_to_verify" in classify_sentence_v2(
        r"I'll put $\boxed{12^{\mathrm{th}}\text{ grade}}$.")


def test_chunked_stream_only_emits_after_completion_and_never_after_closure():
    detector = StreamingTransitionDetectorV2()
    assert detector.observe(prefix("<think>$400/40 = 10$."))["events"] == []
    state = detector.observe(prefix("<think>$400/40 = 10$.\n"))
    assert [e.transition for e in state["events"]] == ["candidate_to_verify"]
    assert detector.observe(prefix("<think>$400/40 = 10$.\nStill working"))["events"] == []
    closed = detector.observe(prefix("<think>$400/40 = 10$.\nStill working\n</think>"))
    assert closed["closure"] is True and closed["events"] == []


def test_future_and_forbidden_fields_cannot_change_prefix_decision():
    text = "<think>Maybe use substitution to solve it.\n"
    a = StreamingTransitionDetectorV2().observe(prefix(text, gold_answer="3",
                                                         future_completion="Wrong."))
    b = StreamingTransitionDetectorV2().observe(prefix(text, gold_answer="999",
                                                         future_completion="Verified."))
    assert a == b
    assert [e.transition for e in a["events"]] == ["approach_to_commit"]


def test_visible_failure_requires_check_evidence():
    assert "failed_check_to_revise" not in classify_sentence_v2(
        "Simplification leads to $x(x-5)=0$.")
    assert "failed_check_to_revise" in classify_sentence_v2(
        "Substituting x=5 gives 9, which violates the required value 7.")


def test_multiline_display_math_stays_one_incomplete_unit():
    detector = StreamingTransitionDetectorV2()
    first = "<think>$$T-3 =\n43-3"
    assert detector.observe(prefix(first))["events"] == []
    complete = detector.observe(prefix(first + " = 40.$$\n"))
    assert [e.transition for e in complete["events"]] == ["candidate_to_verify"]
