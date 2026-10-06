from moe_exp.routing_control.transitions_v22 import StreamingTransitionDetectorV2


def prefix(text):
    return {"problem": "Find x.", "emitted_token_ids": list(range(len(text))),
            "emitted_text": text}


def test_complete_candidate_cannot_fire_in_later_unrelated_sentence():
    detector = StreamingTransitionDetectorV2()
    first = "<think>The answer is 3.\n"
    original = detector.observe(prefix(first))
    assert [e.transition for e in original["events"]] == ["candidate_to_verify"]
    later = detector.observe(prefix(first + "This is a longer sentence.\n"))
    assert later["events"] == []


def test_same_sentence_answer_remains_detectable():
    events = StreamingTransitionDetectorV2().observe(
        prefix("<think>We get $400/40 = 10$.\n"))["events"]
    assert [e.transition for e in events] == ["candidate_to_verify"]


def test_case_heading_is_hard_negative_and_closed_equality_is_a_proposal():
    detector = StreamingTransitionDetectorV2()
    heading = "<think>2. Exactly two equal.\n"
    assert detector.observe(prefix(heading))["events"] == []
    incomplete = heading + "Thus $r = 205/60 = 41/12$"
    assert detector.observe(prefix(incomplete))["events"] == []
    complete = incomplete + ".\n"
    events = detector.observe(prefix(complete))["events"]
    assert [event.transition for event in events] == ["candidate_to_verify"]
    assert all(event.evidence_end <= len(complete) for event in events)
