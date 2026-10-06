from scripts.experimental_resume.rate_transition_v22_fullprefix_starts import (
    messages, parse_rating,
)


def row(prefix):
    return {"transition": "candidate_to_verify",
            "reader_input": {"problem": "Find x.",
                             "emitted_prefix": prefix,
                             "triggering_sentence": "$400/40 = 10$."},
            "analysis_meta": {"v22_fired": True, "future_target": "do not read"}}


def test_full_prefix_prompt_uses_only_emitted_text():
    a = messages(row("We need to compute.\n$400/40 = 10$.\n"))
    b_row = row("We need to compute.\n$400/40 = 10$.\n")
    b_row["analysis_meta"]["future_target"] = "substitution fails"
    assert a == messages(b_row)
    assert "Already emitted reasoning prefix" in a[1]["content"]
    assert "substitution fails" not in a[1]["content"]


def test_trigger_must_end_prefix_and_future_fields_are_rejected():
    try:
        messages(row("$400/40 = 10$.\nLater future text."))
    except ValueError:
        pass
    else:
        raise AssertionError("future suffix should not be accepted")
    invalid = row("$400/40 = 10$.\n")
    invalid["reader_input"]["later_sentence"] = "Yes."
    try:
        messages(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("future field should be rejected")
    assert parse_rating('{"start": false}') == {"start": False}
