from scripts.experimental_resume.rate_transition_v21_starts import messages, parse_rating


def row(later):
    return {"transition": "candidate_to_verify",
            "reader_input": {
                "problem": "Find x.",
                "previous_sentence": "Compute the ratio.",
                "triggering_sentence": "$400/40 = 10$.",
                "later_sentences": later}}


def test_start_reader_is_future_invariant():
    a = messages(row([{"text": "Yes."}]))
    b = messages(row([{"text": "Substitution fails."}, {"text": "Revise."}]))
    assert a == b
    assert "400/40" in a[1]["content"]
    assert "Substitution fails" not in a[1]["content"]


def test_start_reader_requires_exact_fixture_fields_and_single_answer():
    invalid = row([])
    invalid["reader_input"]["correct"] = True
    try:
        messages(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("extra input field should be rejected")
    assert parse_rating('{"start": true}') == {"start": True}
    assert parse_rating('{"start": true} {"start": false}') is None
