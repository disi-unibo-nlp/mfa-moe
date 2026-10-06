from scripts.experimental_resume.rate_transition_v22_qwen_scout import messages, parse_rating


def fixture():
    return {"transition": "candidate_to_verify",
            "reader_input": {"problem": "Find x.",
                             "emitted_prefix": "We compute.\n$400/40 = 10$.\n",
                             "triggering_sentence": "$400/40 = 10$."},
            "analysis_meta": {"future_target": "secret later check"}}


def test_scout_prompt_is_prefix_only_and_future_invariant():
    first = fixture()
    expected = messages(first)
    first["analysis_meta"]["future_target"] = "a completely different future"
    assert messages(first) == expected
    assert "secret later check" not in str(expected)
    assert "completely different future" not in str(expected)


def test_scout_rejects_future_suffix_and_extra_reader_field():
    row = fixture()
    row["reader_input"]["emitted_prefix"] += "A later check."
    try:
        messages(row)
    except ValueError:
        pass
    else:
        raise AssertionError("future suffix was accepted")
    row = fixture()
    row["reader_input"]["answer"] = "10"
    try:
        messages(row)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden field was accepted")
    assert parse_rating('{"start": true}') == {"start": True}


def test_scout_requires_exact_single_boolean_json_object():
    assert parse_rating('<think>reasoning</think> {"start": false}') == {"start": False}
    for invalid in ('{"start": true, "explanation": "x"}',
                    '{"start": 1}', '{"start": "true"}',
                    'prefix {"start": true}',
                    '{"start": true} trailing',
                    '{"start": true}{"start": false}',
                    '<think>unfinished {"start": true}'):
        assert parse_rating(invalid) is None
