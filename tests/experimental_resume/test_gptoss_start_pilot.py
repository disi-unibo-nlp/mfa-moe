from scripts.experimental_resume.rate_gptoss_start_pilot import messages, parse_rating


def test_strict_final_channel_json_and_prefix_only_inputs():
    row = {
        "uid": "one", "family": "family-one", "transition": "candidate_to_verify",
        "prefix_tokens": 10,
        "reader_input": {"problem": "Find x.",
                         "emitted_prefix": "Proposed x=2.\n",
                         "triggering_sentence": "Proposed x=2."},
    }
    original = messages(row)
    assert "Find x." in str(original)
    assert parse_rating('<|channel|>final<|message|>{"start":true}<|return|>') == {"start": True}
    assert parse_rating('{"start":true,"reason":"x"}') is None
    row["future_answer"] = "forbidden"
    try:
        messages(row)
    except ValueError:
        pass
    else:
        raise AssertionError("future field accepted by GPT-OSS reader")
