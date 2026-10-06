from scripts.experimental_resume.native_prefix_semantic_veto_v0 import (
    cache_receipt, messages, parse_rating,
)


def row():
    return {"transition": "candidate_to_verify",
            "reader_input": {"problem": "Find a state.",
                             "emitted_prefix": "Earlier computation.\nState becomes $B=3,G=3$.\n",
                             "triggering_sentence": "State becomes $B=3,G=3$."},
            "analysis_meta": {"future_target": "secret"}}


def test_native_side_prompt_is_future_invariant_and_strict():
    first = row()
    expected = messages(first)
    first["analysis_meta"]["future_target"] = "different future"
    assert messages(first) == expected
    assert "secret" not in str(expected)
    assert parse_rating('{"start":true}') == {"start": True}
    assert parse_rating('{"start":true,"why":"x"}') is None
    first["reader_input"]["gold_answer"] = "3"
    try:
        messages(first)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden reader field accepted")


def test_native_snapshot_complete():
    receipt = cache_receipt()
    assert receipt["weight_shard_count"] > 10
    assert receipt["weight_bytes_total"] > 10_000_000_000
