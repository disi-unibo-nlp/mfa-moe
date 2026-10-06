from scripts.experimental_resume.rate_gptoss_start_pilot import (
    messages, model_cache, parse_rating,
)


def fixture():
    return {"uid": "a" * 64, "family": "b" * 64,
            "transition": "candidate_to_verify", "prefix_tokens": 25,
            "reader_input": {"problem": "Find x.",
                             "emitted_prefix": "Try this.\n$x=5$.\n",
                             "triggering_sentence": "$x=5$."}}


def test_gptoss_prefix_only_and_strict_final_json():
    row = fixture()
    prompt = messages(row)
    assert "Find x." in prompt[1]["content"]
    assert parse_rating('<|channel|>final<|message|>{"start": true}<|return|>') == {"start": True}
    assert parse_rating('{"start": false}') == {"start": False}
    assert parse_rating('{"start": true, "other": 1}') is None
    assert parse_rating('<|channel|>analysis<|message|>thinking') is None
    row["reader_input"]["answer"] = "5"
    try:
        messages(row)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden answer field was accepted")


def test_gptoss_cache_has_local_complete_weight_shards():
    cache = model_cache()
    assert cache["snapshot"].endswith("6cee5e81ee83917806bbde320786a8fb61efebee")
    assert len(cache["weight_shard_bytes"]) >= 2
