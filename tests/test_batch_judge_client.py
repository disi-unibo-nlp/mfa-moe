from __future__ import annotations

import pytest

from moe_exp.correlation_pipeline.batch_judge import (
    BATCH_ROUTE,
    build_batch_payload,
    map_batch_choices,
    map_batch_response,
)


def test_route_is_the_real_vllm_batch_endpoint() -> None:
    assert BATCH_ROUTE == "/v1/chat/completions/batch"


def test_payload_sends_sixteen_conversations_in_one_request() -> None:
    conversations = [[{"role": "user", "content": f"s{i}"}] for i in range(16)]
    payload = build_batch_payload(
        conversations,
        model="Qwen/Qwen3.8-27B",
        max_tokens=4096,
        temperature=0.0,
        reasoning_effort="medium",
    )
    assert len(payload["messages"]) == 16
    assert all(isinstance(conversation, list) for conversation in payload["messages"])
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["max_tokens"] == 4096
    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    # Documented /v1/chat/completions/batch limitations.
    assert payload["stream"] is False
    assert payload["n"] == 1
    assert "tools" not in payload


def test_model_card_sampling_is_explicit_and_optional() -> None:
    conversations = [[{"role": "user", "content": f"s{i}"}] for i in range(16)]
    kwargs = dict(model="Qwen/Qwen3.8-27B", max_tokens=4096,
                  temperature=1.0, reasoning_effort="medium")
    sampling = dict(top_p=0.95, top_k=20, min_p=0.0,
                    presence_penalty=0.0, repetition_penalty=1.0)
    payload = build_batch_payload(conversations, **kwargs, **sampling)
    assert all(payload[key] == value for key, value in sampling.items())
    assert payload["temperature"] == 1.0
    assert payload["messages"] == conversations
    assert payload["max_tokens"] == 4096
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": False,
    }
    baseline = build_batch_payload(conversations, **kwargs)
    assert not sampling.keys() & baseline.keys()
    assert {key: value for key, value in payload.items() if key not in sampling} == baseline


def test_response_maps_choices_back_by_index() -> None:
    response = {"choices": [{"index": i, "message": {"content": f"r{i}"}} for i in range(16)]}
    mapped = map_batch_response(response, expected=16)
    assert [mapped[i] for i in range(16)] == [f"r{i}" for i in range(16)]


def test_out_of_order_choices_are_restored_by_index() -> None:
    response = {
        "choices": [
            {"index": 2, "message": {"content": "c"}},
            {"index": 0, "message": {"content": "a"}},
            {"index": 1, "message": {"content": "b"}},
        ]
    }
    assert map_batch_response(response, expected=3) == {0: "a", 1: "b", 2: "c"}


def test_short_response_is_a_hard_error() -> None:
    with pytest.raises(ValueError):
        map_batch_response({"choices": [{"index": 0, "message": {"content": "r0"}}]}, expected=16)


def test_duplicate_index_is_a_hard_error() -> None:
    response = {
        "choices": [
            {"index": 0, "message": {"content": "a"}},
            {"index": 0, "message": {"content": "b"}},
        ]
    }
    with pytest.raises(ValueError):
        map_batch_response(response, expected=2)


@pytest.mark.parametrize("choice", [
    {"index": False, "message": {"content": "x"}},
    {"index": -1, "message": {"content": "x"}},
    {"index": 1, "message": {"content": "x"}},
    {"index": 0, "message": {"content": "x"}, "finish_reason": "length"},
    {"index": 0, "message": []},
    None,
])
def test_invalid_or_truncated_choice_fails_closed(choice):
    with pytest.raises(ValueError):
        map_batch_response({"choices": [choice]}, expected=1)


def test_tolerant_choice_map_preserves_non_stop_choice_and_raw_content() -> None:
    response = {
        "choices": [
            {
                "index": 1,
                "finish_reason": "length",
                "message": {"content": "raw truncated completion"},
            },
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "Analyze"},
            },
        ]
    }

    mapped = map_batch_choices(response, expected=2)

    assert mapped[1]["finish_reason"] == "length"
    assert mapped[1]["message"]["content"] == "raw truncated completion"
    assert mapped[0]["message"]["content"] == "Analyze"
