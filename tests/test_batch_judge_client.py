from __future__ import annotations

import pytest

from moe_exp.correlation_pipeline.batch_judge import (
    BATCH_ROUTE,
    build_batch_payload,
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
    assert payload.get("stream", False) is False
    assert payload.get("n", 1) == 1
    assert "tools" not in payload


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
