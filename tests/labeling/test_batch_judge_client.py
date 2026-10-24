from __future__ import annotations

import urllib.request

import pytest

from labeling.batch_predictor import (
    JUDGE_MODEL,
    SAMPLING,
    batch_payload,
    map_batch_choices,
    post_batch,
)


def test_route_is_the_real_vllm_batch_endpoint(monkeypatch) -> None:
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"choices": []}'

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert post_batch({}, base_url="http://127.0.0.1:8000/v1/") == {"choices": []}
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions/batch"
    assert seen["auth"] == "Bearer local-vllm-key"


def test_payload_sends_sixteen_conversations_in_one_request() -> None:
    conversations = [[{"role": "user", "content": f"s{i}"}] for i in range(16)]
    payload = batch_payload(conversations)
    assert payload["messages"] == conversations
    assert payload["model"] == JUDGE_MODEL == "Qwen/Qwen3.8-27B"
    assert payload["max_tokens"] == 16384
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True, "reasoning_effort": "low", "preserve_thinking": False,
    }
    # The model card's sampling, sent explicitly.
    assert SAMPLING == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 0.0, "repetition_penalty": 1.0}
    assert all(payload[key] == value for key, value in SAMPLING.items())
    # Documented /v1/chat/completions/batch limitations.
    assert payload["stream"] is False
    assert payload["n"] == 1
    assert "tools" not in payload


def test_response_maps_choices_back_by_index() -> None:
    response = {"choices": [{"index": i, "message": {"content": f"r{i}"}} for i in range(16)]}
    mapped = map_batch_choices(response, expected=16)
    assert [mapped[i]["message"]["content"] for i in range(16)] == [f"r{i}" for i in range(16)]


def test_out_of_order_choices_are_restored_by_index() -> None:
    response = {
        "choices": [
            {"index": 2, "message": {"content": "c"}},
            {"index": 0, "message": {"content": "a"}},
            {"index": 1, "message": {"content": "b"}},
        ]
    }
    mapped = map_batch_choices(response, expected=3)
    assert [mapped[i]["message"]["content"] for i in range(3)] == ["a", "b", "c"]


def test_short_response_is_a_hard_error() -> None:
    with pytest.raises(ValueError):
        map_batch_choices({"choices": [{"index": 0, "message": {"content": "r0"}}]}, expected=16)


def test_duplicate_index_is_a_hard_error() -> None:
    response = {
        "choices": [
            {"index": 0, "message": {"content": "a"}},
            {"index": 0, "message": {"content": "b"}},
        ]
    }
    with pytest.raises(ValueError):
        map_batch_choices(response, expected=2)


@pytest.mark.parametrize("choice", [
    {"index": False, "message": {"content": "x"}},
    {"index": -1, "message": {"content": "x"}},
    {"index": 1, "message": {"content": "x"}},
    None,
])
def test_invalid_choice_fails_closed(choice):
    with pytest.raises(ValueError):
        map_batch_choices({"choices": [choice]}, expected=1)


def test_choice_map_preserves_non_stop_choice_and_raw_content() -> None:
    response = {
        "choices": [
            {"index": 1, "finish_reason": "length", "message": {"content": "raw truncated completion"}},
            {"index": 0, "finish_reason": "stop", "message": {"content": "Analyze"}},
        ]
    }

    mapped = map_batch_choices(response, expected=2)

    assert mapped[1]["finish_reason"] == "length"
    assert mapped[1]["message"]["content"] == "raw truncated completion"
    assert mapped[0]["message"]["content"] == "Analyze"
