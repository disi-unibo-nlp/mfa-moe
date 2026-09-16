"""Client for vLLM's real batched chat-completion route.

`/v1/chat/completions/batch` accepts N conversations in a single request and returns
one choice per conversation, indexed 0..N-1. DSPy and LiteLLM do not know this route,
so the judge's transport is implemented here while prompt rendering and response
parsing stay with DSPy's own adapter (see `batch_predictor`), keeping labels identical
to the single-request control path.

Limitations of the route (from vLLM's BatchChatCompletionRequest docstring):
streaming, tools and beam search are unsupported, and `n` must be 1.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

BATCH_ROUTE = "/v1/chat/completions/batch"


def build_batch_payload(
    conversations: list[list[dict[str, Any]]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    enable_thinking: bool = True,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    """Build one batch request carrying every conversation in `conversations`."""
    if not conversations:
        raise ValueError("a batch requires at least one conversation")
    for conversation in conversations:
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("each conversation must be a non-empty list of messages")
    return {
        "model": model,
        "n": 1,
        "stream": False,
        "messages": conversations,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "reasoning_effort": reasoning_effort,
            "preserve_thinking": False,
        },
        **{
            key: value
            for key, value in {
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
            }.items()
            if value is not None
        },
    }


def map_batch_choices(response: dict[str, Any], *, expected: int) -> dict[int, dict[str, Any]]:
    """Map raw `choices` back to conversation positions without judging their content.

    Structural problems stay fatal because a response like that cannot be attributed to
    the input rows. Per-choice problems (truncation, missing content, unparseable text)
    are left to the tolerant caller, which is why the original choice dictionaries are
    returned instead of extracted content strings.
    """
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise ValueError("batch response has no choices list")
    if len(choices) != expected:
        raise ValueError(f"batch response returned {len(choices)} choices, expected {expected}")
    mapped: dict[int, dict[str, Any]] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("batch choice must be an object")
        index = choice.get("index")
        if type(index) is not int:
            raise ValueError("batch choice is missing an integer index")
        if index in mapped:
            raise ValueError(f"batch response repeated index {index}")
        if not 0 <= index < expected:
            raise ValueError(f"batch choice index {index} outside 0..{expected - 1}")
        mapped[index] = choice
    if len(mapped) != expected:
        raise ValueError(f"batch response covered {len(mapped)} indices, expected {expected}")
    return mapped


def map_batch_response(response: dict[str, Any], *, expected: int) -> dict[int, str]:
    """Map `choices` back to conversation position, rejecting anything but a clean answer."""
    mapped: dict[int, str] = {}
    for index, choice in map_batch_choices(response, expected=expected).items():
        if choice.get("finish_reason") not in (None, "stop"):
            raise ValueError("batch choice did not finish normally")
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise ValueError(f"batch choice {index} has no string content")
        mapped[index] = content
    return mapped


def post_batch(
    payload: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """POST one batch request and return the decoded JSON body."""
    url = base_url.rstrip("/").removesuffix("/v1") + BATCH_ROUTE
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))
