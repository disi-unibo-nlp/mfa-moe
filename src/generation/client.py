"""Requests to a running vLLM server: one chat completion, or the continuation of an exact token prefix."""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from generation.model_profiles import API_KEY


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning_content: str
    usage: dict[str, Any]
    finish_reason: str | None
    prompt_token_ids: list[int] | None
    token_ids: list[int] | None


@dataclass(frozen=True)
class TokenContinuation:
    token_ids: list[int]
    finish_reason: str | None
    usage: dict[str, Any]
    prompt_token_ids: list[int]


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_with_retries(url, payload, timeout, max_retries, first_wait, longest_wait, parse):
    """POST, hand the reply to `parse`, and try again (waiting 2x longer each time) when that fails."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return parse(_post_json(url, payload, timeout))
        except (OSError, KeyError, ValueError) as exc:  # OSError includes HTTP errors
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(first_wait * 2**attempt, longest_wait))
    raise RuntimeError(f"Request to {url} failed after {max_retries + 1} attempts") from last_error


def _sampling_fields(sampler: dict[str, Any], seed: int) -> dict[str, Any]:
    """Request fields of a model-card sampler: unset (None) ones are left out, top_k 0 (off) is sent as -1."""
    fields = {key: value for key, value in sampler.items() if value is not None}
    fields["top_k"] = fields["top_k"] or -1
    return {**fields, "seed": seed, "stream": False}


def generate_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    sampler: dict[str, Any],
    reasoning_effort: str | None,
    seed: int,
    timeout: int,
    chat_template_kwargs: dict[str, Any],
) -> Completion:
    """One chat completion that also returns the exact prompt and completion token IDs."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **_sampling_fields(sampler, seed),
        "return_token_ids": True,
        "include_reasoning": True,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    def parse(response: dict[str, Any]) -> Completion:
        choice = response["choices"][0]
        message = choice["message"]
        content = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if not (content or reasoning):
            raise ValueError("Inference server returned an empty assistant message")
        return Completion(
            content=content,
            reasoning_content=reasoning,
            usage=dict(response.get("usage") or {}),
            finish_reason=choice.get("finish_reason"),
            prompt_token_ids=response.get("prompt_token_ids"),
            token_ids=choice.get("token_ids"),
        )

    return _post_with_retries(f"{base_url.rstrip('/')}/chat/completions", payload, timeout, 4, 1, 30, parse)


def complete_tokens(
    *,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    sampler: dict[str, Any],
    seed: int,
    timeout: int,
) -> TokenContinuation:
    """Continue an exact integer-token prefix through /v1/completions.

    No chat template or special tokens are added, and the echoed prompt IDs must equal
    the request, so the continuation extends exactly the saved token sequence.
    """
    payload = {
        "model": model,
        "prompt": list(prompt_token_ids),
        "max_tokens": max_tokens,
        **_sampling_fields(sampler, seed),
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "return_token_ids": True,
        "truncate_prompt_tokens": None,
    }

    def parse(response: dict[str, Any]) -> TokenContinuation:
        choice = response["choices"][0]
        echoed = choice.get("prompt_token_ids") or response.get("prompt_token_ids")
        if echoed is not None and list(echoed) != list(prompt_token_ids):
            raise RuntimeError("Server altered the continuation prompt token IDs")  # not retried
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list):
            raise ValueError("Server did not return continuation token IDs")
        return TokenContinuation(
            token_ids=token_ids,
            finish_reason=choice.get("finish_reason"),
            usage=dict(response.get("usage") or {}),
            prompt_token_ids=list(echoed) if echoed is not None else [],
        )

    return _post_with_retries(f"{base_url.rstrip('/')}/completions", payload, timeout, 2, 10, 120, parse)
