from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


SAMPLING_FIELDS = ("min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
                   "reasoning_effort")


@dataclass(frozen=True)
class Completion:
    text: str
    content: str
    reasoning_content: str
    usage: dict[str, Any]
    finish_reason: str | None
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None


def _post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _assistant_text(message: dict[str, Any]) -> tuple[str, str, str]:
    content = str(message.get("content") or "").strip()
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
    if reasoning:
        text = f"<think>\n{reasoning}\n</think>"
        if content:
            text += f"\n{content}"
    else:
        text = content
    return text, content, reasoning


def generate_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    timeout: int = 3600,
    max_retries: int = 4,
    return_token_ids: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
    extra_sampling: dict[str, Any] | None = None,
) -> Completion:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1 if top_k == 0 else top_k,
        "seed": seed,
        "stream": False,
    }
    # Card-profile fields (min_p, presence/frequency/repetition penalty, reasoning_effort)
    # are sent only when set, so legacy request payloads are unchanged.
    for key, value in (extra_sampling or {}).items():
        if key not in SAMPLING_FIELDS:
            raise ValueError(f"Unsupported sampling field: {key}")
        if value is not None:
            payload[key] = value
    if return_token_ids:
        payload["return_token_ids"] = True
        payload["include_reasoning"] = True
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _post_json(url, api_key, payload, timeout)
            choice = response["choices"][0]
            text, content, reasoning = _assistant_text(choice["message"])
            if not text:
                raise ValueError("Inference server returned an empty assistant message")
            return Completion(
                text=text,
                content=content,
                reasoning_content=reasoning,
                usage=dict(response.get("usage") or {}),
                finish_reason=choice.get("finish_reason"),
                prompt_token_ids=response.get("prompt_token_ids"),
                token_ids=choice.get("token_ids"),
            )
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Inference request failed after {max_retries + 1} attempts") from last_error


@dataclass(frozen=True)
class TokenContinuation:
    token_ids: list[int]
    text: str
    finish_reason: str | None
    usage: dict[str, Any]
    prompt_token_ids: list[int]


def complete_tokens(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    timeout: int = 21600,
    max_retries: int = 2,
    extra_sampling: dict[str, Any] | None = None,
) -> TokenContinuation:
    """Continue an exact integer-token prefix through /v1/completions.

    No chat template or special tokens are added, and the echoed prompt IDs must equal
    the request, so the continuation extends exactly the saved token sequence.
    """
    payload = {
        "model": model,
        "prompt": list(prompt_token_ids),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1 if top_k == 0 else top_k,
        "seed": seed,
        "stream": False,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "return_token_ids": True,
        "truncate_prompt_tokens": None,
    }
    for key, value in (extra_sampling or {}).items():
        if key not in SAMPLING_FIELDS:
            raise ValueError(f"Unsupported sampling field: {key}")
        if value is not None:
            payload[key] = value
    url = f"{base_url.rstrip('/')}/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _post_json(url, api_key, payload, timeout)
            choice = response["choices"][0]
            echoed = choice.get("prompt_token_ids") or response.get("prompt_token_ids")
            if echoed is not None and list(echoed) != list(prompt_token_ids):
                raise RuntimeError("Server altered the continuation prompt token IDs")
            token_ids = choice.get("token_ids")
            if not isinstance(token_ids, list):
                raise ValueError("Server did not return continuation token IDs")
            return TokenContinuation(
                token_ids=token_ids,
                text=choice.get("text") or "",
                finish_reason=choice.get("finish_reason"),
                usage=dict(response.get("usage") or {}),
                prompt_token_ids=list(echoed) if echoed is not None else [],
            )
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(10 * 2**attempt, 120))
    raise RuntimeError(f"Continuation request failed after {max_retries + 1} attempts") from last_error
