from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Completion:
    text: str
    content: str
    reasoning_content: str
    usage: dict[str, Any]
    finish_reason: str | None


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
    reasoning = str(message.get("reasoning_content") or "").strip()
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
) -> Completion:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": False,
    }
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _post_json(url, api_key, payload, timeout)
            choice = response["choices"][0]
            text, content, reasoning = _assistant_text(choice["message"])
            if not text:
                raise ValueError("llama.cpp returned an empty assistant message")
            return Completion(
                text=text,
                content=content,
                reasoning_content=reasoning,
                usage=dict(response.get("usage") or {}),
                finish_reason=choice.get("finish_reason"),
            )
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"llama.cpp request failed after {max_retries + 1} attempts") from last_error