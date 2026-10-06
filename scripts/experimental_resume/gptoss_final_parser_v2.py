"""Strict GPT-OSS final-channel boolean parser for vLLM 0.29 output.

vLLM's detokenized GPT-OSS text on LEONARDO can render the channel switch as
``assistantfinal``. Only an exact one-key JSON object after that marker, after
the explicit special-token final marker, or as the entire response is accepted.
Analysis prose is never searched for a JSON-looking substring.
"""
from __future__ import annotations

import json


def parse_rating(text: str) -> dict[str, bool] | None:
    if not isinstance(text, str):
        return None
    special = "<|channel|>final<|message|>"
    plain = "assistantfinal"
    if special in text:
        if text.count(special) != 1:
            return None
        tail = text.split(special, 1)[1]
        for terminator in ("<|return|>", "<|end|>"):
            if terminator in tail:
                if tail.count(terminator) != 1 or not tail.endswith(terminator):
                    return None
                tail = tail[: -len(terminator)]
                break
    elif plain in text:
        if text.count(plain) != 1 or not text.startswith("analysis"):
            return None
        tail = text.split(plain, 1)[1]
    else:
        tail = text
    try:
        value = json.loads(tail.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and set(value) == {"start"} and type(value["start"]) is bool else None
