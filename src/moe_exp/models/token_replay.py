"""Exact vLLM token replay and character offsets, independent of chat format."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

TOKEN_REPLAY_VERSION = 1


@lru_cache(maxsize=8)
def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _token_ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(i) is not int or i < 0 for i in value):
        raise ValueError(f"Exact replay requires non-empty {name} from vLLM return_token_ids")
    return value


def make_token_replay(tokenizer: Any, prompt_ids: Any, completion_ids: Any) -> tuple[str, dict]:
    from tokenizers.decoders import DecodeStream

    prompt_ids = _token_ids(prompt_ids, "prompt_token_ids")
    completion_ids = _token_ids(completion_ids, "completion_token_ids")
    text = tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    # Re-encoding generated text can merge adjacent tokens and change their
    # count. DecodeStream preserves the actual token sequence, including split
    # UTF-8 characters. Tokens contributing to one character share its span.
    stream = DecodeStream(skip_special_tokens=False)
    offsets: list[list[int]] = []
    pending = 0
    decoded = ""
    for token_id in completion_ids:
        pending += 1
        piece = stream.step(tokenizer.backend_tokenizer, token_id)
        if piece is not None:
            offsets.extend([[len(decoded), len(decoded) + len(piece)] for _ in range(pending)])
            decoded += piece
            pending = 0
    if not text.startswith(decoded):
        raise ValueError("Incremental tokenizer decoding differs from the saved completion")
    if pending:
        offsets.extend([[len(decoded), len(text)] for _ in range(pending)])
    elif decoded != text:
        raise ValueError("Tokenizer decoding left unaligned completion text")
    return text, {
        "schema_version": TOKEN_REPLAY_VERSION,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "completion_offsets": offsets,
    }


def validate_token_replay(replay: dict, tokenizer: Any, cot_text: str) -> tuple[list[int], list[int]]:
    if replay.get("schema_version") != TOKEN_REPLAY_VERSION:
        raise ValueError("Unsupported token replay schema")
    if replay.get("tokenizer_sha256") != tokenizer_fingerprint(tokenizer):
        raise ValueError("Generation and forward tokenizers have different vocabularies")
    prompt = _token_ids(replay.get("prompt_token_ids"), "prompt_token_ids")
    completion = _token_ids(replay.get("completion_token_ids"), "completion_token_ids")
    if tokenizer.decode(
        completion, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    ) != cot_text:
        raise ValueError("Saved completion token IDs do not decode to cot_text")
    offsets = replay.get("completion_offsets")
    if not isinstance(offsets, list) or len(offsets) != len(completion):
        raise ValueError("Missing or incomplete completion token offsets")
    last_start = 0
    for span in offsets:
        if (not isinstance(span, list) or len(span) != 2
                or any(type(i) is not int for i in span)
                or not last_start <= span[0] <= span[1] <= len(cot_text)):
            raise ValueError("Invalid completion token character offsets")
        last_start = span[0]
    return prompt, completion
