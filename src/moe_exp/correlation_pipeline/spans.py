"""Deterministic reasoning units and offsets into the exact teacher-forced input."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from moe_exp.gepaLLMAsJudge.data import SENTENCE_LABELS
from moe_exp.models.inference import _find_prompt_length, _format_prompt

SPAN_SCHEMA_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def trace_digest(trace: Any) -> str:
    return digest(
        {
            "prompt": trace.prompt,
            "messages": trace.generation_messages,
            "system_prompt": trace.system_prompt,
            "cot_text": trace.cot_text,
            **({"token_replay": trace.metadata["token_replay"]}
               if "token_replay" in trace.metadata else {}),
        }
    )


def reasoning_bounds(trace: Any) -> tuple[int, int]:
    """Return the envelope of the reasoning (which may have message gaps)."""
    ranges = reasoning_ranges(trace)
    return ranges[0][0], ranges[-1][1]


def reasoning_ranges(trace: Any) -> list[tuple[int, int]]:
    """Locate reasoning without including native message control tokens."""
    text = trace.cot_text
    reasoning = trace.metadata.get("reasoning_content")
    if reasoning:
        start = text.find(reasoning)
        if start < 0:
            # GPT-OSS can emit consecutive analysis messages. vLLM joins their
            # bodies with newlines, while exact replay retains Harmony headers.
            if "token_replay" in trace.metadata:
                messages = list(re.finditer(
                    r"<\|channel\|>analysis<\|message\|>(.*?)"
                    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|\Z)",
                    text, flags=re.DOTALL,
                ))
                if messages and "\n".join(m[1] for m in messages).strip() == reasoning:
                    return [m.span(1) for m in messages]
            raise ValueError("Saved reasoning_content does not occur in cot_text")
        return [(start, start + len(reasoning))]
    opening = text.find("<think>")
    if opening >= 0:
        start = opening + len("<think>")
        closing = text.find("</think>", start)
        return [(start, closing if closing >= 0 else len(text))]
    # Exact replay includes native control tokens. With no reasoning channel,
    # restrict units to the parsed final content rather than annotating markers.
    if "token_replay" in trace.metadata:
        content = trace.metadata.get("assistant_content")
        if content and content in text:
            start = text.find(content)
            return [(start, start + len(content))]
        return [(0, 0)]
    return [(0, len(text))]


def sentence_spans(trace: Any) -> list[dict[str, Any]]:
    """Split punctuation/newlines, retaining source offsets and math expressions.

    This is a versioned deterministic segmentation, not a linguistic parser.
    Decimal points and punctuation inside LaTeX math do not create boundaries.
    """
    units = []
    for start, end in reasoning_ranges(trace):
        for unit in _sentence_spans_in_range(trace.cot_text, start, end):
            units.append({**unit, "index": len(units)})
    return units


def _sentence_spans_in_range(text: str, start: int, end: int) -> list[dict[str, Any]]:
    # Disjoint alternatives avoid exponential backtracking on unclosed math.
    # The fallback preserves the original last-escaped-dollar closing behavior.
    protected = [
        match.span()
        for match in re.finditer(
            r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:(?:\\.|[^$\\])*?\$|(?:\\.|[^$\\])*\\\$)",
            text,
            flags=re.DOTALL,
        )
    ]
    spans = []
    cursor = start
    boundaries = [match for match in re.finditer(r"(?<=[.!?])[^\S\n]+|\n+", text[start:end])]
    for match in boundaries:
        boundary = start + match.start()
        if any(left <= boundary < right for left, right in protected):
            continue
        if re.fullmatch(r"\s*(?:\d+[.)]|[Ss]tep\s+\d+[.:])", text[cursor:boundary]):
            continue
        spans.append((cursor, boundary))
        cursor = start + match.end()
    spans.append((cursor, end))
    units = []
    for left, right in spans:
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            units.append(
                {"index": len(units), "start": left, "end": right, "text": text[left:right]}
            )
    return units


def selected_sentence_indices(trace: Any, units: list[dict[str, Any]]) -> list[int]:
    """Return the fixed sampling selection, or every unit for ordinary runs."""
    selection = trace.metadata.get("sentence_selection")
    if selection is None:
        return list(range(len(units)))
    if not isinstance(selection, dict) or selection.get("schema_version") != 1:
        raise ValueError("Unsupported sentence selection")
    indices = selection.get("indices")
    if (
        not isinstance(indices, list)
        or any(type(index) is not int or not 0 <= index < len(units) for index in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("Sentence selection must contain sorted unique valid indices")
    return indices


def validate_annotation(trace: Any, annotation: dict[str, Any]) -> None:
    if annotation.get("schema_version") != SPAN_SCHEMA_VERSION:
        raise ValueError("Unsupported reasoning annotation schema")
    if annotation.get("trace_sha256") != trace_digest(trace):
        raise ValueError(f"Stale annotation for {trace.dataset}/{trace.problem_id}")
    if annotation.get("sentence_selection") != trace.metadata.get("sentence_selection"):
        raise ValueError("Annotation belongs to a different sentence selection")
    units = sentence_spans(trace)
    expected = [units[index] for index in selected_sentence_indices(trace, units)]
    actual = annotation.get("units", [])
    if len(expected) != len(actual):
        raise ValueError("Incomplete sentence annotations")
    for unit, labeled in zip(expected, actual, strict=True):
        if any(labeled.get(key) != value for key, value in unit.items()):
            raise ValueError("Annotation sentence offsets/text do not match generation")
        if labeled.get("label") not in SENTENCE_LABELS:
            raise ValueError("Invalid reasoning class")


def token_layout(trace: Any, tokenizer: Any) -> dict[str, Any]:
    """Align with the jointly tokenized prompt+completion used by inference.

    A token belongs to the unit with greatest character overlap (earlier unit
    wins ties). Whitespace-only gaps attach to the preceding unit. Every token
    overlapping reasoning is owned exactly once; tags/final answers are excluded.
    Fast tokenizer offsets are required; approximate retokenization is unsafe.
    """
    replay = trace.metadata.get("token_replay")
    if replay is not None:
        from moe_exp.models.token_replay import validate_token_replay
        validate_token_replay(replay, tokenizer, trace.cot_text)
        offsets = replay["completion_offsets"]
        prompt, prompt_len = "", 0
    else:
        prompt = _format_prompt(tokenizer, trace.prompt, trace.system_prompt, trace.generation_messages)
        full_text = prompt + trace.cot_text
        try:
            encoded = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded["offset_mapping"][0].tolist()
        except (KeyError, NotImplementedError, TypeError) as error:
            raise ValueError(
                "Reasoning views require a tokenizer with exact character offsets"
            ) from error
        prompt_len = _find_prompt_length(tokenizer, prompt, full_text)
    units = sentence_spans(trace)
    ranges = reasoning_ranges(trace)
    owners = [[] for _ in units]
    reasoning_tokens = []
    unit_index = 0
    for token, (left, right) in enumerate(offsets[prompt_len:]):
        left, right = left - len(prompt), right - len(prompt)
        if right <= left or not any(right > start and left < end for start, end in ranges):
            continue
        if not units:
            continue
        reasoning_tokens.append(token)
        while unit_index + 1 < len(units) and units[unit_index]["end"] <= left:
            unit_index += 1
        candidates = []
        index = max(0, unit_index - 1)
        while index < len(units) and units[index]["start"] < right:
            overlap = max(0, min(right, units[index]["end"]) - max(left, units[index]["start"]))
            candidates.append((overlap, -index))
            index += 1
        overlap, negative_index = max(candidates, default=(0, -unit_index))
        owner = -negative_index
        if overlap == 0:
            owner = max(0, unit_index - int(units[unit_index]["start"] >= right))
        owners[owner].append(token)
    return {
        "token_count": len(offsets) - prompt_len,
        "units": units,
        "unit_tokens": owners,
        "reasoning_tokens": reasoning_tokens,
    }


def position_windows(tokens: list[int], mean_length: float, bins: int = 10) -> list[dict[str, Any]]:
    if mean_length <= 0 or not math.isfinite(mean_length) or bins < 1:
        raise ValueError("Position reference length and bin count must be positive")
    boundaries = [math.ceil(mean_length * index / bins) for index in range(bins + 1)]
    windows = [
        {
            "name": f"bin_{index:02d}",
            "start": boundaries[index],
            "end": boundaries[index + 1],
            "tokens": tokens[boundaries[index] : boundaries[index + 1]],
        }
        for index in range(bins)
    ]
    windows.append(
        {
            "name": "overflow",
            "start": boundaries[-1],
            "end": None,
            "tokens": tokens[boundaries[-1] :],
        }
    )
    return windows
