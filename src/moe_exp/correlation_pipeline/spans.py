"""Deterministic reasoning units and offsets into the exact teacher-forced input."""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
import math
import re
from typing import Any, Iterable

from moe_exp.gepaLLMAsJudge.data import SENTENCE_LABELS

SPAN_SCHEMA_VERSION = 1

# Version 2 single-`$` protection limits: a stray delimiter may not swallow prose.
SINGLE_DOLLAR_MAX_CHARS = 1000
SINGLE_DOLLAR_MAX_BOUNDARIES = 2
# A plausible single-`$` opener follows whitespace or one of these characters.
OPENING_CONTEXT = "([{=,;:"
_BRACKET_MARKER = re.compile(r"\\[\[\(]")
_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])[^\S\n]+|\n+")
_STEP_PATTERN = re.compile(r"\s*(?:\d+[.)]|[Ss]tep\s+\d+[.:])")
_V1_MATH_PATTERN = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|(?<!\\)\$(?:(?:\\.|[^$\\])*?\$|(?:\\.|[^$\\])*\\\$)",
    flags=re.DOTALL,
)


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

    This is a versioned deterministic segmentation, not a linguistic parser. Version 2
    keeps the frozen version 1 math protection but refuses a single-`$` span that starts
    after a word character and is longer than `SINGLE_DOLLAR_MAX_CHARS` or hides
    `SINGLE_DOLLAR_MAX_BOUNDARIES` sentence boundaries. A stray closing `$` therefore
    cannot swallow prose up to the next real formula, while a plausible `$...$` pair is
    protected exactly as before.
    """
    return _index_units(
        trace.cot_text,
        reasoning_ranges(trace),
        math_protection_spans,
    )


def sentence_spans_v1(trace: Any) -> list[dict[str, Any]]:
    """Frozen version 1 segmentation, kept only to map stored v1 selections to v2."""
    return _index_units(trace.cot_text, reasoning_ranges(trace), legacy_math_protection_spans)


def _index_units(
    text: str, ranges: list[tuple[int, int]], protection: Any
) -> list[dict[str, Any]]:
    protected = protection(text)
    units = []
    for start, end in ranges:
        for unit in _split_with_protection(text, start, end, protected):
            units.append({**unit, "index": len(units)})
    return units


def _boundary_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [
        (start + match.start(), start + match.end())
        for match in _BOUNDARY_PATTERN.finditer(text[start:end])
    ]


def _split_with_protection(
    text: str, start: int, end: int, protected: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    spans = []
    cursor = start
    for boundary, after in _boundary_spans(text, start, end):
        if any(left <= boundary < right for left, right in protected):
            continue
        if _STEP_PATTERN.fullmatch(text[cursor:boundary]):
            continue
        spans.append((cursor, boundary))
        cursor = after
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


def legacy_math_protection_spans(text: str) -> list[tuple[int, int]]:
    """Frozen version 1 protected spans (regex over the whole text)."""
    return [match.span() for match in _V1_MATH_PATTERN.finditer(text)]


def _is_escaped_dollar(text: str, index: int) -> bool:
    """True when `$` closes an odd-length backslash run, i.e. version 1 `\\$`."""
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _dollar_markers(text: str) -> list[tuple[int, str]]:
    """Every `$` and every `\\[`/`\\(` opener, in text order."""
    markers: list[tuple[int, str]] = []
    index = text.find("$")
    while index >= 0:
        markers.append((index, "$"))
        index = text.find("$", index + 1)
    markers.extend(
        (match.start(), match.group()) for match in _BRACKET_MARKER.finditer(text)
    )
    markers.sort()
    return markers


def _hidden_by_stray_delimiter(text: str, left: int, right: int) -> bool:
    """Version 2 refusal: a suspicious opener may not hide a long prose span.

    A span that starts right after a word character is a closing delimiter that the
    frozen pattern read as an opener. It is dropped when it is longer than
    `SINGLE_DOLLAR_MAX_CHARS` characters or covers `SINGLE_DOLLAR_MAX_BOUNDARIES`
    sentence boundaries; short ones stay protected so unaffected traces keep the exact
    version 1 segmentation.
    """
    span = text[left:right]
    if not span.startswith("$") or span.startswith("$$"):
        return False
    previous = text[left - 1] if left > 0 else ""
    if not previous or previous.isspace() or previous in OPENING_CONTEXT:
        return False
    inner = span[1:-1]
    if len(inner) > SINGLE_DOLLAR_MAX_CHARS:
        return True
    return len(_BOUNDARY_PATTERN.findall(inner)) >= SINGLE_DOLLAR_MAX_BOUNDARIES


def math_protection_spans(text: str) -> list[tuple[int, int]]:
    """Version 2 protected spans, matched once per text and shared by every range.

    The scanner walks the markers once, so it stays linear on unclosed math. Single `$`
    delimiters are paired exactly like the frozen pattern: the opener takes the nearest
    later unescaped `$` and only falls back to the nearest `\\$` when no unescaped `$`
    follows at all; an opener with no partner stays literal. Everything the winning
    expression covers, including a nested `$$`, `\\[..\\]` or `\\(..\\)` block, is
    consumed by it, and a suspicious span is dropped by `_hidden_by_stray_delimiter`.
    """
    markers = _dollar_markers(text)
    total = len(markers)
    spans: list[tuple[int, int]] = []
    skip_until = -1
    position = 0
    while position < total:
        index, marker = markers[position]
        if index < skip_until:
            position += 1
            continue
        if marker != "$":
            closing = text.find("\\]" if marker == "\\[" else "\\)", index + 2)
            if closing >= 0:
                spans.append((index, closing + 2))
                skip_until = closing + 2
            position += 1
            continue
        if _is_escaped_dollar(text, index):
            position += 1
            continue
        if text.startswith("$$", index):
            closing = text.find("$$", index + 2)
            if closing >= 0:
                spans.append((index, closing + 2))
                skip_until = closing + 2
                position += 1
                continue
        closing = closing_position = fallback = fallback_position = None
        cursor = position + 1
        while cursor < total:
            other, other_marker = markers[cursor]
            if other_marker == "$":
                if not _is_escaped_dollar(text, other):
                    closing, closing_position = other, cursor
                    break
                if fallback is None:
                    fallback, fallback_position = other, cursor
            cursor += 1
        if closing is None:
            closing, closing_position = fallback, fallback_position
        if closing is None:
            position += 1
            continue
        if not _hidden_by_stray_delimiter(text, index, closing + 1):
            spans.append((index, closing + 1))
        skip_until = closing + 1
        position = closing_position + 1
    return sorted(spans)


def map_selection_to_v2(trace: Any, indices: Iterable[int]) -> list[int]:
    """Map stored version 1 unit indices onto the version 2 sub-units they contain.

    The fix only ever splits a version 1 unit further, so one stored v1 index can expand
    into several v2 indices but never into characters outside the old unit's span.
    """
    legacy = sentence_spans_v1(trace)
    current = sentence_spans(trace)
    starts = [unit["start"] for unit in current]
    mapped: list[int] = []
    for index in indices:
        if type(index) is not int or not 0 <= index < len(legacy):
            raise ValueError(f"version 1 unit index out of range: {index!r}")
        left, right = legacy[index]["start"], legacy[index]["end"]
        position = bisect_left(starts, left)
        while position < len(current) and current[position]["end"] <= right:
            mapped.append(current[position]["index"])
            position += 1
    return sorted(set(mapped))


def splitter_delta(trace: Any) -> dict[str, Any]:
    """Describe how the fixed splitter changes one trace relative to the frozen v1 one."""
    legacy = sentence_spans_v1(trace)
    current = sentence_spans(trace)
    legacy_keys = {(unit["start"], unit["end"], unit["text"]) for unit in legacy}
    current_keys = {(unit["start"], unit["end"], unit["text"]) for unit in current}
    superseded = [
        unit for unit in legacy if (unit["start"], unit["end"], unit["text"]) not in current_keys
    ]
    mapped = map_selection_to_v2(trace, [unit["index"] for unit in legacy])
    return {
        "v1_units": len(legacy),
        "v2_units": len(current),
        "mapped_units": len(mapped),
        "added_units": len(current_keys - legacy_keys),
        "superseded_units": len(superseded),
        "superseded": [
            {
                "index": unit["index"],
                "start": unit["start"],
                "end": unit["end"],
                "text": unit["text"],
            }
            for unit in superseded
        ],
        "affected": legacy_keys != current_keys or len(mapped) != len(legacy),
    }


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
    # Imported lazily so unit segmentation and its tests do not need the torch stack.
    from moe_exp.models.inference import _find_prompt_length, _format_prompt

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
