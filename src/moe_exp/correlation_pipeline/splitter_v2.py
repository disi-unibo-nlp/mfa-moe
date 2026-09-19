"""Math protection used by the corrected September 2026 imported labels.

Copied from label producer commit 1e97c5c758b64d07bcecf7da91ffc5152c3afd95.
"""

from __future__ import annotations

import re

SINGLE_DOLLAR_MAX_CHARS = 1000


SINGLE_DOLLAR_MAX_BOUNDARIES = 2


OPENING_CONTEXT = "([{=,;:"


_BRACKET_MARKER = re.compile(r"\\[\[\(]")


_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])[^\S\n]+|\n+")


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
