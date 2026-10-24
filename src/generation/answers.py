"""Read the final answer out of generated text, and compare two answers loosely."""
from __future__ import annotations

import re

_GSM_GOLD_RE = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces.

    Returns None when there is no complete box. Nested braces such as
    ``\\boxed{\\frac{11}{2}}`` are kept whole; an unclosed final box is skipped
    in favour of the previous complete one.
    """
    position = len(text)
    while True:
        start = max(text.rfind("\\boxed", 0, position), text.rfind("\\fbox", 0, position))
        if start < 0:
            return None
        brace = start + (6 if text.startswith("\\boxed", start) else 5)
        while brace < len(text) and text[brace] == " ":
            brace += 1
        if brace < len(text) and text[brace] == "{":
            depth = 0
            for index in range(brace, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1:index]
        position = start


def extract_model_answer(text: str) -> str:
    """Best-effort final-answer extraction from generated CoT text.

    Priority: last balanced \\boxed{} → #### marker → "the answer is …" → last number.
    Returns empty string when nothing is found.
    """
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed.strip()

    m = _GSM_GOLD_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")

    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")

    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else ""


def answers_match(model_answer: str, gold_answer: str) -> bool | None:
    """Fuzzy answer comparison.

    Returns True/False, or None when comparison is ambiguous (missing answers).
    """
    if not model_answer or not gold_answer:
        return None

    ma = model_answer.strip().lower().replace(",", "").rstrip(".")
    ga = gold_answer.strip().lower().replace(",", "").rstrip(".")

    if ma == ga:
        return True

    try:
        return abs(float(ma) - float(ga)) < 1e-6
    except ValueError:
        return False  # non-numeric answers match only when equal (no substring fallback)
