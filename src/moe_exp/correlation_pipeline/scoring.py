from __future__ import annotations

import re
from typing import Any

from moe_exp.utils import answers_match, extract_model_answer

_CHOICE_RE = re.compile(r"\\boxed\s*\{\s*([A-J])\s*\}", re.IGNORECASE)


def extract_choice(text: str) -> str:
    matches = _CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else ""


def extract_answer(text: str, answer_type: str) -> str:
    return extract_choice(text) if answer_type == "choice" else extract_model_answer(text)


def _math_verify(model_text: str, gold_answer: str) -> bool | None:
    try:
        from math_verify import parse, verify

        gold = parse(f"${gold_answer}$")
        prediction = parse(model_text)
        if not gold or not prediction:
            return None
        return bool(verify(gold, prediction))
    except Exception:  # noqa: BLE001 - symbolic parsers expose heterogeneous failures
        return None


def score_completion(
    example: dict[str, Any],
    *,
    answer_type: str,
    model_text: str,
) -> tuple[str, bool | None, str]:
    gold = str(example.get("gold_answer") or "").strip()
    model_answer = extract_answer(model_text, answer_type)
    if not gold:
        return model_answer, None, "unscored_no_gold_answer"
    if answer_type == "choice":
        return model_answer, model_answer == gold.upper(), "exact_multiple_choice"
    verified = _math_verify(model_text, gold)
    if verified is not None:
        return model_answer, verified, "math_verify"
    return model_answer, answers_match(model_answer, gold), "normalized_exact_numeric_fallback"