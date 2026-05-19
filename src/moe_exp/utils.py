from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import jsonlines

# ---------------------------------------------------------------------------
# Answer-extraction regexes
# ---------------------------------------------------------------------------

_GSM_GOLD_RE = re.compile(r"####\s*([^\n]+)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gold_answer_gsm8k(raw: str) -> str:
    m = _GSM_GOLD_RE.search(raw)
    return m.group(1).strip().replace(",", "") if m else raw.strip()


def extract_model_answer(text: str) -> str:
    """Best-effort final-answer extraction from generated CoT text.

    Priority: \\boxed{} → #### marker → "the answer is …" → last number.
    Returns empty string when nothing is found.
    """
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _GSM_GOLD_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")

    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")

    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else ""


def answers_match(model_answer: str, gold_answer: str) -> Optional[bool]:
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
        pass

    # Exact match only for non-numeric answers (no substring fallback)
    return ma == ga


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for rec in records:
            writer.write(rec.model_dump() if hasattr(rec, "model_dump") else rec)


def read_json(path: Path) -> dict | list | None:
    """Read a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
