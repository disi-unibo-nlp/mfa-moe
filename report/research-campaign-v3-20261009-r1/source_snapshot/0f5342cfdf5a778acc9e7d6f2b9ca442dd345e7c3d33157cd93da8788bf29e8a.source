from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Answer-extraction regexes
# ---------------------------------------------------------------------------

_GSM_GOLD_RE = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gold_answer_gsm8k(raw: str) -> str:
    m = _GSM_GOLD_RE.search(raw)
    return m.group(1).strip().replace(",", "") if m else raw.strip()


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
        pass

    # Exact match only for non-numeric answers (no substring fallback)
    return ma == ga


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(records: list, path: Path) -> None:
    """Write JSONL atomically so an interrupted run is never a completion marker."""
    import jsonlines

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        with jsonlines.open(tmp_path, mode="w") as writer:
            for rec in records:
                writer.write(rec.model_dump() if hasattr(rec, "model_dump") else rec)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict | list | None:
    """Read a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
