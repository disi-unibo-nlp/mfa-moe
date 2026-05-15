from __future__ import annotations

import re

# Numbered step markers: "1.", "2)", "Step 1:", "Step 2." etc.
_NUMBERED_STEP = re.compile(
    r"(?:^|\n)\s*(?:step\s+)?\d+[.)]\s+",
    re.IGNORECASE | re.MULTILINE,
)


def split_steps(text: str) -> list[str]:
    """Split CoT text into individual reasoning steps.

    Priority:
    1. Numbered markers — "1.", "Step 2:", "3)" etc.
    2. Double-newline paragraph breaks.
    3. Single-newline breaks.
    """
    # 1. Numbered step markers
    parts = _NUMBERED_STEP.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts

    # 2. Blank-line paragraph breaks
    parts = re.split(r"\n{2,}", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts

    # 3. Single newlines
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts if parts else [text.strip()]
