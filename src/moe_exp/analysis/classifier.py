from __future__ import annotations

import re

from moe_exp.schemas import StepLabels
from moe_exp.utils import extract_model_answer

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Explicit backtracking / self-correction phrases
_BACKTRACK = re.compile(
    r"\b(?:"
    r"wait"
    r"|actually"
    r"|let me reconsider"
    r"|let['’]s reconsider"
    r"|let['’]s try again"
    r"|let me re-?(?:do|check|try|think|start|approach|calculate|compute|evaluate|examine)"
    r"|let['’]s re-?(?:do|check|try|think|start|approach|calculate|compute|evaluate|examine)"
    r"|let (?:me|us) start over"
    r"|on second thought"
    r"|scratch that"
    r"|never\s?mind"
    r"|(?:i|we) (?:need to |should )?(?:try|use) a different approach"
    r"|(?:i|we) (?:made|got) (?:a|an|another) (?:mistake|error)"
    r"|(?:i|we) (?:was|were) wrong"
    r"|my (?:mistake|bad|apologies)"
    r"|i apologize"
    r"|error in (?:my|our) (?:reasoning|calculations|logic)"
    r"|that(?:'s| is) (?:wrong|incorrect|not right)"
    r"|hmm+"
    r"|oops"
    r"|going back"
    r"|revisiting"
    r"|correction:"
    r"|hold on"
    r"|no,?\s+wait"
    r")\b",
    re.IGNORECASE,
)

# Explicit contradiction phrases
_CONTRADICTION = re.compile(
    r"\b(?:"
    r"contradict(?:s|ion)?"
    r"|inconsistent"
    r"|impossible(?: situation)?"
    r"|not (?:a )?valid"
    r"|(?:which|this) (?:is|leads to) (?:a|an)?\s*(?:impossible|contradiction)"
    r"|violates"
    r"|this (?:can't|cannot) be right"
    r"|this doesn't add up"
    r"|this equation is not true"
    r"|not possible"
    r"|but (?:earlier|above|i (?:said|calculated|stated))"
    r"|but that (?:means|implies|gives)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intermediate_answers(steps: list[str]) -> list[str | None]:
    """For each step, try to extract a numeric/expression answer. None if absent."""
    return [extract_model_answer(s) or None for s in steps]


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_trace(steps: list[str], full_text: str) -> StepLabels:  # noqa: ARG001
    """Return step-level taxonomy labels for a single CoT trace.

    Detects:
    - backtracking_steps   : steps containing explicit backtrack phrases
    - contradiction_steps  : steps containing explicit contradiction phrases
    - self_correction_steps: backtrack steps where the extracted answer
                             differs from the answer stated before the event
    - final_answer_reversal: the last extracted answer differs from the first
    - first_error_step     : heuristic — earliest backtracking or contradiction
                             step; None when neither is detected. For datasets
                             with ground-truth labels (e.g. ProcessBench) the
                             caller should overwrite this with the gold label.
    """
    backtracking_steps: list[int] = []
    contradiction_steps: list[int] = []

    for i, step in enumerate(steps):
        if _BACKTRACK.search(step):
            backtracking_steps.append(i)
        if _CONTRADICTION.search(step):
            contradiction_steps.append(i)

    # Self-correction: backtrack event + answer changes in subsequent steps
    intermediates = _intermediate_answers(steps)
    self_correction_steps: list[int] = []
    for bt_idx in backtracking_steps:
        before = [a for a in intermediates[:bt_idx] if a is not None]
        after = [a for a in intermediates[bt_idx + 1 :] if a is not None]
        if before and after and before[-1] != after[-1]:
            self_correction_steps.append(bt_idx)

    # Final-answer reversal: first vs last extracted answer differ
    non_null = [a for a in intermediates if a is not None]
    final_answer_reversal = len(non_null) >= 2 and non_null[0] != non_null[-1]

    # Heuristic first_error_step: earliest signal of an error in the trace
    candidates = sorted(set(backtracking_steps) | set(contradiction_steps))
    first_error_step: int | None = candidates[0] if candidates else None

    return StepLabels(
        first_error_step=first_error_step,
        backtracking_steps=backtracking_steps,
        contradiction_steps=contradiction_steps,
        self_correction_steps=self_correction_steps,
        final_answer_reversal=final_answer_reversal,
    )
