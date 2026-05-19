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
    r"|let['\u2019]s reconsider"
    r"|let['\u2019]s try again"
    r"|let me re-?(?:do|check|try|think|start|approach|calculate|compute|evaluate|examine)"
    r"|let['\u2019]s re-?(?:do|check|try|think|start|approach|calculate|compute|evaluate|examine)"
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

# Patterns indicating the model is explicitly stating its final answer
_FINAL_ANSWER_STATEMENT = re.compile(
    r"(?:"
    r"(?:the\s+)?(?:final\s+)?answer\s+is"
    r"|\\boxed\{"
    r"|####"
    r"|therefore,?\s+(?:the\s+)?answer"
    r"|so,?\s+(?:the\s+)?answer"
    r"|thus,?\s+(?:the\s+)?answer"
    r"|hence,?\s+(?:the\s+)?answer"
    r"|in conclusion"
    r"|my answer is"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intermediate_answers(steps: list[str]) -> list[str | None]:
    """For each step, try to extract a numeric/expression answer. None if absent."""
    return [extract_model_answer(s) or None for s in steps]


def _extract_stated_final_answers(steps: list[str]) -> list[tuple[int, str]]:
    """Extract (step_index, answer) pairs where the model explicitly states a final answer.

    Only considers steps that contain explicit final-answer language (e.g.,
    "the answer is", "\\boxed{}", "####"), not arbitrary intermediate computations.
    """
    results = []
    for i, step in enumerate(steps):
        if _FINAL_ANSWER_STATEMENT.search(step):
            answer = extract_model_answer(step)
            if answer:
                results.append((i, answer))
    return results


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


def classify_trace(steps: list[str], full_text: str) -> StepLabels:  # noqa: ARG001
    """Return step-level taxonomy labels for a single CoT trace.

    Detects:
    - backtracking_steps        : steps containing explicit backtrack phrases
    - contradiction_steps       : steps containing explicit contradiction phrases
    - self_correction_steps     : backtrack steps where the extracted answer
                                  differs from the answer stated before the event
    - final_answer_reversal     : the model explicitly states a final answer and
                                  later explicitly states a DIFFERENT final answer
    - first_reasoning_event_step: earliest backtracking or contradiction step
                                  (where the model SIGNALS an issue, not where
                                  the actual error occurs)
    - first_error_step          : None by default; should be set by the caller
                                  from gold labels (ProcessBench/PRM800K)
    """
    backtracking_steps: list[int] = []
    contradiction_steps: list[int] = []

    for i, step in enumerate(steps):
        if _BACKTRACK.search(step):
            backtracking_steps.append(i)
        if _CONTRADICTION.search(step):
            contradiction_steps.append(i)

    # Self-correction: backtrack event where the immediate next answer differs
    # from the last answer before the backtrack (indicating a correction attempt)
    intermediates = _intermediate_answers(steps)
    self_correction_steps: list[int] = []
    for bt_idx in backtracking_steps:
        before = [a for a in intermediates[:bt_idx] if a is not None]
        # Look only at answers immediately following the backtrack (not the whole tail)
        after_immediate = [a for a in intermediates[bt_idx:bt_idx + 3] if a is not None]
        if before and after_immediate and before[-1] != after_immediate[0]:
            self_correction_steps.append(bt_idx)

    # Final-answer reversal: the model explicitly states a final answer (via
    # "the answer is", \boxed{}, ####, etc.) and then later states a DIFFERENT one.
    # This is NOT triggered by intermediate computation values changing.
    stated_answers = _extract_stated_final_answers(steps)
    final_answer_reversal = False
    if len(stated_answers) >= 2:
        first_answer = stated_answers[0][1]
        last_answer = stated_answers[-1][1]
        # Normalize for comparison
        first_norm = first_answer.strip().lower().replace(",", "").rstrip(".")
        last_norm = last_answer.strip().lower().replace(",", "").rstrip(".")
        final_answer_reversal = first_norm != last_norm

    # first_reasoning_event_step: earliest signal where the model flags an issue.
    # This is NOT the actual first error - just where the model reacts.
    candidates = sorted(set(backtracking_steps) | set(contradiction_steps))
    first_reasoning_event_step: int | None = candidates[0] if candidates else None

    return StepLabels(
        # first_error_step left as None - caller sets from gold labels if available
        first_error_step=None,
        first_reasoning_event_step=first_reasoning_event_step,
        backtracking_steps=backtracking_steps,
        contradiction_steps=contradiction_steps,
        self_correction_steps=self_correction_steps,
        final_answer_reversal=final_answer_reversal,
    )
