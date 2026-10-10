"""Frozen deterministic benchmark scorer plus conservative semantic bounds."""
from __future__ import annotations

from .contracts import require


def score(result, solution, *, scorer=None):
    from moe_exp.utils import extract_model_answer
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    scorer = scorer or score_completion_detailed
    gold = extract_model_answer(solution)
    require(bool(gold), "reference solution has no extractable final answer")
    if not result["execution_complete"]:
        missing = result.get("type") == "AssignmentMissingAtCloseout"
        return {"strict_score": 0, "semantic_lower": 0, "semantic_upper": 1,
                "status": "ASSIGNED_MISSING" if missing else "EXECUTION_FAILURE", "assigned_failure_retained": True}
    detail = scorer({"gold_answer": gold}, answer_type="math", model_text=result["text"])
    answer, accepted = detail["model_answer"], detail["is_correct"]
    strict = int(accepted is True)
    if strict:
        lower, upper, status = 1, 1, "STRICT_ACCEPTED"
    elif result["finish_reason"] == "length" or not answer:
        lower, upper, status = 0, 0, "CAPPED_OR_MISSING_FINAL"
    else:
        lower, upper, status = 0, 1, "UNCAPPED_REJECT_UNRESOLVED"
    return {"strict_score": strict, "semantic_lower": lower, "semantic_upper": upper,
            "status": status, "detail": detail, "new_semantic_judge": False}
