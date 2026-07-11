from __future__ import annotations

from moe_exp.schemas import TraceRecord


def build_row(model_id: str, dataset_name: str, traces: list[TraceRecord]) -> dict:
    """Aggregate taxonomy counts for one (model, dataset) pair.

    Produces the columns from the Experiment 1 table:
      Model | Dataset | % correct traces | % backtracking | % contradiction |
      % self-correction | % final-answer reversal |
      # self-correction traces ending correct | # ending incorrect

    ``is_correct`` is a final trace label.  It does not establish that an
    intermediate self-correction caused the final outcome; the field names make
    that distinction explicit.
    """
    n = len(traces)
    if n == 0:
        return {"model": model_id, "dataset": dataset_name, "n_examples": 0}

    n_correct = sum(1 for t in traces if t.is_correct is True)
    n_backtrack = sum(1 for t in traces if t.step_labels.backtracking_steps)
    n_contradiction = sum(1 for t in traces if t.step_labels.contradiction_steps)
    n_self_corr = sum(1 for t in traces if t.step_labels.self_correction_steps)
    n_reversal = sum(1 for t in traces if t.step_labels.final_answer_reversal)

    self_corr_traces = [t for t in traces if t.step_labels.self_correction_steps]
    n_sc_ending_correct = sum(1 for t in self_corr_traces if t.is_correct is True)
    n_sc_ending_incorrect = sum(1 for t in self_corr_traces if t.is_correct is False)

    def pct(k: int) -> float:
        return round(k / n, 4)

    return {
        "model": model_id,
        "dataset": dataset_name,
        "n_examples": n,
        "pct_correct_traces": pct(n_correct),
        "pct_backtracking": pct(n_backtrack),
        "pct_contradiction": pct(n_contradiction),
        "pct_self_correction": pct(n_self_corr),
        "pct_final_answer_reversal": pct(n_reversal),
        "n_self_correction_traces_ending_correct": n_sc_ending_correct,
        "n_self_correction_traces_ending_incorrect": n_sc_ending_incorrect,
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        "experiment": 1,
        "description": "Basic reasoning and failure taxonomy",
        "columns": [
            "model",
            "dataset",
            "n_examples",
            "pct_correct_traces",
            "pct_backtracking",
            "pct_contradiction",
            "pct_self_correction",
            "pct_final_answer_reversal",
            "n_self_correction_traces_ending_correct",
            "n_self_correction_traces_ending_incorrect",
        ],
        "rows": rows,
    }
