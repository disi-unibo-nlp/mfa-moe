"""Reapply the current event taxonomy to existing trace files.

This refreshes heuristic labels without re-generating chains or re-extracting
routing tensors. Gold first-error labels are retained exactly.

Usage
-----
    python -m moe_exp.analysis.relabel_traces \
        --input results/exp2/.../traces_with_routing.jsonl \
        --output results/exp2/.../traces_with_routing_relabelled.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from moe_exp.analysis.classifier import classify_trace
from moe_exp.schemas import TraceRecord


LABEL_FIELDS = (
    "backtracking_steps",
    "contradiction_steps",
    "self_correction_steps",
    "first_reasoning_event_step",
    "final_answer_reversal",
)


def _label_counts(traces: Iterable[TraceRecord]) -> dict[str, int]:
    """Count traces carrying each heuristic label (not individual event tokens)."""
    counts: Counter[str] = Counter()
    for trace in traces:
        labels = trace.step_labels
        for field in LABEL_FIELDS:
            value = getattr(labels, field)
            present = bool(value) if isinstance(value, list) else value is not None and value is not False
            if present:
                counts[field] += 1
    return dict(counts)


def relabel_trace(trace: TraceRecord) -> TraceRecord:
    """Replace heuristic labels and keep independently supplied gold labels."""
    relabelled = classify_trace(trace.steps, trace.cot_text)
    relabelled.first_error_step = trace.step_labels.first_error_step
    trace.step_labels = relabelled
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reapply the current high-precision event taxonomy to trace JSONL"
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing trace JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Relabelled trace JSONL")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: beside --output)",
    )
    args = parser.parse_args()

    traces: list[TraceRecord] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(TraceRecord.model_validate_json(line))

    before = _label_counts(traces)
    relabelled = [relabel_trace(trace) for trace in traces]
    after = _label_counts(relabelled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trace in relabelled:
            handle.write(trace.model_dump_json() + "\n")

    summary_path = args.summary_output or args.output.with_suffix(".relabel_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input": args.input.as_posix(),
                "output": args.output.as_posix(),
                "n_traces": len(traces),
                "before_trace_counts": before,
                "after_trace_counts": after,
            },
            handle,
            indent=2,
        )

    print(f"Relabelled {len(traces)} traces: {args.output}")
    for field in LABEL_FIELDS:
        print(f"{field}: {before.get(field, 0)} -> {after.get(field, 0)}")


if __name__ == "__main__":
    main()
