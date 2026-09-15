"""CPU-side probe payload builder for the MTP compatibility job.

Builds 16 real judge inputs from saved generation traces so the probe exercises the
same prompt shape as production, not a synthetic string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.spans import selected_sentence_indices, sentence_spans
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord


def build_probe_items(traces_path: Path, count: int = 16) -> list[dict[str, Any]]:
    """Return `count` real judge inputs drawn from a saved traces.jsonl file."""
    items: list[dict[str, Any]] = []
    for row in iter_jsonl(traces_path):
        trace = TraceRecord(**row)
        units = sentence_spans(trace)
        selected = selected_sentence_indices(trace, units)
        question = (
            "\n\n".join(
                message["content"]
                for message in (trace.generation_messages or [])
                if message["role"] == "user"
            )
            or trace.prompt
        )
        for index in selected:
            items.append(
                {
                    "problem_statement": question,
                    "previous_sentence": units[index - 1]["text"]
                    if index
                    else "<START OF RESPONSE>",
                    "sentence": units[index]["text"],
                    "next_sentence": units[index + 1]["text"]
                    if index + 1 < len(units)
                    else "<END OF RESPONSE>",
                }
            )
            if len(items) == count:
                return items
    raise ValueError(f"only found {len(items)} judge units, needed {count}")
