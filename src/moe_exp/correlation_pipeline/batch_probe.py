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


def build_balanced_slice(traces_paths: list[Path], slice_size: int) -> list[dict[str, Any]]:
    """Take an equal number of judge items from each trace file.

    Every sweep cell must see identical inputs, so the split is exact by construction:
    an uneven request is a hard error rather than a silently truncated slice. Each item
    is tagged with the dataset directory it came from so the split is reportable.
    """
    if not traces_paths:
        raise ValueError("at least one traces file is required")
    if slice_size % len(traces_paths) != 0:
        raise ValueError(
            f"slice_size {slice_size} does not divide evenly across "
            f"{len(traces_paths)} trace files"
        )
    per_file = slice_size // len(traces_paths)
    items: list[dict[str, Any]] = []
    for path in traces_paths:
        source = path.parent.name
        for item in build_probe_items(path, per_file):
            items.append({**item, "source": source})
    if len(items) != slice_size:
        raise ValueError(f"assembled {len(items)} items, expected {slice_size}")
    return items
