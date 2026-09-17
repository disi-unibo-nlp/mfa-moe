"""Build deterministic sentence identities and exact production partitions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.spans import (
    digest,
    map_selection_to_v2,
    selected_sentence_indices,
    sentence_spans,
    sentence_spans_v1,
    trace_digest,
)
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

DATASETS = ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva")
PART_SIZE = 25_000
PARTS = 4
TOTAL = PART_SIZE * PARTS


def capped_quotas(capacities: dict, weights: dict, total: int) -> tuple[dict, list]:
    """Retain the existing capped proportional allocation helper."""
    if type(total) is not int or total <= 0 or set(capacities) != set(weights):
        raise ValueError("Positive total and matching capacities/weights required")
    if any(type(n) is not int or n < 0 for n in capacities.values()):
        raise ValueError("Invalid capacities")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Invalid weights")
    from fractions import Fraction
    w = {k: Fraction(str(v)) for k, v in weights.items()}
    active = {k for k in w if w[k] > 0 and capacities[k] > 0}
    if sum(capacities[k] for k in active) < total:
        raise ValueError("Insufficient positive-weight sentence supply")
    quotas = dict.fromkeys(capacities, 0)
    rounds = []
    remaining = total
    while remaining:
        weight = sum(w[k] for k in active)
        shares = {k: remaining * w[k] / weight for k in active}
        capped = sorted(k for k in active if shares[k] >= capacities[k])
        rounds.append({"remaining": remaining, "active": sorted(active), "capped": capped})
        if capped:
            for k in capped:
                quotas[k] = capacities[k]
                remaining -= quotas[k]
                active.remove(k)
            continue
        for k in active:
            quotas[k] = int(shares[k])
        left = remaining - sum(quotas[k] for k in active)
        for k in sorted(active, key=lambda k: (-(shares[k] - quotas[k]), k))[:left]:
            quotas[k] += 1
        remaining = 0
    return quotas, rounds


def _question(trace: TraceRecord) -> str:
    return "\n\n".join(
        str(message["content"])
        for message in (trace.generation_messages or [])
        if message.get("role") == "user"
    ) or trace.prompt


def _inputs(question: str, units: list[dict[str, Any]], index: int) -> dict[str, str]:
    return {
        "problem_statement": question,
        "previous_sentence": units[index - 1]["text"] if index else "<START OF RESPONSE>",
        "sentence": units[index]["text"],
        "next_sentence": units[index + 1]["text"] if index + 1 < len(units) else "<END OF RESPONSE>",
    }


def trace_key(trace: TraceRecord) -> tuple[str, str, int]:
    """The exact source key used by `--include-traces` and the affected list."""
    return (trace.dataset, trace.problem_id, trace.sample_id)


def enumerate_items(
    trace_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    *,
    limit: int | None = None,
    include: set[tuple[str, str, int]] | None = None,
) -> list[tuple[dict, TraceRecord, dict, dict]]:
    """Enumerate source traces in file order without padding or deduplication.

    `limit` stops after that many items. `include` restricts the enumeration to those
    trace keys and stops as soon as every requested key has been seen, so a targeted
    re-label only reads the traces it needs; a key that is missing from the source is
    an error rather than a silently smaller part.
    """
    remaining = set(include) if include is not None else None
    if remaining is not None and not remaining:
        raise ValueError("include must name at least one trace")
    items = []
    for dataset in datasets:
        path = trace_root / dataset / "traces.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            trace = TraceRecord(**row)
            key = trace_key(trace)
            if remaining is not None and key not in remaining:
                continue
            units = sentence_spans(trace)
            trace_sha256 = trace_digest(trace)
            question = _question(trace)
            if remaining is None:
                indices = selected_sentence_indices(trace, units)
            else:
                # A stored sentence_selection always indexes the frozen version 1 units,
                # so a targeted re-label maps it onto the current splitter's sub-units.
                stored = selected_sentence_indices(trace, sentence_spans_v1(trace))
                indices = map_selection_to_v2(trace, stored)
            for index in indices:
                identity = {
                    "dataset": trace.dataset,
                    "problem_id": trace.problem_id,
                    "source_problem_id": trace.source_problem_id,
                    "sample_id": trace.sample_id,
                    "sentence_index": index,
                    "start": units[index]["start"],
                    "end": units[index]["end"],
                    "trace_sha256": trace_sha256,
                }
                items.append((identity, trace, units[index], _inputs(question, units, index)))
                if limit is not None and len(items) >= limit:
                    break
            if remaining is not None:
                remaining.discard(key)
                if not remaining:
                    break
            if limit is not None and len(items) >= limit:
                break
        if remaining is not None and not remaining:
            break
        if limit is not None and len(items) >= limit:
            break
    if remaining:
        raise ValueError(f"include traces missing from source: {sorted(remaining)}")
    identities = [item[0] for item in items]
    if len({digest(identity) for identity in identities}) != len(identities):
        raise ValueError("source contains duplicate sentence identities")
    return items


def partition_items(items: list[tuple], *, part_size: int = PART_SIZE, parts: int = PARTS) -> list[list[tuple]]:
    if type(part_size) is not int or type(parts) is not int or part_size < 1 or parts < 1:
        raise ValueError("part_size and parts must be positive integers")
    if len(items) != part_size * parts:
        raise ValueError(f"expected {part_size * parts} source identities, found {len(items)}")
    result = [items[offset:offset + part_size] for offset in range(0, len(items), part_size)]
    if any(len(part) != part_size for part in result):
        raise ValueError("partition size mismatch")
    flat = [digest(item[0]) for part in result for item in part]
    if len(flat) != len(set(flat)):
        raise ValueError("partitions overlap")
    return result


def plan(trace_root: Path, *, total: int = TOTAL, part_size: int = PART_SIZE, parts: int = PARTS) -> dict[str, Any]:
    items = enumerate_items(trace_root)
    if len(items) < total:
        raise ValueError(f"source has {len(items)} identities, fewer than required {total}")
    selected = items[:total]
    partitions = partition_items(selected, part_size=part_size, parts=parts)
    return {
        "schema_version": 1,
        "trace_root": str(trace_root),
        "datasets": list(DATASETS),
        "source_identities": len(items),
        "selected_identities": len(selected),
        "part_size": part_size,
        "parts": parts,
        "part_counts": [len(part) for part in partitions],
        "selected_sha256": hashlib.sha256(
            "\n".join(digest(item[0]) for item in selected).encode()
        ).hexdigest(),
        "first_identity": selected[0][0],
        "last_identity": selected[-1][0],
    }
