"""Run one exact 25,000-sentence Qwen batch-64 annotation part."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from moe_exp.correlation_pipeline.annotation_partition import (
    DATASETS,
    PARTS,
    PART_SIZE,
    TOTAL,
    enumerate_items,
    partition_items,
    plan,
    trace_key,
)
from moe_exp.correlation_pipeline.batch_predictor import (
    classify_batch_outcomes,
    load_program_and_adapter,
)

BATCH_SIZE = 64
LABELS = frozenset(("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor"))
UNKNOWN_FIELDS = frozenset(("status", "raw_completion", "failure"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _publish(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _unknown_record(row: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Build one persisted `unknown` record, rejecting metadata we cannot re-read."""
    raw = outcome.get("raw_completion")
    if raw is not None and not isinstance(raw, str):
        raise ValueError("unknown outcome raw_completion must be a string or null")
    failure = outcome.get("failure")
    if not isinstance(failure, dict):
        raise ValueError("unknown outcome is missing failure metadata")
    missing = [
        key for key in ("kind", "error_type", "message") if not isinstance(failure.get(key), str)
    ]
    if missing:
        raise ValueError(f"unknown outcome failure is missing {missing}")
    finish_reason = failure.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ValueError("unknown outcome finish_reason must be a string or null")
    return dict(
        row,
        status="unknown",
        raw_completion=raw,
        failure={
            "kind": failure["kind"],
            "error_type": failure["error_type"],
            "message": failure["message"],
            "finish_reason": finish_reason,
        },
    )


def validate_record(row: dict[str, Any], record: Any) -> None:
    """Check one checkpoint record against its source row; raise `ValueError` if it drifts.

    Legacy valid records (`label` only, no status field) and new `unknown` records are
    both accepted, so checkpoints written before this change still resume.
    """
    if not isinstance(record, dict):
        raise ValueError("invalid checkpoint record")
    if any(record.get(key) != value for key, value in row.items()):
        raise ValueError("invalid checkpoint record")
    added = set(record) - set(row)
    if added == {"label"} and record["label"] in LABELS:
        return
    if added == set(UNKNOWN_FIELDS) and record.get("status") == "unknown":
        _unknown_record(row, record)
        return
    raise ValueError("invalid checkpoint record")


def outcome_record(row: dict[str, Any], outcome: Any) -> dict[str, Any]:
    """Normalise one classifier outcome into the record that gets checkpointed.

    A canonical label yields the legacy record shape. An `unknown` outcome from the
    tolerant predictor yields a record carrying the exact raw completion and its
    failure metadata, so a bad choice is persisted work instead of a lost batch.
    Anything else is a programming or checkpoint contract error.
    """
    if isinstance(outcome, str):
        if outcome not in LABELS:
            raise ValueError(f"invalid classifier label {outcome!r}")
        record = dict(row, label=outcome)
    elif isinstance(outcome, dict) and outcome.get("status") == "unknown":
        record = _unknown_record(row, outcome)
    elif isinstance(outcome, dict):
        label = outcome.get("label")
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"invalid classifier outcome {outcome!r}")
        record = dict(row, label=label)
    else:
        raise ValueError(f"invalid classifier outcome {outcome!r}")
    validate_record(row, record)
    return record


def run_part(items, config, *, output_dir, classify, expected_count=PART_SIZE, dry_run=False):
    rows = [dict(identity=identity, unit=unit, inputs=inputs) for identity, _trace, unit, inputs in items]
    if len(rows) != expected_count:
        raise ValueError(f"part count mismatch: expected {expected_count}, found {len(rows)}")
    binding = dict(schema_version=1, config=config, items_sha256=digest(rows), expected_count=expected_count)
    root = Path(output_dir)
    checkpoints = root / "checkpoints"
    saved = []
    for index, path in enumerate(sorted(checkpoints.glob("*.json"))):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name != f"batch-{index:06d}.json" or value.get("binding") != binding:
            raise ValueError("stale checkpoint")
        records = value.get("records")
        if not isinstance(records, list) or not 1 <= len(records) <= BATCH_SIZE:
            raise ValueError("invalid batch size")
        for record in records:
            position = len(saved)
            if position >= len(rows):
                raise ValueError("invalid checkpoint record")
            validate_record(rows[position], record)
            saved.append(record)
    if dry_run:
        return {"status": "dry_run", "completed": len(saved), "expected": expected_count, "binding_sha256": digest(binding)}
    for offset in range(len(saved), len(rows), BATCH_SIZE):
        batch = rows[offset:offset + BATCH_SIZE]
        outcomes = classify([row["inputs"] for row in batch])
        if not isinstance(outcomes, list) or len(outcomes) != len(batch):
            raise ValueError("invalid batch outcomes")
        records = [outcome_record(row, outcome) for row, outcome in zip(batch, outcomes, strict=True)]
        _publish(checkpoints / f"batch-{len(saved) // BATCH_SIZE:06d}.json", {"binding": binding, "records": records})
        saved.extend(records)
        if len(saved) % (BATCH_SIZE * 10) == 0 or len(saved) == len(rows):
            print(f"PART_PROGRESS completed={len(saved)} expected={len(rows)}", flush=True)
    summary = {"status": "complete", "completed": len(saved), "expected": len(rows), "binding_sha256": digest(binding), "annotations_sha256": digest(saved)}
    unknown_count = sum(1 for record in saved if record.get("status") == "unknown")
    if unknown_count:
        summary["unknown_count"] = unknown_count
    for path, value in ((root / "annotations.json", saved), (root / "summary.json", summary)):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("stale completed output")
        else:
            _publish(path, value)
    return summary


def load_include_traces(path: Path) -> set[tuple[str, str, int]]:
    """Read the committed affected-trace list used for targeted re-labeling."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported include-traces schema in {path}")
    entries = document.get("trace_keys")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"include-traces file {path} has no trace keys")
    keys = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid include-traces entry in {path}: {entry!r}")
        keys.add((entry["dataset"], entry["problem_id"], entry["sample_id"]))
    return keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument(
        "--include-traces",
        type=Path,
        help="JSON list of trace keys to label instead of the first --total units",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, default=PARTS)
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--judge-program", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("MFA_FAULTHANDLER") == "1":
        import faulthandler

        # Slurm batch runs enable this so a stalled process writes a Python stack
        # to stderr before the launcher's stalled-write watchdog kills it.
        faulthandler.dump_traceback_later(1200, repeat=True)
    if args.batch_size != BATCH_SIZE:
        raise ValueError("production batch size is fixed at 64")
    if not 0 <= args.part < args.parts or args.part_size * args.parts != args.total:
        raise ValueError("part, parts, part-size and total must define exact disjoint parts")
    include = load_include_traces(args.include_traces) if args.include_traces else None
    items = enumerate_items(
        args.trace_root, DATASETS, limit=None if include is not None else args.total, include=include
    )
    if include is not None and not args.plan_only and len(items) != args.total:
        raise ValueError(
            f"include-traces selection has {len(items)} units, expected exactly {args.total}"
        )
    if len(items) < args.total:
        raise ValueError(f"source has {len(items)} identities, fewer than required {args.total}")
    selected = items[:args.total]
    parts = partition_items(selected, part_size=args.part_size, parts=args.parts)
    source_plan = {
        "schema_version": 1,
        "trace_root": str(args.trace_root),
        "datasets": list(DATASETS),
        "include_traces_sha256": (
            digest(sorted(include)) if include is not None else None
        ),
        "source_identities": len(items),
        "selected_identities": len(selected),
        "part_size": args.part_size,
        "parts": args.parts,
        "part_counts": [len(part) for part in parts],
        "selected_sha256": hashlib.sha256("\n".join(digest(item[0]) for item in selected).encode()).hexdigest(),
        "first_identity": selected[0][0],
        "last_identity": selected[-1][0],
    }
    print(json.dumps({"source_plan": source_plan, "part": args.part, "part_count": len(parts[args.part])}, sort_keys=True), flush=True)
    if args.plan_only:
        return
    program_bytes = args.judge_program.read_bytes()
    config = {
        "source_plan_sha256": digest(source_plan),
        "program_sha256": hashlib.sha256(program_bytes).hexdigest(),
        "judge_model": args.model,
        "base_url": args.base_url,
        "batch_size": BATCH_SIZE,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking": True,
        "preserve_thinking": False,
    }
    _program, predict, adapter = load_program_and_adapter(str(args.judge_program))
    def classify(batch):
        return classify_batch_outcomes(batch, adapter=adapter, predict=predict, model=args.model, base_url=args.base_url, api_key=args.api_key, max_tokens=args.max_tokens, temperature=args.temperature, reasoning_effort=args.reasoning_effort, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty)
    result = run_part(
        parts[args.part],
        config,
        output_dir=args.output_dir,
        classify=classify,
        expected_count=len(parts[args.part]),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
