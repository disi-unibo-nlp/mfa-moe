"""Label the sentences of one trace root with the judge, 64 at a time, resuming after every finished
batch.

Input: a trace root `ROOT/<dataset>/traces.jsonl` whose rows name the sentences to label in
`metadata.sentence_selection` (written by `labeling.sample_stratified` and
`labeling.choose_sentences`), and a vLLM server that serves the judge (`start_server $JUDGE_MODEL`, from
`scripts/slurm/env.sh`; `scripts/labeling/label_sentences.sbatch TRACE_ROOT OUT_DIR` starts both).
Output in OUT: `annotations.json` (one record per sentence, in file order), `summary.json` and
`checkpoints/batch-NNNNNN.json`. A rerun with the same inputs skips the finished batches.

    python -m labeling.label_sentences --trace-root ROOT --output-dir OUT
        --base-url http://127.0.0.1:PORT/v1

With `--include-traces LIST.json` only the listed traces are labelled. Their stored
`sentence_selection` indexes the units of the first sentence splitter (version 1), so it is mapped to the
units of the current one (version 2).
`--plan-only` prints what would be labelled and stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from labeling.batch_predictor import (
    CHAT_TEMPLATE_KWARGS, JUDGE_MODEL, JUDGE_PROGRAM, MAX_TOKENS, SAMPLING, classify_batch_outcomes,
    judge_inputs, load_program_and_adapter, question_of,
)
from labeling.spans import (
    check_unknown, digest, map_selection_to_v2, selected_sentence_indices, sentence_spans,
    sentence_spans_v1, trace_digest,
)
from schemas import TraceRecord
from store import CLASSES, DATASETS, iter_jsonl

BATCH_SIZE = 64
UNKNOWN_FIELDS = frozenset(("status", "raw_completion", "failure"))


def _publish(path: Path, value: Any) -> None:
    """Write `value` as JSON; fails if `path` exists, and a killed writer never leaves a partial
    file."""
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
        Path(name).unlink(missing_ok=True)


def _unknown_record(row: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """One persisted `unknown` record: the row plus the raw completion and failure."""
    raw, failure = check_unknown(outcome)
    return dict(row, status="unknown", raw_completion=raw, failure=failure)


def validate_record(row: dict[str, Any], record: Any) -> None:
    """Check one checkpoint record against its source row: it is the row plus a `label`, or the row plus
    the fields of an `unknown` outcome (status, raw completion and failure)."""
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in row.items()):
        raise ValueError("invalid checkpoint record")
    added = set(record) - set(row)
    if added == {"label"} and record["label"] in CLASSES:
        return
    if added == UNKNOWN_FIELDS and record.get("status") == "unknown":
        _unknown_record(row, record)
        return
    raise ValueError("invalid checkpoint record")


def outcome_record(row: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """The record to checkpoint for one classifier outcome: the row plus its label, or plus the raw
    completion and failure of an `unknown` outcome, so that a bad answer is saved work instead of a
    lost batch."""
    if outcome.get("status") == "unknown":
        return _unknown_record(row, outcome)
    label = outcome.get("label")
    if label not in CLASSES:
        raise ValueError(f"invalid classifier outcome {outcome!r}")
    return dict(row, label=label)


def _without_server(value: dict) -> dict:
    """A binding or summary without the server address (`config.base_url`). Checkpoints may store the
    address in their key, but a rerun on another port must still match them."""
    value = dict(value or {})
    if "base_url" in value.get("config", {}):
        value["config"] = {k: v for k, v in value["config"].items() if k != "base_url"}
    return value


def run_part(rows, config, *, output_dir, classify):
    """Label every row (identity, unit, inputs), checkpointing each batch; returns the summary."""
    binding = dict(schema_version=1, config=config, items_sha256=digest(rows),
                   expected_count=len(rows))
    root = Path(output_dir)
    checkpoints = root / "checkpoints"
    saved = []
    for index, path in enumerate(sorted(checkpoints.glob("*.json"))):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name != f"batch-{index:06d}.json" or _without_server(value.get("binding")) != binding:
            raise ValueError("stale checkpoint")
        records = value.get("records")
        if not isinstance(records, list) or not 1 <= len(records) <= BATCH_SIZE:
            raise ValueError("invalid batch size")
        for record in records:
            if len(saved) >= len(rows):
                raise ValueError("invalid checkpoint record")
            validate_record(rows[len(saved)], record)
            saved.append(record)
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
    summary = {"status": "complete", "completed": len(saved), "expected": len(rows),
               "binding_sha256": digest(binding), "annotations_sha256": digest(saved)}
    unknown_count = sum(1 for record in saved if record.get("status") == "unknown")
    if unknown_count:
        summary["unknown_count"] = unknown_count
    for path, value in ((root / "annotations.json", saved), (root / "summary.json", summary)):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "summary.json" and existing.get("binding_sha256") != value["binding_sha256"]:
                stored = json.loads(sorted(checkpoints.glob("*.json"))[0].read_text(encoding="utf-8"))["binding"]
                if digest(stored) == existing["binding_sha256"] and _without_server(stored) == binding:
                    existing["binding_sha256"] = value["binding_sha256"]
            if existing != value:
                raise ValueError("stale completed output")
        else:
            _publish(path, value)
    return summary


def load_include_traces(path: Path, source: str | None = None) -> set[tuple[str, str, int]]:
    """Read the list of traces to relabel.

    `source` keeps only the entries that belong to the corpus being labelled, because the same
    problem id can exist in two corpora.
    """
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
        if source is not None and entry.get("source") != source:
            continue
        keys.add((entry["dataset"], entry["problem_id"], entry["sample_id"]))
    if not keys:
        raise ValueError(f"include-traces file {path} has no entries for source {source!r}")
    return keys


def checkpoint_config(source_plan: dict, program_bytes: bytes) -> dict:
    """What a checkpoint is bound to: the traces, the judge program, the model and the decoding
    settings.

    The server address is not part of it (every Slurm job serves on its own port), so a rerun on
    another node or port resumes the same checkpoints.
    """
    return {
        "source_plan_sha256": digest(source_plan),
        "program_sha256": hashlib.sha256(program_bytes).hexdigest(),
        "judge_model": JUDGE_MODEL,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        **SAMPLING,
        **CHAT_TEMPLATE_KWARGS,
    }


def enumerate_items(
    trace_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    *,
    include: set[tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """The sentences to label, in file order and without padding or deduplication.

    Each comes as `{identity, unit, inputs}`.

    `include` restricts the enumeration to those (dataset, problem id, sample id) traces and stops
    as soon as every one has been seen; a key that is missing from the source is an error rather
    than a silently smaller part.
    """
    remaining = set(include) if include is not None else None
    rows = []
    for dataset in datasets:
        path = trace_root / dataset / "traces.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for raw in iter_jsonl(path):
            trace = TraceRecord(**raw)
            key = (trace.dataset, trace.problem_id, trace.sample_id)
            if remaining is not None and key not in remaining:
                continue
            units = sentence_spans(trace)
            if remaining is None:
                indices = selected_sentence_indices(trace, units)
            else:
                # A stored sentence_selection indexes the version 1 units, so a targeted relabel maps it
                # onto the sub-units of the current splitter.
                stored = selected_sentence_indices(trace, sentence_spans_v1(trace))
                indices = map_selection_to_v2(trace, stored)
            question = question_of(trace.generation_messages, trace.prompt)
            sha = trace_digest(trace)
            for index in indices:
                identity = {
                    "dataset": trace.dataset,
                    "problem_id": trace.problem_id,
                    "source_problem_id": trace.source_problem_id,
                    "sample_id": trace.sample_id,
                    "sentence_index": index,
                    "start": units[index]["start"],
                    "end": units[index]["end"],
                    "trace_sha256": sha,
                }
                rows.append(dict(identity=identity, unit=units[index],
                                 inputs=judge_inputs(question, units, index)))
            if remaining is not None:
                remaining.discard(key)
                if not remaining:
                    break
        if remaining is not None and not remaining:
            break
    if remaining:
        raise ValueError(f"include traces missing from source: {sorted(remaining)}")
    if len({digest(row["identity"]) for row in rows}) != len(rows):
        raise ValueError("source contains duplicate sentence identities")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-program", type=Path, default=JUDGE_PROGRAM)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--include-traces", type=Path, help="JSON list of the only traces to label")
    parser.add_argument("--include-source", choices=("gpt", "gemma"),
                        help="keep only the include entries attributed to this source")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    include = None
    if args.include_traces:
        include = load_include_traces(args.include_traces, args.include_source)
    rows = enumerate_items(args.trace_root, include=include)
    if not rows:
        raise ValueError(f"no sentences to label under {args.trace_root}")
    # The plan's hash is part of every checkpoint key; the part fields describe the single part that holds
    # every sentence.
    source_plan = {
        "schema_version": 1,
        "trace_root": str(args.trace_root),
        "datasets": list(DATASETS),
        "include_traces_sha256": digest(sorted(include)) if include is not None else None,
        "source_identities": len(rows),
        "selected_identities": len(rows),
        "part_size": len(rows),
        "parts": 1,
        "part_counts": [len(rows)],
        "selected_sha256": hashlib.sha256(
            "\n".join(digest(row["identity"]) for row in rows).encode()).hexdigest(),
        "first_identity": rows[0]["identity"],
        "last_identity": rows[-1]["identity"],
    }
    print(json.dumps({"source_plan": source_plan}, sort_keys=True), flush=True)
    if args.plan_only:
        return
    if include is not None and args.include_source:
        listed = json.loads(args.include_traces.read_text(encoding="utf-8"))
        expected = listed["expected_units"][args.include_source]
        if len(rows) != expected:
            raise ValueError(
                f"include-traces selection has {len(rows)} units, expected exactly {expected}")
    config = checkpoint_config(source_plan, args.judge_program.read_bytes())
    predict, adapter = load_program_and_adapter(str(args.judge_program))

    def classify(batch):
        return classify_batch_outcomes(batch, adapter=adapter, predict=predict,
                                       base_url=args.base_url)

    summary = run_part(rows, config, output_dir=args.output_dir, classify=classify)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
