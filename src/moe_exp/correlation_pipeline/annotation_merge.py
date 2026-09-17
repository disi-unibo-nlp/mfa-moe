"""Merge the frozen v1 labels with the targeted v2 re-label of the splitter-affected traces.

The fixed (version 2) splitter segments every trace exactly like the frozen version 1
splitter except for the traces listed in `data/annotation/affected_traces_v2.json`, so the
completed v1 parts stay authoritative everywhere else. This tool rebuilds the full v2
sequence per source: version 1 rows are kept unchanged, the affected traces are replaced by
the version 2 sub-units contained in their v1 selection (the same containment rule as
`spans.map_selection_to_v2`), and a coverage report names every dropped v1 record.

`--verify-splitter` re-checks the blast radius over the covered corpus; `--verify-units`
re-derives both unit sets from the source traces while merging. `--dry-run` writes nothing.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.annotation_batch import LABELS, UNKNOWN_FIELDS, _publish, digest
from moe_exp.correlation_pipeline.annotation_partition import (
    DATASETS,
    PART_SIZE,
    PARTS,
    TOTAL,
    trace_key,
)
from moe_exp.correlation_pipeline.spans import (
    map_selection_to_v2,
    selected_sentence_indices,
    sentence_spans,
    sentence_spans_v1,
)
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

SOURCES = ("gpt", "gemma")
DEFAULT_V1_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production"
)
DEFAULT_V2_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production-v2-affected"
)
DEFAULT_AFFECTED = (
    Path(__file__).resolve().parents[3] / "data/annotation/affected_traces_v2.json"
)
DEFAULT_TRACE_ROOTS = {
    "gpt": Path(
        "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/"
        "gpt-oss-20b/generation/openai--gpt-oss-20b"
    ),
    "gemma": Path(
        "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/"
        "gemma-nvfp4-nf4/reasoning-vllm-v1/sampling/generation/nvidia--Gemma-4-26B-A4B-NVFP4"
    ),
}
SUPERSEDED_TEXT_LIMIT = 200


def load_affected(path: Path) -> dict[str, Any]:
    """Read the committed affected-trace list with its verified expectations."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported affected-traces schema in {path}")
    entries = document.get("trace_keys")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no trace keys")
    keys = set()
    by_source = {source: set() for source in SOURCES}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid affected-trace entry in {path}: {entry!r}")
        key = (entry["dataset"], entry["problem_id"], entry["sample_id"])
        if entry.get("source") not in SOURCES:
            raise ValueError(f"{path}: entry {key} has no valid source: {entry.get('source')!r}")
        keys.add(key)
        by_source[entry["source"]].add(key)
    if len(keys) != len(entries):
        raise ValueError(f"{path} repeats an affected trace")
    return {
        "keys": keys,
        "by_source": by_source,
        "expected_units": document.get("expected_units") or {},
        "verified_units": document.get("verified_units") or {},
        "document": document,
    }


def trace_key_of(record: dict[str, Any]) -> tuple[str, str, int]:
    identity = record["identity"]
    return (identity["dataset"], identity["problem_id"], identity["sample_id"])


def unit_span(record: dict[str, Any]) -> tuple[int, int, str]:
    unit = record["unit"]
    return (unit["start"], unit["end"], unit["text"])


def validate_records(records: Any, where: str) -> dict[str, Any]:
    """Check one annotations.json list and return its label statistics."""
    if not isinstance(records, list) or not records:
        raise ValueError(f"{where}: expected a non-empty record list")
    stats = {"records": len(records), "labels": 0, "unknown": 0}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{where}: record is not an object")
        identity, unit = record.get("identity"), record.get("unit")
        if not isinstance(identity, dict) or not isinstance(unit, dict):
            raise ValueError(f"{where}: record lacks identity/unit")
        for key in ("dataset", "problem_id", "sample_id", "sentence_index", "start", "end"):
            if key not in identity:
                raise ValueError(f"{where}: identity lacks {key}")
        for key in ("index", "start", "end", "text"):
            if key not in unit:
                raise ValueError(f"{where}: unit lacks {key}")
        if (
            identity["sentence_index"] != unit["index"]
            or identity["start"] != unit["start"]
            or identity["end"] != unit["end"]
        ):
            raise ValueError(f"{where}: identity and unit disagree")
        if not isinstance(unit["text"], str) or not unit["text"]:
            raise ValueError(f"{where}: unit text must be a non-empty string")
        if not isinstance(record.get("inputs"), dict):
            raise ValueError(f"{where}: record lacks judge inputs")
        added = set(record) - {"identity", "unit", "inputs"}
        if added == {"label"}:
            if record["label"] not in LABELS:
                raise ValueError(f"{where}: non-canonical label {record['label']!r}")
            stats["labels"] += 1
        elif added == UNKNOWN_FIELDS and record.get("status") == "unknown":
            stats["unknown"] += 1
        else:
            raise ValueError(f"{where}: unexpected record shape {sorted(added)}")
    return stats


def group_by_trace(records: list[dict[str, Any]], where: str) -> OrderedDict:
    """Group records per trace, rejecting interleaved traces."""
    groups: OrderedDict[tuple[str, str, int], list[dict[str, Any]]] = OrderedDict()
    previous: tuple[str, str, int] | None = None
    for record in records:
        key = trace_key_of(record)
        if key != previous:
            if key in groups:
                raise ValueError(f"{where}: records for {key} are not contiguous")
            previous = key
        groups.setdefault(key, []).append(record)
    return groups


def check_sub_units(
    key: tuple[str, str, int],
    legacy: list[tuple[int, int, str]],
    fresh: list[tuple[int, int, str]],
) -> None:
    """The fresh units must tile the legacy spans exactly, whitespace aside."""
    inside = {}
    for start, end, text in fresh:
        for left, right, legacy_text in legacy:
            if left <= start and end <= right:
                inside.setdefault((left, right, legacy_text), []).append((start, end, text))
                break
    for left, right, legacy_text in legacy:
        cursor = left
        for start, end, text in sorted(inside.get((left, right, legacy_text), [])):
            if start < cursor:
                if text.strip():
                    raise ValueError(f"{key}: overlapping v2 sub-units")
                continue
            if legacy_text[cursor - left : start - left].strip():
                raise ValueError(f"{key}: v2 sub-units leave a gap inside the v1 span")
            cursor = end
        if cursor != right:
            raise ValueError(f"{key}: v2 sub-units do not reach the end of the v1 span")


def fetch_trace(trace_root: Path, key: tuple[str, str, int]) -> TraceRecord:
    dataset, problem_id, sample_id = key
    for row in iter_jsonl(trace_root / dataset / "traces.jsonl"):
        trace = TraceRecord(**row)
        if (trace.dataset, trace.problem_id, trace.sample_id) == key:
            return trace
    raise ValueError(f"trace {key} not found under {trace_root}")


def selection_units(trace: TraceRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    legacy = sentence_spans_v1(trace)
    current = sentence_spans(trace)
    indices = selected_sentence_indices(trace, legacy)
    return legacy, current, indices


def merge_source(
    source: str,
    v1_records: list[dict[str, Any]],
    v2_records: list[dict[str, Any]],
    affected_keys: set[tuple[str, str, int]],
    *,
    expected_rows: int | None,
    trace_root: Path | None = None,
    verify_units: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = group_by_trace(v1_records, f"{source} v1")
    v2_groups = group_by_trace(v2_records, f"{source} v2")
    outside = sorted(set(v2_groups) - affected_keys)
    if outside:
        raise ValueError(f"{source}: v2 labels exist for traces outside the affected list: {outside}")

    rows: list[dict[str, Any]] = []
    reused = relabeled = superseded = 0
    superseded_records: list[dict[str, Any]] = []
    affected_report: list[dict[str, Any]] = []
    unchanged_v2_units = 0
    for key, group in groups.items():
        legacy = [unit_span(record) for record in group]
        if key in affected_keys:
            fresh_records = v2_groups.get(key)
            if not fresh_records:
                raise ValueError(f"{source}: affected trace {key} has no v2 labels")
            fresh = [unit_span(record) for record in fresh_records]
            mapped = [
                span
                for left, right, _ in legacy
                for span in fresh
                if left <= span[0] and span[1] <= right
            ]
            if len(set(mapped)) != len(mapped):
                raise ValueError(f"{source}: v2 units of {key} are contained in more than one v1 span")
            if not set(mapped) <= set(fresh):
                raise ValueError(f"{source}: v2 labels of {key} do not cover every mapped sub-unit")
            check_sub_units(key, legacy, mapped)
            fresh_set = set(fresh)
            dropped = [record for record in group if unit_span(record) not in fresh_set]
            superseded += len(dropped)
            superseded_records.extend(
                {
                    "dataset": key[0],
                    "problem_id": key[1],
                    "sample_id": key[2],
                    "sentence_index": record["identity"]["sentence_index"],
                    "start": record["unit"]["start"],
                    "end": record["unit"]["end"],
                    "text_chars": len(record["unit"]["text"]),
                    "text_prefix": record["unit"]["text"][:SUPERSEDED_TEXT_LIMIT],
                }
                for record in dropped
            )
            mapped_set = set(mapped)
            chosen = [record for record in fresh_records if unit_span(record) in mapped_set]
            relabeled += len(chosen)
            rows.extend(chosen)
            affected_report.append(
                {
                    "dataset": key[0],
                    "problem_id": key[1],
                    "sample_id": key[2],
                    "v1_rows": len(group),
                    "v2_rows": len(chosen),
                    "v2_rows_available": len(fresh_records),
                    "superseded": len(dropped),
                }
            )
            continue
        rows.extend(group)
        reused += len(group)
        if verify_units:
            if trace_root is None:
                raise ValueError("--verify-units needs a trace root")
            trace = fetch_trace(trace_root, key)
            legacy_units, current_units, indices = selection_units(trace)
            expected = [legacy_units[index] for index in indices[: len(group)]]
            derived = [(unit["start"], unit["end"], unit["text"]) for unit in expected]
            if derived != legacy:
                raise ValueError(f"{source}: v1 labels for {key} do not match the frozen splitter")
            mapped_indices = map_selection_to_v2(trace, indices[: len(group)])
            mapped_spans = [(current_units[index]["start"], current_units[index]["end"], current_units[index]["text"]) for index in mapped_indices]
            if mapped_spans != legacy:
                raise ValueError(f"{source}: {key} is not listed as affected but version 2 changes it")
            if any(
                (unit["start"], unit["end"], unit["text"]) != (legacy_units[index]["start"], legacy_units[index]["end"], legacy_units[index]["text"])
                for index, unit in zip(mapped_indices, current_units, strict=False)
            ):
                raise ValueError(f"{source}: {key} unit identity drifted")

    for key, group in group_by_trace(rows, f"{source} merged").items():
        previous = None
        for record in group:
            unit = record["unit"]
            if previous is not None and unit["start"] < previous:
                raise ValueError(f"{source}: {key} units are not in text order")
            previous = unit["end"]
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{source}: merged {len(rows)} rows, expected {expected_rows}")
    digests = [digest(record["identity"]) for record in rows]
    if len(set(digests)) != len(digests):
        raise ValueError(f"{source}: merged rows contain duplicate identities")
    stats = validate_records(rows, f"{source} merged")
    report = {
        "source": source,
        "rows": len(rows),
        "reused_from_v1": reused,
        "relabeled_from_v2": relabeled,
        "superseded_v1_records": superseded,
        "superseded_details": superseded_records,
        "traces": len(groups),
        "affected_traces": affected_report,
        "labels": stats["labels"],
        "unknown": stats["unknown"],
        "annotations_sha256": digest(rows),
        "expected_rows": expected_rows,
    }
    return rows, report


def verify_splitter(
    trace_roots: dict[str, Path], affected: dict[str, Any], *, total: int = TOTAL
) -> dict[str, Any]:
    """Corpus dry-run: only the affected traces may change, with the verified counts."""
    report = {}
    observed: dict[tuple[str, str, int], str] = {}
    for source in SOURCES:
        trace_root = trace_roots[source]
        old_total = new_total = 0
        changed: list[dict[str, Any]] = []
        for dataset in DATASETS:
            if old_total >= total:
                break
            path = trace_root / dataset / "traces.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            for row in iter_jsonl(path):
                if old_total >= total:
                    break
                trace = TraceRecord(**row)
                legacy, current, indices = selection_units(trace)
                taken = indices[: total - old_total]
                mapped = map_selection_to_v2(trace, taken)
                old_keys = [
                    (legacy[index]["start"], legacy[index]["end"], legacy[index]["text"]) for index in taken
                ]
                new_keys = [
                    (current[index]["start"], current[index]["end"], current[index]["text"]) for index in mapped
                ]
                if new_keys != old_keys:
                    changed.append(
                        {
                            "dataset": trace.dataset,
                            "problem_id": trace.problem_id,
                            "sample_id": trace.sample_id,
                            "v1_units": len(legacy),
                            "v2_units": len(current),
                            "v1_selected": len(taken),
                            "v2_mapped": len(mapped),
                        }
                    )
                old_total += len(taken)
                new_total += len(mapped)
        keys = {(item["dataset"], item["problem_id"], item["sample_id"]) for item in changed}
        expected = (affected["verified_units"].get(source) or {}).get("v2")
        unexpected = sorted(keys - affected["keys"])
        repeated = sorted(key for key in keys if key in observed)
        status = {
            "old_items": old_total,
            "new_items": new_total,
            "delta": new_total - old_total,
            "changed_traces": changed,
            "expected_new_items": expected,
            "unexpected_traces": unexpected,
            "changed_in_another_source": repeated,
        }
        if unexpected or repeated:
            raise ValueError(f"{source}: splitter blast radius does not match the affected list: {status}")
        misattributed = sorted(keys - affected["by_source"][source])
        if misattributed:
            raise ValueError(
                f"{source}: affected list attributes {misattributed} to another source"
            )
        if expected is not None and new_total != expected:
            raise ValueError(f"{source}: splitter produced {new_total} units, expected {expected}")
        for key in keys:
            observed[key] = source
        report[source] = status
    missing = sorted(affected["keys"] - set(observed))
    if missing:
        raise ValueError(
            f"affected traces that the splitter never changes: {missing}"
        )
    for source, status in report.items():
        status["affected_traces"] = sorted(
            key for key, owner in observed.items() if owner == source
        )
    return report


def load_parts(
    root: Path, source: str, *, parts: int = PARTS, part_size: int = PART_SIZE
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Concatenate the per-part annotations of one source, reporting missing parts."""
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    part_stats: list[dict[str, Any]] = []
    for part in range(parts):
        path = root / source / f"part-{part:02d}" / "annotations.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        part_records = json.loads(path.read_text(encoding="utf-8"))
        stats = validate_records(part_records, str(path))
        if len(part_records) != part_size:
            raise ValueError(f"{path}: {len(part_records)} records, expected {part_size}")
        records.extend(part_records)
        part_stats.append({"part": part, "path": str(path), **stats})
    return records, missing, part_stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_V2_ROOT / "merged")
    parser.add_argument("--affected-traces", type=Path, default=DEFAULT_AFFECTED)
    parser.add_argument("--trace-root", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-units", action="store_true")
    parser.add_argument("--verify-splitter", action="store_true")
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--parts", type=int, default=PARTS)
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    args = parser.parse_args(argv)
    if args.parts * args.part_size != args.total:
        raise SystemExit("--parts, --part-size and --total must describe exact disjoint parts")

    trace_roots = dict(DEFAULT_TRACE_ROOTS)
    for value in args.trace_root:
        source, _, path = value.partition("=")
        if source not in SOURCES or not path:
            raise ValueError(f"--trace-root must look like gpt=/path, got {value!r}")
        trace_roots[source] = Path(path)
    affected = load_affected(args.affected_traces)

    if args.verify_splitter:
        report = verify_splitter(trace_roots, affected, total=args.total)
        print(json.dumps({"splitter_verification": report}, sort_keys=True), flush=True)
        return

    summary: dict[str, Any] = {
        "status": "complete",
        "schema_version": 1,
        "v1_root": str(args.v1_root),
        "v2_root": str(args.v2_root),
        "affected_traces": str(args.affected_traces),
        "sources": {},
    }
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCES:
        v1_records, missing, part_stats = load_parts(
            args.v1_root, source, parts=args.parts, part_size=args.part_size
        )
        if missing:
            raise SystemExit(f"{source}: incomplete v1 parts, missing {missing}")
        if len(v1_records) != args.total:
            raise SystemExit(f"{source}: {len(v1_records)} v1 rows, expected {args.total}")
        v2_path = args.v2_root / source / "part-00" / "annotations.json"
        if not v2_path.is_file():
            raise SystemExit(f"{source}: missing v2 labels at {v2_path}")
        v2_records = json.loads(v2_path.read_text(encoding="utf-8"))
        v2_stats = validate_records(v2_records, str(v2_path))
        expected_units = affected["expected_units"].get(source)
        if expected_units is not None and v2_stats["records"] != expected_units:
            raise SystemExit(
                f"{source}: v2 run has {v2_stats['records']} units, expected {expected_units}"
            )
        expected_rows = (affected["verified_units"].get(source) or {}).get("v2")
        rows, report = merge_source(
            source,
            v1_records,
            v2_records,
            affected["by_source"][source],
            expected_rows=expected_rows,
            trace_root=trace_roots[source],
            verify_units=args.verify_units,
        )
        rows_by_source[source] = rows
        report["v1_parts"] = part_stats
        report["v2_source"] = {"path": str(v2_path), **v2_stats}
        summary["sources"][source] = report
    summary["rows"] = sum(item["rows"] for item in summary["sources"].values())
    summary["reused_from_v1"] = sum(item["reused_from_v1"] for item in summary["sources"].values())
    summary["relabeled_from_v2"] = sum(
        item["relabeled_from_v2"] for item in summary["sources"].values()
    )
    summary["superseded_v1_records"] = sum(
        item["superseded_v1_records"] for item in summary["sources"].values()
    )
    if not args.dry_run:
        for source in SOURCES:
            _publish(args.merged_root / source / "annotations.json", rows_by_source[source])
            _publish(args.merged_root / source / "summary.json", summary["sources"][source])
        _publish(args.merged_root / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "sources"}, sort_keys=True), flush=True)
    for source, report in summary["sources"].items():
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("v1_parts",)
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
