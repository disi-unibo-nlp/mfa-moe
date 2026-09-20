"""Collect, verify and package the balanced 100k sentence labels of every source.

Each source was labelled as four self-contained 25,000-identity parts produced by
``sample_stratified.py`` (``dataset x correct|incorrect x Q1..Q4`` strata) and
``annotation_batch.py`` (one part per Slurm job). This tool rebuilds the deliverable:

* per source, the four ``annotations.json`` parts are checked against their
  ``summary.json`` (complete, 25,000 rows, matching ``annotations_sha256``), the
  ``part_manifest.json`` of the sampled root (identity and per-dataset trace counts) and
  the record schema (``annotation_merge.validate_records``); identities must be unique
  across parts;
* the parts are concatenated in order into ``<merged-root>/<source>/annotations.json``
  with a ``summary.json`` (rows, labels, datasets, strata, digests);
* ``<merged-root>/manifest.json``, ``SHA256SUMS.txt`` and ``README.txt`` describe the
  whole set (judge settings, code revision, Slurm launch record, per-source sampling
  summaries), and everything is zipped into ``data/labels/``.

``--verify-only`` performs every check without writing. Part roots are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.annotation_batch import LABELS, digest
from moe_exp.correlation_pipeline.annotation_merge import _publish, validate_records
from moe_exp.correlation_pipeline.annotation_partition import DATASETS, PART_SIZE, PARTS

REPO = Path(__file__).resolve().parents[3]
SCRATCH = Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe")
STRATIFIED_ROOT = SCRATCH / "qwen38-mtp3-stratified"
RESULTS = REPO / "results/correlation_pipeline"
SAMPLING = "reasoning-vllm-v1/sampling-100k-stratified"
SOURCES: dict[str, dict[str, Any]] = {
    "gpt": {
        "generation_model": "openai/gpt-oss-20b",
        "label_root": STRATIFIED_ROOT / "gpt",
        # v2 sample: see sample_stratified.reasoning_tokens; the harmony path
        # reports a constant reasoning_tokens=0, so the first gpt sample had
        # degenerate length quartiles.
        "sampling_root": RESULTS / "gpt-oss-20b" / "reasoning-vllm-v1/sampling-100k-stratified-v2",
    },
    "gemma": {
        "generation_model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "label_root": STRATIFIED_ROOT / "gemma",
        "sampling_root": RESULTS / "gemma-nvfp4-nf4" / SAMPLING,
    },
    "qwen36": {
        "generation_model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "label_root": STRATIFIED_ROOT / "qwen36",
        "sampling_root": RESULTS / "qwen36-35b-a3b-fp8" / SAMPLING,
    },
    "nemotron": {
        "generation_model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "label_root": STRATIFIED_ROOT / "nemotron",
        "sampling_root": RESULTS / "nemotron-nvfp4-dspark" / SAMPLING,
    },
    "glm": {
        "generation_model": "zai-org/GLM-4.7-Flash",
        "label_root": STRATIFIED_ROOT / "glm",
        "sampling_root": RESULTS / "glm-4.7-flash" / SAMPLING,
    },
    "qwen330b": {
        "generation_model": "Qwen/Qwen3-30B-A3B",
        "label_root": STRATIFIED_ROOT / "qwen330b",
        "sampling_root": RESULTS / "qwen3-30b-a3b" / SAMPLING,
    },
}
JUDGE = {
    "model": "Qwen/Qwen3.8-27B",
    "program": "results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json",
    "sampling": {
        "batch_size": 64, "max_tokens": 16384, "temperature": 1.0, "top_p": 0.95,
        "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "reasoning_effort": "low", "max_model_len": 49152,
    },
    "serving": "vLLM 0.29 +cu129, tensor parallel 2, MTP 3 speculative tokens, FLASH_ATTN, bfloat16 KV cache",
}
SUPERSEDES = "data/labels/mfa-moe-labels-final-2026-09-18.zip"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _identity_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    identity = record["identity"]
    return (
        identity["dataset"],
        identity["problem_id"],
        identity["sample_id"],
        identity["sentence_index"],
    )


def collect_source(
    name: str,
    label_root: Path,
    sampling_root: Path,
    *,
    parts: int = PARTS,
    part_size: int = PART_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify the labelled parts of one source and return the merged rows plus a summary."""
    sampling = _read_json(sampling_root / "sampling_manifest.json")
    plan_hash = sampling.get("manifest_sha256")
    if sampling.get("parts") != parts or sampling.get("part_size") != part_size:
        raise ValueError(f"{name}: sampling manifest is not {parts} x {part_size}")
    if sampling.get("selected_sentences") != parts * part_size:
        raise ValueError(f"{name}: sampling manifest selected {sampling.get('selected_sentences')} sentences")

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    part_reports: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    unknown = 0
    for part in range(parts):
        part_name = f"part-{part:02d}"
        part_manifest = _read_json(sampling_root / "parts" / part_name / "part_manifest.json")
        if part_manifest.get("part") != part or part_manifest.get("identities") != part_size:
            raise ValueError(f"{name}/{part_name}: part manifest does not describe {part_size} identities")
        if part_manifest.get("sampling_manifest_sha256") != plan_hash:
            raise ValueError(f"{name}/{part_name}: part manifest belongs to a different sampling plan")
        part_dir = label_root / part_name
        summary = _read_json(part_dir / "summary.json")
        if summary.get("status") != "complete" or not (
            summary.get("completed") == summary.get("expected") == part_size
        ):
            raise ValueError(f"{name}/{part_name}: summary is not a complete {part_size}-row part: {summary}")
        records = _read_json(part_dir / "annotations.json")
        stats = validate_records(records, f"{name}/{part_name}")
        if len(records) != part_size:
            raise ValueError(f"{name}/{part_name}: {len(records)} records, expected {part_size}")
        records_sha256 = digest(records)
        if summary.get("annotations_sha256") != records_sha256:
            raise ValueError(f"{name}/{part_name}: annotations.json does not match summary annotations_sha256")
        datasets: Counter[str] = Counter()
        traces: dict[str, set[tuple[str, int]]] = {}
        for record in records:
            key = _identity_key(record)
            if key in seen:
                raise ValueError(f"{name}/{part_name}: duplicate identity {key}")
            seen.add(key)
            if key[0] not in DATASETS:
                raise ValueError(f"{name}/{part_name}: unexpected dataset {key[0]!r}")
            datasets[key[0]] += 1
            traces.setdefault(key[0], set()).add((key[1], key[2]))
            if "label" in record:
                labels[record["label"]] += 1
            else:
                unknown += 1
        trace_counts = {dataset: len(keys) for dataset, keys in sorted(traces.items())}
        if trace_counts != dict(sorted(part_manifest.get("datasets", {}).items())):
            raise ValueError(
                f"{name}/{part_name}: labelled traces per dataset {trace_counts} differ from "
                f"the part manifest {part_manifest.get('datasets')}"
            )
        cells.update(part_manifest.get("cells", {}))
        merged.extend(records)
        part_reports.append(
            {
                "part": part,
                "label_dir": str(part_dir),
                "trace_root": part_manifest.get("trace_root"),
                "rows": len(records),
                "unknown": stats.get("unknown", 0),
                "annotations_sha256": records_sha256,
                "binding_sha256": summary.get("binding_sha256"),
                "datasets": dict(sorted(datasets.items())),
                "traces": trace_counts,
            }
        )
    if len(merged) != parts * part_size:
        raise ValueError(f"{name}: merged {len(merged)} rows, expected {parts * part_size}")
    if sum(cells.values()) != len(merged):
        raise ValueError(f"{name}: strata counts {sum(cells.values())} do not cover {len(merged)} rows")
    summary = {
        "schema_version": 1,
        "source": name,
        "generation_model": sampling.get("generation_model"),
        "selection": sampling.get("selection"),
        "seed": sampling.get("seed"),
        "sampling_manifest": str(sampling_root / "sampling_manifest.json"),
        "sampling_manifest_sha256": plan_hash,
        "rows": len(merged),
        "labels": {label: labels.get(label, 0) for label in sorted(LABELS)},
        "unknown": unknown,
        "datasets": dict(sorted(Counter(record["identity"]["dataset"] for record in merged).items())),
        "traces": len({_identity_key(record)[:3] for record in merged}),
        "problems": len({record["identity"].get("source_problem_id") or record["identity"]["problem_id"] for record in merged}),
        "cells": dict(sorted(cells.items())),
        "parts": part_reports,
        "annotations_sha256": digest(merged),
    }
    return merged, summary


def _git_revision() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return {"commit": head, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _readme(manifest: dict[str, Any]) -> str:
    lines = [
        f"MFA-MoE balanced sentence labelling (generated {manifest['generated_at']})",
        "",
        "Every source holds exactly 100,000 sentence labels chosen by",
        "sample_stratified.py: one completion per source problem, per-dataset",
        "reasoning-length quartiles, quotas proportional to the",
        "dataset x correct|incorrect x Q1..Q4 supply, four 25,000-identity parts,",
        "each part internally stratified (seed 42).",
        "",
        "Contents",
    ]
    for source, entry in sorted(manifest["sources"].items()):
        summary = entry["summary"]
        lines.append(
            f"  merged/{source}/annotations.json   {summary['rows']:,} rows, "
            f"{summary['traces']:,} traces, {summary['problems']:,} problems "
            f"({summary['generation_model']})"
        )
        lines.append(f"  merged/{source}/summary.json       labels, datasets, strata, part digests")
        lines.append(f"  sampling/{source}/sampling_manifest.json  the sampling plan behind the parts")
    lines += [
        "  merged/summary.json              all sources",
        "  manifest.json, SHA256SUMS.txt    judge settings, code revision, Slurm jobs, hashes",
        "",
        f"Totals: {manifest['totals']['rows']:,} rows over {len(manifest['sources'])} sources; "
        f"{manifest['totals']['unknown']} unknown outcomes.",
        "",
        f"Judge: {JUDGE['model']}, program {JUDGE['program'].rsplit('/', 2)[-2]}, batch 64, "
        "max_tokens 16,384, temperature 1.0, top_p 0.95, top_k 20, min_p 0, no penalties, "
        "reasoning_effort low, max_model_len 49,152.",
        "",
        f"All {len(manifest['sources'])} sources were labelled by "
        "sbatch/native_stratified_annotate.sbatch on the",
        "same code snapshot and judge settings (see manifest.json for the launch record).",
        f"Supersedes {SUPERSEDES} (GPT-OSS labelled only a corpus prefix, Gemma used the",
        "error-weighted sample) and the 2026-09-18 Nemotron parts under",
        "qwen38-mtp3-production/nemotron (same sampling plan, earlier launcher).",
        "",
    ]
    return "\n".join(lines)


def export(
    sources: dict[str, dict[str, Any]],
    *,
    merged_root: Path,
    zip_path: Path | None,
    launch_record: Path | None,
    verify_only: bool,
    parts: int = PARTS,
    part_size: int = PART_SIZE,
) -> dict[str, Any]:
    """Verify every source and, unless ``verify_only``, publish the merged tree and zip."""
    collected: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for name, spec in sources.items():
        collected[name] = collect_source(
            name, Path(spec["label_root"]), Path(spec["sampling_root"]), parts=parts, part_size=part_size
        )
        print(f"verified source={name} rows={collected[name][1]['rows']} unknown={collected[name][1]['unknown']}", flush=True)
    launch = _read_json(launch_record) if launch_record else None
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": (
            "MFA-MoE sentence labelling of the balanced 100k stratified samples "
            "(one completion per problem, correctness x reasoning-length quartile strata) "
            "for every benchmark source, four 25,000-identity parts per source."
        ),
        "judge": JUDGE,
        "code": _git_revision(),
        "launch_record": launch,
        "supersedes": SUPERSEDES,
        "sources": {},
        "totals": {
            "rows": sum(summary["rows"] for _records, summary in collected.values()),
            "unknown": sum(summary["unknown"] for _records, summary in collected.values()),
            "per_source": {name: summary["rows"] for name, (_records, summary) in collected.items()},
        },
    }
    if verify_only:
        manifest["verify_only"] = True
        for name, (_records, summary) in collected.items():
            manifest["sources"][name] = {"summary": summary}
        return manifest

    sums: list[tuple[str, Path]] = []
    for name, (records, summary) in collected.items():
        source_dir = merged_root / name
        _publish(source_dir / "annotations.json", records)
        _publish(source_dir / "summary.json", summary)
        sampling_target = merged_root / "sampling" / name / "sampling_manifest.json"
        _publish(sampling_target, _read_json(Path(sources[name]["sampling_root"]) / "sampling_manifest.json"))
        entry: dict[str, Any] = {"summary": summary, "files": {}}
        for relative in (
            f"merged/{name}/annotations.json",
            f"merged/{name}/summary.json",
            f"sampling/{name}/sampling_manifest.json",
        ):
            path = merged_root / relative.split("/", 1)[1] if relative.startswith("merged/") else merged_root / relative
            entry["files"][relative] = {"bytes": path.stat().st_size, "file_sha256": _sha256_file(path)}
            sums.append((relative, path))
        manifest["sources"][name] = entry
    overall = {
        "schema_version": 1,
        "rows": manifest["totals"]["rows"],
        "unknown": manifest["totals"]["unknown"],
        "sources": {name: summary for name, (_records, summary) in collected.items()},
    }
    _publish(merged_root / "summary.json", overall)
    sums.append(("merged/summary.json", merged_root / "summary.json"))
    _publish(merged_root / "manifest.json", manifest)
    sums.append(("manifest.json", merged_root / "manifest.json"))
    (merged_root / "README.txt").write_text(_readme(manifest), encoding="utf-8")
    sums.append(("README.txt", merged_root / "README.txt"))
    (merged_root / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256_file(path)}  {relative}\n" for relative, path in sums), encoding="utf-8"
    )
    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        pending = zip_path.with_name(f".{zip_path.name}.pending")
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, path in sums:
                archive.write(path, relative)
            archive.write(merged_root / "SHA256SUMS.txt", "SHA256SUMS.txt")
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"zip integrity check failed for {pending}")
        pending.replace(zip_path)
        manifest["zip"] = {"path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": _sha256_file(zip_path)}
        print(f"zip={zip_path} bytes={manifest['zip']['bytes']} sha256={manifest['zip']['sha256']}", flush=True)
    return manifest


def _parse_overrides(values: list[str], sources: dict[str, dict[str, Any]], field: str) -> None:
    for value in values:
        name, _sep, path = value.partition("=")
        if name not in sources or not path:
            raise ValueError(f"--{field.replace('_', '-')} expects SOURCE=PATH with a selected source, got {value!r}")
        sources[name][field] = Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--label-root", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--sampling-root", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--merged-root", type=Path, default=STRATIFIED_ROOT / "merged")
    parser.add_argument(
        "--zip",
        type=Path,
        default=REPO / "data/labels" / f"mfa-moe-labels-stratified-{datetime.now(timezone.utc):%Y-%m-%d}.zip",
    )
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--launch-record", type=Path)
    parser.add_argument("--parts", type=int, default=PARTS)
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    sources = {name: dict(SOURCES[name]) for name in args.sources}
    _parse_overrides(args.label_root, sources, "label_root")
    _parse_overrides(args.sampling_root, sources, "sampling_root")
    manifest = export(
        sources,
        merged_root=args.merged_root,
        zip_path=None if args.no_zip else args.zip,
        launch_record=args.launch_record,
        verify_only=args.verify_only,
        parts=args.parts,
        part_size=args.part_size,
    )
    print(json.dumps({"totals": manifest["totals"], "zip": manifest.get("zip"), "verify_only": args.verify_only}, sort_keys=True))


if __name__ == "__main__":
    main()
