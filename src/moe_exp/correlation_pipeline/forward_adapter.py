"""Offline conversion of verified batch labels into forward annotation JSONL.

No judge is loaded or called. Source parts and sampled traces are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from moe_exp.correlation_pipeline.annotation_batch import BATCH_SIZE, digest, validate_record
from moe_exp.correlation_pipeline.annotation_partition import enumerate_items
from moe_exp.correlation_pipeline.labels_export import SOURCES, collect_source, _read_json
from moe_exp.correlation_pipeline.spans import (
    SPAN_SCHEMA_VERSION, reasoning_bounds, trace_digest, validate_annotation,
)

CLASSIFIER_FIELDS = (
    "program_sha256", "judge_model", "batch_size", "max_tokens", "temperature",
    "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
    "reasoning_effort", "enable_thinking", "preserve_thinking",
)


def identity_key(model, identity):
    return (model, identity["dataset"], identity["problem_id"],
            identity["sample_id"], identity["sentence_index"])


def _source_rows(root, datasets, model, plan_hash):
    items = enumerate_items(root, tuple(datasets))
    for identity, trace, unit, inputs in items:
        selection = trace.metadata.get("sentence_selection") or {}
        if trace.model_id != model or selection.get("manifest_sha256") != plan_hash:
            raise ValueError("Source model or sampling digest mismatch")
        yield identity, trace, unit, inputs


def adapt_source(name, label_root, sampling_root, output_root=None, *,
                 parts=4, part_size=25000):
    """Verify everything before publishing; output_root=None is a read-only audit."""
    plan = _read_json(sampling_root / "sampling_manifest.json")
    plan_hash = plan.get("manifest_sha256")
    if not plan_hash or digest({k: v for k, v in plan.items() if k != "manifest_sha256"}) != plan_hash:
        raise ValueError("Sampling manifest digest mismatch")
    model = plan["generation_model"]
    if model != SOURCES[name]["generation_model"]:
        raise ValueError("Sampling manifest belongs to a different source model")
    if plan.get("selection") != "stratified-accuracy-reasoning-length":
        raise ValueError("Expected the actual stratified sampling manifest")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", model).strip("-")
    if plan.get("generation_slug") != slug:
        raise ValueError("Sampling generation slug mismatch")
    datasets = plan["datasets"]
    completions = plan.get("completions")
    if not isinstance(completions, list) or not completions:
        raise ValueError("Sampling manifest lacks selected completion membership")
    all_members = {(model, row["dataset"], row["problem_id"], row["sample_id"]): row
                   for row in completions}
    if len(all_members) != len(completions):
        raise ValueError("Duplicate completion in sampling manifest")
    # The plan also records candidates allocated zero sentences.
    members = {key: row for key, row in all_members.items() if row["selected_sentences"] > 0}
    records, report = collect_source(
        name, label_root, sampling_root, parts=parts, part_size=part_size,
    )
    by_identity = {identity_key(model, row["identity"]): row for row in records}
    classifier = None
    bindings = []
    for part in range(parts):
        part_name = f"part-{part:02d}"
        root = sampling_root / "parts" / part_name / "generation" / slug
        items = list(_source_rows(root, datasets, model, plan_hash))
        rows = [dict(identity=i, unit=u, inputs=inputs) for i, _t, u, inputs in items]
        part_records = records[part * part_size:(part + 1) * part_size]
        if len(rows) != part_size:
            raise ValueError("Sampled part has missing or extra identities")
        for row, record in zip(rows, part_records, strict=True):
            validate_record(row, record)
        checkpoints = sorted((label_root / part_name / "checkpoints").glob("*.json"))
        if not checkpoints:
            raise ValueError("Classifier checkpoint binding is missing")
        saved = []
        binding = None
        for index, path in enumerate(checkpoints):
            value = _read_json(path)
            current = value.get("binding")
            if path.name != f"batch-{index:06d}.json" or not isinstance(current, dict):
                raise ValueError("Invalid classifier checkpoint sequence")
            if binding is not None and current != binding:
                raise ValueError("Inconsistent classifier checkpoint bindings")
            binding = current
            batch = value.get("records")
            if not isinstance(batch, list) or not 1 <= len(batch) <= BATCH_SIZE:
                raise ValueError("Invalid classifier checkpoint batch")
            saved.extend(batch)
        summary = _read_json(label_root / part_name / "summary.json")
        if (saved != part_records or binding.get("items_sha256") != digest(rows)
                or binding.get("expected_count") != part_size
                or binding.get("schema_version") != 1
                or summary.get("binding_sha256") != digest(binding)):
            raise ValueError("Stale classifier checkpoint or source identity binding")
        config = binding.get("config", {})
        if any(field not in config for field in CLASSIFIER_FIELDS):
            raise ValueError("Incomplete classifier settings")
        current_classifier = {field: config[field] for field in CLASSIFIER_FIELDS}
        if classifier is not None and current_classifier != classifier:
            raise ValueError("Different classifier settings across parts")
        classifier = current_classifier
        bindings.append(digest(binding))

    annotations = {}
    seen = set()
    for identity, trace, unit, inputs in _source_rows(
        sampling_root / "generation" / slug, datasets, model, plan_hash,
    ):
        key = identity_key(model, identity)
        if key in seen or key not in by_identity:
            raise ValueError("Duplicate or missing composite sentence identity")
        seen.add(key)
        record = by_identity[key]
        validate_record(dict(identity=identity, unit=unit, inputs=inputs), record)
        trace_key = key[:4]
        if trace_key not in annotations:
            member = members.get(trace_key)
            selection = trace.metadata["sentence_selection"]
            if (member is None or member.get("trace_sha256") != trace_digest(trace)
                    or member.get("source_problem_id") != trace.source_problem_id
                    or member.get("selected_sentences") != len(selection["indices"])
                    or selection.get("schema_version") != plan["schema_version"]
                    or selection.get("selection") != plan["selection"]):
                raise ValueError("Trace does not match actual sampling manifest membership")
            expected_stratum = {"dataset": trace.dataset, "correct": member["is_correct"],
                                "quartile": member["quartile"], "reasoning_tokens": member["reasoning_tokens"]}
            if selection.get("stratum") != expected_stratum:
                raise ValueError("Trace stratum differs from the sampling manifest")
            annotations[trace_key] = (trace, {
                "schema_version": SPAN_SCHEMA_VERSION, "status": "complete",
                "source_model": model, "dataset": trace.dataset,
                "problem_id": trace.problem_id, "sample_id": trace.sample_id,
                "trace_sha256": trace_digest(trace), "classifier": classifier,
                "sentence_selection": trace.metadata["sentence_selection"],
                "reasoning_span": list(reasoning_bounds(trace)), "units": [],
            })
        annotation = annotations[trace_key][1]
        outcome = {k: v for k, v in record.items() if k not in {"identity", "unit", "inputs"}}
        annotation["units"].append({**unit, **outcome})
    if set(annotations) != set(members):
        raise ValueError("Combined sample omits manifest completions")
    if seen != set(by_identity):
        raise ValueError("Labels contain identities missing from the combined sample")
    output = defaultdict(list)
    for trace, annotation in annotations.values():
        validate_annotation(trace, annotation)
        units = annotation["units"]
        annotation["coverage"] = {
            "selected": len(units), "labeled": sum("label" in u for u in units),
            "unknown": sum(u.get("status") == "unknown" for u in units),
        }
        output[trace.dataset].append(annotation)
    result = {
        **report, "status": "complete", "classifier": classifier,
        "bindings": bindings, "adapter_schema_version": 1,
        "sampling_manifest_sha256": plan_hash,
        "behavioral_sampling": "selected attempts only; not complete avg@32",
    }
    if output_root is not None:
        files = {
            output_root / slug / dataset / "annotations.jsonl":
                "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
            for dataset, rows in output.items()
        }
        files[output_root / slug / "adapter_summary.json"] = json.dumps(result, indent=2) + "\n"
        # Preflight all existing outputs so a stale bundle is never partially replaced.
        for path, payload in files.items():
            if path.exists() and path.read_text() != payload:
                raise ValueError(f"Existing adapter output differs: {path}")
        for path, payload in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(payload)
            os.replace(temporary, path)
            if path.read_text() != payload:
                raise OSError(f"Adapter read-back failed: {path}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=tuple(SOURCES))
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--sampling-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_only and args.output_dir is None:
        parser.error("--output-dir is required unless --verify-only")
    spec = SOURCES[args.source]
    report = adapt_source(
        args.source, args.label_root or spec["label_root"],
        args.sampling_root or spec["sampling_root"],
        None if args.verify_only else args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
