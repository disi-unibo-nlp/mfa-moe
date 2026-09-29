"""Freeze a four-trace replay panel, then audit forwards inside Slurm only."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from pathlib import Path

from moe_exp.correlation_pipeline.provenance import file_sha256
from moe_exp.correlation_pipeline.spans import digest
from moe_exp.jsonl import iter_jsonl

SOURCES = {
    "qwen330b": ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "qwen3-30b-a3b"),
    "glm": ("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67", "glm-4.7-flash"),
    "qwen36": ("Qwen/Qwen3.6-35B-A3B-FP8", "2ab40a9acc6d567889ca4d4e59feb2da56121454", "qwen36-35b-a3b-fp8"),
}
REPO = Path(__file__).resolve().parents[3]


def slug(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model).strip("-")


def publish(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
    if path.read_text() != payload:
        raise OSError(f"Read-back failed: {path}")


def freeze_panel(source, source_root, output_root, *, annotation_root=None):
    model, revision, _leaf = SOURCES[source]
    forward_model = "unsloth/Qwen3.6-35B-A3B" if source == "qwen36" else model
    annotations = {}
    if annotation_root is not None:
        for annotation in iter_jsonl(annotation_root / slug(model) / "math500/annotations.jsonl"):
            key = (annotation["problem_id"], annotation.get("sample_id", 0))
            if key in annotations:
                raise ValueError("Duplicate panel annotation")
            annotations[key] = annotation
    if source == "qwen36" and not annotations:
        raise ValueError("Qwen3.6 debug requires existing verified labels")
    source_file = source_root / slug(model) / "math500/traces.jsonl"
    manifest_file = source_file.with_name("manifest.json")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("status") != "complete" or manifest.get("target_model_id") != forward_model:
        raise ValueError("Panel requires a completed matching per-dataset manifest")
    selected, seen = [], set()
    for row in itertools.islice(iter_jsonl(source_file), 64):
        metadata = row.get("metadata", {})
        replay = metadata.get("token_replay", {})
        tokens = replay.get("completion_token_ids", [])
        budget = (metadata.get("generation_config") or {}).get("max_tokens")
        usage = (metadata.get("usage") or {}).get("completion_tokens")
        if (not tokens or len(tokens) > 4096 or metadata.get("finish_reason") == "length"
                or (budget is not None and usage is not None and usage >= budget)):
            continue
        if row["model_id"] != model or row["dataset"] != "math500":
            raise ValueError("Source panel identity mismatch")
        identity = (model, row["dataset"], row["problem_id"], row.get("sample_id", 0))
        if identity in seen:
            raise ValueError("Duplicate panel identity")
        seen.add(identity)
        if replay.get("schema_version") != 1 or len(replay.get("completion_offsets", [])) != len(tokens):
            raise ValueError("Panel requires saved token IDs and exact offsets")
        if source == "qwen36":
            annotation = annotations.get((row["problem_id"], row.get("sample_id", 0)))
            if annotation is None or not any("label" in u for u in annotation["units"]):
                continue
            from moe_exp.correlation_pipeline.spans import validate_annotation
            from moe_exp.schemas import TraceRecord
            validate_annotation(TraceRecord(**row), annotation)
            outcome = row.get("is_correct")
            if outcome not in (True, False) or sum(r.get("is_correct") is outcome for r in selected) >= 2:
                continue
        if len(selected) < 4:
            selected.append(row)
    if len(selected) != 4:
        raise ValueError("Fewer than four eligible traces in the first 64 records")
    report = {
        "schema_version": 1, "source": source, "model": model, "revision": revision,
        "forward_model": forward_model,
        "selection": ("first two correct and two incorrect eligible in first 64; <=4096 completion tokens"
                      if source == "qwen36" else "first four eligible in first 64; complete; <=4096 completion tokens"),
        "source_manifest_sha256": file_sha256(manifest_file),
        "records_sha256": digest(selected),
        "identities": [[model, r["dataset"], r["problem_id"], r.get("sample_id", 0)] for r in selected],
        "completion_tokens": [len(r["metadata"]["token_replay"]["completion_token_ids"]) for r in selected],
        "outcome_coverage": {str(value): sum(r.get("is_correct") is value for r in selected)
                             for value in (True, False, None)},
        "label_coverage": ("existing verified labels" if annotations else
                           "unlabeled panel; real class-conditioned replay not validated"),
    }
    target = output_root / "generation" / slug(model) / "math500"
    files = {
        target / "traces.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected),
        target / "manifest.json": json.dumps({"status": "complete", "dataset": "math500",
                                             "model": model, "target_model_id": forward_model,
                                             "traces": 4, "problems": 4, "samples_per_problem": 1,
                                             "scored_traces": sum(r.get("is_correct") is not None for r in selected),
                                             "correct_traces": sum(r.get("is_correct") is True for r in selected),
                                             "trace_path": str(target / "traces.jsonl"),
                                             "debug_panel_sha256": digest(selected)}, indent=2) + "\n",
        output_root / "panel.json": json.dumps(report, indent=2) + "\n",
    }
    if annotations:
        panel_annotations = [annotations[(r["problem_id"], r.get("sample_id", 0))] for r in selected]
        report["annotations_sha256"] = digest(panel_annotations)
        files[output_root / "annotations" / slug(model) / "math500/annotations.jsonl"] = "".join(
            json.dumps(a, ensure_ascii=False) + "\n" for a in panel_annotations
        )
        files[output_root / "panel.json"] = json.dumps(report, indent=2) + "\n"
    for path, payload in files.items():
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"Frozen panel differs: {path}")
    for path, payload in files.items():
        publish(path, payload)
    return report


def prepare_qwen36(root):
    """Verify existing production labels and freeze a tiny labeled replay panel."""
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    from moe_exp.correlation_pipeline.labels_export import SOURCES as LABEL_SOURCES
    spec = LABEL_SOURCES["qwen36"]
    summary = adapt_source(
        "qwen36", spec["label_root"], spec["sampling_root"], root / "annotations",
    )
    panel = freeze_panel(
        "qwen36", spec["sampling_root"] / "generation", root / "panels/qwen36",
        annotation_root=root / "annotations",
    )
    return {"adapter": summary, "panel": panel}


def run_panel(source, panel_root, output_root, cpu_gate, *, generation_model=None,
              forward_model=None, revision=None, quantization=None, annotation_dir=None):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Forward debug requires an allocated srun step")
    from moe_exp.correlation_pipeline.provenance import code_provenance
    gate = json.loads(cpu_gate.read_text())
    if gate.get("status") != "passed" or gate.get("code") != code_provenance():
        raise ValueError("CPU gates must pass for this exact code before GPU validation")
    import resource
    import torch
    from moe_exp.correlation_pipeline import analyze, extract
    from moe_exp.correlation_pipeline.forward_audit import audit_bundle
    from moe_exp.models import routing_extraction
    from moe_exp.models.loader import load_model_and_tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Forward debug requires exactly two allocated CUDA devices")
    source_model, source_revision, _leaf = SOURCES[source]
    generation_model = generation_model or source_model
    model_id = forward_model or ("unsloth/Qwen3.6-35B-A3B" if source == "qwen36" else source_model)
    revision = revision or source_revision
    quantization = quantization or ("bnb-4bit" if source == "qwen36" else "none")
    if source == "qwen36":
        annotation_dir = annotation_dir or panel_root / "annotations"
        if quantization not in {"bnb-4bit", "bnb-8bit"}:
            raise ValueError("Qwen3.6 debug requires quantized expert loading")
    modes = ["full", "class", "position"] if annotation_dir else ["full", "position"]
    panel = json.loads((panel_root / "panel.json").read_text())
    records = list(iter_jsonl(panel_root / "generation" / slug(generation_model) / "math500/traces.jsonl"))
    if (panel["records_sha256"] != digest(records) or len(records) != 4
            or panel["model"] != generation_model or panel["revision"] != revision
            or panel.get("forward_model", model_id) != model_id):
        raise ValueError("Frozen panel integrity mismatch")
    if annotation_dir:
        annotations = list(iter_jsonl(annotation_dir / slug(generation_model) / "math500/annotations.jsonl"))
        if digest(annotations) != panel.get("annotations_sha256"):
            raise ValueError("Frozen panel annotations changed")
    started = time.monotonic()
    print(f"MODEL_LOAD_START model={model_id} quantization={quantization}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_id, quantization=quantization, revision=revision, local_files_only=True,
        offload_folder=str(output_root / "offload"),
    )
    print("MODEL_LOAD_COMPLETE", flush=True)
    args = extract.build_parser().parse_args([
        "--model-id", model_id, "--revision", revision, "--local-files-only",
        "--generation-model", generation_model, "--generation-dir", str(panel_root / "generation"),
        "--output-dir", str(output_root / "forward"), "--datasets", "math500",
        "--quantization", quantization, "--all-router-layers", "--views", *modes,
        "--save-raw-tensors",
        *(["--annotation-dir", str(annotation_dir)] if annotation_dir else []),
    ])
    first = extract.extract_all(args, model_and_tokenizer=(model, tokenizer))
    first_seconds = time.monotonic() - started
    forward_path = output_root / "forward" / slug(model_id) / "math500/traces_with_routing.jsonl"
    forward_records = list(iter_jsonl(forward_path))
    provenance = forward_records[0]["metadata"]["forward_provenance"]
    if any(provenance.get(k) != revision for k in ("resolved_model_revision", "resolved_tokenizer_revision")):
        raise ValueError("Resolved checkpoint/tokenizer revisions do not match the pin")
    for source_row, forward_row in zip(records, forward_records, strict=True):
        if source_row["metadata"]["token_replay"] != forward_row["metadata"]["token_replay"]:
            raise ValueError("Replay tokens changed during extraction")
    original = routing_extraction.extract_logs_single_pass
    def forbidden_forward(**kwargs):
        raise RuntimeError("Unchanged resume attempted another forward")
    routing_extraction.extract_logs_single_pass = forbidden_forward
    try:
        resumed = extract.extract_all(args, model_and_tokenizer=(model, tokenizer))
    finally:
        routing_extraction.extract_logs_single_pass = original
    if resumed[0]["execution"]["forwards"] != 0 or resumed[0]["execution"]["reused"] != 4:
        raise ValueError("Resume did not reuse all four traces")
    audit = audit_bundle(forward_path, require_raw=True, require_replay=True)
    publish(output_root / "audit.json", json.dumps(audit, indent=2, allow_nan=False) + "\n")
    if annotation_dir and not all(row["expert_by_class_ready"] for row in audit["traces"]):
        raise RuntimeError("Class-conditioned inputs are incomplete; see audit.json")
    if audit["status"] != "complete":
        raise RuntimeError("Saved-artifact audit failed; see audit.json")
    analysis = analyze.analyze(analyze.build_parser().parse_args([
        "--model-id", model_id, "--forward-dir", str(output_root / "forward"),
        "--output-dir", str(output_root / "analysis"), "--datasets", "math500",
        "--views", *modes, "--bootstrap-samples", "20",
        "--expert-min-trace-support", "1",
    ]))
    analysis_root = output_root / "analysis" / slug(model_id)
    for name in ("trace_features.csv", "problem_features.csv", "expert_trace_features.csv", "correlations.json"):
        if not (analysis_root / name).is_file() or not (analysis_root / name).stat().st_size:
            raise RuntimeError(f"Missing analysis artifact: {name}")
    import pandas as pd
    trace_table = pd.read_csv(analysis_root / "trace_features.csv")
    problem_table = pd.read_csv(analysis_root / "problem_features.csv")
    expert_table = pd.read_csv(analysis_root / "expert_trace_features.csv")
    if len(trace_table) != 4 or len(problem_table) != 4 or expert_table.empty or analysis["n_traces"] != 4:
        raise RuntimeError("Analysis row counts do not match the frozen panel")
    json.loads((analysis_root / "correlations.json").read_text())
    from moe_exp.correlation_pipeline.spans import SPAN_SCHEMA_VERSION
    view_root = analysis_root / f"views-v{SPAN_SCHEMA_VERSION}"
    view_checks = []
    for path in sorted(view_root.glob("*/*/trace_features.csv")):
        frame = pd.read_csv(path)
        if len(frame) != 4:
            raise RuntimeError(f"Incorrect view row count: {path}")
        json.loads((path.parent / "correlations.json").read_text())
        view_checks.append({"view": str(path.parent.relative_to(view_root)), "rows": len(frame),
                            "traces_with_tokens": int(frame.selected_token_count.gt(0).sum())})
    from moe_exp.correlation_pipeline.spans import SENTENCE_LABELS
    expected_views = 12 + (len(SENTENCE_LABELS) if annotation_dir else 0)
    if len(view_checks) != expected_views:
        raise RuntimeError("Missing full/class/position analysis outputs")
    report = {
        "status": "complete", "panel": panel, "initial_extraction": first, "resume": resumed,
        "model_load_and_first_forward_seconds": first_seconds,
        "elapsed_seconds": time.monotonic()-started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
        "peak_cuda_reserved_bytes": [torch.cuda.max_memory_reserved(i) for i in range(2)],
        "analysis_traces": analysis["n_traces"], "provenance": provenance,
        "class_conditioned_gpu_acceptance": "passed" if annotation_dir else "not tested",
        "quantization": getattr(model, "_replay_quantization", None),
        "view_outputs": view_checks,
        "statistical_scope": "execution check only; four traces do not establish scientific correlations",
    }
    publish(output_root / "acceptance.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--source", required=True, choices=tuple(SOURCES))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cpu-gate", type=Path)
    parser.add_argument("--generation-model")
    parser.add_argument("--forward-model")
    parser.add_argument("--revision")
    parser.add_argument("--quantization", choices=("none", "bnb-4bit", "bnb-8bit", "unsloth-4bit"))
    parser.add_argument("--annotation-dir", type=Path)
    args = parser.parse_args(argv)
    for path in (args.panel_root, args.output_root):
        if path is not None and not path.resolve().is_relative_to(REPO):
            parser.error("Debug writes must remain inside this repository")
    if args.action == "freeze":
        root = args.source_root or REPO / "results/correlation_pipeline" / SOURCES[args.source][2] / "generation"
        print(json.dumps(freeze_panel(args.source, root, args.panel_root, annotation_root=args.annotation_dir), sort_keys=True))
    else:
        if args.output_root is None or args.cpu_gate is None:
            parser.error("run requires --output-root and --cpu-gate")
        run_panel(args.source, args.panel_root, args.output_root, args.cpu_gate,
                  generation_model=args.generation_model, forward_model=args.forward_model,
                  revision=args.revision, quantization=args.quantization, annotation_dir=args.annotation_dir)


if __name__ == "__main__":
    main()
