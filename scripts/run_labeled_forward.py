"""Native full-corpus orchestration using the existing label/forward/analysis code."""
import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/correlation_pipeline/labeled-forward-v1"
SPEC = ROOT / "plan.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)
    assert path.read_text() == payload


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip("-")


def check_code(plan):
    from moe_exp.correlation_pipeline.provenance import code_provenance
    if code_provenance() != plan["code"]:
        raise RuntimeError("Source changed after the production plan was frozen")


def prepare(name, spec, plan):
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    from moe_exp.correlation_pipeline.labels_export import SOURCES
    from moe_exp.jsonl import iter_jsonl
    source = SOURCES[name]
    result = adapt_source(name, source["label_root"], source["sampling_root"], ROOT / "annotations")
    generation = source["sampling_root"] / "generation" / slug(spec["generation_model"])
    counts = {}
    tokens = 0
    shortest = longest = None
    for dataset in spec["datasets"]:
        counts[dataset] = 0
        for row in iter_jsonl(generation / dataset / "traces.jsonl"):
            replay = row.get("metadata", {}).get("token_replay")
            if not replay or not replay.get("completion_token_ids"):
                raise ValueError("Every production trace requires exact token replay")
            length = len(replay["completion_token_ids"])
            counts[dataset] += 1
            tokens += length
            if shortest is None or length < shortest[0]:
                shortest = (length, dataset, row)
            if longest is None or length > longest[0]:
                longest = (length, dataset, row)
    if sum(counts.values()) != spec["traces"] or result["rows"] != 100000:
        raise ValueError("Prepared corpus does not match the reviewed model inventory")
    panel = {}
    identities = set()
    for length, dataset, row in (shortest, longest):
        identity = (dataset, row["problem_id"], row.get("sample_id", 0))
        if identity not in identities:
            panel.setdefault(dataset, []).append(row)
            identities.add(identity)
    panel_root = ROOT / "sources" / name / "smoke-generation"
    for dataset, rows in panel.items():
        path = panel_root / slug(spec["generation_model"]) / dataset / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write(ROOT / "sources" / name / "prepared.json", {
        "status": "complete", "source": name, "code": plan["code"],
        "adapter": result, "dataset_traces": counts, "completion_tokens": tokens,
        "smoke_datasets": list(panel), "smoke_traces": len(identities),
        "smoke_completion_tokens": [shortest[0], longest[0]],
    })


def extract_args(spec, output, generation, datasets):
    from moe_exp.correlation_pipeline import extract
    return extract.build_parser().parse_args([
        "--model-id", spec["forward_model"], "--revision", spec["revision"], "--local-files-only",
        "--generation-model", spec["generation_model"], "--quantization", spec["quantization"],
        "--generation-dir", str(generation), "--output-dir", str(output),
        "--annotation-dir", str(ROOT / "annotations"), "--datasets", *datasets,
        "--all-router-layers", "--views", "full", "class", "position",
    ])


def validate_precision(model, name):
    if name == "gpt":
        from transformers.integrations.mxfp4 import Mxfp4GptOssExperts
        quantizer = getattr(model, "hf_quantizer", None)
        if quantizer is None or quantizer.quantization_config.dequantize:
            raise RuntimeError("GPT replay must retain native MXFP4 weights")
        blocks = [layer.mlp.experts for layer in model.model.layers]
        if not blocks or not all(isinstance(block, Mxfp4GptOssExperts) for block in blocks):
            raise RuntimeError("GPT experts are not native MXFP4")
        info = {"method": "mxfp4", "bits": 4, "expert_blocks": len(blocks)}
    else:
        module = {"gemma": "gemma_quantized", "nemotron": "nemotron_quantized", "qwen36": "qwen_quantized"}[name]
        from importlib import import_module
        count = import_module("moe_exp.models." + module).validate_quantized_experts(model)
        info = {"method": "bitsandbytes", "bits": 4, "expert_linear_count": count,
                "compute_dtype": "bfloat16"}
    model._replay_quantization = info
    return info


def forward(name, spec, plan):
    import torch
    from moe_exp.correlation_pipeline import extract
    from moe_exp.correlation_pipeline.forward_audit import audit_bundle
    from moe_exp.correlation_pipeline.labels_export import SOURCES
    from moe_exp.jsonl import iter_jsonl
    from moe_exp.models.loader import load_model_and_tokenizer
    prepared = json.loads((ROOT / "sources" / name / "prepared.json").read_text())
    gate = json.loads(Path(plan["cpu_gate"]).read_text())
    if gate["status"] != "passed" or gate["code"] != plan["code"]:
        raise RuntimeError("Production requires passing CPU tests for the frozen source")
    checkpoint = json.loads(Path(spec["checkpoint_report"]).read_text())
    if (checkpoint["status"] != "complete" or checkpoint["revision"] != spec["revision"]
            or checkpoint["model"] != spec["forward_model"]):
        raise RuntimeError("Pinned checkpoint download is incomplete")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Production forward expects two allocated CUDA devices")
    if name == "gpt":
        from transformers.utils import is_kernels_available, is_triton_available
        if not is_kernels_available() or not is_triton_available("3.4.0"):
            raise RuntimeError("GPT MXFP4 dependencies unavailable; refusing BF16 fallback")
        if torch.cuda.get_device_capability() < (7, 5):
            raise RuntimeError("GPT MXFP4 requires SM75 or newer")
    started = time.monotonic()
    print("MODEL_LOAD_START", name, flush=True)
    model, tokenizer = load_model_and_tokenizer(
        spec["forward_model"], revision=spec["revision"], local_files_only=True,
        quantization=spec["quantization"], offload_folder=str(ROOT / "sources" / name / "offload"),
    )
    precision = validate_precision(model, name)
    print("MODEL_LOAD_COMPLETE", name, precision, flush=True)
    # Shortest and longest complete labeled traces test loading, exact replay,
    # long-context memory and retained analysis inputs before the corpus loop.
    smoke = ROOT / "sources" / name / "smoke-forward"
    args = extract_args(spec, smoke, ROOT / "sources" / name / "smoke-generation", prepared["smoke_datasets"])
    extract.extract_all(args, model_and_tokenizer=(model, tokenizer))
    audits = []
    for dataset in prepared["smoke_datasets"]:
        report = audit_bundle(smoke / slug(spec["forward_model"]) / dataset / "traces_with_routing.jsonl", require_replay=True)
        if report["status"] != "complete" or not all(t["expert_by_class_ready"] for t in report["traces"]):
            raise RuntimeError("Production smoke artifact audit failed")
        audits.append({"dataset": dataset, "status": report["status"], "traces": len(report["traces"])})
    write(ROOT / "sources" / name / "smoke-acceptance.json", {
        "status": "complete", "quantization": precision, "audits": audits,
        "completion_tokens": prepared["smoke_completion_tokens"], "code": plan["code"],
    })
    print("SMOKE_COMPLETE", name, flush=True)
    args = extract_args(spec, ROOT / "forward", SOURCES[name]["sampling_root"] / "generation", spec["datasets"])
    summaries = extract.extract_all(args, model_and_tokenizer=(model, tokenizer))
    for entry in summaries:
        if entry["traces"] != prepared["dataset_traces"][entry["dataset"]]:
            raise RuntimeError("Forward output count differs from complete labeled corpus")
        for row in iter_jsonl(Path(entry["output"])):
            metadata = row["metadata"]
            if not metadata.get("reasoning_annotation") or len(metadata["correlation_views"]["scopes"]) != 19:
                raise RuntimeError("Forward omitted required label-conditioned views")
    write(ROOT / "sources" / name / "forward-acceptance.json", {
        "status": "complete", "source": name, "job_id": os.environ["SLURM_JOB_ID"],
        "code": plan["code"], "quantization": precision, "datasets": summaries,
        "elapsed_seconds": time.monotonic()-started,
        "peak_cuda_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
        "storage_policy": plan["storage_policy"],
    })


def analyze(name, spec, plan):
    from moe_exp.correlation_pipeline import analyze as analysis
    from moe_exp.correlation_pipeline.forward_audit import audit_bundle
    done = json.loads((ROOT / "sources" / name / "forward-acceptance.json").read_text())
    if done["status"] != "complete" or done["code"] != plan["code"]:
        raise RuntimeError("Forward completion required before analysis")
    audits = []
    for dataset in spec["datasets"]:
        path = ROOT / "forward" / slug(spec["forward_model"]) / dataset / "traces_with_routing.jsonl"
        report = audit_bundle(path, require_replay=True)
        if report["status"] != "complete" or not all(t["expert_by_class_ready"] for t in report["traces"]):
            raise RuntimeError("Full-corpus artifact audit failed")
        for trace in report["traces"]:
            trace.pop("expert_frequencies_by_layer", None)
        write(ROOT / "sources" / name / ("audit-" + dataset + ".json"), report)
        audits.append({"dataset": dataset, "traces": len(report["traces"]), "status": report["status"]})
    result = analysis.analyze(analysis.build_parser().parse_args([
        "--model-id", spec["forward_model"], "--forward-dir", str(ROOT / "forward"),
        "--output-dir", str(ROOT / "analysis"), "--datasets", *spec["datasets"],
        "--views", "full", "class", "position", "--bootstrap-samples", "500",
    ]))
    excluded = result["excluded_truncated_generations"]
    if result["n_traces"] + sum(excluded.values()) != spec["traces"]:
        raise RuntimeError("Analysis accounting omitted labeled traces")
    ar = ROOT / "analysis" / slug(spec["forward_model"])
    tables = {}
    for name_ in ("trace_features.csv", "problem_features.csv", "expert_trace_features.csv"):
        with (ar / name_).open() as handle:
            reader = csv.DictReader(handle)
            tables[name_] = {"columns": len(reader.fieldnames), "rows": sum(1 for _ in reader)}
    from moe_exp.correlation_pipeline.spans import SPAN_SCHEMA_VERSION
    views = []
    for path in sorted((ar / f"views-v{SPAN_SCHEMA_VERSION}").glob("*/*/trace_features.csv")):
        with path.open() as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != result["n_traces"]:
            raise RuntimeError("Analysis view omitted eligible traces")
        json.loads((path.parent / "correlations.json").read_text())
        views.append({"path": str(path.parent), "rows": count})
    if len(views) != 19:
        raise RuntimeError("Analysis omitted full/class/position views")
    write(ROOT / "sources" / name / "analysis-acceptance.json", {
        "status": "complete", "audits": audits, "n_traces": result["n_traces"],
        "excluded_truncated_generations": excluded, "view_outputs": views,
        "tables": tables, "code": plan["code"], "job_id": os.environ["SLURM_JOB_ID"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "forward", "analyze"))
    parser.add_argument("source", choices=("gpt", "gemma", "qwen36", "nemotron"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run this driver in an allocated Slurm step")
    plan = json.loads(SPEC.read_text())
    check_code(plan)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    globals()[args.phase](args.source, plan["sources"][args.source], plan)


if __name__ == "__main__":
    main()
