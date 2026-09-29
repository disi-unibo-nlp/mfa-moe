#!/usr/bin/env python3
"""Prepared pilot actions. GPU actions require the reviewed Slurm allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket

from moe_exp.correlation_pipeline.dynamics.common import digest, write_json, write_rows
from moe_exp.correlation_pipeline.dynamics.pilot import (
    JUDGE_REVISION, REVISION, SLUG, SOURCE, TARGET, verify, launcher_hashes,
)
from moe_exp.correlation_pipeline.provenance import code_provenance, file_sha256

PROJECT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe")
REPO = PROJECT / "repo"


def preflight(root, *, require_code=True):
    manifest = verify(root)
    if require_code and (manifest["code"] != code_provenance() or manifest.get("launchers") != launcher_hashes()):
        raise ValueError("Pilot code provenance changed; rebind only after rerunning validation")
    cache = PROJECT / "cache/hf/hub"
    snapshots = {
        "forward": cache / "models--unsloth--Qwen3.6-35B-A3B/snapshots" / REVISION,
        "judge": cache / "models--Qwen--Qwen3.8-27B/snapshots" / JUDGE_REVISION,
    }
    checked = {}
    for name, path in snapshots.items():
        index = json.loads((path / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        if not all((path / shard).is_file() for shard in shards):
            raise ValueError(name + " checkpoint is incomplete")
        checked[name] = dict(revision=path.name, shards=len(shards),
                            bytes=sum((path / shard).stat().st_size for shard in shards))
    return dict(status="passed", manifest_sha256=manifest["manifest_sha256"], code=code_provenance(),
                snapshots=checked, traces=len(manifest["traces"]),
                sentences=manifest["storage_estimate"]["reasoning_sentences"])


def guard():
    import getpass
    if (not os.environ.get("SLURM_JOB_ID") or getpass.getuser() != "lmolfett" or
            not socket.gethostname().startswith("lrdn") or
            os.environ.get("SLURM_JOB_ACCOUNT") != "iscrc_miosr" or
            os.environ.get("SLURM_JOB_PARTITION") != "boost_usr_prod"):
        raise RuntimeError("Pilot compute requires the explicitly authorized Booster allocation")


def label(root, base_url):
    guard()
    manifest = verify(root)
    from moe_exp.correlation_pipeline.annotation_partition import enumerate_items
    from moe_exp.correlation_pipeline.annotation_batch import run_part
    from moe_exp.correlation_pipeline.batch_predictor import load_program_and_adapter, classify_batch_outcomes
    from moe_exp.correlation_pipeline.spans import reasoning_bounds, trace_digest, validate_annotation
    from moe_exp.jsonl import iter_jsonl
    from moe_exp.schemas import TraceRecord
    items = enumerate_items(root / "generation" / SLUG, ("math500",))
    config = {**manifest["judge"], "pilot_manifest_sha256": manifest["manifest_sha256"]}
    _, predict, adapter = load_program_and_adapter(str(root / "selected_program.json"))
    def classify(batch):
        j = manifest["judge"]
        return classify_batch_outcomes(batch, adapter=adapter, predict=predict, model=j["judge_model"],
            base_url=base_url, api_key="local-vllm-key", max_tokens=j["max_tokens"],
            temperature=j["temperature"], reasoning_effort=j["reasoning_effort"], top_p=j["top_p"],
            top_k=j["top_k"], min_p=j["min_p"], presence_penalty=j["presence_penalty"],
            repetition_penalty=j["repetition_penalty"])
    result = run_part(items, config, output_dir=root / "label-checkpoints", classify=classify,
                      expected_count=manifest["storage_estimate"]["reasoning_sentences"])
    saved = json.loads((root / "label-checkpoints/annotations.json").read_text())
    by_trace = {}
    for record in saved:
        i = record["identity"]
        key = (i["dataset"], i["problem_id"], i.get("sample_id", 0))
        outcome = {k: v for k, v in record.items() if k not in ("identity", "unit", "inputs")}
        by_trace.setdefault(key, []).append({**record["unit"], **outcome})
    annotations = []
    for row in iter_jsonl(root / "generation" / SLUG / "math500/traces.jsonl"):
        t = TraceRecord(**row)
        a = dict(schema_version=1, trace_sha256=trace_digest(t), source_model=t.model_id,
            dataset=t.dataset, problem_id=t.problem_id, sample_id=t.sample_id,
            reasoning_span=list(reasoning_bounds(t)), sentence_selection=None,
            classifier=config, status="complete",
            units=by_trace[(t.dataset, t.problem_id, t.sample_id)])
        validate_annotation(t, a)
        annotations.append(a)
    write_rows(root / "annotations" / SLUG / "math500/annotations.jsonl", annotations)
    write_json(root / "label-acceptance.json", {**result, "manifest_sha256": manifest["manifest_sha256"],
        "traces": len(annotations), "labeled_sentences": len(saved) - result.get("unknown_count", 0),
        "unknown_sentences": result.get("unknown_count", 0)})
    return result


def replay(root):
    guard()
    manifest = verify(root)
    acceptance = json.loads((root / "label-acceptance.json").read_text())
    if acceptance.get("status") != "complete" or acceptance.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Labeling has not completed for this pilot")
    from moe_exp.correlation_pipeline.extract import build_parser, extract_all
    args = build_parser().parse_args([
        "--model-id", TARGET, "--generation-model", SOURCE, "--datasets", "math500",
        "--generation-dir", str(root / "generation"), "--output-dir", str(root / "forward"),
        "--annotation-dir", str(root / "annotations"), "--views", "full", "class",
        "--quantization", "bnb-4bit", "--revision", REVISION, "--local-files-only",
        "--all-router-layers", "--router-only", "--dynamics",
    ])
    summary = extract_all(args)
    write_json(root / "replay-execution.json", summary)
    return summary


def acceptance(root):
    from moe_exp.jsonl import iter_jsonl
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import validate_annotation
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    import math
    manifest = verify(root)
    path = root / "forward/unsloth--Qwen3.6-35B-A3B/math500/traces_with_routing.jsonl"
    rows = list(iter_jsonl(path))
    if len(rows) != 64:
        raise ValueError("Incomplete pilot replay")
    observed = {tuple([r["model_id"], r["dataset"], r["problem_id"], r["sample_id"]]) for r in rows}
    if observed != {tuple(r["identity"]) for r in manifest["traces"]}:
        raise ValueError("Replayed membership differs from frozen pilot")
    count = 0
    for row in rows:
        trace = TraceRecord(**row)
        validate_annotation(trace, trace.metadata["reasoning_annotation"])
        layout = saved_layout(trace)
        compact = trace.metadata["routing_dynamics"]
        if trace.model_logs.layer_indices != list(range(40)) or not compact["semantics"]["native_selection"]:
            raise ValueError("All native router layers were not collected")
        if not compact["windows"]:
            raise ValueError("No routing windows collected")
        for window in compact["windows"]:
            for name, value in window.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Nonfinite statistic: " + name)
            if window["full_distribution_status"] != "available" or window["executed_mixture_status"] != "available":
                raise ValueError("Missing full-router or executed-mixture collection")
        if layout["token_count"] != trace.metadata["correlation_features"]["token_count"]:
            raise ValueError("Token alignment differs from forward")
        count += len(compact["windows"])
    result = dict(status="complete", traces=64, windows=count, manifest_sha256=manifest["manifest_sha256"],
        labels=json.loads((root / "label-acceptance.json").read_text()),
        code=code_provenance(), forward_sha256=file_sha256(path))
    write_json(root / "acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "label", "replay", "accept", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:43842/v1")
    args = parser.parse_args()
    if args.phase == "preflight":
        result = preflight(args.root)
        write_json(args.root / "preflight.json", result)
    else:
        preflight(args.root)
        if args.phase == "label": result = label(args.root, args.base_url)
        elif args.phase == "replay": result = replay(args.root)
        elif args.phase == "accept": result = acceptance(args.root)
        else:
            guard()
            from moe_exp.correlation_pipeline.dynamics.cli import analyze
            result = analyze([args.root / "forward/unsloth--Qwen3.6-35B-A3B/math500/traces_with_routing.jsonl"],
                args.root / "analysis", evaluate_models=True, plots=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
