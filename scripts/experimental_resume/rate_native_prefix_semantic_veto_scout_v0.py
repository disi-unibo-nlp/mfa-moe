"""Sealed 12-prefix native Qwen semantic-veto execution/parse scout.

Two seeded, unsteered draws per frozen discovery prefix. This is a detector
qualification scout, not a behavioral or intervention experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import native_prefix_semantic_veto_v0 as veto

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FRAME = veto.BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json"
PRICE = REPO / "report/experimental-resume-v1/NATIVE_PREFIX_SEMANTIC_VETO_PRICE_v0.json"
OUT = veto.BASE / "ratings-native-veto-scout-v0-d8f50b7e-8d2b37d2"
BATCH = 12
READERS = 2


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sealed(path, body):
    if path.exists():
        raise FileExistsError(path)
    value = {**body, "sha256": veto.digest(body)}
    temp = path.with_name(path.name + ".part-" + os.environ["SLURM_JOB_ID"])
    with temp.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return value


def seed(uid, reader):
    return int(veto.digest(["native-prefix-veto-scout-v0", uid, reader])[:8], 16) % 2_000_000_000


def run(out):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("native veto scout requires GPU Slurm")
    frame, price = veto.sealed(FRAME), veto.sealed(PRICE)
    if (frame["schema"] != "transition-v22-qwen-12-scout-frame-v1"
            or frame["rows"] != 12 or frame["families"] != 12
            or len(frame["records"]) != 12
            or len({row["uid"] for row in frame["records"]}) != 12):
        raise ValueError("native veto scout frame changed")
    if (price["schema"] != "native-prefix-semantic-veto-price-v0"
            or price["status"] != "HOLD_NATIVE_SCOUT"
            or price["scout_frame_sha256"] != frame["sha256"]
            or price["prompt_driver_sha256"] != file_sha(veto.__file__)
            or price["max_tokens_per_rating"] != veto.MAX_TOKENS
            or price["scout"]["ratings"] != READERS * len(frame["records"])
            or price["scout"]["projected_GPU_h"] > price["proposed_ceilings_GPU_h"]["scout"]
            or price["model_cache"] != veto.cache_receipt()):
        raise ValueError("sealed native veto GPU price does not bind frame/prompt/model")
    for row in frame["records"]:
        veto.messages(row)
    binding_body = {
        "schema": "native-prefix-semantic-veto-scout-binding-v0",
        "frame_sha256": frame["sha256"], "price_sha256": price["sha256"],
        "prompt_driver_sha256": file_sha(veto.__file__),
        "rating_driver_sha256": file_sha(__file__),
        "model_snapshot": str(veto.MODEL), "model_cache": price["model_cache"],
        "visible_input_allowlist": sorted(veto.INPUT_ALLOWLIST),
        "readers": READERS, "max_tokens": veto.MAX_TOKENS,
        "sampler": {"temperature": 0.2, "top_p": 0.95, "enable_thinking": False},
        "purpose": "prefix-only unsteered discovery detector execution and parse scout",
    }
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "WRITER.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_path = out / "BINDING.json"
    binding = veto.sealed(binding_path) if binding_path.exists() else write_sealed(binding_path, binding_body)
    if binding != {**binding_body, "sha256": veto.digest(binding_body)}:
        raise ValueError("native veto scout output directory rebound")
    assignments = out / "assignments"
    batches = out / "batches"
    loads = out / "loads"
    for path in (assignments, batches, loads):
        path.mkdir(exist_ok=True)
    rows = frame["records"]
    remaining = []
    for reader in range(READERS):
        path = batches / f"reader{reader}.json"
        if path.exists():
            saved = veto.sealed(path)
            if (saved["binding_sha256"] != binding["sha256"]
                    or saved["reader"] != reader
                    or [r["uid"] for r in saved["records"]] != [r["uid"] for r in rows]
                    or any(not (assignments / f"{row['uid']}-reader{reader}.json").exists()
                           for row in rows)):
                raise ValueError("saved native veto reader batch differs or lacks attempts")
        else:
            remaining.append(reader)
    if remaining:
        from vllm import LLM, SamplingParams
        start_load = time.monotonic()
        model = LLM(model=str(veto.MODEL), tokenizer=str(veto.MODEL),
                    tensor_parallel_size=2, dtype="bfloat16", kv_cache_dtype="bfloat16",
                    max_model_len=16384, max_num_seqs=16,
                    max_num_batched_tokens=8192, gpu_memory_utilization=.85,
                    enforce_eager=True, generation_config="vllm",
                    language_model_only=True,
                    attention_config={"backend": "FLASH_ATTN"})
        load_seconds = time.monotonic() - start_load
        write_sealed(loads / f"{os.environ['SLURM_JOB_ID']}.json",
                     {"schema": "native-prefix-veto-scout-load-v0",
                      "binding_sha256": binding["sha256"],
                      "job_id": os.environ["SLURM_JOB_ID"],
                      "load_seconds": load_seconds})
        print(json.dumps({"load_seconds": load_seconds, "remaining_readers": remaining}), flush=True)
        conversations = [veto.messages(row) for row in rows]
        for reader in remaining:
            batch_path = batches / f"reader{reader}.json"
            if any((assignments / f"{row['uid']}-reader{reader}.json").exists() for row in rows):
                raise RuntimeError("ambiguous prior attempt needs sealed recovery disposition")
            for row in rows:
                write_sealed(assignments / f"{row['uid']}-reader{reader}.json",
                             {"schema": "native-prefix-veto-scout-assignment-v0",
                              "binding_sha256": binding["sha256"],
                              "uid": row["uid"], "reader": reader,
                              "job_id": os.environ["SLURM_JOB_ID"],
                              "state": "attempted_before_generation"})
            params = [SamplingParams(temperature=.2, top_p=.95,
                                     max_tokens=veto.MAX_TOKENS,
                                     seed=seed(row["uid"], reader)) for row in rows]
            started = time.monotonic()
            try:
                outputs = model.chat(conversations, sampling_params=params,
                                     chat_template_kwargs={"enable_thinking": False},
                                     use_tqdm=False)
            except Exception as exc:
                for row in rows:
                    write_sealed(assignments / f"{row['uid']}-reader{reader}-failure.json",
                                 {"schema": "native-prefix-veto-scout-failure-v0",
                                  "binding_sha256": binding["sha256"], "uid": row["uid"],
                                  "reader": reader, "job_id": os.environ["SLURM_JOB_ID"],
                                  "exception_type": type(exc).__name__,
                                  "exception_message": str(exc)[:500]})
                raise
            if len(outputs) != len(rows) or any(len(output.outputs) != 1 for output in outputs):
                raise ValueError("native veto scout model output count differs")
            records = []
            for row, output in zip(rows, outputs, strict=True):
                result = output.outputs[0]
                records.append({"uid": row["uid"], "transition": row["transition"],
                                "rating": veto.parse_rating(result.text),
                                "finish_reason": result.finish_reason,
                                "generated_tokens": len(result.token_ids),
                                "prompt_tokens": len(output.prompt_token_ids),
                                "raw_completion": result.text})
            timing = {"wall_seconds": time.monotonic() - started,
                      "prompt_tokens": sum(r["prompt_tokens"] for r in records),
                      "generated_tokens": sum(r["generated_tokens"] for r in records)}
            write_sealed(batch_path,
                         {"schema": "native-prefix-veto-scout-reader-batch-v0",
                          "binding_sha256": binding["sha256"], "reader": reader,
                          "job_id": os.environ["SLURM_JOB_ID"], "timing": timing,
                          "records": records})
            print(json.dumps({"reader": reader, "timing": timing}), flush=True)
    parts = [veto.sealed(batches / f"reader{reader}.json") for reader in range(READERS)]
    if any([r["uid"] for r in part["records"]] != [r["uid"] for r in rows] for part in parts):
        raise ValueError("native veto scout final UID order differs")
    counts = Counter()
    for part in parts:
        for result in part["records"]:
            counts["ratings"] += 1
            counts["parsed"] += result["rating"] is not None
            counts["natural"] += result["finish_reason"] == "stop"
            counts["positive"] += bool(result["rating"] and result["rating"]["start"])
    summary_path = out / "SUMMARY.json"
    summary_body = {"schema": "native-prefix-veto-scout-summary-v0",
                    "binding_sha256": binding["sha256"], "families": 12,
                    "rows": len(rows), "counts": dict(counts),
                    "reader_batch_shas": [part["sha256"] for part in parts],
                    "reader_timings": [part["timing"] for part in parts]}
    summary = veto.sealed(summary_path) if summary_path.exists() else write_sealed(summary_path, summary_body)
    if summary != {**summary_body, "sha256": veto.digest(summary_body)}:
        raise ValueError("native veto scout summary changed")
    print(json.dumps({"summary": str(summary_path), "counts": dict(counts)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)
