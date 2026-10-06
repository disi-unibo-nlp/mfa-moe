"""Checkpointed, one-A100 GPT-OSS audit of the frozen 27-prefix panel.

This is a different-family LLM measurement audit. It is not human ground truth
or an online detector. Qwen/native ratings and selection tags never enter the
model prompt. Batch and per-assignment receipts expose incomplete attempts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_gptoss_start_pilot as pilot
from gptoss_final_parser_v2 import parse_rating

FRAME = BASE / "GPTOSS_START_PANEL_v2.json"
PRICE = REPO / "report/experimental-resume-v1/GPTOSS_START_PANEL_PRICE_v2.json"
PARSER = REPO / "scripts/experimental_resume/gptoss_final_parser_v2.py"
PROMPT = pilot.QWEN_PROMPT_DRIVER
MAX_TOKENS = 512
BATCH = 4


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def atomic_json(path, body):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".part-" + os.environ["SLURM_JOB_ID"])
    value = {**body, "sha256": digest(body)}
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return value


def seed(uid):
    return int(digest(["gptoss-start-panel-v2", uid])[:8], 16)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("GPT-OSS panel requires allocated GPU Slurm")
    frame, price = sealed(FRAME), sealed(PRICE)
    cache = pilot.model_cache()
    rows = frame["records"]
    if (frame["schema"] != "gptoss-start-panel-v2" or len(rows) != 27
            or frame["selected_start_rows"] != 23 or frame["frozen_control_rows"] != 4
            or len({r["uid"] for r in rows}) != 27
            or any(set(r) != {"uid", "family", "transition", "reader_input", "prefix_tokens"}
                   for r in rows)):
        raise ValueError("expanded GPT-OSS panel changed")
    if (price["schema"] != "gptoss-start-panel-price-v2"
            or price["status"] != "PASS_COMPLETE_1_GPUH"
            or price["frame_sha256"] != frame["sha256"]
            or price["rating_driver_sha256"] != file_sha(__file__)
            or price["parser_sha256"] != file_sha(PARSER)
            or price["prompt_driver_sha256"] != file_sha(PROMPT)
            or price["model_cache"] != cache
            or price["rows"] != len(rows)
            or price["max_decode_tokens"] != len(rows) * MAX_TOKENS):
        raise ValueError("sealed price no longer binds panel code, frame or cache")
    conversations = [pilot.messages(row) for row in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "WRITER.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding_body = {
            "schema": "gptoss-start-panel-binding-v2",
            "frame_sha256": frame["sha256"], "price_sha256": price["sha256"],
            "rating_driver_sha256": file_sha(__file__),
            "parser_sha256": file_sha(PARSER),
            "prompt_driver_sha256": file_sha(PROMPT),
            "model_cache": cache, "batch_size": BATCH, "max_tokens": MAX_TOKENS,
            "scope": "one different-family LLM start vote per 27 frozen discovery prefixes; not human truth",
        }
        binding_path = args.out / "BINDING.json"
        if binding_path.exists():
            binding = sealed(binding_path)
            if binding != {**binding_body, "sha256": digest(binding_body)}:
                raise ValueError("GPT-OSS panel output directory rebound")
        else:
            binding = atomic_json(binding_path, binding_body)
        batches = args.out / "batches"
        attempts = args.out / "attempts"
        batches.mkdir(exist_ok=True)
        attempts.mkdir(exist_ok=True)
        pending = []
        for begin in range(0, len(rows), BATCH):
            path = batches / f"{begin:04d}.json"
            block = rows[begin:begin + BATCH]
            if path.exists():
                saved = sealed(path)
                if saved["binding_sha256"] != binding["sha256"] or [r["uid"] for r in saved["records"]] != [r["uid"] for r in block]:
                    raise ValueError("committed panel batch differs")
            else:
                prior = list(attempts.glob(f"{begin:04d}-*.json"))
                if prior and not args.recover:
                    raise RuntimeError(f"uncommitted batch {begin} has prior attempts; explicit --recover required")
                pending.append((begin, path, block))
        load_seconds = 0.0
        if pending:
            from vllm import LLM, SamplingParams
            start_load = time.monotonic()
            model = LLM(model=str(pilot.MODEL), tokenizer=str(pilot.MODEL), tensor_parallel_size=1,
                        dtype="bfloat16", max_model_len=16384, max_num_seqs=BATCH,
                        max_num_batched_tokens=16384, gpu_memory_utilization=.8,
                        enforce_eager=True, generation_config="vllm")
            load_seconds = time.monotonic() - start_load
            for begin, path, block in pending:
                attempt_file = attempts / f"{begin:04d}-{os.environ['SLURM_JOB_ID']}.json"
                atomic_json(attempt_file, {
                    "schema": "gptoss-start-panel-attempt-v2", "binding_sha256": binding["sha256"],
                    "begin": begin, "uids": [r["uid"] for r in block],
                    "job_id": os.environ["SLURM_JOB_ID"], "state": "attempted_before_generation",
                    "recovery": bool(args.recover),
                })
                params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                         seed=seed(r["uid"])) for r in block]
                start = time.monotonic()
                try:
                    output = model.chat([pilot.messages(r) for r in block], sampling_params=params,
                                        chat_template_kwargs={"reasoning_effort": "low"}, use_tqdm=False)
                except Exception as exc:
                    atomic_json(attempts / f"{begin:04d}-{os.environ['SLURM_JOB_ID']}-failure.json", {
                        "schema": "gptoss-start-panel-failure-v2", "binding_sha256": binding["sha256"],
                        "begin": begin, "uids": [r["uid"] for r in block],
                        "job_id": os.environ["SLURM_JOB_ID"],
                        "exception_type": type(exc).__name__, "exception_message": str(exc)[:500],
                    })
                    raise
                if len(output) != len(block):
                    raise ValueError("GPT-OSS panel output count differs")
                records = [{"uid": r["uid"], "rating": parse_rating(o.outputs[0].text),
                            "raw_completion": o.outputs[0].text,
                            "finish_reason": o.outputs[0].finish_reason,
                            "prompt_tokens": len(o.prompt_token_ids),
                            "generated_tokens": len(o.outputs[0].token_ids)}
                           for r, o in zip(block, output)]
                expected = price["prompt_tokens_per_row"][begin:begin + len(block)]
                if [r["prompt_tokens"] for r in records] != expected:
                    raise ValueError("actual GPT-OSS prompt-token lengths differ from sealed price")
                atomic_json(path, {"schema": "gptoss-start-panel-batch-v2",
                                   "binding_sha256": binding["sha256"], "begin": begin,
                                   "job_id": os.environ["SLURM_JOB_ID"],
                                   "generation_seconds": time.monotonic() - start,
                                   "records": records})
                print(json.dumps({"committed_batch": begin, "rows": len(records),
                                  "parsed_stops": sum(r["rating"] is not None and r["finish_reason"] == "stop" for r in records)}), flush=True)
        parts = [sealed(batches / f"{begin:04d}.json") for begin in range(0, len(rows), BATCH)]
        records = [row for part in parts for row in part["records"]]
        if [r["uid"] for r in records] != [r["uid"] for r in rows]:
            raise ValueError("GPT-OSS panel assignments incomplete")
        summary_path = args.out / "SUMMARY.json"
        if summary_path.exists():
            summary = sealed(summary_path)
            if summary["binding_sha256"] != binding["sha256"] or summary["batch_sha256s"] != [p["sha256"] for p in parts]:
                raise ValueError("existing GPT-OSS summary differs")
        else:
            summary = atomic_json(summary_path, {
                "schema": "gptoss-start-panel-summary-v2",
                "binding_sha256": binding["sha256"], "batch_sha256s": [p["sha256"] for p in parts],
                "rows": len(records), "parsed_natural_stops": sum(r["rating"] is not None and r["finish_reason"] == "stop" for r in records),
                "positive_starts": sum(r["rating"] == {"start": True} and r["finish_reason"] == "stop" for r in records),
                "load_seconds_this_job": load_seconds,
                "interpretation": "Different-model-family LLM audit; raw completions are saved; no human-truth or causal claim",
            })
        print(json.dumps({"summary": str(summary_path), "sha256": summary["sha256"],
                          "parsed_natural_stops": summary["parsed_natural_stops"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    run(parser.parse_args())
