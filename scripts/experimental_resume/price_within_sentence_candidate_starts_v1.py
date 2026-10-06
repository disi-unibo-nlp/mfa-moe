"""Exact-token, complete-stage price for early-cut Qwen3.8 start audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
PRIOR = BASE / "ratings-v22-fullprefix-v2-6c10499b-1ef8863f"
OUT = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_START_RATING_PRICE_v1.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_within_sentence_candidate_starts_v1 as rating
from price_transition_ratings_v3 import count_prompt_tokens


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != rating.digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("exact early-cut price requires CPU Slurm")
    from transformers import AutoTokenizer
    frame = sealed(rating.FRAME)
    prior_binding = sealed(PRIOR / "BINDING.json")
    prior_summary = sealed(PRIOR / "SUMMARY.json")
    if (frame["schema"] != "within-sentence-start-audit-frame-v1"
            or frame["rows"] != 43 or frame["families"] != 35
            or prior_summary["binding_sha256"] != prior_binding["sha256"]
            or prior_summary["counts"].get("ratings") != 744):
        raise ValueError("frozen timing frame or prior Qwen throughput source changed")
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        rating.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="low")) for row in frame["records"]]
    if max(lengths) + rating.MAX_TOKENS > 49152:
        raise ValueError("early-cut audit prompt exceeds Qwen context")
    reader_timings = [r for b in prior_summary["timings"] for r in b["reader_timings"]]
    observed_generated = sum(r["generated_tokens"] for r in reader_timings)
    observed_wall = sum(r["wall_seconds"] for r in reader_timings)
    if observed_generated != prior_summary["counts"]["generated_tokens"] or observed_wall <= 0:
        raise ValueError("prior Qwen timing receipt incomplete")
    bounded_decode_tps = observed_generated / observed_wall * .5
    decode_tokens = 2 * len(lengths) * rating.MAX_TOKENS
    prefill_tokens = 2 * sum(lengths)
    load_seconds = max(r["load_seconds"] for r in prior_summary["load_receipts"]) * 1.25
    decode_seconds = decode_tokens / bounded_decode_tps
    prefill_seconds = prefill_tokens / 9800.0
    shutdown_seconds = 180.0
    projected_seconds = load_seconds + decode_seconds + prefill_seconds + shutdown_seconds
    projected_gpu_h = 2 * projected_seconds / 3600
    status = "PASS_COMPLETE_2_GPUH" if projected_seconds <= 3600 else "HOLD_EXCEEDS_FIRST_HOUR"
    body = {
        "schema": "within-sentence-start-rating-price-v1",
        "job_id": os.environ["SLURM_JOB_ID"], "status": status,
        "frame_sha256": frame["sha256"],
        "rating_driver_sha256": hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rating.RUBRIC.read_bytes()).hexdigest(),
        "prior_qwen_binding_sha256": prior_binding["sha256"],
        "prior_qwen_summary_sha256": prior_summary["sha256"],
        "rows": len(lengths), "ratings": 2 * len(lengths),
        "prompt_tokens_exact_twice": prefill_tokens,
        "prompt_tokens_per_row": lengths,
        "prompt_tokens_max": max(lengths),
        "max_decode_tokens": decode_tokens,
        "prior_generated_tokens": observed_generated,
        "prior_generation_seconds_including_prefill": observed_wall,
        "bounded_decode_tokens_per_second": bounded_decode_tps,
        "components_seconds": {"cold_load": load_seconds,
                               "all_capped_decode": decode_seconds,
                               "prefill_extra_conservative": prefill_seconds,
                               "shutdown": shutdown_seconds},
        "projected_gpu_hours": projected_gpu_h,
        "allocation_plan": {"first_slice_GPU_h_max": 2.0,
                            "one_recovery_GPU_h_max": 2.0,
                            "combined_GPU_h_ceiling": 4.0},
        "interpretation": "Two A100, one-hour first allocation and one-hour recovery ceiling; all 86 responses at unchanged 1024 cap, exact tokenizer prefill, half measured Qwen effective decode throughput, 1.25x measured cold load, extra prefill and shutdown. Timing start audit is separate from original transition primary."
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": rating.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": status,
                      "projected_gpu_hours": projected_gpu_h,
                      "prompt_tokens_exact_twice": prefill_tokens}), flush=True)


if __name__ == "__main__":
    main()
