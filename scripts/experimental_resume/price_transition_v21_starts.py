"""Exact-template, conservative complete-stage price for fresh start-only audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
PARITY = ROOT / "steering-v1/runs/routing-control-v1/dense-judge-parity/results-8e939bee-2341bac1/PARITY.json"
OUT = REPO / "report/experimental-resume-v1/TRANSITION_V21_START_RATING_PRICE.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_transition_v21_starts as rating
from price_transition_ratings_v3 import count_prompt_tokens


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != rating.digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed price input: {path}")
    return value


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("exact prompt pricing requires CPU Slurm")
    from transformers import AutoTokenizer
    frame = sealed(rating.FRAME)
    parity = sealed(PARITY)
    if frame["schema"] != "transition-v21-independent-audit-frame-v1":
        raise ValueError("unrecognized v2.1 discovery frame")
    if parity["job_id"] != "59112590" or parity["coverage"] < .99:
        raise ValueError("prior judge parity throughput source not qualified")
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = []
    for row in frame["records"]:
        tokens = tokenizer.apply_chat_template(rating.messages(row), tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=True, reasoning_effort="low")
        lengths.append(count_prompt_tokens(tokens))
    if not lengths or max(lengths) + rating.MAX_TOKENS > 49152:
        raise ValueError("start rating prompt exceeds judge context")
    n_ratings = 2 * len(lengths)
    prefill = 2 * sum(lengths)
    decode = n_ratings * rating.MAX_TOKENS
    observed_decode_tps = parity["total_generated_tokens"] / parity["generation_seconds"]
    pessimistic_decode_tps = observed_decode_tps * .65
    pessimistic_prefill_tps = 19656.064021098042 * .5
    repeat_work = 1.25
    decode_seconds = decode / pessimistic_decode_tps * repeat_work
    prefill_seconds = prefill / pessimistic_prefill_tps * repeat_work
    load_seconds = 2 * parity["load_seconds"] * 1.25
    shutdown_seconds = 2 * 196
    total_gpu_h = 2 * (decode_seconds + prefill_seconds + load_seconds + shutdown_seconds) / 3600
    ceiling = 10.0  # 2 GPUs x (4-hour first slice + 1-hour single recovery)
    body = {
        "schema": "transition-v21-start-rating-price-v1",
        "job_id": os.environ["SLURM_JOB_ID"],
        "status": "PASS_COMPLETE_STAGE" if total_gpu_h <= ceiling else "HOLD_PRICE_EXCEEDS_CEILING",
        "frame_sha256": frame["sha256"],
        "rating_driver_sha256": hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rating.RUBRIC.read_bytes()).hexdigest(),
        "parity_sha256": parity["sha256"],
        "rows": len(lengths), "ratings": n_ratings,
        "prompt_tokens_exact_twice": prefill,
        "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
        "max_decode_tokens": decode,
        "observed_parity_decode_tokens_per_s": observed_decode_tps,
        "pessimistic_decode_tokens_per_s": pessimistic_decode_tps,
        "pessimistic_prefill_tokens_per_s": pessimistic_prefill_tps,
        "repeat_work_factor": repeat_work,
        "components_seconds": {"decode": decode_seconds, "prefill": prefill_seconds,
                               "two_cold_loads": load_seconds, "two_shutdowns": shutdown_seconds},
        "complete_stage_projected_GPU_h": total_gpu_h,
        "allocation_plan": {"first_slice_GPU_h_max": 8.0,
                            "one_recovery_GPU_h_max": 2.0,
                            "combined_GPU_h_ceiling": ceiling},
        "interpretation": "Pessimistic all-capped 1024-token pricing; two same-model prefix-only start readers; disjoint sampled windows but same discovery families; target-window ratings are a separate stage",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": rating.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "rows": len(lengths),
                      "complete_projected_GPU_h": total_gpu_h,
                      "ceiling_GPU_h": ceiling}))


if __name__ == "__main__":
    main()
