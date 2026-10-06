"""Exact-template, conservative price for a 12-prefix full-context Qwen scout."""
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
OUT = REPO / "report/experimental-resume-v1/TRANSITION_V22_QWEN_12_SCOUT_PRICE_v2.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_transition_v22_qwen_scout as rating
from price_transition_ratings_v3 import count_prompt_tokens


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != rating.digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("exact prompt pricing requires CPU Slurm")
    from transformers import AutoTokenizer
    frame = sealed(rating.FRAME)
    parity = sealed(PARITY)
    if frame["schema"] != "transition-v22-qwen-12-scout-frame-v1" or len(frame["records"]) != 12:
        raise ValueError("unexpected scout frame")
    if parity["job_id"] != "59112590" or parity["coverage"] < .99:
        raise ValueError("prior Qwen parity source not qualified")
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        rating.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="low")) for row in frame["records"]]
    if max(lengths) + rating.MAX_TOKENS > 49152:
        raise ValueError("scout prompt exceeds judge context")
    prefill = 2 * sum(lengths)
    decode = 2 * len(lengths) * rating.MAX_TOKENS
    prior_decode = parity["total_generated_tokens"] / parity["generation_seconds"]
    stress_decode = prior_decode * .65 / 2.0
    stress_prefill = 19656.064021098042 * .5 / 2.0
    components = {
        "cold_load": parity["load_seconds"] * 1.25,
        "decode": decode / stress_decode * 1.25,
        "prefill": prefill / stress_prefill * 1.25,
        "shutdown": 196.0,
    }
    projected = sum(components.values())
    body = {
        "schema": "transition-v22-qwen-12-scout-price-v2",
        "job_id": os.environ["SLURM_JOB_ID"],
        "status": "PASS_NORMAL_1H_ONLY" if projected < 3600 else "HOLD_PRICE_EXCEEDS_1H",
        "frame_sha256": frame["sha256"],
        "rating_driver_sha256": hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rating.RUBRIC.read_bytes()).hexdigest(),
        "parity_sha256": parity["sha256"],
        "rows": len(lengths), "ratings": 2 * len(lengths),
        "prompt_tokens_exact_twice": prefill,
        "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
        "prompt_tokens_sorted": sorted(lengths),
        "max_decode_tokens": decode,
        "prior_short_context_decode_tokens_per_s": prior_decode,
        "stress_decode_tokens_per_s": stress_decode,
        "stress_prefill_tokens_per_s": stress_prefill,
        "stress_assumptions": {"context_slowdown": 2.0, "rate_factor": .65,
                               "repeat_work": 1.25, "cold_load_factor": 1.25},
        "components_seconds": components,
        "projected_wall_seconds": projected,
        "debug_30m_gate": "PASS" if projected <= 1500 else "FAIL",
        "normal_1h_gpu_hour_ceiling": 2.0,
        "projected_gpu_hours": 2 * projected / 3600,
        "interpretation": "All-capped 1024-token pricing with 2x long-context slowdown stress. Scout directly measures full-prefix load/prefill/decode and parse coverage; it does not estimate detector accuracy. A 30-minute debug slot requires additional margin and is not assumed safe.",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": rating.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "projected_wall_seconds": projected,
                      "projected_gpu_hours": body["projected_gpu_hours"]}), flush=True)


if __name__ == "__main__":
    main()
