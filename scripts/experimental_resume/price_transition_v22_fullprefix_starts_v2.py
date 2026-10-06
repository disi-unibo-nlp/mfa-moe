"""Complete-stage full-prefix rating price gated by measured 12-prefix scout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery"
SCOUT = BASE / "ratings-v22-qwen-scout-d8f50b7e-06f8fabd"
SCOUT_FRAME = BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json"
PARITY = ROOT / "steering-v1/runs/routing-control-v1/dense-judge-parity/results-8e939bee-2341bac1/PARITY.json"
OUT = REPO / "report/experimental-resume-v1/TRANSITION_V22_FULL_PREFIX_START_RATING_PRICE_v2.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_transition_v22_fullprefix_starts_v2 as rating
from price_transition_ratings_v3 import count_prompt_tokens


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != rating.digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("exact full-stage pricing requires CPU Slurm")
    from transformers import AutoTokenizer
    frame = sealed(rating.FRAME)
    scout_frame = sealed(SCOUT_FRAME)
    scout_binding = sealed(SCOUT / "BINDING.json")
    scout_summary = sealed(SCOUT / "SUMMARY.json")
    scout_batch = sealed(SCOUT / "batches/000000.json")
    parity = sealed(PARITY)
    if (frame["schema"] != "transition-v22-full-prefix-start-frame-v1" or len(frame["records"]) != 372
            or scout_frame["schema"] != "transition-v22-qwen-12-scout-frame-v1"
            or scout_frame["source_frame_sha256"] != frame["sha256"]
            or scout_binding["frame_sha256"] != scout_frame["sha256"]
            or scout_summary["binding_sha256"] != scout_binding["sha256"]
            or scout_batch["binding_sha256"] != scout_binding["sha256"]
            or scout_summary["counts"].get("ratings") != 24
            or parity["job_id"] != "59112590"):
        raise ValueError("full frame, independent scout, or historical parity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        rating.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="low")) for row in frame["records"]]
    if max(lengths) + rating.MAX_TOKENS > 49152:
        raise ValueError("full-stage prompt exceeds declared context")
    n_ratings = 2 * len(lengths)
    prefill_tokens = 2 * sum(lengths)
    decode_tokens = n_ratings * rating.MAX_TOKENS
    reader_timings = scout_batch["reader_timings"]
    observed_generated = sum(t["generated_tokens"] for t in reader_timings)
    observed_generation_s = sum(t["wall_seconds"] for t in reader_timings)
    if observed_generated <= 0 or observed_generation_s <= 0:
        raise ValueError("scout lacks measured generation throughput")
    scout_tps = observed_generated / observed_generation_s
    prior_stress_tps = (parity["total_generated_tokens"] / parity["generation_seconds"]) * .65 / 2.0
    bounded_tps = min(prior_stress_tps, scout_tps * .8)
    stress_prefill_tps = 19656.064021098042 * .5 / 2.0
    load_s = scout_batch["load_seconds"] * 1.25 * 2
    decode_s = decode_tokens / bounded_tps * 1.25
    prefill_s = prefill_tokens / stress_prefill_tps * 1.25
    shutdown_s = 196.0 * 2
    total_gpu_h = 2 * (load_s + decode_s + prefill_s + shutdown_s) / 3600
    parsed = scout_summary["counts"].get("parsed_stop", 0)
    by_uid = {r["uid"]: r for r in scout_frame["records"]}
    long_parsed, long_total = 0, 0
    for row in scout_batch["records"]:
        if by_uid[row["uid"]]["prefix_tokens"] >= 4096:
            for reader in row["readers"]:
                long_total += 1
                long_parsed += reader["rating"] is not None and reader["finish_reason"] == "stop"
    parse_pass = parsed >= 22 and long_total == 8 and long_parsed >= 7
    status = ("PASS_COMPLETE_16_GPUH" if parse_pass and total_gpu_h <= 16.0
              else "HOLD_SCOUT_PARSE" if not parse_pass else "HOLD_PRICE_EXCEEDS_16_GPUH")
    body = {
        "schema": "transition-v22-fullprefix-start-rating-price-v2",
        "job_id": os.environ["SLURM_JOB_ID"], "status": status,
        "frame_sha256": frame["sha256"],
        "rating_driver_sha256": hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rating.RUBRIC.read_bytes()).hexdigest(),
        "scout_frame_sha256": scout_frame["sha256"],
        "scout_binding_sha256": scout_binding["sha256"],
        "scout_summary_sha256": scout_summary["sha256"],
        "scout_batch_sha256": scout_batch["sha256"],
        "parity_sha256": parity["sha256"],
        "rows": len(lengths), "ratings": n_ratings,
        "prompt_tokens_exact_twice": prefill_tokens,
        "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
        "max_decode_tokens": decode_tokens,
        "scout_observed_generated_tokens": observed_generated,
        "scout_observed_generation_seconds": observed_generation_s,
        "scout_observed_tokens_per_s_including_prefill": scout_tps,
        "bounded_decode_tokens_per_s": bounded_tps,
        "bounded_prefill_tokens_per_s": stress_prefill_tps,
        "scout_parse_gate": {"parsed_stop": parsed, "required": 22,
                             "long_parsed_stop": long_parsed, "long_total": long_total,
                             "long_required": 7, "pass": parse_pass},
        "components_seconds": {"two_cold_loads": load_s, "decode": decode_s,
                               "prefill": prefill_s, "two_shutdowns": shutdown_s},
        "complete_stage_projected_GPU_h": total_gpu_h,
        "allocation_plan": {"first_slice_GPU_h_max": 12.0,
                            "one_recovery_GPU_h_max": 4.0,
                            "combined_GPU_h_ceiling": 16.0},
        "interpretation": "All 744 assignments at 1024-token cap, exact full-prefix prompts, two cold loads, all prefill/recovery/shutdown. Decode rate is the lower of prior stressed parity and 0.8x context-matched scout effective token rate; this deliberately double-counts some prefill. Parsed coverage gate is qualification only, not semantic detector validation."
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": rating.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": status,
                      "complete_projected_GPU_h": total_gpu_h,
                      "scout_parsed_stop": parsed}), flush=True)


if __name__ == "__main__":
    main()
