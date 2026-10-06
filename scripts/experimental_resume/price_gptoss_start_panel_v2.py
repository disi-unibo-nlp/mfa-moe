"""Exact tokenizer and measured-runtime price for frozen GPT-OSS panel."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
OUT = REPO / "report/experimental-resume-v1/GPTOSS_START_PANEL_PRICE_v2.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_gptoss_start_panel_v2 as panel
import rate_gptoss_start_pilot as pilot
from price_transition_ratings_v3 import count_prompt_tokens


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("GPT-OSS tokenizer price requires allocated CPU Slurm")
    from transformers import AutoTokenizer
    frame = panel.sealed(panel.FRAME)
    pilot_binding = panel.sealed(BASE / "ratings-gptoss-start-pilot-v1-18a7f804-eda3cc45/BINDING.json")
    pilot_results = panel.sealed(BASE / "ratings-gptoss-start-pilot-v1-18a7f804-eda3cc45/RESULTS.json")
    posthoc = panel.sealed(REPO / "report/experimental-resume-v1/GPTOSS_START_PILOT_POSTHOC_PARSE_v2.json")
    if (frame["schema"] != "gptoss-start-panel-v2" or len(frame["records"]) != 27
            or pilot_results["binding_sha256"] != pilot_binding["sha256"]
            or posthoc["result_sha256"] != pilot_results["sha256"]
            or posthoc["parsed_natural_stops"] != 4):
        raise ValueError("GPT-OSS pilot or panel qualification changed")
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        pilot.messages(row), tokenize=True, add_generation_prompt=True,
        reasoning_effort="low")) for row in frame["records"]]
    if max(lengths) + panel.MAX_TOKENS > 16384:
        raise ValueError("frozen GPT-OSS panel exceeds vLLM model context")
    measured = pilot_results["load_seconds"] + pilot_results["generation_seconds"]
    # Pilot: one cold load, 12,371 prompt tokens, four responses. The 30-minute
    # slot leaves >4x this measured load+generation plus queue/start overhead.
    if measured >= 360 or sum(lengths) > 120000:
        raise ValueError("measured pilot scaling does not support debug ceiling")
    body = {
        "schema": "gptoss-start-panel-price-v2", "job_id": os.environ["SLURM_JOB_ID"],
        "status": "PASS_COMPLETE_1_GPUH",
        "frame_sha256": frame["sha256"],
        "pilot_binding_sha256": pilot_binding["sha256"],
        "pilot_result_sha256": pilot_results["sha256"],
        "pilot_posthoc_parse_sha256": posthoc["sha256"],
        "rating_driver_sha256": panel.file_sha(panel.__file__),
        "parser_sha256": panel.file_sha(panel.PARSER),
        "prompt_driver_sha256": panel.file_sha(panel.PROMPT),
        "model_cache": pilot.model_cache(),
        "rows": 27, "prompt_tokens_exact": sum(lengths),
        "prompt_tokens_per_row": lengths,
        "max_prompt_tokens": max(lengths),
        "max_decode_tokens": 27 * panel.MAX_TOKENS,
        "pilot_measured": {"cold_load_seconds": pilot_results["load_seconds"],
                            "generation_seconds_4": pilot_results["generation_seconds"],
                            "prompt_tokens_4": sum(r["prompt_tokens"] for r in pilot_results["records"]),
                            "generated_tokens_4": sum(r["generated_tokens"] for r in pilot_results["records"])},
        "allocation_plan": {"first_gpus": 1, "first_qos": "boost_qos_dbg",
                            "first_wall_seconds": 1800, "first_gpu_hour_ceiling": .5,
                            "single_recovery_gpu_hour_ceiling": .5,
                            "combined_gpu_hour_ceiling": 1.0},
        "price_logic": "one measured GPT-OSS cold load and seven checkpointed batches of at most four ratings; exact tokenizer prefill, all-capped 512 decode, 30-minute first slot plus separately checked recovery slot. No continuation cap is reduced.",
        "interpretation": "Prospectively frozen different-family LLM measurement audit; not human truth or semantic steering",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": panel.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "prompt_tokens": sum(lengths),
                      "max_decode_tokens": body["max_decode_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
