"""Tokenize all 372 audit prefixes for a conservative GPT-OSS pilot bound."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
OUT = REPO / "report/experimental-resume-v1/GPTOSS_START_PILOT_PREPRICE_v3.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_gptoss_start_pilot as pilot
from price_transition_ratings_v3 import count_prompt_tokens


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("GPT-OSS preprice requires CPU Slurm")
    from transformers import AutoTokenizer
    source = pilot.sealed(pilot.BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json")
    if source["schema"] != "transition-v22-full-prefix-start-frame-v1" or len(source["records"]) != 372:
        raise ValueError("unexpected source frame")
    cache = pilot.model_cache()
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, local_files_only=True)
    lengths = []
    for row in source["records"]:
        visible = {k: row[k] for k in ("uid", "family", "transition", "reader_input", "prefix_tokens")}
        lengths.append(count_prompt_tokens(tokenizer.apply_chat_template(
            pilot.messages(visible), tokenize=True, add_generation_prompt=True,
            reasoning_effort="low")))
    body = {"schema": "gptoss-start-pilot-preprice-v3",
            "job_id": os.environ["SLURM_JOB_ID"], "status": "HOLD_SELECTED_FRAME",
            "source_frame_sha256": source["sha256"],
            "driver_sha256": pilot.file_sha(Path(pilot.__file__)),
            "qwen_prompt_driver_sha256": pilot.file_sha(pilot.QWEN_PROMPT_DRIVER),
            "model_cache": cache, "source_rows": 372,
            "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
            "prompt_tokens_sum": sum(lengths),
            "max_selected_four_prefill_tokens": 4 * max(lengths),
            "max_decode_tokens": 4 * pilot.MAX_TOKENS,
            "max_context_tokens": max(lengths) + pilot.MAX_TOKENS,
            "context_limit": 16384,
            "qualification_ceiling": {"gpu_type": "A100", "gpus": 1,
                                      "wall_seconds": 3600, "gpu_hours": 1.0},
            "interpretation": "Upper bound over all 372 eligible discovery full-prefix rows; no four-row ambiguity/control frame exists yet. GPT-OSS cold load and decoding are unmeasured; exact selected-frame CPU price and runtime parity gate remain required."}
    if body["max_context_tokens"] > body["context_limit"]:
        raise ValueError("at least one source context exceeds GPT-OSS pilot limit")
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": pilot.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "max_selected_four_prefill_tokens": body["max_selected_four_prefill_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
