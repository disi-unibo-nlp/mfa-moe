"""Exact prompt and one-A100 allocation price for GPT-OSS parity pilot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
OUT = REPO / "report/experimental-resume-v1/GPTOSS_START_PILOT_PRICE_v1.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import rate_gptoss_start_pilot as pilot
from price_transition_ratings_v3 import count_prompt_tokens


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("GPT-OSS exact token pricing requires CPU Slurm")
    from transformers import AutoTokenizer
    frame = pilot.sealed(pilot.FRAME)
    if frame["schema"] != "gptoss-start-pilot-frame-v1" or len(frame["records"]) != 4:
        raise ValueError("not the frozen four-row parity frame")
    cache = pilot.model_cache()
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, local_files_only=True)
    prompt_lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        pilot.messages(row), tokenize=True, add_generation_prompt=True,
        reasoning_effort="low")) for row in frame["records"]]
    if max(prompt_lengths) + pilot.MAX_TOKENS > 16384:
        raise ValueError("pilot context exceeds the declared 16,384-token vLLM limit")
    body = {"schema": "gptoss-start-pilot-price-v1", "job_id": os.environ["SLURM_JOB_ID"],
            "status": "PASS_BOUNDED_PARITY_PILOT",
            "frame_sha256": frame["sha256"],
            "driver_sha256": pilot.file_sha(Path(pilot.__file__)),
            "qwen_prompt_driver_sha256": pilot.file_sha(pilot.QWEN_PROMPT_DRIVER),
            "model_cache": cache, "rows": 4,
            "prompt_tokens_exact": sum(prompt_lengths),
            "prompt_tokens_per_row": prompt_lengths,
            "max_decode_tokens": 4 * pilot.MAX_TOKENS,
            "allocation": {"gpus": 1, "gpu_type": "A100", "wall_seconds": 3600,
                           "gpu_hour_ceiling": 1.0, "qos": "normal"},
            "price_logic": "single cold load + four prompts, exact tokenizer prefill and all-capped 512-token decode; GPT-OSS runtime unknown, so this is a bounded qualification allocation, not a throughput projection. At walltime the job stops and cannot be extended automatically.",
            "interpretation": "Different-model-family start rubric and parse parity on two Qwen reader disagreements and two controls; not an accuracy estimate or human adjudication"}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": pilot.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "prompt_tokens": sum(prompt_lengths),
                      "max_decode_tokens": body["max_decode_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
