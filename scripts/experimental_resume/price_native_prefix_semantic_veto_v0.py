"""Exact prompt counts and provisional native-model side-judge resource price."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
OUT = REPO / "report/experimental-resume-v1/NATIVE_PREFIX_SEMANTIC_VETO_PRICE_v0.json"
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
import native_prefix_semantic_veto_v0 as veto
from price_transition_ratings_v3 import count_prompt_tokens


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("native semantic-veto prompt pricing requires CPU Slurm")
    from transformers import AutoTokenizer
    full = veto.sealed(veto.FRAME)
    scout = veto.sealed(veto.BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json")
    if (full["schema"] != "transition-v22-full-prefix-start-frame-v1"
            or len(full["records"]) != 372
            or scout["schema"] != "transition-v22-qwen-12-scout-frame-v1"
            or len(scout["records"]) != 12
            or scout["source_frame_sha256"] != full["sha256"]):
        raise ValueError("native side-judge pricing frame binding differs")
    cache = veto.cache_receipt()
    tokenizer = AutoTokenizer.from_pretrained(veto.MODEL, local_files_only=True)
    def lengths(rows):
        return [count_prompt_tokens(tokenizer.apply_chat_template(
            veto.messages(row), tokenize=True, add_generation_prompt=True,
            enable_thinking=False)) for row in rows]
    full_lengths = lengths(full["records"])
    scout_lengths = lengths(scout["records"])
    max_context = max(full_lengths) + veto.MAX_TOKENS
    if max_context > 16384:
        raise ValueError("native semantic-veto prompt exceeds declared 16k context")
    assumptions = {"cold_load_seconds": 900.0, "shutdown_seconds": 196.0,
                   "prefill_tokens_per_s": 2500.0, "decode_tokens_per_s": 30.0,
                   "work_multiplier": 1.25}
    def cost(values):
        prompt = 2 * sum(values)
        decode = 2 * len(values) * veto.MAX_TOKENS
        seconds = (assumptions["cold_load_seconds"] + assumptions["shutdown_seconds"] +
                   assumptions["work_multiplier"] *
                   (prompt / assumptions["prefill_tokens_per_s"] +
                    decode / assumptions["decode_tokens_per_s"]))
        return {"rows": len(values), "ratings": 2 * len(values),
                "prompt_tokens_exact_twice": prompt,
                "max_decode_tokens": decode,
                "projected_wall_seconds": seconds,
                "projected_GPU_h": 2 * seconds / 3600}
    scout_price, full_price = cost(scout_lengths), cost(full_lengths)
    body = {"schema": "native-prefix-semantic-veto-price-v0",
            "job_id": os.environ["SLURM_JOB_ID"],
            "status": "HOLD_NATIVE_SCOUT" if scout_price["projected_GPU_h"] <= 2 and full_price["projected_GPU_h"] <= 6 else "HOLD_PRICE_EXCEEDS_PROPOSAL",
            "full_frame_sha256": full["sha256"], "scout_frame_sha256": scout["sha256"],
            "prompt_driver_sha256": hashlib.sha256(Path(veto.__file__).read_bytes()).hexdigest(),
            "model_cache": cache, "max_tokens_per_rating": veto.MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "full_prompt_tokens_min": min(full_lengths),
            "full_prompt_tokens_max": max(full_lengths),
            "full_prompt_tokens_sum": sum(full_lengths),
            "scout_prompt_tokens_min": min(scout_lengths),
            "scout_prompt_tokens_max": max(scout_lengths),
            "scout_prompt_tokens_sum": sum(scout_lengths),
            "declared_context_limit": 16384,
            "assumptions": assumptions,
            "scout": scout_price, "full": full_price,
            "proposed_ceilings_GPU_h": {"scout": 2.0, "full": 6.0},
            "interpretation": "Provisional stress price for unsteered native Qwen3.6 side judgments, two seeded draws per prefix. Native semantic-judge cold load, prefill, parse and decode have not been measured with this prompt. Scout must pass before full-stage submission. Detection effects are compared only after independent Qwen3.8 full-prefix audit; side-request tokens are part of controller utility cost."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": veto.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": body["status"],
                      "scout_GPU_h": scout_price["projected_GPU_h"],
                      "full_GPU_h": full_price["projected_GPU_h"]}), flush=True)


if __name__ == "__main__":
    main()
