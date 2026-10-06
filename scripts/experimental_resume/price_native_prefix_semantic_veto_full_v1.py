"""Seal measured-scout-qualified full native prefix-veto GPU resource price."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import native_prefix_semantic_veto_v0 as veto

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE_PRICE = REPO / "report/experimental-resume-v1/NATIVE_PREFIX_SEMANTIC_VETO_PRICE_v0.json"
DRIVER = REPO / "scripts/experimental_resume/rate_native_prefix_semantic_veto_full_v0.py"
SCOUT = veto.BASE / "ratings-native-veto-scout-v0-d8f50b7e-8d2b37d2"
OUT = REPO / "report/experimental-resume-v1/NATIVE_PREFIX_SEMANTIC_VETO_FULL_PRICE_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = veto.sealed(BASE_PRICE)
    frame = veto.sealed(veto.FRAME)
    binding = veto.sealed(SCOUT / "BINDING.json")
    summary = veto.sealed(SCOUT / "SUMMARY.json")
    if (base["schema"] != "native-prefix-semantic-veto-price-v0"
            or base["status"] != "HOLD_NATIVE_SCOUT"
            or base["full_frame_sha256"] != frame["sha256"]
            or base["prompt_driver_sha256"] != sha(Path(veto.__file__))
            or base["model_cache"] != veto.cache_receipt()
            or base["declared_context_limit"] != 16384
            or base["full_prompt_tokens_max"] + veto.MAX_TOKENS > 16384
            or binding["frame_sha256"] != base["scout_frame_sha256"]
            or binding["price_sha256"] != base["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or summary["counts"].get("ratings") != 24
            or summary["counts"].get("parsed") != 24
            or summary["counts"].get("natural") != 24):
        raise ValueError("native veto scout, frame or model qualification differs")
    batches = [veto.sealed(SCOUT / "batches" / f"reader{i}.json") for i in range(2)]
    if any(part["binding_sha256"] != binding["sha256"] for part in batches):
        raise ValueError("native veto scout reader batch binding differs")
    observed = {
        "load_seconds": veto.sealed(SCOUT / "loads/59193843.json")["load_seconds"],
        "generation_seconds": sum(part["timing"]["wall_seconds"] for part in batches),
        "prefill_tokens": sum(part["timing"]["prompt_tokens"] for part in batches),
        "generated_tokens": sum(part["timing"]["generated_tokens"] for part in batches),
        "parsed": summary["counts"]["parsed"],
        "natural": summary["counts"]["natural"]}
    if observed["prefill_tokens"] != base["scout"]["prompt_tokens_exact_twice"]:
        raise ValueError("native scout exact prompt token parity differs")
    assumptions = base["assumptions"]
    one_load = assumptions["cold_load_seconds"]
    one_shutdown = assumptions["shutdown_seconds"]
    work = assumptions["work_multiplier"] * (
        base["full"]["prompt_tokens_exact_twice"] / assumptions["prefill_tokens_per_s"]
        + base["full"]["max_decode_tokens"] / assumptions["decode_tokens_per_s"])
    full_one_wall = one_load + work + one_shutdown
    full_two_wall = 2 * (one_load + one_shutdown) + work
    first_gpu_h = 2 * full_one_wall / 3600
    complete_gpu_h = 2 * full_two_wall / 3600
    status = "PASS_COMPLETE_8_GPUH" if first_gpu_h <= 4 and complete_gpu_h <= 8 else "HOLD_PRICE_EXCEEDS_CEILING"
    body = {
        "schema": "native-prefix-semantic-veto-full-price-v1",
        "status": status,
        "frame_sha256": frame["sha256"],
        "source_price_sha256": base["sha256"],
        "scout_binding_sha256": binding["sha256"],
        "scout_summary_sha256": summary["sha256"],
        "scout_reader_batch_shas": [part["sha256"] for part in batches],
        "rating_driver_sha256": sha(DRIVER),
        "prompt_driver_sha256": sha(Path(veto.__file__)),
        "model_cache": base["model_cache"],
        "rows": 372, "ratings": 744, "max_tokens_per_rating": veto.MAX_TOKENS,
        "prompt_tokens_exact_twice": base["full"]["prompt_tokens_exact_twice"],
        "max_decode_tokens": base["full"]["max_decode_tokens"],
        "prompt_tokens_max": base["full_prompt_tokens_max"],
        "context_limit": 16384, "scout_observed": observed,
        "assumptions": assumptions,
        "full_one_load_projected_GPU_h": first_gpu_h,
        "complete_with_one_recovery_projected_GPU_h": complete_gpu_h,
        "allocation_plan": {"first_slice_GPU_h_ceiling": 4.0,
                            "one_recovery_GPU_h_ceiling": 4.0,
                            "combined_GPU_h_ceiling": 8.0,
                            "each_job": "two A100, at most two hours, normal QoS"},
        "interpretation": "Full frozen 372-window unsteered native-model side-rating audit; 24/24 short-scout responses parsed and naturally stopped. Maximum64 output tokens, exact prompts, pessimistic30 decode tokens/s and2500 prefill tokens/s, 900s cold loads and196s shutdowns. This measures detector agreement and side-request burden, not human truth or causal steering."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": veto.digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "status": status,
                      "first_GPU_h": first_gpu_h, "complete_GPU_h": complete_gpu_h}))


if __name__ == "__main__":
    main()
