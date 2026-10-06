"""Four-row GPT-OSS different-family full-prefix start-rating parity pilot.

This is an LLM measurement audit, not independent human truth. The four-row
frame is frozen only after Qwen's reader disagreements and controls are known.
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
sys.path.insert(0, str(REPO))
from scripts.experimental_resume.rate_transition_v22_qwen_scout import messages as prefix_messages
FRAME = BASE / "GPTOSS_START_PILOT_FRAME_v1.json"
PRICE = REPO / "report/experimental-resume-v1/GPTOSS_START_PILOT_PRICE_v1.json"
QWEN_PROMPT_DRIVER = REPO / "scripts/experimental_resume/rate_transition_v22_qwen_scout.py"
MODEL = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")
MAX_TOKENS = 512


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def model_cache():
    index = MODEL / "model.safetensors.index.json"
    config = MODEL / "config.json"
    tokenizer = MODEL / "tokenizer.json"
    template = MODEL / "chat_template.jinja"
    for path in (index, config, tokenizer, template):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = json.loads(config.read_text())
    if cfg.get("architectures") != ["GptOssForCausalLM"] or cfg.get("quantization_config", {}).get("quant_method") != "mxfp4":
        raise ValueError("unexpected GPT-OSS model architecture or quantization")
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = {name: (MODEL / name).stat().st_size for name in set(weight_map.values())}
    if len(shards) < 2 or any(size < 100_000_000 for size in shards.values()):
        raise ValueError("GPT-OSS weight cache incomplete")
    return {"snapshot": str(MODEL), "config_sha256": file_sha(config),
            "index_sha256": file_sha(index), "tokenizer_sha256": file_sha(tokenizer),
            "template_sha256": file_sha(template),
            "weight_shard_bytes": dict(sorted(shards.items()))}


def messages(row):
    if set(row) != {"uid", "family", "transition", "reader_input", "prefix_tokens"}:
        raise ValueError("pilot row has unexpected fields")
    if row["transition"] not in {"candidate_to_verify", "approach_to_commit", "failed_check_to_revise"}:
        raise ValueError("pilot transition changed")
    return prefix_messages(row)


def parse_rating(text):
    if "<|channel|>final<|message|>" in text:
        text = text.rsplit("<|channel|>final<|message|>", 1)[1]
    text = text.split("<|return|>", 1)[0].split("<|end|>", 1)[0]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and set(value) == {"start"} and type(value["start"]) is bool else None


def atomic_json(path, body):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".part-" + os.environ["SLURM_JOB_ID"])
    value = {**body, "sha256": digest(body)}
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return value


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("GPT-OSS pilot requires allocated GPU Slurm")
    frame, price = sealed(FRAME), sealed(PRICE)
    cache = model_cache()
    rows = frame["records"]
    if (frame["schema"] != "gptoss-start-pilot-frame-v1" or len(rows) != 4
            or len({row["family"] for row in rows}) != 4):
        raise ValueError("pilot frame not frozen from four distinct families")
    if (price["schema"] != "gptoss-start-pilot-price-v1"
            or price["status"] != "PASS_BOUNDED_PARITY_PILOT"
            or price["frame_sha256"] != frame["sha256"]
            or price["driver_sha256"] != file_sha(Path(__file__))
            or price["qwen_prompt_driver_sha256"] != file_sha(QWEN_PROMPT_DRIVER)
            or price["model_cache"] != cache
            or price["rows"] != 4 or price["max_decode_tokens"] != 4 * MAX_TOKENS):
        raise ValueError("price no longer binds GPT-OSS pilot inputs/code/model")
    conversations = [messages(row) for row in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "WRITER.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_body = {"schema": "gptoss-start-pilot-binding-v1",
                    "frame_sha256": frame["sha256"], "price_sha256": price["sha256"],
                    "driver_sha256": file_sha(Path(__file__)),
                    "qwen_prompt_driver_sha256": file_sha(QWEN_PROMPT_DRIVER),
                    "model_cache": cache, "max_tokens": MAX_TOKENS,
                    "scope": "one different-model-family reader per four Qwen disagreement/control prefixes; parity/parse only"}
    binding_path = args.out / "BINDING.json"
    if binding_path.exists():
        binding = sealed(binding_path)
        if binding != {**binding_body, "sha256": digest(binding_body)}:
            raise ValueError("GPT-OSS output directory rebound")
    else:
        binding = atomic_json(binding_path, binding_body)
    result_path = args.out / "RESULTS.json"
    if result_path.exists():
        result = sealed(result_path)
        if result["binding_sha256"] != binding["sha256"] or [r["uid"] for r in result["records"]] != [r["uid"] for r in rows]:
            raise ValueError("saved GPT-OSS result differs")
        print(json.dumps({"resume": "complete", "result": str(result_path)}), flush=True)
        return
    assignments = args.out / "assignments"
    assignments.mkdir(exist_ok=True)
    if any((assignments / f"{row['uid']}.json").exists() for row in rows):
        raise RuntimeError("incomplete attempted GPT-OSS pilot requires manual adjudication")
    from vllm import LLM, SamplingParams
    load_started = time.monotonic()
    model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=1,
                dtype="bfloat16", max_model_len=16384, max_num_seqs=4,
                max_num_batched_tokens=16384, gpu_memory_utilization=.8,
                enforce_eager=True, generation_config="vllm")
    load_seconds = time.monotonic() - load_started
    for row in rows:
        atomic_json(assignments / f"{row['uid']}.json",
                    {"schema": "gptoss-start-pilot-assignment-v1",
                     "binding_sha256": binding["sha256"], "uid": row["uid"],
                     "state": "attempted_before_generation", "job_id": os.environ["SLURM_JOB_ID"]})
    params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                             seed=int(digest(["gptoss-pilot-v1", row["uid"]])[:8], 16)) for row in rows]
    started = time.monotonic()
    try:
        outputs = model.chat(conversations, sampling_params=params,
                             chat_template_kwargs={"reasoning_effort": "low"}, use_tqdm=False)
    except Exception as exc:
        for row in rows:
            atomic_json(assignments / f"{row['uid']}-failure.json",
                        {"schema": "gptoss-start-pilot-failure-v1",
                         "binding_sha256": binding["sha256"], "uid": row["uid"],
                         "exception_type": type(exc).__name__,
                         "exception_message": str(exc)[:500],
                         "job_id": os.environ["SLURM_JOB_ID"]})
        raise
    if len(outputs) != 4:
        raise ValueError("GPT-OSS pilot output count differs")
    results = [{"uid": row["uid"], "rating": parse_rating(out.outputs[0].text),
                "raw_completion": out.outputs[0].text,
                "finish_reason": out.outputs[0].finish_reason,
                "prompt_tokens": len(out.prompt_token_ids),
                "generated_tokens": len(out.outputs[0].token_ids)}
               for row, out in zip(rows, outputs)]
    atomic_json(result_path,
                {"schema": "gptoss-start-pilot-results-v1",
                 "binding_sha256": binding["sha256"],
                 "load_seconds": load_seconds,
                 "generation_seconds": time.monotonic() - started,
                 "records": results})
    print(json.dumps({"result": str(result_path), "load_seconds": load_seconds,
                      "parsed_stop": sum(r["rating"] is not None and r["finish_reason"] == "stop" for r in results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
