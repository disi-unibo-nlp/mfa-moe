"""Twelve-row, context-stratified Qwen throughput and parsing scout.

The model receives only the original problem and already emitted reasoning
through the trigger. Later sentences are absent from the sealed reader frame.
These are two independent same-model draws, not human truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FRAME = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_V22_QWEN_12_SCOUT_FRAME.json"
RUBRIC = REPO / "report/experimental-resume-v1/TRANSITION_RUBRIC_v0.1.md"
PRICE = REPO / "report/experimental-resume-v1/TRANSITION_V22_QWEN_12_SCOUT_PRICE_v2.json"
MODEL = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
BATCH = 12
MAX_TOKENS = 1024
CRITERIA = {
    "candidate_to_verify": "The current sentence states a complete proposed value or answer, not yet substantively checked. A computed intermediate value can count. An unfinished expression or mere intention to compute cannot.",
    "approach_to_commit": "The current sentence tentatively names a specific method, construction, case split, formula, or representation and the operation it would perform. General uncertainty, an already executed step without a tentative approach, or a bare 'try another way' cannot.",
    "failed_check_to_revise": "The current sentence visibly computes a contradiction, invalid value, or violated original condition, with enough computation or constraint detail to see what failed. An ordinary equation or simplification is not a failed check.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed frame: {path}")
    return value


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages(row):
    if set(row["reader_input"]) != {"problem", "emitted_prefix", "triggering_sentence"}:
        raise ValueError("reader fixture field set changed")
    start = {key: row["reader_input"][key] for key in
             ("problem", "emitted_prefix", "triggering_sentence")}
    if not all(isinstance(v, str) for v in start.values()):
        raise ValueError("prefix text is not a string")
    if not start["emitted_prefix"].rstrip().endswith(start["triggering_sentence"].rstrip()):
        raise ValueError("triggering sentence is not at the end of emitted prefix")
    system = ("Independently rate a *starting condition* in visible mathematical reasoning. "
              "Use only the original problem and emitted reasoning prefix below. "
              "The current sentence is repeated for focus and already appears at the end of that prefix. "
              "Do not infer future text, correctness, or hidden thoughts. "
              'Return exactly one JSON object: {"start": true} or {"start": false}. '
              "When evidence is ambiguous, use false. No explanation.\nCriterion: " +
              CRITERIA[row["transition"]])
    user = ("Original problem:\n" + start["problem"] +
            "\n\nAlready emitted reasoning prefix:\n" + start["emitted_prefix"] +
            "\n\nCurrent sentence:\n" + start["triggering_sentence"] + "\n\nJSON:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_rating(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and set(value) == {"start"} and type(value["start"]) is bool else None


def rating_seed(uid, reader):
    return int(digest(["transition-v22-qwen-scout-rating-v1", uid, reader])[:8], 16) % 2_000_000_000


def atomic_json(path, body):
    if path.exists():
        raise FileExistsError(path)
    value = {**body, "sha256": digest(body)}
    temp = path.with_name(path.name + ".part-" + os.environ["SLURM_JOB_ID"])
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return value


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("start-only semantic audit requires GPU Slurm")
    frame = sealed(FRAME)
    if frame["schema"] != "transition-v22-qwen-12-scout-frame-v1" or frame["families"] != 12 or frame["rows"] != 12:
        raise ValueError("not the frozen discovery-only v2.1 frame")
    rows = frame["records"]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("duplicate rating UID")
    price = sealed(PRICE)
    if (price["schema"] != "transition-v22-qwen-12-scout-price-v2"
            or price["status"] != "PASS_NORMAL_1H_ONLY" or price["frame_sha256"] != frame["sha256"]
            or price["rating_driver_sha256"] != file_sha(Path(__file__))
            or price["rubric_sha256"] != file_sha(RUBRIC)
            or price["rows"] != len(rows) or price["ratings"] != 2 * len(rows)
            or price["normal_1h_gpu_hour_ceiling"] != 2.0):
        raise ValueError("sealed GPU price no longer binds scout code/frame/rubric")
    binding_body = {
        "schema": "transition-v22-qwen-scout-rating-binding-v1",
        "frame_sha256": frame["sha256"],
        "driver_sha256": file_sha(Path(__file__)),
        "rubric_sha256": file_sha(RUBRIC),
        "model_snapshot": str(MODEL), "model_revision": MODEL.name,
        "batch_size": BATCH, "max_tokens_per_rating": MAX_TOKENS,
        "readers": 2,
        "sampler": {"temperature": .2, "top_p": .95,
                    "thinking": True, "reasoning_effort": "low"},
        "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_sentence"],
        "scope": "12 independent previously unrated discovery windows stratified by native context length and v2.2 screen status; two same-model draws; throughput/parse pilot, not detector precision",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "WRITER.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_path = args.out / "BINDING.json"
    if binding_path.exists():
        binding = sealed(binding_path)
        if binding != {**binding_body, "sha256": digest(binding_body)}:
            raise ValueError("rating output directory rebound to changed input/code")
    else:
        binding = atomic_json(binding_path, binding_body)
    (args.out / "batches").mkdir(exist_ok=True)
    (args.out / "assignments").mkdir(exist_ok=True)
    pending = []
    for start in range(0, len(rows), BATCH):
        path = args.out / "batches" / f"{start:06d}.json"
        if path.exists():
            part = sealed(path)
            if part["binding_sha256"] != binding["sha256"] or part["start"] != start or [
                r["uid"] for r in part["records"]] != [r["uid"] for r in rows[start:start+BATCH]]:
                raise ValueError("saved rating batch UID/binding differs")
            if any(not (args.out / "assignments" / f"{row['uid']}-reader{reader}.json").exists()
                   for row in rows[start:start+BATCH] for reader in (0, 1)):
                raise ValueError("saved batch lacks per-assignment attempt receipts")
        else:
            pending.append((start, path))
    if pending:
        from vllm import LLM, SamplingParams
        load_started = time.monotonic()
        model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                    dtype="bfloat16", kv_cache_dtype="bfloat16", max_model_len=49152,
                    max_num_seqs=32, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config="vllm", language_model_only=True,
                    attention_config={"backend": "FLASH_ATTN"})
        load_seconds = time.monotonic() - load_started
        started = time.monotonic()
        for start, path in pending:
            block = rows[start:start+BATCH]
            conversations = [messages(row) for row in block]
            per_reader = []
            timing = []
            for reader in (0, 1):
                reader_path = args.out / "batches" / f"{start:06d}-reader{reader}.json"
                if reader_path.exists():
                    saved = sealed(reader_path)
                    if saved["binding_sha256"] != binding["sha256"] or saved["start"] != start or saved["reader"] != reader or [x["uid"] for x in saved["records"]] != [x["uid"] for x in block]:
                        raise ValueError("saved reader batch differs")
                    if any(not (args.out / "assignments" / f"{row['uid']}-reader{reader}.json").exists()
                           for row in block):
                        raise ValueError("saved reader results lack assignment receipts")
                    per_reader.append([x["result"] for x in saved["records"]])
                    timing.append(saved["timing"])
                    continue
                for row in block:
                    assignment_path = args.out / "assignments" / f"{row['uid']}-reader{reader}.json"
                    if assignment_path.exists():
                        raise RuntimeError(f"incomplete attempted rating requires manual adjudication: {assignment_path}")
                for row in block:
                    atomic_json(args.out / "assignments" / f"{row['uid']}-reader{reader}.json",
                                {"schema": "transition-v22-qwen-scout-assignment-v1",
                                 "binding_sha256": binding["sha256"], "uid": row["uid"],
                                 "reader": reader, "start": start,
                                 "state": "attempted_before_generation",
                                 "job_id": os.environ["SLURM_JOB_ID"]})
                params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                         seed=rating_seed(row["uid"], reader)) for row in block]
                reader_started = time.monotonic()
                try:
                    outputs = model.chat(conversations, sampling_params=params,
                                         chat_template_kwargs={"enable_thinking": True,
                                                               "reasoning_effort": "low"},
                                         use_tqdm=False)
                except Exception as exc:
                    for row in block:
                        atomic_json(args.out / "assignments" / f"{row['uid']}-reader{reader}-failure.json",
                                    {"schema": "transition-v22-qwen-scout-failure-v1",
                                     "binding_sha256": binding["sha256"], "uid": row["uid"],
                                     "reader": reader, "start": start,
                                     "job_id": os.environ["SLURM_JOB_ID"],
                                     "exception_type": type(exc).__name__,
                                     "exception_message": str(exc)[:500]})
                    raise
                reader_elapsed = time.monotonic() - reader_started
                if len(outputs) != len(block):
                    for row in block:
                        atomic_json(args.out / "assignments" / f"{row['uid']}-reader{reader}-failure.json",
                                    {"schema": "transition-v22-qwen-scout-failure-v1",
                                     "binding_sha256": binding["sha256"], "uid": row["uid"],
                                     "reader": reader, "start": start,
                                     "job_id": os.environ["SLURM_JOB_ID"],
                                     "exception_type": "OutputCountMismatch",
                                     "exception_message": f"got {len(outputs)} for {len(block)} assignments"})
                    raise ValueError("start rating model output count differs")
                prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
                generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                one_timing = {"reader": reader, "wall_seconds": reader_elapsed,
                              "prompt_tokens": prompt_tokens,
                              "generated_tokens": generated_tokens}
                one_results = [{"rating": parse_rating(o.outputs[0].text),
                                "finish_reason": o.outputs[0].finish_reason,
                                "generated_tokens": len(o.outputs[0].token_ids),
                                "raw_completion": o.outputs[0].text}
                               for o in outputs]
                atomic_json(reader_path,
                            {"schema": "transition-v22-qwen-scout-reader-batch-v1",
                             "binding_sha256": binding["sha256"], "start": start,
                             "reader": reader, "timing": one_timing,
                             "records": [{"uid": row["uid"], "result": one_results[i]}
                                         for i, row in enumerate(block)]})
                timing.append(one_timing)
                per_reader.append(one_results)
            records = [{"uid": row["uid"], "transition": row["transition"],
                        "readers": [per_reader[0][i], per_reader[1][i]]}
                       for i, row in enumerate(block)]
            atomic_json(path, {"schema": "transition-v22-qwen-scout-rating-batch-v1",
                               "binding_sha256": binding["sha256"],
                               "start": start, "load_seconds": load_seconds,
                               "reader_timings": timing, "records": records})
            print(json.dumps({"complete": min(start+BATCH, len(rows)), "of": len(rows),
                              "load_seconds": load_seconds,
                              "reader_timings": timing,
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
    complete = []
    saved_timings = []
    for start in range(0, len(rows), BATCH):
        part = sealed(args.out / "batches" / f"{start:06d}.json")
        complete.extend(part["records"])
        saved_timings.append({"load_seconds": part["load_seconds"],
                              "reader_timings": part["reader_timings"]})
    if [r["uid"] for r in complete] != [r["uid"] for r in rows]:
        raise ValueError("rating completion UID coverage differs")
    counts = Counter()
    for row in complete:
        counts["rows"] += 1
        for reader in row["readers"]:
            counts["ratings"] += 1
            counts["generated_tokens"] += reader["generated_tokens"]
            if reader["rating"] is not None and reader["finish_reason"] == "stop":
                counts["parsed_stop"] += 1
        a, b = row["readers"]
        if all(x["rating"] is not None and x["finish_reason"] == "stop" for x in (a, b)):
            counts["pair_covered"] += 1
            counts["start_agree"] += a["rating"]["start"] == b["rating"]["start"]
    summary = {"schema": "transition-v22-qwen-scout-rating-summary-v1",
               "binding_sha256": binding["sha256"], "counts": dict(counts),
               "timings": saved_timings,
               "interpretation": "Context-matched startup, prefill, decode and parse scout; only 12 previously unrated discovery families, not semantic detector precision or human truth"}
    summary_path = args.out / "SUMMARY.json"
    if summary_path.exists():
        if sealed(summary_path) != {**summary, "sha256": digest(summary)}:
            raise ValueError("saved summary differs from resumed results")
    else:
        atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(args.out / "SUMMARY.json"),
                      "counts": summary["counts"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
