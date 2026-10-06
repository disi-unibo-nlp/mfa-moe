"""Sealed 43-window earlier-cut Qwen candidate-start timing audit.

The model receives only the original problem and already emitted reasoning
through the exact token cut inside a triggering sentence. Its later clause is
absent from the reader frame. These are same-model draws, not human truth.
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
FRAME = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery/WITHIN_SENTENCE_START_AUDIT_FRAME_v1.json"
RUBRIC = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_START_RUBRIC_v1.md"
PRICE = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_START_RATING_PRICE_v1.json"
MODEL = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
BATCH = 16
MAX_TOKENS = 1024
CRITERION = ("The visible current fragment states a complete, concrete proposed value or answer "
             "that has not already been substantively checked in the visible prefix. A computed "
             "intermediate value may count. A premise/setup relation, partial algebraic chain, "
             "unfinished expression, check result, or already checked conclusion does not count.")


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
    if set(row["reader_input"]) != {"problem", "emitted_prefix", "triggering_fragment"}:
        raise ValueError("reader fixture field set changed")
    start = {key: row["reader_input"][key] for key in
             ("problem", "emitted_prefix", "triggering_fragment")}
    if not all(isinstance(v, str) for v in start.values()):
        raise ValueError("prefix text is not a string")
    if not start["emitted_prefix"].rstrip().endswith(start["triggering_fragment"].rstrip()):
        raise ValueError("triggering fragment is not at the end of emitted prefix")
    system = ("Independently rate a *starting condition* in visible mathematical reasoning. "
              "Use only the original problem and emitted reasoning prefix below. "
              "The current fragment is repeated for focus and already appears at the end of that prefix. "
              "Its original sentence may continue later; judge only what is visible now. "
              "Do not infer any later clause, future text, correctness, or hidden thoughts. "
              'Return exactly one JSON object: {"start": true} or {"start": false}. '
              "When evidence is ambiguous, use false. No explanation.\nCriterion: " +
              CRITERION)
    user = ("Original problem:\n" + start["problem"] +
            "\n\nAlready emitted reasoning prefix:\n" + start["emitted_prefix"] +
            "\n\nCurrent emitted fragment:\n" + start["triggering_fragment"] + "\n\nJSON:")
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
    return int(digest(["within-sentence-start-rating-v1", uid, reader])[:8], 16) % 2_000_000_000


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
        raise RuntimeError("within-sentence start audit requires GPU Slurm")
    frame = sealed(FRAME)
    if (frame["schema"] != "within-sentence-start-audit-frame-v1"
            or frame["families"] != 35 or frame["rows"] != 43
            or frame["meaningful_rows"] != 31 or frame["short_gap_control_rows"] != 12):
        raise ValueError("not the frozen discovery-only early-cut frame")
    rows = frame["records"]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("duplicate rating UID")
    price = sealed(PRICE)
    if (price["schema"] != "within-sentence-start-rating-price-v1"
            or price["status"] != "PASS_COMPLETE_2_GPUH" or price["frame_sha256"] != frame["sha256"]
            or price["rating_driver_sha256"] != file_sha(Path(__file__))
            or price["rubric_sha256"] != file_sha(RUBRIC)
            or price["rows"] != len(rows) or price["ratings"] != 2 * len(rows)
            or price["allocation_plan"]["combined_GPU_h_ceiling"] != 4.0):
        raise ValueError("sealed GPU price no longer binds early-cut code/frame/rubric")
    binding_body = {
        "schema": "within-sentence-start-rating-binding-v1",
        "frame_sha256": frame["sha256"],
        "price_sha256": price["sha256"],
        "driver_sha256": file_sha(Path(__file__)),
        "rubric_sha256": file_sha(RUBRIC),
        "model_snapshot": str(MODEL), "model_revision": MODEL.name,
        "batch_size": BATCH, "max_tokens_per_rating": MAX_TOKENS,
        "readers": 2,
        "sampler": {"temperature": .2, "top_p": .95,
                    "thinking": True, "reasoning_effort": "low"},
        "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_fragment"],
        "scope": "43 frozen early-cut discovery prefixes, 86 assigned Qwen draws; separate timing diagnostic, original trigger-sentence-excluded primary unchanged; not human truth",
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
    (args.out / "loads").mkdir(exist_ok=True)
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
    needs_inference = any(not (args.out / "batches" / f"{start:06d}-reader{reader}.json").exists()
                          for start, _ in pending for reader in (0, 1))
    load_seconds = 0.0
    if needs_inference:
        from vllm import LLM, SamplingParams
        load_started = time.monotonic()
        model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                    dtype="bfloat16", kv_cache_dtype="bfloat16", max_model_len=49152,
                    max_num_seqs=32, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config="vllm", language_model_only=True,
                    attention_config={"backend": "FLASH_ATTN"})
        load_seconds = time.monotonic() - load_started
        atomic_json(args.out / "loads" / f"{os.environ['SLURM_JOB_ID']}.json",
                    {"schema": "within-sentence-start-load-v1",
                     "binding_sha256": binding["sha256"],
                     "job_id": os.environ["SLURM_JOB_ID"],
                     "load_seconds": load_seconds})
    if pending:
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
                                {"schema": "within-sentence-start-assignment-v1",
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
                                    {"schema": "within-sentence-start-failure-v1",
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
                                    {"schema": "within-sentence-start-failure-v1",
                                     "binding_sha256": binding["sha256"], "uid": row["uid"],
                                     "reader": reader, "start": start,
                                     "job_id": os.environ["SLURM_JOB_ID"],
                                     "exception_type": "OutputCountMismatch",
                                     "exception_message": f"got {len(outputs)} for {len(block)} assignments"})
                    raise ValueError("start rating model output count differs")
                prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
                generated_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                one_timing = {"reader": reader, "job_id": os.environ["SLURM_JOB_ID"],
                              "wall_seconds": reader_elapsed,
                              "prompt_tokens": prompt_tokens,
                              "generated_tokens": generated_tokens}
                one_results = [{"rating": parse_rating(o.outputs[0].text),
                                "finish_reason": o.outputs[0].finish_reason,
                                "generated_tokens": len(o.outputs[0].token_ids),
                                "raw_completion": o.outputs[0].text}
                               for o in outputs]
                atomic_json(reader_path,
                            {"schema": "within-sentence-start-reader-batch-v1",
                             "binding_sha256": binding["sha256"], "start": start,
                             "reader": reader, "timing": one_timing,
                             "records": [{"uid": row["uid"], "result": one_results[i]}
                                         for i, row in enumerate(block)]})
                timing.append(one_timing)
                per_reader.append(one_results)
            records = [{"uid": row["uid"], "transition": row["transition"],
                        "readers": [per_reader[0][i], per_reader[1][i]]}
                       for i, row in enumerate(block)]
            atomic_json(path, {"schema": "within-sentence-start-rating-batch-v1",
                               "binding_sha256": binding["sha256"],
                               "start": start,
                               "reader_timings": timing, "records": records})
            print(json.dumps({"complete": min(start+BATCH, len(rows)), "of": len(rows),
                              "current_job_load_seconds": load_seconds,
                              "reader_timings": timing,
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
    complete = []
    saved_timings = []
    for start in range(0, len(rows), BATCH):
        part = sealed(args.out / "batches" / f"{start:06d}.json")
        complete.extend(part["records"])
        saved_timings.append({"reader_timings": part["reader_timings"]})
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
    load_receipts = [sealed(path) for path in sorted((args.out / "loads").glob("*.json"))]
    if any(rec["binding_sha256"] != binding["sha256"] for rec in load_receipts):
        raise ValueError("load receipt binding differs")
    summary = {"schema": "within-sentence-start-rating-summary-v1",
               "binding_sha256": binding["sha256"], "counts": dict(counts),
               "timings": saved_timings, "load_receipts": load_receipts,
               "interpretation": "Complete assigned 43-window discovery earlier-cut Qwen audit; two same-model readers, not independent human truth or original primary endpoint"}
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
