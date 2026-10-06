"""Checkpointed, start-only, arm-blind Qwen audit of fresh v2.1 windows.

The model receives only the original problem and sentences emitted through the
trigger.  Later sentences in the sealed frame are deliberately excluded here.
These are two independent same-model draws, not human truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FRAME = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_V21_INDEPENDENT_AUDIT_FRAME.json"
RUBRIC = REPO / "report/experimental-resume-v1/TRANSITION_RUBRIC_v0.1.md"
MODEL = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
BATCH = 32
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
    if set(row["reader_input"]) != {"problem", "previous_sentence", "triggering_sentence", "later_sentences"}:
        raise ValueError("reader fixture field set changed")
    start = {key: row["reader_input"][key] for key in
             ("problem", "previous_sentence", "triggering_sentence")}
    if not all(isinstance(v, str) for v in start.values()):
        raise ValueError("prefix text is not a string")
    system = ("Independently rate a *starting condition* in visible mathematical reasoning. "
              "Use only the original problem, preceding sentence, and current sentence below. "
              "Do not infer future text, correctness, or hidden thoughts. "
              'Return exactly one JSON object: {"start": true} or {"start": false}. '
              "When evidence is ambiguous, use false. No explanation.\nCriterion: " +
              CRITERIA[row["transition"]])
    user = ("Original problem:\n" + start["problem"] +
            "\n\nPrevious sentence:\n" + start["previous_sentence"] +
            "\n\nCurrent sentence:\n" + start["triggering_sentence"] + "\n\nJSON:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_rating(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    matches = re.findall(r'"start"\s*:\s*(true|false)', text, re.I)
    return {"start": matches[0].lower() == "true"} if len(matches) == 1 else None


def rating_seed(uid, reader):
    return int(digest(["transition-v21-start-rating-v1", uid, reader])[:8], 16) % 2_000_000_000


def atomic_json(path, body):
    value = {**body, "sha256": digest(body)}
    temp = path.with_name(path.name + ".part-" + os.environ["SLURM_JOB_ID"])
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return value


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("start-only semantic audit requires GPU Slurm")
    frame = sealed(FRAME)
    if frame["schema"] != "transition-v21-independent-audit-frame-v1" or frame["families"] != 48:
        raise ValueError("not the frozen discovery-only v2.1 frame")
    rows = frame["records"]
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("duplicate rating UID")
    binding_body = {
        "schema": "transition-v21-start-rating-binding-v1",
        "frame_sha256": frame["sha256"],
        "driver_sha256": file_sha(Path(__file__)),
        "rubric_sha256": file_sha(RUBRIC),
        "model_snapshot": str(MODEL), "model_revision": MODEL.name,
        "batch_size": BATCH, "max_tokens_per_rating": MAX_TOKENS,
        "readers": 2,
        "sampler": {"temperature": .2, "top_p": .95,
                    "thinking": True, "reasoning_effort": "low"},
        "visible_input_allowlist": ["problem", "previous_sentence", "triggering_sentence"],
        "scope": "discovery-only, independent previously unrated windows; two same-model draws; LLM audit, not human truth",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    binding_path = args.out / "BINDING.json"
    if binding_path.exists():
        binding = sealed(binding_path)
        if binding != {**binding_body, "sha256": digest(binding_body)}:
            raise ValueError("rating output directory rebound to changed input/code")
    else:
        binding = atomic_json(binding_path, binding_body)
    (args.out / "batches").mkdir(exist_ok=True)
    pending = []
    for start in range(0, len(rows), BATCH):
        path = args.out / "batches" / f"{start:06d}.json"
        if path.exists():
            part = sealed(path)
            if part["binding_sha256"] != binding["sha256"] or part["start"] != start or [
                r["uid"] for r in part["records"]] != [r["uid"] for r in rows[start:start+BATCH]]:
                raise ValueError("saved rating batch UID/binding differs")
        else:
            pending.append((start, path))
    if pending:
        from vllm import LLM, SamplingParams
        model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                    dtype="bfloat16", kv_cache_dtype="bfloat16", max_model_len=49152,
                    max_num_seqs=32, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config="vllm", language_model_only=True,
                    attention_config={"backend": "FLASH_ATTN"})
        started = time.monotonic()
        for start, path in pending:
            block = rows[start:start+BATCH]
            conversations = [messages(row) for row in block]
            per_reader = []
            for reader in (0, 1):
                params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                         seed=rating_seed(row["uid"], reader)) for row in block]
                outputs = model.chat(conversations, sampling_params=params,
                                     chat_template_kwargs={"enable_thinking": True,
                                                           "reasoning_effort": "low"},
                                     use_tqdm=False)
                if len(outputs) != len(block):
                    raise ValueError("start rating model output count differs")
                per_reader.append([{"rating": parse_rating(o.outputs[0].text),
                                    "finish_reason": o.outputs[0].finish_reason,
                                    "generated_tokens": len(o.outputs[0].token_ids),
                                    "raw_completion": o.outputs[0].text}
                                   for o in outputs])
            records = [{"uid": row["uid"], "transition": row["transition"],
                        "readers": [per_reader[0][i], per_reader[1][i]]}
                       for i, row in enumerate(block)]
            atomic_json(path, {"schema": "transition-v21-start-rating-batch-v1",
                               "binding_sha256": binding["sha256"],
                               "start": start, "records": records})
            print(json.dumps({"complete": min(start+BATCH, len(rows)), "of": len(rows),
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
    complete = []
    for start in range(0, len(rows), BATCH):
        complete.extend(sealed(args.out / "batches" / f"{start:06d}.json")["records"])
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
    summary = {"schema": "transition-v21-start-rating-summary-v1",
               "binding_sha256": binding["sha256"], "counts": dict(counts),
               "interpretation": "Fresh previously unrated discovery windows; prefix-only Qwen start judgments, not human truth or independent-family validation"}
    atomic_json(args.out / "SUMMARY.json", summary)
    print(json.dumps({"summary": str(args.out / "SUMMARY.json"),
                      "counts": summary["counts"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
