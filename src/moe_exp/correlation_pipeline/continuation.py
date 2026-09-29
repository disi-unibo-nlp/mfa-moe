"""Continue length-capped historical traces from their exact saved token IDs.

The continuation prompt is the saved prompt_token_ids + completion_token_ids, sent to
/v1/completions with the trace's own historical sampler (temperature/top_p/top_k; the
historical runs used no presence/frequency/repetition penalty), so each branch extends
the same conditional sampling law to a larger cumulative budget. Only the suffix is
stored; outcomes are parsed from the combined original+suffix token sequence.

  python -m moe_exp.correlation_pipeline.continuation plan  --model qwen36 --out DIR
  python -m moe_exp.correlation_pipeline.continuation run   --model qwen36 --out DIR --base-url URL
  python -m moe_exp.correlation_pipeline.continuation score --model qwen36 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.client import complete_tokens
from moe_exp.correlation_pipeline.model_profiles import CARD_PROFILES

RESULTS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline")
HISTORICAL = {
    "gpt": RESULTS / "gpt-oss-20b/generation/openai--gpt-oss-20b",
    "glm": RESULTS / "glm-4.7-flash/generation/zai-org--GLM-4.7-Flash",
    "gemma": RESULTS / "gemma-nvfp4-nf4/generation/nvidia--Gemma-4-26B-A4B-NVFP4",
    "qwen36": RESULTS / "qwen36-35b-a3b-fp8/generation/Qwen--Qwen3.6-35B-A3B-FP8",
    "nemotron": RESULTS / "nemotron-nvfp4-dspark/generation/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "qwen330b": RESULTS / "qwen3-30b-a3b/generation/Qwen--Qwen3-30B-A3B",
    "qwen35": RESULTS / "generation/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4",
}
DATASETS = ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva")
CUMULATIVE_BUDGET = 131_072
SEED_OFFSET = 7_000_003  # continuation seed = original sample seed + offset (distinct stream)
CONTINUATION_CONTRACT_VERSION = 1


def original_generation_config(record: dict) -> dict:
    """Resolve both historical flat and card-v3 nested sampler/budget contracts.

    Card-v3 max_tokens=None means context-bounded generation; its recorded
    effective_max_tokens is the actual ORIGINAL budget, not 131072 extra tokens.
    """
    metadata = record["metadata"]
    saved = metadata["generation_config"]
    budget = metadata.get("effective_max_tokens") or saved.get("max_tokens")
    if type(budget) is not int or budget <= 0:
        raise ValueError("Missing recorded original effective generation budget")
    fields = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
              "frequency_penalty", "repetition_penalty", "stop", "stop_token_ids", "ignore_eos")
    sampler = saved.get("sampler", saved)
    result = {k: sampler[k] for k in fields if sampler.get(k) is not None}
    if "temperature" not in result or "top_p" not in result:
        raise ValueError("Missing recorded original sampling settings")
    result.setdefault("top_k", 0)
    result.update(max_tokens=budget, seed=saved.get("seed"))
    return result


def exact_prefix(record: dict, completion_tokens: int) -> dict:
    """Branch from any saved exact prefix, charged to the ORIGINAL request budget.

    Historical length-capped continuation functions below retain their separate
    131072-token extension contract. This function does not consult final outcome.
    """
    from moe_exp.correlation_pipeline.dynamics.common import digest
    replay = record["metadata"].get("token_replay") or {}
    prompt = replay.get("prompt_token_ids")
    completion = replay.get("completion_token_ids")
    if not prompt or completion is None or type(completion_tokens) is not int or not 0 <= completion_tokens <= len(completion):
        raise ValueError("Exact saved prompt and requested completion prefix required")
    config = original_generation_config(record)
    if completion_tokens >= config["max_tokens"]:
        raise ValueError("No original generation budget remains")
    prefix = completion[:completion_tokens]
    return dict(prompt_token_ids=list(prompt), prefix_token_ids=prefix,
                prefix_tokens=completion_tokens, prefix_sha256=digest([prompt, prefix]),
                generation_config=dict(config), budget_population="original",
                original_generation_config=record["metadata"]["generation_config"],
                max_new_tokens=config["max_tokens"] - completion_tokens)


def run_branches(manifest, output, generate):
    """Resumable branch execution with an injected qualified native backend.

    The backend receives the frozen request and must return exact suffix IDs and
    intervention telemetry. Immutable files and per-branch locks prevent duplicate
    results or mixing of manifests. A crash before persistence may repeat compute,
    but cannot count a branch twice.
    """
    import fcntl
    from moe_exp.correlation_pipeline.dynamics.contracts import load, save, seal, validate
    from moe_exp.correlation_pipeline.dynamics.common import digest
    from moe_exp.correlation_pipeline.provenance import code_provenance
    validate(manifest, "branches")
    code = code_provenance()
    if code["source_sha256"] != manifest["code"]["source_sha256"]:
        raise ValueError("Branch code differs from the frozen manifest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "branches.json", manifest)
    results = []
    for branch in manifest["payload"]["branches"]:
        branch_id = branch["branch_id"]
        if branch_id != digest({k: v for k, v in branch.items() if k != "branch_id"}):
            raise ValueError("Branch binding mismatch")
        if branch["max_new_tokens"] != branch["original_max_tokens"] - len(branch["prefix_token_ids"]):
            raise ValueError("Branch changed original cumulative budget")
        path = output / (branch_id + ".json")
        with (output / (branch_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                artifact = load(path, "analysis")
                if artifact["inputs"] != {"branches": manifest["binding"], "branch": branch_id}:
                    raise ValueError("Incompatible resumed branch result")
            else:
                start = time.monotonic()
                value = generate(branch)
                ids = value.get("suffix_token_ids")
                if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                    raise ValueError("Backend did not return exact generated token IDs")
                if len(ids) > branch["max_new_tokens"] or not value.get("prompt_echo_verified"):
                    raise ValueError("Backend changed the exact prefix or generation budget")
                if not value.get("telemetry_verified"):
                    raise ValueError("Missing native request/token telemetry")
                result = {**value, "branch_id": branch_id, "manifest_binding": manifest["binding"],
                    "status": "complete", "seconds": time.monotonic() - start,
                    "total_tokens": len(branch["prefix_token_ids"]) + len(ids),
                    "is_correct": None, "capped": value.get("finish_reason") == "length"}
                artifact = seal("analysis", result,
                    inputs={"branches": manifest["binding"], "branch": branch_id},
                    config=dict(decoding=branch["decoding"], seed=branch["seed"], policy=branch["policy"]),
                    population=manifest["population"], code=code)
                save(path, artifact)
            results.append(artifact["payload"])
    return results


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def key_of(row: dict) -> str:
    return f"{row['dataset']}|{row['problem_id']}|{row['sample_id']}"


def iter_parents(model: str):
    """Genuine length-capped parents: stop reason 'length' at exactly the request cap."""
    for dataset in DATASETS:
        path = HISTORICAL[model] / dataset / "traces.jsonl"
        with path.open() as handle:
            for line_index, line in enumerate(handle):
                record = json.loads(line)
                meta = record["metadata"]
                if meta.get("finish_reason") != "length":
                    continue
                replay = meta.get("token_replay") or {}
                prompt_ids = replay.get("prompt_token_ids")
                completion_ids = replay.get("completion_token_ids")
                config = meta["generation_config"]
                parent = dict(
                    model=model, dataset=dataset, problem_id=record["problem_id"],
                    sample_id=record.get("sample_id"), line_index=line_index,
                    gold_answer=record.get("gold_answer"), seed=config.get("seed"),
                    temperature=config["temperature"], top_p=config["top_p"],
                    top_k=config.get("top_k", 0), historical_max_tokens=config["max_tokens"],
                    prompt_len=len(prompt_ids or []), completion_len=len(completion_ids or []),
                    exclusion=None)
                if not prompt_ids or not completion_ids:
                    parent["exclusion"] = "missing_token_ids"
                elif len(completion_ids) != config["max_tokens"]:
                    parent["exclusion"] = "length_stop_below_request_cap"
                parent["parent_sha256"] = _sha([prompt_ids, completion_ids])
                yield parent, prompt_ids, completion_ids


def plan(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    context = CARD_PROFILES[args.model]["max_model_len"]
    rows = []
    for parent, _, _ in iter_parents(args.model):
        total = parent["prompt_len"] + parent["completion_len"]
        budget = min(CUMULATIVE_BUDGET - parent["completion_len"], context - total)
        parent["extra_max_tokens"] = budget
        if parent["exclusion"] is None and budget <= 0:
            parent["exclusion"] = "no_context_room"
        rows.append(parent)
    random.Random(20260925).shuffle(rows)  # fixed processing order, not length/difficulty ordered
    manifest = dict(schema_version=CONTINUATION_CONTRACT_VERSION, model=args.model,
                    cumulative_budget=CUMULATIVE_BUDGET, server_context=context,
                    seed_offset=SEED_OFFSET, parents=rows,
                    eligible=sum(r["exclusion"] is None for r in rows),
                    excluded={k: sum(r["exclusion"] == k for r in rows)
                              for k in {r["exclusion"] for r in rows} if k})
    manifest["manifest_sha256"] = _sha(manifest)
    (out / "plan.json").write_text(json.dumps(manifest))
    print(json.dumps({k: manifest[k] for k in ("model", "eligible", "excluded", "server_context")}))
    return manifest


def _done_keys(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from an interrupted writer is ignored
            if row.get("status") == "complete":
                done.add(row["key"])
    return done


def run(args) -> None:
    out = Path(args.out)
    manifest = json.loads((out / "plan.json").read_text())
    ids = {key_of(p): (p, pi, ci) for p, pi, ci in iter_parents(args.model)}
    results = out / "continuations.jsonl"
    done = _done_keys(results)
    eligible = [p for p in manifest["parents"] if p["exclusion"] is None]
    if args.pilot:
        # validation subset: the longest parent (context/KV stress) plus the first others in
        # the fixed shuffled order; accepted pilot branches are reused by the main run
        longest = max(eligible, key=lambda p: p["prompt_len"] + p["completion_len"])
        eligible = [longest] + [p for p in eligible if p is not longest][:args.pilot - 1]
    todo = [p for p in eligible if key_of(p) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(json.dumps(dict(model=args.model, todo=len(todo), done=len(done))), flush=True)
    lock = threading.Lock()
    served = CARD_PROFILES[args.model]["model"]

    def one(parent):
        if args.deadline_epoch and time.time() > args.deadline_epoch:
            return None
        live, prompt_ids, completion_ids = ids[key_of(parent)]
        if live["parent_sha256"] != parent["parent_sha256"]:
            raise RuntimeError(f"Parent changed since planning: {key_of(parent)}")
        started = time.time()
        result = complete_tokens(
            base_url=args.base_url, api_key=args.api_key, model=served,
            prompt_token_ids=prompt_ids + completion_ids, max_tokens=parent["extra_max_tokens"],
            temperature=parent["temperature"], top_p=parent["top_p"], top_k=parent["top_k"],
            seed=(parent["seed"] or 0) + SEED_OFFSET, timeout=args.timeout)
        row = dict(key=key_of(parent), status="complete", model=args.model,
                   dataset=parent["dataset"], problem_id=parent["problem_id"],
                   sample_id=parent["sample_id"], parent_sha256=parent["parent_sha256"],
                   seed=(parent["seed"] or 0) + SEED_OFFSET,
                   extra_max_tokens=parent["extra_max_tokens"], suffix_token_ids=result.token_ids,
                   suffix_len=len(result.token_ids), finish_reason=result.finish_reason,
                   usage=result.usage, prompt_echo_verified=bool(result.prompt_token_ids),
                   seconds=round(time.time() - started, 2),
                   contract=CONTINUATION_CONTRACT_VERSION)
        with lock, results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    pending = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, parent) for parent in todo]
        for index, future in enumerate(as_completed(futures)):
            row = future.result()
            pending += row is None
            if index % 10 == 0:
                print(json.dumps(dict(finished=index + 1, of=len(todo))), flush=True)
    if pending:
        raise SystemExit(f"{pending} parents left for the resumed task (deadline)")


FINAL_MARKERS = {
    "qwen3": ("</think>", None), "glm45": ("</think>", None), "nemotron_v3": ("</think>", None),
    "gemma4": ("<channel|>", None),
    "openai_gptoss": ("<|channel|>final<|message|>", ("<|return|>", "<|end|>")),
}


def final_answer(text: str, parser: str) -> str | None:
    start, stops = FINAL_MARKERS[parser]
    index = text.rfind(start)
    if index < 0:
        return None
    answer = text[index + len(start):]
    for stop in stops or ():
        cut = answer.find(stop)
        if cut >= 0:
            answer = answer[:cut]
    return answer


def score(args) -> None:
    """CPU: decode combined sequences, extract the final answer and score in the main thread."""
    from transformers import AutoTokenizer
    from moe_exp.correlation_pipeline.scoring import score_completion_detailed
    card = CARD_PROFILES[args.model]
    tokenizer = (AutoTokenizer.from_pretrained(card["weights"]) if card["weights"] else
                 AutoTokenizer.from_pretrained(card["tokenizer"] or card["model"], revision=card["revision"]))
    out = Path(args.out)
    parents = {key_of(p): (p, pi, ci) for p, pi, ci in iter_parents(args.model)}
    rows = []
    for line in (out / "continuations.jsonl").open():
        row = json.loads(line)
        if row.get("status") != "complete":
            continue
        parent, _, completion_ids = parents[row["key"]]
        text = tokenizer.decode(completion_ids + row["suffix_token_ids"], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        answer = final_answer(text, card["parser"])
        scored = score_completion_detailed({"gold_answer": parent["gold_answer"]},
                                           answer_type="math", model_text=answer or "")
        rows.append(dict(key=row["key"], model=args.model, dataset=row["dataset"],
                         problem_id=row["problem_id"], sample_id=row["sample_id"],
                         cumulative_tokens=len(completion_ids) + row["suffix_len"],
                         finish_reason=row["finish_reason"], finished_reasoning=answer is not None,
                         strict_correct=scored["is_correct"] if answer is not None else False,
                         scoring_method=scored["method"],
                         scoring_status=scored["math_verify_status"],
                         model_answer=scored["model_answer"][:500],
                         final_answer_text=(answer or "")[-4000:]))
    (out / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = dict(model=args.model, scored=len(rows),
                   still_capped=sum(r["finish_reason"] == "length" for r in rows),
                   correct=sum(r["strict_correct"] is True for r in rows))
    (out / "scored_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "run", "score"])
    ap.add_argument("--model", required=True, choices=sorted(HISTORICAL))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--api-key", default="local-vllm-key")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--deadline-epoch", type=float)
    args = ap.parse_args(argv)
    {"plan": plan, "run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
