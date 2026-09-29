"""J1: LLM answer-equivalence adjudication of finished answers that math_verify rejects.

The judge (Qwen3.8-27B, card thinking sampler) compares the model's FINAL answer with the
reference answer; it is not asked to solve the problem. Three independent votes per item,
three-state output (EQUIVALENT / NOT_EQUIVALENT / UNCERTAIN), per-item checkpoints.

  build-hist  CPU: residual items from the historical corpus + rescored table
  audit       CPU: freeze the stratified 100-item audit set (before any judging)
  run         GPU client: judge an items file against a running server
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from moe_exp.utils import last_boxed

JUDGE_MODEL = "Qwen/Qwen3.8-27B"
JUDGE_SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
                     repetition_penalty=1.0)  # Qwen3.8-27B README.md:252 (thinking mode)
VOTES = 3
PROMPT_VERSION = 1
PARSER_VERSION = 2  # v2: regex verdict extraction (LaTeX backslashes make "reason" invalid JSON)
SYSTEM = (
    "You check whether a candidate's final answer to a math or science problem is equivalent "
    "to the reference answer. Do not re-solve the problem and do not judge the reasoning; "
    "only compare the two final answers.\n"
    "EQUIVALENT: same mathematical value, expression, set, interval or tuple (any order-"
    "insensitive answer lists match as sets), possibly written differently (fractions vs "
    "decimals only if exactly equal, simplified vs unsimplified, x=5 vs 5, degrees symbol, "
    "\\dfrac vs \\frac, inequality vs interval notation for the same set). For physical "
    "quantities, the same value in consistent units, where the candidate rounds to the "
    "reference at the reference's stated precision.\n"
    "NOT_EQUIVALENT: a different value/set, a missing or extra solution, wrong units that "
    "change the value, or no final answer.\n"
    "UNCERTAIN: the reference itself is ambiguous or the comparison cannot be decided.\n"
    'Reply with a JSON object only: {"verdict": "EQUIVALENT|NOT_EQUIVALENT|UNCERTAIN", '
    '"reason": "<one sentence>"}'
)


def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidate_text(answer_text: str) -> str:
    boxed = last_boxed(answer_text or "")
    tail = (answer_text or "")[-600:]
    return (f"Boxed final answer: {boxed}\n" if boxed is not None else "No boxed answer.\n") + \
        f"Closing text of the response:\n{tail}"


def build_item(*, corpus, model, dataset, problem_id, sample_id, problem, gold, answer_text,
               stratum=None) -> dict:
    item = dict(corpus=corpus, model=model, dataset=dataset, problem_id=problem_id,
                sample_id=sample_id, problem=(problem or "")[:4000], gold=gold,
                candidate=candidate_text(answer_text), stratum=stratum)
    item["item_id"] = _sha([corpus, model, dataset, problem_id, sample_id, item["candidate"], gold])[:24]
    return item


def build_hist(args) -> None:
    """Residual finished-but-rejected historical answers (strict math_verify False, stop)."""
    import pandas as pd
    from moe_exp.correlation_pipeline.continuation import DATASETS, HISTORICAL
    rescored = pd.read_parquet(args.rescored)
    wanted = rescored[(rescored.mv_correct == False) & (rescored.finish_reason != "length")]  # noqa: E712
    keys = {(r.model if r.model != "qwen35gptq" else "qwen35", r.problem_id) for r in wanted.itertuples()}
    items = []
    for model, root in HISTORICAL.items():
        for dataset in DATASETS:
            with (root / dataset / "traces.jsonl").open() as handle:
                for line in handle:
                    record = json.loads(line)
                    if (model, record["problem_id"]) not in keys:
                        continue
                    meta = record["metadata"]
                    items.append(build_item(
                        corpus="historical", model=model, dataset=dataset,
                        problem_id=record["problem_id"], sample_id=record.get("sample_id"),
                        problem=record.get("prompt"), gold=record.get("gold_answer"),
                        answer_text=meta.get("assistant_content") or record.get("cot_text")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(i) + "\n" for i in items))
    print(json.dumps(dict(items=len(items), expected=len(wanted),
                          by_dataset=Counter(i["dataset"] for i in items))))


def build_corpus(args) -> None:
    """Residual finished-but-rejected answers of a card-v3 corpus (strict False, not capped)."""
    items = []
    root = Path(args.gen_root)
    for dataset in ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva"):
        for line in (root / dataset / "traces.jsonl").open():
            record = json.loads(line)
            meta = record["metadata"]
            if record.get("is_correct") is not False or (meta.get("termination") or meta.get("finish_reason")) == "length":
                continue
            items.append(build_item(
                corpus=args.corpus, model=args.model, dataset=dataset,
                problem_id=record["problem_id"], sample_id=record.get("sample_id"),
                problem=record.get("prompt"), gold=record.get("gold_answer"),
                answer_text=meta.get("assistant_content") or record.get("cot_text")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(i) + "\n" for i in items))
    print(json.dumps(dict(model=args.model, items=len(items), by_dataset=Counter(i["dataset"] for i in items))))


def build_cont(args) -> None:
    """Residual continuation branches: finished reasoning, answer present, strict False."""
    from moe_exp.correlation_pipeline.continuation import HISTORICAL, DATASETS
    scored = {}
    for line in open(Path(args.cont_dir) / "scored.jsonl"):
        row = json.loads(line)
        if row.get("finished_reasoning") and row.get("strict_correct") is False \
                and row.get("finish_reason") != "length":
            scored[(row["dataset"], row["problem_id"])] = row
    problems = {}
    for dataset in DATASETS:
        for line in (HISTORICAL[args.model] / dataset / "traces.jsonl").open():
            record = json.loads(line)
            if (dataset, record["problem_id"]) in scored:
                problems[(dataset, record["problem_id"])] = (record.get("prompt"), record.get("gold_answer"))
    items = []
    for (dataset, problem_id), row in scored.items():
        prompt, gold = problems[(dataset, problem_id)]
        items.append(build_item(corpus="continuation", model=args.model, dataset=dataset,
                                problem_id=problem_id, sample_id=row["sample_id"], problem=prompt,
                                gold=gold, answer_text=row.get("final_answer_text") or row.get("model_answer")))
    Path(args.out).write_text("".join(json.dumps(i) + "\n" for i in items))
    print(json.dumps(dict(model=args.model, items=len(items))))


def audit(args) -> None:
    """Freeze a stratified 100-item audit set BEFORE judging (blind agent adjudication)."""
    import pandas as pd
    from moe_exp.correlation_pipeline.continuation import DATASETS, HISTORICAL
    items = [json.loads(line) for line in open(args.items)]
    excluded = set()
    for path in args.exclude or []:
        excluded |= {i["item_id"] for i in json.load(open(path))["items"]}
        excluded |= {(i["model"], i["problem_id"]) for i in json.load(open(path))["items"]}
    items = [i for i in items if i["item_id"] not in excluded]
    rng = random.Random(args.seed)
    by = lambda pred: [i for i in items if pred(i)]  # noqa: E731
    pick = lambda pool, n: rng.sample(pool, min(n, len(pool)))  # noqa: E731
    chosen = ([dict(i, stratum="minerva_residual") for i in pick(by(lambda i: i["dataset"] == "minerva"), 35)]
              + [dict(i, stratum="olympiad_residual") for i in pick(by(lambda i: i["dataset"] == "olympiad"), 25)]
              + [dict(i, stratum="other_residual") for i in pick(by(lambda i: i["dataset"] not in ("minerva", "olympiad")), 15)])
    rescored = pd.read_parquet(args.rescored)
    pos = rescored[(rescored.mv_correct == True) & (rescored.finish_reason != "length")]  # noqa: E712
    wrong = rescored[(rescored.mv_correct == False) & (rescored.finish_reason != "length")  # noqa: E712
                     & rescored.dataset.isin(["aime24", "aime25", "amc23"])]
    wrong = wrong[wrong.boxed.str.fullmatch(r"\s*-?\d+\s*") & wrong.gold.str.fullmatch(r"\s*-?\d+\s*")]
    control_keys = {}
    for name, frame, n in (("mv_positive_control", pos, 15), ("clear_wrong_control", wrong, 10)):
        frame = frame[[(("qwen35" if m == "qwen35gptq" else m), p) not in excluded
                       for m, p in zip(frame.model, frame.problem_id)]]
        for r in frame.sample(n, random_state=args.seed).itertuples():
            control_keys[(r.model if r.model != "qwen35gptq" else "qwen35", r.problem_id)] = name
    for model, root in HISTORICAL.items():
        for dataset in DATASETS:
            for line in (root / dataset / "traces.jsonl").open():
                record = json.loads(line)
                stratum = control_keys.get((model, record["problem_id"]))
                if stratum:
                    meta = record["metadata"]
                    chosen.append(build_item(
                        corpus="historical", model=model, dataset=dataset,
                        problem_id=record["problem_id"], sample_id=record.get("sample_id"),
                        problem=record.get("prompt"), gold=record.get("gold_answer"),
                        answer_text=meta.get("assistant_content") or record.get("cot_text"),
                        stratum=stratum))
    rng.shuffle(chosen)
    frozen = dict(schema_version=1, prompt_version=PROMPT_VERSION, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  strata=Counter(i["stratum"] for i in chosen), items=chosen)
    frozen["audit_sha256"] = _sha(frozen["items"])
    Path(args.out).write_text(json.dumps(frozen, indent=1))
    print(json.dumps(dict(items=len(chosen), strata=frozen["strata"], sha=frozen["audit_sha256"])))


def _post(url, payload, timeout):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer local-vllm-key"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"\s*(NOT_EQUIVALENT|EQUIVALENT|UNCERTAIN)\s*"', re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Last "verdict": "..." field; tolerant of invalid JSON escapes in the free-text reason."""
    matches = _VERDICT_RE.findall(text or "")
    return matches[-1].upper() if matches else "UNPARSED"


def judge_item(item, base_url, timeout, seed_base):
    user = (f"Problem:\n{item['problem']}\n\nReference answer:\n{item['gold']}\n\n"
            f"Candidate:\n{item['candidate']}")
    votes, raw = [], []
    for vote in range(VOTES):
        payload = dict(model=JUDGE_MODEL, messages=[{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": user}],
                       max_tokens=8192, seed=seed_base + vote,
                       chat_template_kwargs={"enable_thinking": True}, **JUDGE_SAMPLER)
        for attempt in range(3):
            try:
                out = _post(base_url.rstrip("/") + "/chat/completions", payload, timeout)
                choice = out["choices"][0]
                content = choice["message"].get("content") or ""
                raw.append(content[-2000:])
                if choice.get("finish_reason") == "length":
                    votes.append("TRUNCATED")  # judge ran out of tokens: not a real "uncertain"
                else:
                    votes.append(parse_verdict(content))
                break
            except (OSError, KeyError, ValueError) as error:
                if attempt == 2:
                    votes.append(f"ERROR:{type(error).__name__}")
                time.sleep(5 * (attempt + 1))
    counts = Counter(votes)
    top, n = counts.most_common(1)[0]
    verdict = top if n >= 2 and top in ("EQUIVALENT", "NOT_EQUIVALENT") else "UNCERTAIN"
    return dict(item_id=item["item_id"], votes=votes, verdict=verdict, unanimous=n == VOTES,
                truncated_votes=sum(v == "TRUNCATED" for v in votes), raw_tail=raw,
                parser_version=PARSER_VERSION,
                prompt_version=PROMPT_VERSION, sampler=JUDGE_SAMPLER)


def run(args) -> None:
    items = [json.loads(line) for line in open(args.items)] if args.items.endswith(".jsonl") else \
        json.loads(open(args.items).read())["items"]
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [i for i in items if i["item_id"] not in done]
    print(json.dumps(dict(items=len(items), todo=len(todo))), flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as handle:
        futures = {pool.submit(judge_item, i, args.base_url, args.timeout,
                               int(i["item_id"][:8], 16) % 1_000_000): i for i in todo}
        for index, future in enumerate(as_completed(futures)):
            row = future.result()
            with lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if index % 50 == 0:
                print(json.dumps(dict(judged=index + 1, of=len(todo))), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build-hist", "build-corpus", "build-cont", "audit", "run"])
    ap.add_argument("--cont-dir")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--exclude", nargs="*", help="earlier audit files whose items must not recur")
    ap.add_argument("--gen-root")
    ap.add_argument("--corpus", default="card-v3")
    ap.add_argument("--model")
    ap.add_argument("--rescored")
    ap.add_argument("--items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(argv)
    {"build-hist": build_hist, "build-corpus": build_corpus, "build-cont": build_cont, "audit": audit,
     "run": run}[args.command](args)


if __name__ == "__main__":
    main()
