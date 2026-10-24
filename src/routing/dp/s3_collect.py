"""Addendum 1 (S3) stage 1: token ids and top-1 depth paths per cohort-A attempt (no outcomes are read).

Per attempt of the analysed cohort-A table (qwen36 = split-v1 dev+tune) of every model:
- `tok` int32 [n_r]: completion token id of every reasoning token (card-v3 trace metadata.token_replay,
  hash-checked read via the v3an source location), indexed by reasoning rank;
- labelled-sentence tokens (dynamics-routing sentences.parquet spans, read-only): `lab_rank`, `lab_sid`
  (sentence_index), `lab_top1` uint8 [L, m] and `lab_adj` (the previous reasoning token is adjacent in the
  completion AND in the same sentence);
- gpt only: `top1_all` uint8 [L, n_r] for every reasoning token (F1', item (i)).
Consistency checks: tensor/trace/attempt-table token counts, and for the attempts of the Part 1 sample the
recomputed top-1 at the cached ranks must equal the Part 1 cache.

CLI: python -m dp.s3_collect [--models M ...] [--workers W]
Output: results/s3/_tok/<model>/<row>.npz (never overwritten) + manifest-<jobid>.json per model.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import common as cm
from . import part1 as p1
from . import pathfeat as pf

OUT = cm.RESULTS / "s3"
TOK = OUT / "_tok"
FULL_PATH_MODELS = ("gpt",)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_one(task: dict) -> dict:
    """Worker: one attempt -> npz of token ids and top-1 paths (skipped when present)."""
    from v3an.data import read_location
    from v3an.extract import _load_tensors
    path = Path(task["out"])
    aid = task["attempt_id"]
    if path.exists():
        return dict(row=task["row"], skipped=True)
    ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
    del gaps
    L, T, k = ids.shape
    if T != int(task["completion_tokens"]) or (L, k) != (task["num_layers"], task["top_k"]):
        raise ValueError(f"{aid}: tensor layout differs from the attempt table/spec")
    if ids.min() < 0 or ids.max() >= task["num_experts"] or task["num_experts"] > 256:
        raise ValueError(f"{aid}: expert id out of range for uint8 storage")
    n_r = len(reasoning)
    if n_r != int(task["n_reasoning"]):
        raise ValueError(f"{aid}: reasoning tokens {n_r} != attempt table {task['n_reasoning']}")
    trace = read_location(json.loads(task["source_location"]))
    comp = np.asarray(trace["metadata"]["token_replay"]["completion_token_ids"], dtype=np.int64)
    if len(comp) != T:
        raise ValueError(f"{aid}: token ids {len(comp)} != completion tokens {T}")
    tok = comp[reasoning].astype(np.int32)
    out = dict(tok=tok, n_r=n_r)
    rows = task.get("sent")
    ranks, sids = [], []
    if rows is not None and len(rows):
        if not (rows["n_reasoning"] == n_r).all():
            raise ValueError(f"{aid}: sentence table n_reasoning differs")
        for s, e, si in zip(rows["tok_start"].to_numpy(), rows["tok_end"].to_numpy(),
                            rows["sentence_index"].to_numpy()):
            ranks.append(np.arange(int(s), int(e), dtype=np.int64))
            sids.append(np.full(int(e) - int(s), int(si), np.int32))
    lab_rank = np.concatenate(ranks) if ranks else np.zeros(0, np.int64)
    lab_sid = np.concatenate(sids) if sids else np.zeros(0, np.int32)
    if len(lab_rank) and (np.diff(lab_rank) <= 0).any():
        raise ValueError(f"{aid}: labelled spans overlap or are unsorted")
    if task["full"]:
        top1_all, _ = pf.top1_paths(ids, weights, reasoning)
        out["top1_all"] = top1_all.astype(np.uint8)
        lab_top1 = top1_all[:, lab_rank]
    else:
        lab_top1, _ = pf.top1_paths(ids, weights, reasoning[lab_rank])
    pos = reasoning[lab_rank]
    adj = np.zeros(len(lab_rank), bool)
    if len(lab_rank) > 1:
        adj[1:] = (np.diff(lab_rank) == 1) & (np.diff(pos) == 1) & (lab_sid[1:] == lab_sid[:-1])
    out.update(lab_rank=lab_rank.astype(np.int32), lab_sid=lab_sid, lab_top1=lab_top1.astype(np.uint8),
               lab_adj=adj)
    checked = False
    if task.get("p1_cache"):
        cached = np.load(task["p1_cache"])
        if not np.array_equal(p1.token_sample(n_r, task["row"]), cached["ranks"]):
            raise ValueError(f"{aid}: Part 1 token sample differs from the cache")
        t1, _ = pf.top1_paths(ids, weights, reasoning[cached["ranks"]])
        if not np.array_equal(t1, cached["top1"].astype(np.int32)):
            raise ValueError(f"{aid}: recomputed top-1 differs from the Part 1 cache")
        checked = True
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return dict(row=task["row"], skipped=False, n_r=n_r, n_lab=int(len(lab_rank)), p1_checked=checked,
                contiguous=bool(n_r <= 1 or (np.diff(reasoning) == 1).all()))


def build_tasks(model: str) -> list[dict]:
    att = cm.load_attempts(model, "A")
    spec = cm.model_spec(model)
    sent = cm.load_sentence_table(model, "A")
    by = {k: g for k, g in sent.groupby("attempt_id")}
    _, p1_idx = p1.sample_attempts(model)
    p1_rows = set(int(i) for i in p1_idx)
    d = TOK / model
    d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(len(att)):
        r = att.iloc[int(i)]
        tasks.append(dict(model=model, row=int(i), attempt_id=r.attempt_id, tensor_path=r.tensor_path,
                          source_location=r.source_location, completion_tokens=int(r.completion_tokens),
                          n_reasoning=int(r.n_reasoning), sent=by.get(r.attempt_id),
                          out=str(d / f"{int(i):05d}.npz"), full=model in FULL_PATH_MODELS,
                          p1_cache=str(p1.CACHE / model / f"{int(i):05d}.npz") if int(i) in p1_rows else None,
                          **spec))
    return tasks


def tok_path(model: str, row: int) -> Path:
    return TOK / model / f"{int(row):05d}.npz"


def run(models: list[str], workers: int) -> None:
    tasks = {m: build_tasks(m) for m in models}
    flat = [t for m in models for t in tasks[m]]
    flat.sort(key=lambda t: -t["n_reasoning"])
    log(f"collect: {len(flat)} attempts over {models}, {workers} workers")
    os.environ.update({v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    res: dict = {m: [] for m in models}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(collect_one, t): t["model"] for t in flat}
        for i, f in enumerate(as_completed(futs), 1):
            res[futs[f]].append(f.result())
            if i % 200 == 0 or i == len(flat):
                log(f"collect: {i}/{len(flat)} done, {time.time() - t0:.0f}s")
    for m in models:
        done = [r for r in res[m] if not r["skipped"]]
        present = sum(tok_path(m, t["row"]).exists() for t in tasks[m])
        man = dict(model=m, attempts=len(tasks[m]), files_present=int(present), written_this_run=len(done),
                   skipped_existing=len(res[m]) - len(done),
                   p1_checked_this_run=int(sum(r.get("p1_checked", False) for r in done)),
                   labelled_tokens_this_run=int(sum(r.get("n_lab", 0) for r in done)),
                   reasoning_tokens_this_run=int(sum(r.get("n_r", 0) for r in done)),
                   noncontiguous_this_run=int(sum(not r.get("contiguous", True) for r in done)),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"), slurm_job=os.environ.get("SLURM_JOB_ID"))
        if present != len(tasks[m]):
            raise RuntimeError(f"{m}: {present} of {len(tasks[m])} token files present")
        (TOK / m / f"manifest-{man['slurm_job']}.json").write_text(json.dumps(man, indent=1))
        log(f"{m}: {json.dumps(man)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(cm.MODELS), choices=cm.MODELS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args(argv)
    run(a.models, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
