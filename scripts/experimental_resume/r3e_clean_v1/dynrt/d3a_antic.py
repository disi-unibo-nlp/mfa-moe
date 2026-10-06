"""P-A1 anticipation variant: histograms of the 16 reasoning tokens ENDING 16 tokens before sentence s's end.

For a pair source sentence s with (reasoning-rank) span [tok_start, tok_end) the window is the ranks
[tok_end - 32, tok_end - 16) of the attempt's reasoning-token stream (it may reach into earlier sentences),
truncated at rank 0; an empty window (tok_end <= 16) is marked missing. The same pass recomputes the
D1 `last16` histogram of s (ranks [max(tok_start, tok_end - 16), tok_end)) and compares it with the stored
one, which validates the rank arithmetic and the tensor alignment. Only pair sources of the analysed
questions (GPT B questions are reserved) are computed.

CLI: python -m dynrt.d3a_antic [--limit N] [--out DIR] [--workers W]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FAST, RESULTS, V3R2, W, sha256_file
from .extract import EDGE, HIST_TOL16, hist_frequencies

OUT = RESULTS / "A1"
GAP = 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def window_ranks(tok_end: int) -> tuple[int, int]:
    """Half-open reasoning-rank range of the anticipation window (may be empty)."""
    hi = tok_end - GAP
    lo = max(0, hi - EDGE)
    return lo, max(lo, hi)


def antic_attempt(task: dict) -> dict:
    """Worker: anticipation and last-16 histograms of the pair sources of one attempt."""
    from v3an.extract import _load_tensors
    t0 = time.time()
    ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
    del weights, gaps
    L, T, k = ids.shape
    E = int(task["num_experts"])
    if T != int(task["completion_tokens"]) or (L, k) != (int(task["num_layers"]), int(task["top_k"])):
        raise ValueError(f"{task['attempt_id']}: tensor layout {ids.shape} differs from the spec")
    if ids.min() < 0 or ids.max() >= E:
        raise ValueError(f"{task['attempt_id']}: expert id out of range")
    n_r = len(reasoning)
    if n_r != int(task["n_reasoning"]):
        raise ValueError(f"{task['attempt_id']}: reasoning tokens {n_r} != D1 table {task['n_reasoning']}")
    starts, ends = np.asarray(task["tok_start"]), np.asarray(task["tok_end"])
    if len(ends) and (ends.max() > n_r or starts.min() < 0):
        raise ValueError(f"{task['attempt_id']}: sentence span outside the reasoning stream")
    win_lists, win_idx, chk_lists = [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        lo, hi = window_ranks(int(e))
        if hi > lo:
            win_lists.append(reasoning[lo:hi])
            win_idx.append(i)
        chk_lists.append(reasoning[max(int(s), int(e) - EDGE):int(e)])
    win = hist_frequencies(ids, win_lists, E) if win_lists else np.zeros((0, L, E), np.float32)
    chk = hist_frequencies(ids, chk_lists, E) if chk_lists else np.zeros((0, L, E), np.float32)
    return dict(attempt_id=task["attempt_id"], pair_rows=np.asarray(task["pair_rows"]),
                win=win.astype(np.float16), win_idx=np.asarray(win_idx, np.int64),
                win_len=np.asarray([len(x) for x in win_lists], np.int64), chk=chk.astype(np.float16),
                seconds=time.time() - t0)


def build_tasks(limit: int | None) -> tuple[list[dict], pd.DataFrame, dict]:
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    att = pd.read_parquet(V3R2 / "gpt" / "A" / "attempts.parquet").sort_values("attempt_id").reset_index(drop=True)
    sent = pd.read_parquet(RESULTS / "gpt" / "A" / "sentences.parquet")
    ready = json.loads((FAST / "manifests" / "gpt" / "A" / "ready.json").read_text())
    spec = ready["spec"]
    meta = dict(num_experts=int(spec["experts"]), num_layers=len(spec["layers"]), top_k=int(spec["top_k"]))
    aid_of = att["attempt_id"].to_numpy()
    # `pairs.attempt` indexes the D1 attempt table (kept questions, sorted by attempt id): same order as here
    # because gpt keeps every question; verify through the sentence table.
    if not (sent["attempt_id"].to_numpy()[pairs["src"].to_numpy()] == aid_of[pairs["attempt"].to_numpy()]).all():
        raise ValueError("pair table attempt index differs from the attempt table order")
    by_att = {a: g.index.to_numpy() for a, g in pairs.groupby("attempt", sort=True)}
    a_ids = sorted(by_att)
    if limit:
        step = max(1, len(a_ids) // limit)
        a_ids = a_ids[::step][:limit]
    tasks = []
    for a in a_ids:
        rows = by_att[a]
        src = pairs.loc[rows, "src"].to_numpy()
        r = att.iloc[a]
        tasks.append(dict(attempt_id=r["attempt_id"], tensor_path=r["tensor_path"],
                          completion_tokens=int(r["completion_tokens"]), n_reasoning=int(r["n_reasoning"]),
                          pair_rows=rows.tolist(), tok_start=sent["tok_start"].to_numpy()[src].tolist(),
                          tok_end=sent["tok_end"].to_numpy()[src].tolist(),
                          size=Path(r["tensor_path"]).stat().st_size, **meta))
    tasks.sort(key=lambda t: -t["size"])
    return tasks, pairs, meta


def run(out: Path, *, workers: int, limit: int | None) -> dict:
    t0 = time.time()
    if out.exists():
        raise FileExistsError(f"output exists (never overwritten): {out}")
    tmp = out.parent / f".{out.name}.tmp-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    tmp.mkdir(parents=True)
    tasks, pairs, meta = build_tasks(limit)
    n_pairs = int(sum(len(t["pair_rows"]) for t in tasks))
    log(f"antic: {len(tasks)} attempts, {n_pairs} pair sources, {workers} workers, spec={meta}")
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: "1" for v in thread_vars})
    results = []
    t_pool = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(antic_attempt, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 50 == 0 or i == len(tasks):
                    log(f"antic: {i}/{len(tasks)} attempts, {time.time() - t_pool:.0f}s")
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    pool_seconds = time.time() - t_pool
    results.sort(key=lambda r: r["attempt_id"])
    n = len(pairs)
    L, E = results[0]["win"].shape[1:] if len(results[0]["win"]) else results[0]["chk"].shape[1:]
    h = np.zeros((n, L, E), np.float16)
    has = np.zeros(n, bool)
    win_len = np.zeros(n, np.int64)
    filled = np.zeros(n, bool)
    dev_chk = 0.0
    dev_win = 0.0
    worst_last16 = 0.0
    sent_rows = pairs["src"].to_numpy()
    z = np.load(RESULTS / "gpt" / "A" / "hist.npz")
    last16, last_rows = z["last16"], z["last16_rows"]
    pos = np.searchsorted(last_rows, sent_rows)
    if not (last_rows[np.minimum(pos, len(last_rows) - 1)] == sent_rows).all():
        raise ValueError("stored last16 rows missing for a pair source")
    for r in results:
        rows = r["pair_rows"]
        if filled[rows].any():
            raise ValueError("pair row computed twice")
        filled[rows] = True
        if len(r["win_idx"]):
            tgt = rows[r["win_idx"]]
            h[tgt] = r["win"]
            has[tgt] = True
            win_len[tgt] = r["win_len"]
            dev_win = max(dev_win, float(np.abs(r["win"].astype(np.float32).sum(-1) - 1).max()))
        dev_chk = max(dev_chk, float(np.abs(r["chk"].astype(np.float32).sum(-1) - 1).max()))
        stored = last16[pos[rows]].astype(np.float32)
        worst_last16 = max(worst_last16, float(np.abs(stored - r["chk"].astype(np.float32)).max()))
    if not filled[np.concatenate([t["pair_rows"] for t in tasks])].all():
        raise ValueError("missing pair rows")
    checks = dict(pairs_computed=int(filled.sum()), windows_present=int(has.sum()),
                  windows_partial=int((has & (win_len < EDGE)).sum()), windows_empty=int((filled & ~has).sum()),
                  win_sum_dev=dev_win, last16_sum_dev=dev_chk, last16_max_abs_diff_vs_D1=worst_last16,
                  last16_ok=bool(worst_last16 < HIST_TOL16 and dev_win < HIST_TOL16 and dev_chk < HIST_TOL16))
    np.savez_compressed(tmp / "antic_hist.npz", h=h, has=has, win_len=win_len, computed=filled)
    prov = dict(limit=limit, attempts=len(tasks), pair_sources=n_pairs, workers=workers, spec=meta,
                pairs_sha256=sha256_file(OUT / "pairs.parquet"), host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), pool_seconds=pool_seconds,
                total_seconds=time.time() - t0, seconds_per_attempt_mean=float(np.mean(
                    [r["seconds"] for r in results])), seconds_per_attempt_max=float(max(
                        r["seconds"] for r in results)), checks=checks,
                time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (tmp / "provenance.json").write_text(json.dumps(prov, indent=1))
    tmp.rename(out)
    log(json.dumps(prov))
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args(argv)
    out = a.out or (OUT / ("antic_bench" if a.limit else "antic"))
    if not str(out.resolve()).startswith(str(W)):
        raise ValueError("outputs must stay under the dynamics-routing workspace")
    prov = run(out, workers=a.workers, limit=a.limit)
    return 0 if prov["checks"]["last16_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
