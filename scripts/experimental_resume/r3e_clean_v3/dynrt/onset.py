"""C6 extraction (qwen36 dev+tune cohort A): first lexicon-v1 onset decision and its 64-token routing window.

Decisions come from the frozen offline reference of steering-v1 (`moe_steer.trigger`, read-only import):
`offline_sentence_matches` on the reasoning text (the function `onsets_offline` is built on; the two
are asserted equal). j is an output index (o = 0 is the first reasoning token). The first decision
with 64 <= j <= 8192 is kept together with the identity of its marker and the per-layer expert
frequencies of the reasoning tokens [j-63, j]. No outcome is read.

CLI: python -m dynrt.onset [--limit N] [--out DIR] [--workers W]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import FAST, ROOT, V3R2, cohort_dir, keep_mask, sha256_file

STEER_CODE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/steering")
LEXICON_PATH = ROOT / "steering-v1/manifests/lexicon-v1.json"
LEXICON_SEAL = "232f39bef891f2c587a12dfc316cc030740c924236c716de27b97b35960cb814"
A4_PATH = ROOT / "steering-v1/runtime/a4-dev-firing.json"
J_MIN, J_MAX, WINDOW = 64, 8192, 64
MARKERS = ("can we", "could", "is it possible", "maybe", "or maybe", "what")   # shortest-match identities


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _trigger():
    if str(STEER_CODE) not in sys.path:
        sys.path.append(str(STEER_CODE))
    from moe_steer import trigger
    return trigger


def decisions_of(record: dict, lexicon, trigger, verify: bool = False) -> dict:
    """All onset decisions of a trace (offline reference)."""
    replay = record["metadata"]["token_replay"]
    ids, offsets, cot = replay["completion_token_ids"], replay["completion_offsets"], record["cot_text"]
    think = ids.index(trigger.THINK_END_ID) if trigger.THINK_END_ID in ids else None
    boundary = offsets[think][0] if think is not None else len(cot)
    text = cot[:boundary]
    rows = trigger.offline_sentence_matches(text, offsets, lexicon, full_text=cot)
    dec = [(int(r["decision"]), r["marker"]) for r in rows if r["onset"] and r["decision"] is not None]
    if verify:
        ref = trigger.onsets_offline(text, offsets, lexicon, full_text=cot)
        if [j for j, _ in dec] != [int(j) for j in ref]:
            raise ValueError("decisions differ from onsets_offline")
    return dict(decisions=dec, think_end=think, n_tokens=len(ids))


def onset_attempt(task: dict) -> dict:
    t0 = time.time()
    from v3an.data import read_location
    from v3an.extract import _load_tensors

    from .extract import hist_frequencies
    trigger = _trigger()
    lexicon = trigger.Lexicon.load(LEXICON_PATH)
    record = read_location(json.loads(task["source_location"]))
    if (record["dataset"], record["problem_id"], int(record["sample_id"])) != (
            task["dataset"], task["problem_id"], task["sample_id"]):
        raise ValueError("trace identity mismatch")
    d = decisions_of(record, lexicon, trigger, task.get("verify", False))
    in_range = [(j, m) for j, m in d["decisions"] if J_MIN <= j <= J_MAX]
    out = dict(attempt_id=task["attempt_id"], n_decisions=len(d["decisions"]),
               n_in_range=len(in_range), think_end=d["think_end"], n_tokens=d["n_tokens"],
               decisions=json.dumps([j for j, _ in d["decisions"][:50]]), j=-1, marker="",
               j_rank=-1, window_ok=False, h=None)
    if in_range:
        j, marker = in_range[0]
        ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
        del weights, gaps
        if ids.shape[1] != d["n_tokens"]:
            raise ValueError("tensor tokens differ from the replay")
        pos = np.searchsorted(reasoning, j)
        ok = pos < len(reasoning) and reasoning[pos] == j and pos >= WINDOW - 1
        out.update(j=j, marker=marker, j_rank=int(pos) if ok else -1, window_ok=bool(ok))
        if ok:
            comp = reasoning[pos - WINDOW + 1: pos + 1].astype(np.int64)
            out["h"] = hist_frequencies(ids, [comp], task["num_experts"])[0].astype(np.float16)
            out["window_contiguous"] = bool(comp[-1] - comp[0] == WINDOW - 1)
    out["seconds"] = time.time() - t0
    return out


def build_tasks(limit: int | None):  # verify onsets_offline equality only on the small benchmark
    att = pd.read_parquet(V3R2 / "qwen36" / "A" / "attempts.parquet")
    att = att[keep_mask("qwen36", att["question"].to_numpy())].sort_values("attempt_id")
    att = att.reset_index(drop=True)
    if limit:
        att = att.iloc[::max(1, len(att) // limit)].iloc[:limit].reset_index(drop=True)
    ready = json.loads((FAST / "manifests/qwen36/A/ready.json").read_text())
    E = int(ready["spec"]["experts"])
    tasks = [dict(attempt_id=r.attempt_id, tensor_path=r.tensor_path, source_location=r.source_location,
                  dataset=r.dataset, problem_id=r.problem_id, sample_id=int(r.sample_id), num_experts=E,
                  verify=bool(limit),
                  size=Path(r.tensor_path).stat().st_size) for r in att.itertuples(index=False)]
    tasks.sort(key=lambda t: -t["size"])
    return att, tasks


def a4_agreement(att: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Compare with the A4 dev firing table (streaming onsets per dev trace; ~1% of traces differ from
    the offline reference by design) where the trace is covered."""
    per = {tuple(r["key"]): r for r in json.loads(A4_PATH.read_text())["per_trace"]}
    m_by = meta.set_index("attempt_id")
    covered = same = first_covered = first_same = 0
    for r in att.itertuples(index=False):
        key = (r.dataset, r.source_problem_id, int(r.sample_id))
        if key not in per:
            continue
        m = m_by.loc[r.attempt_id]
        ref = per[key]["onsets"]
        covered += 1
        same += len(ref) == m["n_decisions"] and json.loads(m["decisions"]) == ref[:50]
        ref_first = next((j for j in ref if J_MIN <= j <= J_MAX), None)
        first_covered += 1
        first_same += (ref_first is None and m["j"] < 0) or ref_first == m["j"]
    return dict(a4_traces_covered=covered, decision_lists_identical_to_a4_streaming=same,
                first_j_covered=first_covered, first_j_identical_to_a4_streaming=first_same)


def run(out: Path, workers: int, limit: int | None) -> dict:
    t0 = time.time()
    if out.exists():
        raise FileExistsError(out)
    tmp = out.parent / f".{out.name}.tmp-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    att, tasks = build_tasks(limit)
    lexicon_sha = sha256_file(LEXICON_PATH)
    trig = _trigger()
    lex = trig.Lexicon.load(LEXICON_PATH)
    body = json.loads(LEXICON_PATH.read_text())
    if body["sha256"] != LEXICON_SEAL:
        raise ValueError("lexicon seal mismatch")
    log(f"onset extraction: {len(tasks)} attempts, lexicon markers {lex.markers}")
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: "1" for v in thread_vars})
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(onset_attempt, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 50 == 0 or i == len(tasks):
                    log(f"onset: {i}/{len(tasks)} done, {time.time() - t0:.0f}s")
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    results.sort(key=lambda r: r["attempt_id"])
    hs = [r.pop("h") for r in results]
    meta = pd.DataFrame(results)
    meta.to_parquet(tmp / "onset_meta.parquet")
    with_h = [i for i, h in enumerate(hs) if h is not None]
    np.savez_compressed(tmp / "onset_hist.npz", rows=np.asarray(with_h, np.int64),
                        h=np.stack([hs[i] for i in with_h]) if with_h else np.zeros((0, 1, 1), np.float16),
                        attempt_id=meta["attempt_id"].to_numpy()[with_h].astype(str))
    checks = dict(
        attempts=int(len(meta)), with_decision_in_range=int((meta["n_in_range"] > 0).sum()),
        window_ok=int(meta["window_ok"].sum()),
        window_noncontiguous=int((meta["window_contiguous"] == False).sum())  # noqa: E712
        if "window_contiguous" in meta else 0,
        j_rank_ne_j=int(((meta["j_rank"] != meta["j"]) & (meta["j"] >= 0)).sum()),
        no_decision_at_all=int((meta["n_decisions"] == 0).sum()),
        markers=meta.loc[meta["j"] >= 0, "marker"].value_counts().to_dict(),
        j_quantiles={q: float(meta.loc[meta["j"] >= 0, "j"].quantile(q)) for q in (0.1, 0.5, 0.9)},
        a4=a4_agreement(att, meta), lexicon_sha256=lexicon_sha, seconds=time.time() - t0)
    (tmp / "onset_provenance.json").write_text(json.dumps(dict(
        checks=checks, code_sha256=sha256_file(Path(__file__)), slurm_job=os.environ.get("SLURM_JOB_ID")),
        indent=1, default=str))
    tmp.rename(out)
    log(json.dumps(checks, default=str))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=cohort_dir("qwen36", "A") / "onset")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = ap.parse_args(argv)
    checks = run(a.out, a.workers, a.limit)
    bad = checks["window_ok"] != checks["with_decision_in_range"] or checks["j_rank_ne_j"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
