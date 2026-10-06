"""Stage 1: sentence-level routing table (sentences.parquet + hist.npz) per model and cohort.

One row per labelled sentence with at least one reasoning token. Alignment and tensor loading are
those of v3an (`v3an.extract._load_tensors`, the campaign-v3 `saved_layout` ownership rule and the
`v3an.align.class_token_map` label-matching rules); `moe_exp` comes from PYTHONPATH (frozen
snapshot). Histograms are per-layer selection frequencies (rows sum to 1 per layer).

CLI: python -m dynrt.extract MODEL COHORT [--limit N] [--out DIR] [--workers W] [--no-crosscheck]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (CLASSES, FAST, RESULTS, V3R2, W, abs_bin, cohort_dir, digest, keep_mask,
                     sha256_file)

CHUNK = 384
EDGE = 16
HIST_TOL16 = 4e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hist_frequencies(ids: np.ndarray, token_lists: list, num_experts: int) -> np.ndarray:
    """Per-sentence, per-layer expert selection frequencies, float32 [n, L, E] (rows sum to 1)."""
    L, _, k = ids.shape
    n = len(token_lists)
    out = np.zeros((n, L, num_experts), dtype=np.float32)
    layer = np.arange(L, dtype=np.int64)[:, None, None]
    for a in range(0, n, CHUNK):
        part = token_lists[a:a + CHUNK]
        lens = np.array([len(t) for t in part], dtype=np.int64)
        toks = np.concatenate(part).astype(np.int64)
        sid = np.repeat(np.arange(len(part), dtype=np.int64), lens)
        sel = np.asarray(ids[:, toks, :]).astype(np.int64)
        key = (sid[None, :, None] * L + layer) * num_experts + sel
        counts = np.bincount(key.ravel(), minlength=len(part) * L * num_experts)
        counts = counts.reshape(len(part), L, num_experts)
        out[a:a + len(part)] = counts / (lens[:, None, None] * k)
    return out


def match_labels(units: list, label_records: list) -> tuple[dict, dict]:
    """v3an.align.class_token_map label-matching rules: (label_of_unit, audit)."""
    audit = dict(records=0, matched=0, offset_mismatch=0, out_of_range=0, unknown_label=0,
                 conflicting=0, units=len(units))
    label_of_unit: dict[int, str] = {}
    conflicted = set()
    for idx, start, end, label in label_records:
        audit["records"] += 1
        if not 0 <= idx < len(units):
            audit["out_of_range"] += 1
            continue
        u = units[idx]
        if u["start"] != start or u["end"] != end:
            audit["offset_mismatch"] += 1
            continue
        if label not in CLASSES:
            audit["unknown_label"] += 1
            continue
        if idx in label_of_unit and label_of_unit[idx] != label:
            conflicted.add(idx)
            continue
        label_of_unit[idx] = label
    for idx in conflicted:
        label_of_unit.pop(idx, None)
        audit["conflicting"] += 1
    audit["matched"] = len(label_of_unit)
    return label_of_unit, audit


def extract_attempt(task: dict) -> dict:
    """Worker: one attempt -> sentence rows, histograms and checks."""
    t0 = time.time()
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest
    from moe_exp.schemas import TraceRecord
    from v3an.data import read_location
    from v3an.extract import _load_tensors

    ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
    del weights, gaps
    L, T, k = ids.shape
    E = int(task["num_experts"])
    aid = task["attempt_id"]
    if T != int(task["completion_tokens"]):
        raise ValueError(f"{aid}: tensor tokens {T} != completion {task['completion_tokens']}")
    if (L, k) != (int(task["num_layers"]), int(task["top_k"])):
        raise ValueError(f"{aid}: tensor layout {ids.shape} differs from spec")
    if ids.min() < 0 or ids.max() >= E:
        raise ValueError(f"{aid}: expert id out of range")
    t_load = time.time() - t0

    t1 = time.time()
    trace_dict = read_location(json.loads(task["source_location"]))
    trace = TraceRecord(**trace_dict)
    if trace_digest(trace) != task["trace_sha256"]:
        raise ValueError(f"{aid}: trace digest differs from the captured trace")
    layout = saved_layout(trace)
    if layout["token_count"] is None or int(layout["token_count"]) != T:
        raise ValueError(f"{aid}: replay token count differs from capture")
    if list(layout["reasoning_tokens"]) != reasoning.tolist():
        raise ValueError(f"{aid}: replayed reasoning-token set differs from the capture tensor")
    units, owners = layout["units"], layout["unit_tokens"]
    ranges = reasoning_ranges(trace)
    n_r = len(reasoning)
    label_of_unit, audit = match_labels(units, task["labels"])
    owned = sum(len(o) for o in owners if o)
    comp_to_rank = np.full(T, -1, dtype=np.int64)
    comp_to_rank[reasoning] = np.arange(n_r)

    idx_rows, tok_lists, seg_rows = [], [], []
    tokenless = 0
    for idx in sorted(label_of_unit):
        toks = owners[idx] or []
        if not toks:
            tokenless += 1
            continue
        idx_rows.append(idx)
        tok_lists.append(np.asarray(toks, dtype=np.int64))
        start = units[idx]["start"]
        seg_rows.append(next(j for j, (a, b) in enumerate(ranges) if a <= start < b))
    n_s = len(idx_rows)
    checks = dict(tokens_outside_reasoning=0, non_ascending=0, span_outside=0, noncontiguous=0,
                  owned_tokens_ne_reasoning=int(owned != n_r), tokenless_labelled=tokenless)
    tok_start = np.zeros(n_s, np.int64)
    tok_end = np.zeros(n_s, np.int64)
    n_tok = np.zeros(n_s, np.int64)
    for i, toks in enumerate(tok_lists):
        r = comp_to_rank[toks]
        if (r < 0).any():
            checks["tokens_outside_reasoning"] += 1
            continue
        if len(toks) > 1 and not (np.diff(toks) > 0).all():
            checks["non_ascending"] += 1
        tok_start[i], tok_end[i], n_tok[i] = r.min(), r.max() + 1, len(toks)
        checks["span_outside"] += int(tok_end[i] > n_r or tok_start[i] < 0)
        checks["noncontiguous"] += int(tok_end[i] - tok_start[i] != len(toks))
    t_align = time.time() - t1

    t2 = time.time()
    labelled_set = set(idx_rows)
    seg_of = dict(zip(idx_rows, seg_rows))
    has_prev = np.array([(i - 1) in labelled_set for i in idx_rows], dtype=bool)
    has_next = np.array([(i + 1) in labelled_set for i in idx_rows], dtype=bool)
    prev_seg = np.array([has_prev[j] and seg_of[i - 1] == seg_of[i] for j, i in enumerate(idx_rows)],
                        dtype=bool)
    next_seg = np.array([has_next[j] and seg_of[i + 1] == seg_of[i] for j, i in enumerate(idx_rows)],
                        dtype=bool)
    h32 = hist_frequencies(ids, tok_lists, E) if n_s else np.zeros((0, L, E), np.float32)
    dev32 = float(np.abs(h32.sum(-1) - 1.0).max()) if n_s else 0.0
    h16 = h32.astype(np.float16)
    dev16 = float(np.abs(h16.astype(np.float32).sum(-1) - 1.0).max()) if n_s else 0.0
    first_rows = np.flatnonzero(has_prev)
    last_rows = np.flatnonzero(has_next)
    f16 = hist_frequencies(ids, [tok_lists[i][:EDGE] for i in first_rows], E).astype(np.float16) \
        if len(first_rows) else np.zeros((0, L, E), np.float16)
    l16 = hist_frequencies(ids, [tok_lists[i][-EDGE:] for i in last_rows], E).astype(np.float16) \
        if len(last_rows) else np.zeros((0, L, E), np.float16)
    t_hist = time.time() - t2

    rows = dict(
        sentence_index=np.asarray(idx_rows, np.int32),
        cls=np.array([CLASSES.index(label_of_unit[i]) for i in idx_rows], np.int8),
        segment=np.asarray(seg_rows, np.int16), tok_start=tok_start.astype(np.int32),
        tok_end=tok_end.astype(np.int32), n_tokens=n_tok.astype(np.int32),
        has_prev_labelled=has_prev, has_next_labelled=has_next,
        prev_same_segment=prev_seg, next_same_segment=next_seg)
    audit.update(labelled_tokens=int(n_tok.sum()), rows=n_s, n_reasoning=n_r, **checks,
                 hist_sum_dev32=dev32, hist_sum_dev16=dev16, edge_sum_dev=0.0)
    if len(f16):
        audit["edge_sum_dev"] = max(float(np.abs(f16.astype(np.float32).sum(-1) - 1).max()),
                                    float(np.abs(l16.astype(np.float32).sum(-1) - 1).max())
                                    if len(l16) else 0.0)
    elif len(l16):
        audit["edge_sum_dev"] = float(np.abs(l16.astype(np.float32).sum(-1) - 1).max())

    if task.get("crosscheck"):
        from v3an.align import class_token_map
        amap = class_token_map(trace_dict, task["labels"], expected_tokens=T,
                               expected_reasoning=reasoning.tolist(),
                               expected_digest=task["trace_sha256"])
        v3 = amap["audit"]
        cot = amap["class_of_token"]
        mine = np.full(T, -1, np.int8)
        for toks, c in zip(tok_lists, rows["cls"]):
            mine[toks] = c
        audit["v3an_matched"] = int(v3["matched"])
        audit["v3an_labelled_tokens"] = int(v3["labelled_tokens"])
        audit["v3an_class_of_token_equal"] = bool(np.array_equal(cot, mine))
        audit["v3an_units_equal"] = bool(v3["units"] == len(units))
    audit["attempt_id"] = aid
    audit.update(seconds=time.time() - t0, load_seconds=t_load, align_seconds=t_align,
                 hist_seconds=t_hist, tensor_tokens=T)
    return dict(attempt_id=aid, rows=rows, h=h16, first=f16, first_rows=first_rows, last=l16,
                last_rows=last_rows, audit=audit)


def build_tasks(model: str, cohort: str, limit: int | None, crosscheck: bool):
    """Attempt tasks (sorted by size, descending) plus provenance."""
    from v3an.data import load_labels
    att = pd.read_parquet(V3R2 / model / cohort / "attempts.parquet")
    ready = json.loads((FAST / "manifests" / model / cohort / "ready.json").read_text())
    spec = ready["spec"]
    meta = dict(num_experts=int(spec["experts"]), num_layers=len(spec["layers"]),
                top_k=int(spec["top_k"]))
    n_all = len(att)
    keep = keep_mask(model, att["question"].to_numpy())
    att = att[keep].sort_values("attempt_id").reset_index(drop=True)
    if limit:
        step = max(1, len(att) // limit)
        att = att.iloc[::step].iloc[:limit].reset_index(drop=True)
    labels, label_prov = load_labels(model)
    tasks, missing = [], 0
    for r in att.itertuples(index=False):
        lab = labels.get((r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256))
        if lab is None:
            missing += 1
            lab = []
        tasks.append(dict(attempt_id=r.attempt_id, tensor_path=r.tensor_path,
                          completion_tokens=int(r.completion_tokens), source_location=r.source_location,
                          trace_sha256=r.trace_sha256, labels=lab, crosscheck=crosscheck, **meta,
                          size=Path(r.tensor_path).stat().st_size))
    tasks.sort(key=lambda t: -t["size"])
    prov = dict(model=model, cohort=cohort, attempts_total=n_all, attempts_used=len(tasks),
                attempts_without_labels=missing, spec=meta, label_files=label_prov,
                attempts_parquet_sha256=sha256_file(V3R2 / model / cohort / "attempts.parquet"),
                ready_sha256=sha256_file(FAST / "manifests" / model / cohort / "ready.json"))
    return att, tasks, prov


def code_hashes() -> dict:
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def run(model: str, cohort: str, out: Path, *, workers: int, limit: int | None, crosscheck: bool):
    t0 = time.time()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output exists (never overwritten): {out}")
    tmp = out.parent / f".{out.name}.tmp-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    att, tasks, prov = build_tasks(model, cohort, limit, crosscheck)
    log(f"extract {model}/{cohort}: {len(tasks)} attempts of {prov['attempts_total']}, "
        f"{workers} workers, crosscheck={crosscheck}, spec={prov['spec']}")
    results = []
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {v: os.environ.get(v) for v in thread_vars}
    os.environ.update({v: "1" for v in thread_vars})
    t_pool = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(extract_attempt, t): t["attempt_id"] for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 25 == 0 or i == len(tasks):
                    log(f"extract: {i}/{len(tasks)} done, {time.time() - t_pool:.0f}s")
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    pool_seconds = time.time() - t_pool
    results.sort(key=lambda r: r["attempt_id"])
    a = att.set_index("attempt_id")
    frames, hist, first, last, first_rows, last_rows = [], [], [], [], [], []
    offset = 0
    for r in results:
        n = len(r["rows"]["cls"])
        row = a.loc[r["attempt_id"]]
        df = pd.DataFrame(r["rows"])
        df.insert(0, "attempt_id", r["attempt_id"])
        df.insert(1, "question", row["question"])
        df.insert(2, "dataset", row["dataset"])
        df["n_reasoning"] = int(r["audit"]["n_reasoning"])
        frames.append(df)
        hist.append(r["h"])
        first.append(r["first"])
        last.append(r["last"])
        first_rows.append(r["first_rows"] + offset)
        last_rows.append(r["last_rows"] + offset)
        offset += n
    sent = pd.concat(frames, ignore_index=True)
    sent["cls_name"] = np.array(CLASSES)[sent["cls"].to_numpy()]
    sent["abs_bin"] = abs_bin(sent["tok_start"].to_numpy())
    sent["rel_pos"] = (sent["tok_start"] / sent["n_reasoning"]).astype(np.float64)
    sent.to_parquet(tmp / "sentences.parquet")
    np.savez_compressed(tmp / "hist.npz", h=np.concatenate(hist), first16=np.concatenate(first),
                        first16_rows=np.concatenate(first_rows).astype(np.int64),
                        last16=np.concatenate(last), last16_rows=np.concatenate(last_rows).astype(np.int64))
    meta = pd.DataFrame([r["audit"] for r in results])
    meta.to_csv(tmp / "extract_meta.csv", index=False)
    checks = summarise_checks(model, cohort, meta, a, pool_seconds)
    prov.update(host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                code_sha256=code_hashes(), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                total_seconds=time.time() - t0, pool_seconds=pool_seconds, limit=limit,
                rows=int(len(sent)), checks=checks)
    (tmp / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    tmp.rename(out)
    log(f"extract {model}/{cohort}: {len(sent)} sentence rows -> {out} in {time.time() - t0:.0f}s")
    log(json.dumps(checks, default=str))
    return checks


def summarise_checks(model: str, cohort: str, meta: pd.DataFrame, att: pd.DataFrame,
                     pool_seconds: float) -> dict:
    """Compare with v3an's alignment_audit (A) / v3an class_token_map (cross-check) and internal checks."""
    out: dict = dict(attempts=int(len(meta)), sentence_rows=int(meta["rows"].sum()),
                     pool_seconds=pool_seconds,
                     seconds_per_attempt_mean=float(meta["seconds"].mean()),
                     seconds_per_attempt_max=float(meta["seconds"].max()))
    for c in ("tokens_outside_reasoning", "non_ascending", "span_outside", "noncontiguous",
              "owned_tokens_ne_reasoning", "tokenless_labelled", "offset_mismatch", "out_of_range",
              "unknown_label", "conflicting"):
        out[c] = int(meta[c].sum())
    out["hist_sum_dev32_max"] = float(meta["hist_sum_dev32"].max())
    out["hist_sum_dev16_max"] = float(meta["hist_sum_dev16"].max())
    out["edge_sum_dev_max"] = float(meta["edge_sum_dev"].max())
    out["hist_ok"] = bool(out["hist_sum_dev32_max"] < 1e-4 and out["hist_sum_dev16_max"] < HIST_TOL16
                          and out["edge_sum_dev_max"] < HIST_TOL16)
    m = meta.set_index("attempt_id")
    nr = att["n_reasoning"].reindex(m.index)
    out["n_reasoning_mismatch_vs_v3an_attempts"] = int((nr.to_numpy() != m["n_reasoning"].to_numpy()).sum())
    if "v3an_matched" in m:
        out["crosscheck_matched_mismatch"] = int((m["matched"] != m["v3an_matched"]).sum())
        out["crosscheck_tokens_mismatch"] = int((m["labelled_tokens"] != m["v3an_labelled_tokens"]).sum())
        out["crosscheck_class_of_token_unequal"] = int((~m["v3an_class_of_token_equal"]).sum())
    audit_csv = V3R2 / model / cohort / "alignment_audit.csv"
    if audit_csv.exists():
        aud = pd.read_csv(audit_csv).set_index("attempt_id").reindex(m.index)
        for col in ("records", "matched", "labelled_tokens", "units", "offset_mismatch"):
            out[f"v3an_audit_{col}_mismatch"] = int((aud[col].to_numpy() != m[col].to_numpy()).sum())
        out["v3an_audit_matched_total"] = int(aud["matched"].sum())
        out["ours_matched_total"] = int(m["matched"].sum())
    else:
        out["v3an_audit"] = "absent for this cohort (v3an loaded labels only for cohort A)"
        out["label_records_total"] = int(m["records"].sum())
        out["ours_matched_total"] = int(m["matched"].sum())
    bad = [k for k, v in out.items() if k.endswith("_mismatch") and v]
    bad += [k for k in ("tokens_outside_reasoning", "non_ascending", "span_outside",
                        "owned_tokens_ne_reasoning", "offset_mismatch", "out_of_range",
                        "unknown_label", "conflicting") if out[k]]
    bad += [] if out["hist_ok"] else ["hist_ok"]
    out["failed_checks"] = bad
    out["digest"] = digest({k: v for k, v in out.items() if k not in ("pool_seconds",)
                            and not k.startswith("seconds")})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=("gpt", "qwen36"))
    p.add_argument("cohort", choices=("A", "B"))
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--no-crosscheck", action="store_true")
    a = p.parse_args(argv)
    out = a.out or cohort_dir(a.model, a.cohort, RESULTS)
    if not str(out.resolve()).startswith(str(W)):
        raise ValueError("outputs must stay under the dynamics-routing workspace")
    checks = run(a.model, a.cohort, out, workers=a.workers, limit=a.limit,
                 crosscheck=not a.no_crosscheck)
    return 1 if checks["failed_checks"] else 0


if __name__ == "__main__":
    sys.exit(main())
