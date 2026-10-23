"""Stage 3: K2 (confirmatory, F-C) anticipation hazard from the last 16 / 32 reasoning tokens of sentence s.

  python -m rkin.k2 prep gpt|qwen36|pilot         pair tables, lexical counts, kinematic columns -> data/k2/<m>/
  python -m rkin.k2 calibrate MODEL               (s_text, s_lex) per target, baseline-only inner CV of (r0, f0)
  python -m rkin.k2 cv MODEL TARGET               base / aug16 / aug32 cross-fit (4 outer x 5 repeats, inner 3)
  python -m rkin.k2 nullsim MODEL TARGET          20 augmentations of 15 iid N(0,1) columns (repeats 1)

TARGET: switch (1[b != a], all pairs, 2-class softmax) or dest (class of s+1 over switch pairs, 7-class).
Machinery = P-A1 (dynamics-routing/dynrt d3a_*; read-only imports): multinomial ridge on a dense block plus
fold-fitted tf-idf sparse blocks, question-weighted log-loss, A1 fold ids, paired question bootstrap.
MODEL pilot = DENSE descriptive (switch only; no text n-grams, no gate columns, 16 questions).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .common import (CLASSES, DATA, DYN_RESULTS, OUT, abs_bin, code_hashes, dep_hashes, log, sha256_file,
                     verify_prereg, write_json)

K2DATA = DATA / "k2"
REPEATS, OUTER, INNER = 5, 4, 3
SCALES = (1.0, 3.0, 10.0)
N_NULL = 20
NULL_SEED = 20261001
GAIN_MIN = 0.005
KIN_NAMES = [f"{f}_{t}" for f in ("v_mean", "u_mean", "half_jsd", "accel", "v_end") for t in ("d1", "d2", "d3")]
NON_GATE = [i for i, n in enumerate(KIN_NAMES) if not n.startswith("u_mean")]
MODELS = {"base": ["base"], "aug16": ["base", "k16"], "aug32": ["base", "k32"]}


# ------------------------------------------------------------------------------------ lexical

def _bucket(prefix: str, value: str) -> int:
    from moe_exp.routing.dynrt.d3a_text import N_BUCKETS
    return zlib.crc32((prefix + value).encode("utf-8")) & (N_BUCKETS - 1)


def lex_counts(tok_last: np.ndarray, tok_last2: np.ndarray, bag: np.ndarray) -> sparse.csr_matrix:
    """Hashed lexical counts [n, 2^18]: last token id of s, last token-id bigram, bag of the window token ids.

    Negative ids (positions before the reasoning start) are skipped.
    """
    from moe_exp.routing.dynrt.d3a_text import N_BUCKETS
    cache: dict = {}

    def b(prefix, value):
        key = (prefix, value)
        if key not in cache:
            cache[key] = _bucket(prefix, value)
        return cache[key]

    rows, cols = [], []
    for i in range(len(tok_last)):
        ids = []
        if tok_last[i] >= 0:
            ids.append(b("t1|", str(int(tok_last[i]))))
            if tok_last2[i] >= 0:
                ids.append(b("t2|", f"{int(tok_last2[i])} {int(tok_last[i])}"))
        ids += [b("tb|", str(int(t))) for t in bag[i] if t >= 0]
        rows.append(np.full(len(ids), i, np.int64))
        cols.append(np.asarray(ids, np.int64))
    r, c = np.concatenate(rows), np.concatenate(cols)
    m = sparse.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(tok_last), N_BUCKETS)).tocsr()
    m.sum_duplicates()
    m.sort_indices()
    return m


# --------------------------------------------------------------------------------------- prep

def _events_join(model: str, keys: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Row positions in data/<model>/events.* of the pairs keyed by (att, sentence_index), plus arrays."""
    ev = pd.read_parquet(DATA / model / "events.parquet")
    idx = pd.MultiIndex.from_frame(ev[["att", "sentence_index"]])
    pos = idx.get_indexer(pd.MultiIndex.from_frame(keys[["att", "sentence_index"]]))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} pairs without an RK2 event row")
    z = np.load(DATA / model / "events.npz")
    return pos, dict(ev=ev.iloc[pos].reset_index(drop=True), k2=z["k2"][pos], lex=z["lex"][pos])


def prep(model: str) -> dict:
    """Pair table + dense baseline + text/lexical counts + kinematic columns for one model."""
    pre = verify_prereg()
    out_dir = K2DATA / model
    if out_dir.exists():
        raise FileExistsError(f"exists (never overwritten): {out_dir}")
    tmp = out_dir.parent / f".{model}.tmp-{os.getpid()}"
    tmp.mkdir(parents=True)
    from moe_exp.routing.dynrt.common import rel_bin
    info: dict = dict(model=model)
    if model == "pilot":
        ev = pd.read_parquet(DATA / "pilot" / "events.parquet")
        att = pd.read_parquet(DATA / "pilot" / "attempts.parquet")
        start = (ev["r"] - ev["n_tokens_s"]).to_numpy(np.int64)
        pairs = pd.DataFrame(dict(att=ev["att"], sentence_index=ev["sentence_index"], question=ev["question"],
                                  a=ev["a"], b=ev["b"], n_tokens=ev["n_tokens_s"], abs_bin=abs_bin(start),
                                  rel_bin=rel_bin(start / ev["n_reasoning"].to_numpy(float)).astype(np.int64)))
        pos, j = _events_join("pilot", pairs)
        att_q = att.sort_values("att")["question"].to_numpy()
        text_src = None
    else:
        from moe_exp.routing.dynrt.d3a_pairs import build_pair_table
        from moe_exp.routing.dynrt.data import load_attempts
        att_dyn = load_attempts(model, "A")
        att_rk = pd.read_parquet(DATA / model / "attempts.parquet").sort_values("att")
        if not np.array_equal(att_dyn["attempt_id"].to_numpy(), att_rk["attempt_id"].to_numpy()):
            raise ValueError("RK2 attempt order differs from the dynrt cohort order")
        att_q = att_dyn["question"].to_numpy()
        sent = pd.read_parquet(DYN_RESULTS / model / "A" / "sentences.parquet", columns=["rel_pos"])
        if model == "gpt":
            p = pd.read_parquet(DYN_RESULTS / "A1" / "pairs.parquet")
            text_src = DYN_RESULTS / "A1" / "text_counts.npz"
            info["a1_inputs"] = {"A1/pairs.parquet": sha256_file(DYN_RESULTS / "A1" / "pairs.parquet"),
                                 "A1/text_counts.npz": sha256_file(text_src)}
        else:
            from moe_exp.routing.dynrt.d3a_text import hash_counts
            ps, summ = build_pair_table(model, "A")
            p = ps.df
            sparse.save_npz(tmp / "text_counts.npz", hash_counts(p["text"].tolist()), compressed=True)
            text_src = out_dir / "text_counts.npz"
            info["pair_summary"] = {k: v for k, v in summ.items() if k != "per_source_class"}
        pairs = pd.DataFrame(dict(att=p["attempt"].to_numpy(np.int64), sentence_index=p["sentence_index"],
                                  question=p["question"], dataset=p["dataset"], a=p["a"], b=p["b"],
                                  n_tokens=p["n_tokens"], abs_bin=p["abs_bin"], difficulty=p["difficulty"],
                                  tok_end=p["tok_end"],
                                  rel_bin=rel_bin(sent["rel_pos"].to_numpy()[p["src"].to_numpy()]).astype(np.int64)))
        pos, j = _events_join(model, pairs)
        e = j["ev"]
        bad = int(((e["a"].to_numpy() != pairs["a"].to_numpy()) | (e["b"].to_numpy() != pairs["b"].to_numpy())
                   | (e["r"].to_numpy() != pairs["tok_end"].to_numpy())).sum())
        if bad:
            raise ValueError(f"{bad} pairs disagree with the RK2 event rows (class or boundary rank)")
    e = j["ev"]
    pairs["event_row"] = pos
    pairs["r"] = e["r"].to_numpy()
    pairs["tok_last"], pairs["tok_last2"] = e["tok_last"].to_numpy(), e["tok_last2"].to_numpy()
    lex = lex_counts(pairs["tok_last"].to_numpy(), pairs["tok_last2"].to_numpy(), j["lex"])
    sparse.save_npz(tmp / "lex_counts.npz", lex, compressed=True)
    np.savez_compressed(tmp / "kin.npz", k16=j["k2"][:, 0].astype(np.float64), k32=j["k2"][:, 1].astype(np.float64),
                        att_questions=att_q.astype(str))
    pairs.to_parquet(tmp / "pairs.parquet")
    info.update(n_pairs=int(len(pairs)), n_switch=int((pairs["a"] != pairs["b"]).sum()),
                n_questions=int(pairs["question"].nunique()), lex_nnz_per_row=float(lex.nnz / max(len(pairs), 1)),
                kin_nan_share={w: np.isnan(j["k2"][:, i]).mean(0).round(4).tolist() for i, w in enumerate(("k16", "k32"))},
                text_counts=str(text_src) if text_src else None, prereg=pre, code=code_hashes(), deps=dep_hashes(),
                host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    write_json(tmp / "provenance.json", info)
    tmp.rename(out_dir)
    log(f"K2 prep {model}: {json.dumps({k: info[k] for k in ('n_pairs', 'n_switch', 'n_questions')})}")
    return info


# ---------------------------------------------------------------------------------------- data

@dataclass
class KData:
    model: str
    target: str
    K: int
    y: np.ndarray
    a: np.ndarray
    groups: np.ndarray
    w: np.ndarray
    dense: dict                       # name -> raw [n, d]
    text: sparse.csr_matrix | None
    lex: sparse.csr_matrix
    outer_folds: np.ndarray           # [R, n]
    s_text: float = 1.0
    s_lex: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)


def baseline_dense(pairs: pd.DataFrame, pilot: bool) -> tuple[np.ndarray, list]:
    """A1 dense block (+ relative quintile); DENSE pilot: class, log length, position bin, relative quintile."""
    if pilot:
        a = pairs["a"].to_numpy(np.int64)
        cols = [(a == c).astype(float) for c in range(7)] + [np.log(pairs["n_tokens"].to_numpy(float))]
        names = [f"cls{c}" for c in range(7)] + ["log_len"]
        ab = pairs["abs_bin"].to_numpy(np.int64)
        cols += [(ab == b).astype(float) for b in range(6)]
        names += [f"abs{b}" for b in range(6)]
        base = np.column_stack(cols)
    else:
        from moe_exp.routing.dynrt.d3a_cv import baseline_matrix
        base, names = baseline_matrix(pairs)
    rb = pairs["rel_bin"].to_numpy(np.int64)
    rel = np.column_stack([(rb == q).astype(float) for q in range(5)])
    return np.column_stack([base, rel]), list(names) + [f"rel{q}" for q in range(5)]


def load_kdata(model: str, target: str, smoke: int = 0) -> KData:
    from moe_exp.routing.dynrt.cv import question_folds
    from moe_exp.routing.dynrt.d3a_cv import question_weights
    d = K2DATA / model
    pairs = pd.read_parquet(d / "pairs.parquet")
    lex = sparse.load_npz(d / "lex_counts.npz").tocsr()
    kin = np.load(d / "kin.npz")
    prov = json.loads((d / "provenance.json").read_text())
    text = sparse.load_npz(prov["text_counts"]).tocsr() if prov["text_counts"] else None
    att_q = kin["att_questions"]
    att_idx = pairs["att"].to_numpy(np.int64)
    folds = np.stack([question_folds(att_q, OUTER, r)[att_idx] for r in range(REPEATS)])
    base, _ = baseline_dense(pairs, model == "pilot")
    a = pairs["a"].to_numpy(np.int64)
    b = pairs["b"].to_numpy(np.int64)
    k16, k32 = kin["k16"], kin["k32"]
    if model == "pilot":
        k16, k32 = k16[:, NON_GATE], k32[:, NON_GATE]
    rows = np.arange(len(pairs)) if target == "switch" else np.flatnonzero(a != b)
    if smoke:
        qs = np.array(sorted(set(pairs["question"])))[:smoke]
        rows = rows[np.isin(pairs["question"].to_numpy()[rows], qs)]
    groups = pairs["question"].to_numpy()[rows]
    y = (b != a).astype(np.int64)[rows] if target == "switch" else b[rows]
    return KData(model=model, target=target, K=2 if target == "switch" else 7, y=y, a=a[rows], groups=groups,
                 w=question_weights(groups), dense=dict(base=base[rows], k16=k16[rows], k32=k32[rows]),
                 text=None if text is None else text[rows], lex=lex[rows], outer_folds=folds[:, rows])


# ------------------------------------------------------------------------------------------ CV

class Blocks:
    """Fold-fitted design (tf-idf of text and lexical counts, standardised dense blocks)."""

    def __init__(self, d: KData, tr: np.ndarray, evs: list[np.ndarray]):
        from moe_exp.routing.dynrt.d3a_text import TextTransform
        self.d, self.tr, self.evs = d, tr, evs
        parts_tr, parts_ev = [], [[] for _ in evs]
        for counts, scale in ((d.text, d.s_text), (d.lex, d.s_lex)):
            if counts is None:
                continue
            tt = TextTransform.fit(counts, tr, scale)
            parts_tr.append(tt.transform(counts, tr))
            for i, e in enumerate(evs):
                parts_ev[i].append(tt.transform(counts, e))
        self.xs_tr = sparse.hstack(parts_tr).tocsr()
        self.xs_ev = [sparse.hstack(p).tocsr() for p in parts_ev]
        self.n_sparse = self.xs_tr.shape[1]
        self._cache: dict = {}

    def _raw(self, key: str) -> np.ndarray:
        return self.d.extra[key] if key in self.d.extra else self.d.dense[key]

    def block(self, key: str):
        from moe_exp.routing.dynrt.cv import Scaler
        if key not in self._cache:
            raw = self._raw(key)
            sc = Scaler.fit(raw[self.tr])
            self._cache[key] = (sc.transform(raw[self.tr]), [sc.transform(raw[e]) for e in self.evs])
        return self._cache[key]

    def dense(self, keys):
        parts = [self.block(k) for k in keys]
        return (np.column_stack([p[0] for p in parts]),
                [np.column_stack([p[1][i] for p in parts]) for i in range(len(self.evs))])


def fit_path(blk: Blocks, keys, grid_desc: np.ndarray, stats: list) -> np.ndarray:
    from moe_exp.routing.dynrt.d3a_softmax import fit_softmax, predict_log_proba, weighted_nll
    d = blk.d
    xd_tr, xd_ev = blk.dense(keys)
    ev = blk.evs[0]
    theta, out = None, np.zeros(len(grid_desc))
    for i, lam in enumerate(grid_desc):
        fit = fit_softmax(xd_tr, blk.xs_tr, d.y[blk.tr], d.w[blk.tr], float(lam), d.K, theta0=theta)
        theta = fit.theta()
        stats.append((fit.n_iter, fit.converged, fit.grad_max))
        out[i] = weighted_nll(predict_log_proba(fit, xd_ev[0], blk.xs_ev[0]), d.y[ev], d.w[ev])
    return out


_STATE: dict = {}


def _set_threads() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _task(args):
    from moe_exp.routing.dynrt.cv import GRID, question_folds
    from moe_exp.routing.dynrt.d3a_cv import choose_penalty
    from moe_exp.routing.dynrt.d3a_softmax import fit_softmax, predict_log_proba
    r, f, names = args
    d: KData = _STATE["data"]
    models = {m: _STATE["models"][m] for m in names}
    grid = GRID.copy()
    train = np.flatnonzero(d.outer_folds[r] != f)
    test = np.flatnonzero(d.outer_folds[r] == f)
    inner = question_folds(d.groups[train], INNER, r * 10 + f)
    losses = {m: np.zeros((INNER, len(grid))) for m in models}
    sizes = np.zeros(INNER)
    stats: list = []
    t0 = time.time()
    for j in range(INNER):
        tr, va = train[inner != j], train[inner == j]
        blk = Blocks(d, tr, [va])
        for m, keys in models.items():
            losses[m][j] = fit_path(blk, keys, grid[::-1], stats)[::-1]
        sizes[j] = d.w[va].sum()
    blk = Blocks(d, train, [test])
    logp, lam_of, curves = {}, {}, {}
    for m, keys in models.items():
        mean = (losses[m] * sizes[:, None]).sum(0) / sizes.sum()
        lam = choose_penalty(mean, grid)[0]
        xd_tr, xd_ev = blk.dense(keys)
        fit = fit_softmax(xd_tr, blk.xs_tr, d.y[train], d.w[train], lam, d.K)
        stats.append((fit.n_iter, fit.converged, fit.grad_max))
        logp[m] = predict_log_proba(fit, xd_ev[0], blk.xs_ev[0]).astype(np.float32)
        lam_of[m], curves[m] = lam, mean
    return r, f, test, logp, lam_of, curves, stats, time.time() - t0


def run_cv(d: KData, models: dict, workers: int, repeats: int = REPEATS, per_task: int = 1) -> dict:
    """Cross-fitted held-out log-probabilities {model: [R, n, K]}; models are split `per_task` per job."""
    from moe_exp.routing.dynrt.cv import GRID
    _STATE["data"], _STATE["models"] = d, models
    names = sorted(models)
    chunks = [names[i:i + per_task] for i in range(0, len(names), per_task)]
    tasks = [(r, f, c) for c in chunks for r in range(repeats) for f in range(OUTER)]
    _set_threads()
    res = []
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for i, item in enumerate(pool.imap_unordered(_task, tasks), 1):
                res.append(item)
                log(f"cv {d.model}/{d.target}: {i}/{len(tasks)} ({item[7]:.0f}s)")
    else:
        res = [_task(t) for t in tasks]
    logp = {m: np.full((repeats, d.n, d.K), np.nan, np.float32) for m in models}
    lams = {m: np.zeros((repeats, OUTER)) for m in models}
    curves = {m: np.zeros((repeats, OUTER, len(GRID))) for m in models}
    stats, secs = [], []
    for r, f, test, lp, lam, cur, st, sec in res:
        for m in lp:
            logp[m][r, test] = lp[m]
            lams[m][r, f] = lam[m]
            curves[m][r, f] = cur[m]
        stats.extend(st)
        secs.append(sec)
    for m in models:
        if np.isnan(logp[m]).any():
            raise RuntimeError("held-out predictions missing")
    st = np.asarray(stats, float)
    diag = dict(n_fits=int(len(st)), frac_converged=float(st[:, 1].mean()), mean_iter=float(st[:, 0].mean()),
                max_iter=int(st[:, 0].max()), max_grad=float(st[:, 2].max()), task_seconds_mean=float(np.mean(secs)))
    return dict(logp=logp, lambdas=lams, curves=curves, diag=diag)


# --------------------------------------------------------------------------------- calibration

def _cal_task(args):
    from moe_exp.routing.dynrt.cv import GRID, question_folds
    from moe_exp.routing.dynrt.d3a_cv import choose_penalty
    target, s_text, s_lex = args
    d: KData = _STATE["cal"][target]
    d.s_text, d.s_lex = s_text, s_lex
    grid = GRID.copy()
    train = np.flatnonzero(d.outer_folds[0] != 0)
    inner = question_folds(d.groups[train], INNER, 0)
    losses, sizes, stats = np.zeros((INNER, len(grid))), np.zeros(INNER), []
    t0 = time.time()
    for j in range(INNER):
        tr, va = train[inner != j], train[inner == j]
        losses[j] = fit_path(Blocks(d, tr, [va]), ["base"], grid[::-1], stats)[::-1]
        sizes[j] = d.w[va].sum()
    mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
    lam, best = choose_penalty(mean, grid)
    st = np.asarray(stats, float)
    return dict(target=target, s_text=s_text, s_lex=s_lex, curve=mean.tolist(), best_lambda=lam,
                best_loss=float(mean[best]), frac_converged=float(st[:, 1].mean()), seconds=time.time() - t0)


def calibrate(model: str, workers: int, smoke: int = 0) -> dict:
    pre = verify_prereg()
    targets = ("switch",) if model == "pilot" else ("switch", "dest")
    _STATE["cal"] = {t: load_kdata(model, t, smoke) for t in targets}
    text_scales = (1.0,) if model == "pilot" else SCALES
    tasks = [(t, st, sl) for t in targets for st in text_scales for sl in SCALES]
    _set_threads()
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        rows = pool.map(_cal_task, tasks)
    chosen = {}
    for t in targets:
        rr = [r for r in rows if r["target"] == t]
        best = min(rr, key=lambda r: r["best_loss"])
        chosen[t] = dict(s_text=best["s_text"], s_lex=best["s_lex"], best_loss=best["best_loss"])
        log(f"calibrate {model}/{t}: chosen {chosen[t]}")
    out = dict(rule="baseline-only inner-CV log-loss on the training set of (repeat 0, fold 0); grid "
                    "(s_text, s_lex) in {1,3,10}^2 (pilot: no text block); no augmentation involved",
               rows=rows, chosen=chosen, prereg=pre, code=code_hashes(), smoke=smoke)
    write_json(_out(model, smoke) / "k2_calibration.json", out)
    return out


def _out(model: str, smoke: int):
    return OUT / (model if not smoke else f"{model}-smoke{smoke}")


def chosen_scales(model: str, target: str, smoke: int) -> tuple[float, float]:
    c = json.loads((_out(model, smoke) / "k2_calibration.json").read_text())["chosen"][target]
    return float(c["s_text"]), float(c["s_lex"])


# ---------------------------------------------------------------------------------- evaluation

def auc_gain(logp_base: np.ndarray, logp_aug: np.ndarray, y: np.ndarray, groups: np.ndarray, w: np.ndarray,
             n_boot: int, seed: int = 20261005) -> dict:
    """Question-weighted switch AUC difference (aug - base), mean over repeats, question bootstrap."""
    from moe_exp.routing.dynrt.cv import weighted_auc
    uq, inv = np.unique(groups, return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(uq), size=(n_boot, len(uq)))
    pb, pa = np.exp(logp_base[..., 1]), np.exp(logp_aug[..., 1])
    point_b = np.mean([weighted_auc(p, y, w[None])[0] for p in pb])
    point_a = np.mean([weighted_auc(p, y, w[None])[0] for p in pa])
    boot = np.zeros(n_boot)
    for lo in range(0, n_boot, 100):
        sl = slice(lo, lo + 100)
        wt = np.stack([np.bincount(i, minlength=len(uq))[inv] for i in idx[sl]]) * w[None]
        boot[sl] = np.mean([weighted_auc(a_, y, wt) - weighted_auc(b_, y, wt) for a_, b_ in zip(pa, pb)], axis=0)
    return dict(auc_base=float(point_b), auc_aug=float(point_a), diff=float(point_a - point_b),
                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)))


def run_main(model: str, target: str, workers: int, n_boot: int, smoke: int = 0) -> dict:
    from moe_exp.routing.dynrt import d3a_eval
    pre = verify_prereg()
    d = load_kdata(model, target, smoke)
    d.s_text, d.s_lex = chosen_scales(model, target, smoke)
    repeats = 1 if smoke else REPEATS
    log(f"K2 cv {model}/{target}: {d.n} pairs, {len(set(d.groups))} questions, scales ({d.s_text}, {d.s_lex})")
    t0 = time.time()
    res = run_cv(d, MODELS, workers, repeats=repeats)
    out_dir = _out(model, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"k2_{target}_preds.npz", y=d.y, a=d.a, w=d.w, groups=d.groups,
                        **{f"logp_{m}": v for m, v in res["logp"].items()})
    lp = res["logp"]
    ev = {}
    for m in ("aug16", "aug32"):
        e = d3a_eval.evaluate(lp["base"], lp[m], d.y, d.groups, d.w, n_boot=n_boot, per_class=d.a, full=False)
        if target == "switch":
            e["auc"] = auc_gain(lp["base"], lp[m], d.y, d.groups, d.w, n_boot)
        ev[m] = e
        g = e["gain"]
        log(f"K2 {model}/{target}/{m}: gain {g['point']:.5f} [{g['ci_lo']:.5f}, {g['ci_hi']:.5f}] "
            f"p1 {g['p_one_sided']:.4f} positive repeats {e['n_positive_repeats']}/{repeats}")
    ev["aug32_vs_aug16"] = d3a_eval.evaluate(lp["aug16"], lp["aug32"], d.y, d.groups, d.w, n_boot=n_boot,
                                             full=False)["gain"]
    out = dict(model=model, target=target, K=d.K, n_pairs=int(d.n), n_questions=int(len(set(d.groups))),
               scales=dict(s_text=d.s_text, s_lex=d.s_lex), models=MODELS, evaluation=ev, diag=res["diag"],
               lambdas={m: v.tolist() for m, v in res["lambdas"].items()},
               kin_columns=KIN_NAMES if model != "pilot" else [KIN_NAMES[i] for i in NON_GATE],
               prereg=pre, code=code_hashes(), deps=dep_hashes(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), seconds=time.time() - t0, smoke=smoke)
    write_json(out_dir / f"k2_{target}_cv.json", out)
    return out


def run_null(model: str, target: str, workers: int, n_boot: int, nsim: int = N_NULL, smoke: int = 0) -> dict:
    from moe_exp.routing.dynrt import d3a_eval
    pre = verify_prereg()
    d = load_kdata(model, target, smoke)
    d.s_text, d.s_lex = chosen_scales(model, target, smoke)
    for s in range(nsim):
        d.extra[f"noise{s}"] = np.random.default_rng(NULL_SEED + s).standard_normal((d.n, 15))
    models = {"base": ["base"], **{f"null{s:02d}": ["base", f"noise{s}"] for s in range(nsim)}}
    t0 = time.time()
    res = run_cv(d, models, workers, repeats=1)
    rows = []
    for s in range(nsim):
        e = d3a_eval.evaluate(res["logp"]["base"], res["logp"][f"null{s:02d}"], d.y, d.groups, d.w,
                              n_boot=n_boot, full=False)
        g = e["gain"]
        rows.append(dict(sim=s, gain=g["point"], ci_lo=g["ci_lo"], ci_hi=g["ci_hi"], boot_sd=g["boot_sd"],
                         p_one_sided=g["p_one_sided"]))
    g = np.array([r["gain"] for r in rows])
    summary = dict(nsim=nsim, mean_gain=float(g.mean()), sd_gain=float(g.std(ddof=1)) if nsim > 1 else None,
                   max_gain=float(g.max()), min_gain=float(g.min()),
                   ci_lo_above_0=int(sum(r["ci_lo"] > 0 for r in rows)),
                   p_below_05=int(sum(r["p_one_sided"] < 0.05 for r in rows)))
    out = dict(model=model, target=target, protocol=dict(nsim=nsim, repeats=1, n_boot=n_boot, full_inner_cv=True,
                                                          augmentation=f"15 iid N(0,1) columns, seeds {NULL_SEED}+s"),
               summary=summary, sims=rows, diag=res["diag"], prereg=pre, code=code_hashes(),
               seconds=time.time() - t0, smoke=smoke)
    out_dir = _out(model, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"k2_{target}_null.json", out)
    log(f"K2 null {model}/{target}: {summary}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "calibrate", "cv", "nullsim"))
    ap.add_argument("model", choices=("gpt", "qwen36", "pilot"))
    ap.add_argument("target", nargs="?", choices=("switch", "dest"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--nsim", type=int, default=N_NULL)
    ap.add_argument("--smoke", type=int, default=0, help="restrict to the first N questions (smoke test only)")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep(a.model)
    elif a.cmd == "calibrate":
        calibrate(a.model, a.workers, a.smoke)
    elif a.cmd == "cv":
        run_main(a.model, a.target, a.workers, a.n_boot, a.smoke)
    else:
        run_null(a.model, a.target, a.workers, a.n_boot, a.nsim, a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
