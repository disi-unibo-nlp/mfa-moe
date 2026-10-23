"""Addendum 1 (S3) item (ii): readout counterpart. Do depth paths predict the reasoning class of a sentence
beyond its token ids? All 7 models, cohort A labelled sentences (qwen36 = split-v1 dev+tune). No outcomes.

Rows: labelled sentences (dynamics-routing sentences.parquet, read-only) with >= 1 reasoning token; y = class
(7). Tokens: results/s3/_tok (top-1 paths and token ids of labelled-sentence tokens).
- baseline: token-id bag (crc32(id) mod 4,096 buckets; sublinear tf x in-fold idf, training df >= 2,
  L2-normalised x 10, as the dynamics-routing A1 text block) + position (absolute-position bin one-hot (6) and
  relative-position decile one-hot (10)) + log sentence length;
- augmented: baseline + path features of the sentence: mean F1 (path surprise under the add-0.5 depth
  bigram fitted on the training sentences' tokens), mean F2 (fraction of layers whose top-1 equals the
  previous token's, over within-sentence adjacent token pairs) and the per-layer top-1 histograms
  (L x E fractions) reduced by PCA to 64 components fitted on the training sentences (seeded randomized SVD).
Everything data-dependent (idf, bigram, PCA, standardisation, ridge penalty) is fitted inside the fold it
is used in, including the inner folds. Model: weighted multinomial ridge (softmax), question weights
(1 / labelled sentences of the question), penalty on the C8 grid 10^{-3..2} chosen by inner 3-fold
question-grouped CV (ties -> stronger penalty). Outer: question-grouped 5-fold CV (seed 0).
Primary number: out-of-fold Delta log-loss (baseline - augmented) in nats per sentence, question-weighted,
with a paired question bootstrap (1,000; seed 20260929). Unweighted (sentence-pooled) gain is also reported.
Null (dynamics-routing B1 analogue): the top-1 path of every token is replaced by that of a random token with
the same token id and absolute-position bin (a within-stratum permutation of the token paths; this is the
B1 within-token-id x position-bin class shuffle expressed on the routing side, since sentence labels are
per sentence); all path features and the augmented CV are recomputed (N_NULL draws).

CLI: python -m dp.s3_readout [--models M ...] [--workers W] [--n-null K]  -> results/s3/readout/<model>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from . import common as cm
from . import cv
from .s3_collect import tok_path

OUT = cm.RESULTS / "s3" / "readout"
K = 7
N_BUCKETS = 4096
MIN_DF = 2
BAG_SCALE = 10.0
N_PCA = 64
PCA_OVERSAMPLE = 16
PCA_ITERS = 4
OUTER, INNER = 5, 3
FOLD_SEED = 0
N_NULL = 5
N_BOOT = 1000
BOOT_SEED = 20260929
SEED = 20260929
ABS_EDGES = (1024, 2048, 4096, 8192, 16384)
N_ABS = len(ABS_EDGES) + 1
N_REL = 10
TOKEN_CHUNK = 1 << 18
SENT_COLUMNS = ["attempt_id", "question", "sentence_index", "cls", "tok_start", "tok_end", "n_tokens", "n_reasoning",
                "abs_bin", "rel_pos"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------ data

def bucket_of(tokid: np.ndarray) -> np.ndarray:
    """crc32 of the little-endian int32 token id, mod N_BUCKETS."""
    uniq, inv = np.unique(np.asarray(tokid, np.int64), return_inverse=True)
    b = np.array([zlib.crc32(int(u).to_bytes(4, "little", signed=True)) & (N_BUCKETS - 1) for u in uniq], np.int64)
    return b[inv]


def abs_bin(rank: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(ABS_EDGES), np.asarray(rank), side="right").astype(np.int64)


@dataclass
class ModelData:
    model: str
    L: int
    E: int
    sent: pd.DataFrame
    y: np.ndarray            # [S] class
    groups: np.ndarray       # [S] question
    w: np.ndarray            # [S] question weights
    base: np.ndarray         # [S, d] raw baseline dense columns
    bag: sparse.csr_matrix   # [S, N_BUCKETS] token-id counts
    top1: np.ndarray         # [L, M] uint8 labelled-token paths
    tok_sent: np.ndarray     # [M] sentence row of each token
    tokid: np.ndarray        # [M]
    tok_abs: np.ndarray      # [M] absolute-position bin of the token's reasoning rank
    adj: np.ndarray          # [M] previous token adjacent and in the same sentence
    n_before_filter: int = 0

    @property
    def n(self) -> int:
        return len(self.y)


def question_weights(groups: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def load_model(model: str) -> ModelData:
    spec = cm.model_spec(model)
    L, E = spec["num_layers"], spec["num_experts"]
    att = cm.load_attempts(model, "A")
    row_of = {a: i for i, a in enumerate(att["attempt_id"])}
    sent = pd.read_parquet(cm.DYN_RESULTS / model / "A" / "sentences.parquet", columns=SENT_COLUMNS)
    sent = sent[sent["attempt_id"].isin(row_of)].copy()
    sent["row"] = sent["attempt_id"].map(row_of).astype(np.int64)
    n_all = len(sent)
    sent = sent[sent["n_tokens"] >= 1].sort_values(["row", "tok_start"]).reset_index(drop=True)
    if not (sent["question"].to_numpy() == att["question"].to_numpy()[sent["row"].to_numpy()]).all():
        raise ValueError(f"{model}: sentence question differs from the attempt table")
    if not ((sent["tok_end"] - sent["tok_start"]) == sent["n_tokens"]).all():
        raise ValueError(f"{model}: sentence span length != n_tokens")
    skey = sent["row"].to_numpy() * 10**7 + sent["sentence_index"].to_numpy()
    if len(np.unique(skey)) != len(skey):
        raise ValueError(f"{model}: duplicate (attempt, sentence_index)")
    order = np.argsort(skey)
    top1, tsent, tokid, tabs, adj = [], [], [], [], []
    for i in np.unique(sent["row"].to_numpy()):
        z = np.load(tok_path(model, int(i)))
        k = int(i) * 10**7 + z["lab_sid"].astype(np.int64)
        pos = np.searchsorted(skey[order], k)
        pos = np.minimum(pos, len(order) - 1)
        if not (skey[order][pos] == k).all():
            raise ValueError(f"{model} row {i}: token sentence not in the sentence table")
        top1.append(z["lab_top1"])
        tsent.append(order[pos])
        tokid.append(z["tok"][z["lab_rank"]])
        tabs.append(abs_bin(z["lab_rank"]))
        adj.append(z["lab_adj"])
    top1 = np.concatenate(top1, axis=1)
    tsent = np.concatenate(tsent)
    if not np.array_equal(np.bincount(tsent, minlength=len(sent)), sent["n_tokens"].to_numpy()):
        raise ValueError(f"{model}: token counts per sentence differ from the sentence table")
    tokid = np.concatenate(tokid).astype(np.int64)
    s = len(sent)
    bag = sparse.coo_matrix((np.ones(len(tokid), np.float64), (tsent, bucket_of(tokid))),
                            shape=(s, N_BUCKETS)).tocsr()
    bag.sum_duplicates()
    ab = sent["abs_bin"].to_numpy(np.int64)
    rb = np.minimum((sent["rel_pos"].to_numpy(float) * N_REL).astype(np.int64), N_REL - 1)
    base = np.column_stack([(ab == b).astype(float) for b in range(N_ABS)]
                           + [(rb == b).astype(float) for b in range(N_REL)]
                           + [np.log(sent["n_tokens"].to_numpy(float))])
    groups = sent["question"].to_numpy()
    return ModelData(model, L, E, sent, sent["cls"].to_numpy(np.int64), groups, question_weights(groups), base, bag,
                     top1, tsent, tokid, np.concatenate(tabs), np.concatenate(adj), n_all)


# ------------------------------------------------------------------------------ path features

@dataclass
class PathVariant:
    """Fold-independent path quantities of one path assignment (observed or a null draw)."""
    keys: np.ndarray         # [L-1, M] int32 layer-pair keys l*E*E + a*E + b
    f2: np.ndarray           # [S] sentence mean F2 (NaN without adjacent pairs)
    hist: sparse.csr_matrix  # [S, L*E] per-layer top-1 fractions


def path_variant(md: ModelData, top1: np.ndarray) -> PathVariant:
    L, E, S = md.L, md.E, md.n
    t = top1.astype(np.int64)
    keys = ((np.arange(L - 1, dtype=np.int64) * E * E)[:, None] + t[:-1] * E + t[1:]).astype(np.int32)
    eq = np.zeros(t.shape[1])
    eq[1:] = (t[:, 1:] == t[:, :-1]).mean(0)
    a = md.adj
    num = np.bincount(md.tok_sent[a], weights=eq[a], minlength=S)
    den = np.bincount(md.tok_sent[a], minlength=S)
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = np.where(den > 0, num / den, np.nan)
    n_tok = np.bincount(md.tok_sent, minlength=S).astype(np.float64)
    rows = np.tile(md.tok_sent, L)
    cols = (np.arange(L, dtype=np.int64)[:, None] * E + t).ravel()
    vals = np.tile(1.0 / n_tok[md.tok_sent], L)
    hist = sparse.coo_matrix((vals, (rows, cols)), shape=(S, L * E)).tocsr()
    hist.sum_duplicates()
    return PathVariant(keys, f2, hist)


def null_paths(md: ModelData, draw: int) -> np.ndarray:
    """Token paths permuted among tokens of the same token id x absolute-position bin (B1 strata)."""
    n = len(md.tokid)
    stratum = md.tokid * 8 + md.tok_abs
    base = np.argsort(stratum, kind="stable")
    rng = np.random.default_rng([SEED, cm.MODELS.index(md.model), draw, 991])
    order = np.lexsort((rng.random(n), stratum))
    perm = np.empty(n, np.int64)
    perm[base] = order
    return md.top1[:, perm]


def sentence_f1(md: ModelData, pv: PathVariant, train_rows: np.ndarray) -> np.ndarray:
    """Sentence mean path surprise under the add-alpha bigram fitted on the tokens of `train_rows`."""
    L, E = md.L, md.E
    base = (L - 1) * E * E
    in_train = np.zeros(md.n, bool)
    in_train[train_rows] = True
    tr_tok = in_train[md.tok_sent]
    m = len(md.tok_sent)
    counts = np.zeros(base)
    for a in range(0, m, TOKEN_CHUNK):
        sl = slice(a, a + TOKEN_CHUNK)
        counts += np.bincount(pv.keys[:, sl][:, tr_tok[sl]].ravel(), minlength=base)
    c = counts.reshape(L - 1, E, E)
    logp = np.log((c + cm.ALPHA) / (c.sum(-1, keepdims=True) + cm.ALPHA * E)).reshape(-1)
    surprise = np.empty(m)
    for a in range(0, m, TOKEN_CHUNK):
        surprise[a:a + TOKEN_CHUNK] = -logp[pv.keys[:, a:a + TOKEN_CHUNK]].mean(0)
    n_tok = np.bincount(md.tok_sent, minlength=md.n)
    return np.bincount(md.tok_sent, weights=surprise, minlength=md.n) / np.maximum(n_tok, 1)


def randomized_pca(x: sparse.csr_matrix, k: int, seed: int, n_iter: int = PCA_ITERS,
                   oversample: int = PCA_OVERSAMPLE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean [D], components [k, D], explained-variance fraction [k]) of the rows of sparse x (centred)."""
    n, dim = x.shape
    k = min(k, dim, n - 1)
    mu = np.asarray(x.mean(axis=0)).ravel()
    xt = x.T.tocsr()
    rng = np.random.default_rng(seed)

    def mv(m):
        return x @ m - np.outer(np.ones(n), mu @ m)

    def rmv(m):
        return xt @ m - np.outer(mu, m.sum(0))

    q, _ = np.linalg.qr(mv(rng.standard_normal((dim, k + oversample))))
    for _ in range(n_iter):
        z, _ = np.linalg.qr(rmv(q))
        q, _ = np.linalg.qr(mv(z))
    b = rmv(q).T
    _, s, vt = np.linalg.svd(b, full_matrices=False)
    total = float(x.multiply(x).sum() - n * (mu @ mu))
    return mu, vt[:k], (s[:k] ** 2) / total


# ------------------------------------------------------------------------------ design

class BagTransform:
    """In-fold tf-idf of the token-id bag (the dynamics-routing A1 TextTransform on N_BUCKETS columns)."""

    def __init__(self, counts: sparse.csr_matrix, train: np.ndarray, scale: float = BAG_SCALE):
        x = counts[train]
        df = np.bincount(x.indices, minlength=counts.shape[1]).astype(np.float64)
        self.keep = df >= MIN_DF
        self.idf = np.where(self.keep, np.log((1.0 + x.shape[0]) / (1.0 + df)) + 1.0, 0.0)
        self.cols = np.flatnonzero(self.keep)
        self.scale = scale

    def transform(self, counts: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
        x = counts[rows].tocsr(copy=True)
        x.data = (1.0 + np.log(x.data)) * self.idf[x.indices]
        x = x[:, self.cols].tocsr()
        norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
        x = sparse.diags(self.scale / np.where(norm > 0, norm, 1.0)) @ x
        return x.tocsr()


def standardise(raw: np.ndarray, tr: np.ndarray, evs: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    sc = cv.Scaler.fit(raw[tr])
    return sc.transform(raw[tr]), [sc.transform(raw[e]) for e in evs]


def fold_design(md: ModelData, pv: PathVariant | None, tr: np.ndarray, evs: list[np.ndarray], seed: int
                ) -> tuple[np.ndarray, sparse.csr_matrix, list[np.ndarray], list[sparse.csr_matrix], dict]:
    """Dense and sparse design blocks of the training rows and every evaluation row set (fitted on tr)."""
    bt = BagTransform(md.bag, tr)
    xs_tr, xs_ev = bt.transform(md.bag, tr), [bt.transform(md.bag, e) for e in evs]
    info = dict(bag_columns=int(len(bt.cols)))
    raw = md.base
    if pv is not None:
        f1 = sentence_f1(md, pv, tr)
        mu, comp, evr = randomized_pca(pv.hist[tr], N_PCA, seed)
        rows = np.concatenate([tr] + list(evs))
        scores = np.full((md.n, comp.shape[0]), np.nan)
        scores[rows] = pv.hist[rows] @ comp.T - (mu @ comp.T)[None, :]
        raw = np.column_stack([md.base, f1, pv.f2, scores])
        info.update(pca_components=int(comp.shape[0]), pca_explained=float(evr.sum()))
    xd_tr, xd_ev = standardise(raw, tr, evs)
    return xd_tr, xs_tr, xd_ev, xs_ev, info


# ------------------------------------------------------------------------------ multinomial ridge

@dataclass
class SoftmaxFit:
    b: np.ndarray
    bd: np.ndarray
    bs: np.ndarray
    n_iter: int
    converged: bool
    grad_max: float

    def theta(self) -> np.ndarray:
        return np.r_[self.b, self.bd.ravel(), self.bs.ravel()]


def _log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def fit_softmax(xd: np.ndarray, xs: sparse.csr_matrix, y: np.ndarray, w: np.ndarray, lam: float,
                theta0: np.ndarray | None = None, maxiter: int = 400) -> SoftmaxFit:
    """Weighted multinomial ridge, objective sum w nll / sum w + lam/2 |B|^2 (dynamics-routing d3a_softmax)."""
    n, pd_ = xd.shape
    ps = xs.shape[1]
    wn = w / w.sum()
    xst = xs.T.tocsr()
    rows = np.arange(n)

    def split(theta):
        return theta[:K], theta[K:K + pd_ * K].reshape(pd_, K), theta[K + pd_ * K:].reshape(ps, K)

    def fun(theta):
        b, bd, bs = split(theta)
        lp = _log_softmax(b[None, :] + xd @ bd + xs @ bs)
        loss = -(wn * lp[rows, y]).sum() + 0.5 * lam * ((bd ** 2).sum() + (bs ** 2).sum())
        gz = np.exp(lp)
        gz[rows, y] -= 1.0
        gz *= wn[:, None]
        return loss, np.r_[gz.sum(0), (xd.T @ gz + lam * bd).ravel(), (xst @ gz + lam * bs).ravel()]

    if theta0 is None:
        prior = np.bincount(y, weights=wn, minlength=K).clip(1e-9)
        theta0 = np.r_[np.log(prior / prior.sum()), np.zeros((pd_ + ps) * K)]
    res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B",
                            options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-9, gtol=1e-6, maxcor=20))
    b, bd, bs = split(res.x)
    return SoftmaxFit(b.copy(), bd.copy(), bs.copy(), int(res.nit), bool(res.success), float(np.abs(res.jac).max()))


def predict_log_proba(fit: SoftmaxFit, xd: np.ndarray, xs: sparse.csr_matrix) -> np.ndarray:
    return _log_softmax(fit.b[None, :] + xd @ fit.bd + xs @ fit.bs)


def weighted_nll(logp: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(-(w * logp[np.arange(len(y)), y]).sum() / w.sum())


# ------------------------------------------------------------------------------ CV

_STATE: dict = {}


def outer_folds(groups: np.ndarray) -> np.ndarray:
    return cv.question_folds(groups, OUTER, FOLD_SEED)


def run_task(args: tuple) -> dict:
    """One (variant, outer fold): inner-CV penalty choice, then the held-out class log-probabilities."""
    variant, f = args
    md: ModelData = _STATE["md"]
    pv = _STATE["variants"].get(variant)
    t0 = time.time()
    folds = outer_folds(md.groups)
    train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    inner = cv.question_folds(md.groups[train], INNER, FOLD_SEED * 1000 + f)
    grid = cv.GRID.copy()
    losses = np.zeros((INNER, len(grid)))
    sizes = np.zeros(INNER)
    stats = []
    vseed = 0 if variant in ("base", "aug") else 1 + int(variant.split("_")[1])
    for j in range(INNER):
        tr, va = train[inner != j], train[inner == j]
        xd_tr, xs_tr, (xd_va,), (xs_va,), _ = fold_design(md, pv, tr, [va], seed=SEED + 100 * f + 10 * j + vseed)
        theta = None
        for gi in range(len(grid) - 1, -1, -1):                     # descending penalty, warm-started
            fit = fit_softmax(xd_tr, xs_tr, md.y[tr], md.w[tr], float(grid[gi]), theta0=theta)
            theta = fit.theta()
            stats.append((fit.n_iter, fit.converged, fit.grad_max))
            losses[j, gi] = weighted_nll(predict_log_proba(fit, xd_va, xs_va), md.y[va], md.w[va])
        sizes[j] = md.w[va].sum()
    mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
    best = int(np.flatnonzero(mean <= mean.min() + 1e-12)[-1])
    lam = float(grid[best])
    xd_tr, xs_tr, (xd_te,), (xs_te,), info = fold_design(md, pv, train, [test], seed=SEED + 100 * f + 9 + vseed)
    fit = fit_softmax(xd_tr, xs_tr, md.y[train], md.w[train], lam)
    stats.append((fit.n_iter, fit.converged, fit.grad_max))
    logp = predict_log_proba(fit, xd_te, xs_te).astype(np.float32)
    return dict(variant=variant, fold=f, test=test, logp=logp, lam=lam, inner_curve=mean, stats=stats, info=info,
                seconds=time.time() - t0)


def question_bootstrap(gain: np.ndarray, groups: np.ndarray, w: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict:
    """Weighted mean sum(w gain) / sum(w) of per-sentence gains with a paired question bootstrap.

    w = question weights gives the mean of question means; w = 1 gives the sentence-pooled mean."""
    uniq, inv = np.unique(groups, return_inverse=True)
    num, den = np.bincount(inv, weights=w * gain), np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(gain=float(num.sum() / den.sum()), ci_lo=float(lo), ci_hi=float(hi),
                p_one_sided=float((1 + (boot <= 0).sum()) / (n_boot + 1)), boot_sd=float(boot.std(ddof=1)),
                n_questions=int(len(uniq)), n_boot=n_boot)


def nll_rows(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -logp[np.arange(len(y)), y].astype(np.float64)


def analyse(md: ModelData, workers: int, n_null: int) -> dict:
    """Base / augmented / null CV of one model's sentence table -> result dict."""
    t0 = time.time()
    model = md.model
    log(f"{model}: {md.n} sentences ({md.n_before_filter} before n_tokens >= 1), "
        f"{len(np.unique(md.groups))} questions, {md.top1.shape[1]} tokens, L={md.L}, E={md.E} "
        f"({time.time() - t0:.0f}s)")
    variants = {"aug": path_variant(md, md.top1)}
    for j in range(n_null):
        variants[f"null_{j}"] = path_variant(md, null_paths(md, j))
    log(f"{model}: path variants built ({time.time() - t0:.0f}s)")
    _STATE.update(md=md, variants=variants)
    names = ["base", "aug"] + [f"null_{j}" for j in range(n_null)]
    tasks = [(v, f) for v in names for f in range(OUTER)]
    tasks.sort(key=lambda t: t[0] == "base")                          # path tasks (slower) first
    res = []
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
                res.append(r)
                log(f"{model}: task {i}/{len(tasks)} {r['variant']} fold {r['fold']} lam {r['lam']:g} "
                    f"({r['seconds']:.0f}s)")
    else:
        res = [run_task(t) for t in tasks]
    logp = {v: np.full((md.n, K), np.nan, np.float32) for v in names}
    lams = {v: [None] * OUTER for v in names}
    infos = {v: [None] * OUTER for v in names}
    stats = []
    for r in res:
        logp[r["variant"]][r["test"]] = r["logp"]
        lams[r["variant"]][r["fold"]] = r["lam"]
        infos[r["variant"]][r["fold"]] = r["info"]
        stats.extend(r["stats"])
    for v in names:
        if np.isnan(logp[v]).any():
            raise RuntimeError(f"{model}: held-out predictions missing for {v}")
    nll = {v: nll_rows(logp[v], md.y) for v in names}
    y, g, w = md.y, md.groups, md.w
    ones = np.ones(md.n)
    out = dict(model=model, n_sentences=int(md.n), n_sentences_before_min_tokens=int(md.n_before_filter),
               n_questions=int(len(np.unique(g))), n_tokens=int(md.top1.shape[1]), layers=md.L, experts=md.E,
               split_restriction="split-v1 dev+tune" if model == "qwen36" else None,
               class_counts={c: int((y == i).sum()) for i, c in enumerate(cm.CLASSES)},
               f2_missing_share=float(np.isnan(variants["aug"].f2).mean()),
               design=dict(bag_buckets=N_BUCKETS, pca=N_PCA, outer=OUTER, inner=INNER, grid=cv.GRID,
                           n_null=n_null, weights="question (1/labelled sentences)"),
               logloss={v: dict(question_weighted=float(np.bincount(np.unique(g, return_inverse=True)[1],
                                                                    weights=w * nll[v]).mean()),
                                pooled=float(nll[v].mean())) for v in names},
               accuracy={v: float((logp[v].argmax(1) == y).mean()) for v in names},
               lambdas=lams, fold_info=infos)
    out["gain_aug_vs_base"] = question_bootstrap(nll["base"] - nll["aug"], g, w)
    out["gain_aug_vs_base_pooled"] = question_bootstrap(nll["base"] - nll["aug"], g, ones)
    nulls = [v for v in names if v.startswith("null_")]
    if nulls:
        null_gain = [question_bootstrap(nll["base"] - nll[v], g, w)["gain"] for v in nulls]
        mean_null_nll = np.mean([nll[v] for v in nulls], axis=0)
        obs = out["gain_aug_vs_base"]["gain"]
        out["null_within_token_id"] = dict(
            gains=null_gain, mean=float(np.mean(null_gain)),
            sd=float(np.std(null_gain, ddof=1)) if len(null_gain) > 1 else None,
            max=float(np.max(null_gain)), retained_fraction=float(np.mean(null_gain) / obs) if obs else None,
            excess_obs_over_null=question_bootstrap(mean_null_nll - nll["aug"], g, w),
            gain_null_mean_vs_base=question_bootstrap(nll["base"] - mean_null_nll, g, w),
            strata="token id x absolute-position bin (0-1k, 1-2k, 2-4k, 4-8k, 8-16k, 16k+)")
    per_class = {}
    for i, c in enumerate(cm.CLASSES):
        m = y == i
        if m.sum() >= 50:
            per_class[c] = dict(n=int(m.sum()), pooled_gain=float((nll["base"] - nll["aug"])[m].mean()),
                                pooled_excess_over_null=float((mean_null_nll - nll["aug"])[m].mean()) if nulls
                                else None)
    out["per_class_pooled"] = per_class
    st = np.asarray(stats, float)
    out["fit_diagnostics"] = dict(n_fits=int(len(st)), frac_converged=float(st[:, 1].mean()),
                                  max_iter=int(st[:, 0].max()), max_grad=float(st[:, 2].max()))
    out["seconds"] = time.time() - t0
    gg = out["gain_aug_vs_base"]
    log(f"{model}: gain {gg['gain']:.5f} [{gg['ci_lo']:.5f}, {gg['ci_hi']:.5f}]"
        + (f", null mean {out['null_within_token_id']['mean']:.5f}, excess "
           f"{out['null_within_token_id']['excess_obs_over_null']['gain']:.5f}" if nulls else "")
        + f" ({out['seconds']:.0f}s)")
    _STATE.clear()
    return out


def run_model(model: str, workers: int, n_null: int) -> dict:
    out = analyse(load_model(model), workers, n_null)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}.json").write_text(json.dumps(cm.clean(out), indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(cm.MODELS), choices=cm.MODELS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--n-null", type=int, default=N_NULL)
    a = ap.parse_args(argv)
    os.environ.update({v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    for m in a.models:
        if (OUT / f"{m}.json").exists():
            log(f"{m}: output exists, skipped")
            continue
        run_model(m, a.workers, a.n_null)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
