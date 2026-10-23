"""P-A1 cross-fitted multinomial ridge (C8 machinery: 4 outer question folds x 5 repeats, inner 3, grid 10^{-3..2}).

Everything data-dependent is fitted inside the training part of the fold it is used in: the tf-idf
vocabulary/idf of the hashed text, the class profiles (attempts of the training questions) behind the
d_c scores, and the standardisation of all dense columns. Outer fold ids are the C8 ones (question_folds over
ALL cohort-A questions, seeds 0-4); the 37 questions of GPT cohort B are dropped afterwards. The inner fold
seed is that of `cv._task` (`seed0 * 1000 + r * 10 + fold`), the penalty of the model is the inner-CV-best
value of the question-weighted log-loss (ties -> stronger penalty).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import features as F
from .common import RESULTS
from .cv import GRID, Scaler, question_folds
from .d3a_softmax import fit_softmax, predict_log_proba, weighted_nll
from .d3a_text import TextTransform
from .data import load_cohort

OUT = RESULTS / "A1"
K = 7
TEXT_SCALE = 10.0
REPEATS, OUTER, INNER = 5, 4, 3
ABS_LEVELS = 6
MODELS = {"base": ["base"], "aug": ["base", "last"], "aug_ant": ["base", "ant"]}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Data:
    y: np.ndarray
    a: np.ndarray
    groups: np.ndarray
    w: np.ndarray
    attempt: np.ndarray
    base: np.ndarray
    base_names: list
    counts: sparse.csr_matrix
    h_last: np.ndarray
    h_ant: np.ndarray | None
    has_ant: np.ndarray | None
    table: F.SentenceTable
    outer_folds: np.ndarray            # [R, n] outer fold id of every pair row
    extra: dict = field(default_factory=dict)      # name -> raw dense [n, d] blocks (e.g. simulated noise)
    perm: dict = field(default_factory=dict)       # name -> row permutation of the last-16 histograms (null sims)
    text_scale: float = TEXT_SCALE

    @property
    def n(self) -> int:
        return len(self.y)


def baseline_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Raw dense baseline columns: current class, log sentence length, absolute position bin, dataset, difficulty."""
    cols, names = [], []
    a = df["a"].to_numpy(np.int64)
    for c in range(K):
        cols.append((a == c).astype(float))
        names.append(f"cls{c}")
    cols.append(np.log(df["n_tokens"].to_numpy(float)))
    names.append("log_len")
    ab = df["abs_bin"].to_numpy(np.int64)
    for b in range(ABS_LEVELS):
        cols.append((ab == b).astype(float))
        names.append(f"abs{b}")
    levels = sorted(df["dataset"].unique())
    ds = df["dataset"].to_numpy()
    for lv in levels[1:]:
        cols.append((ds == lv).astype(float))
        names.append(f"ds_{lv}")
    cols.append(df["difficulty"].to_numpy(float))
    names.append("difficulty")
    return np.column_stack(cols), names


def within_class_permutation(a: np.ndarray, seed: int) -> np.ndarray:
    """Row permutation shuffling rows among pairs of the same current class (breaks every other link)."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(a))
    for c in range(K):
        idx = np.flatnonzero(a == c)
        perm[idx] = rng.permutation(idx)
    return perm


def question_weights(groups: np.ndarray) -> np.ndarray:
    """1 / (pairs of the question): every question weighs 1, a question value is its mean over pairs."""
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def load_data(*, antic_dir=None, text_scale: float = TEXT_SCALE) -> Data:
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    counts = sparse.load_npz(OUT / "text_counts.npz").tocsr()
    if counts.shape[0] != len(pairs):
        raise ValueError("text counts are not aligned to the pair table")
    coh = load_cohort("gpt", "A")
    att_q = coh.attempts["question"].to_numpy()
    if len(set(att_q)) != len(att_q):
        raise ValueError("cohort A must have one attempt per question for the C8 fold reuse")
    hist = coh.hist
    src = pairs["src"].to_numpy()
    pos = np.searchsorted(hist["last16_rows"], src)
    if not (hist["last16_rows"][np.minimum(pos, len(hist["last16_rows"]) - 1)] == src).all():
        raise ValueError("last16 histogram missing for a pair source")
    h_last = hist["last16"][pos].reshape(len(pairs), -1).astype(np.float32)
    h_ant = has_ant = None
    if antic_dir is not None:
        z = np.load(antic_dir / "antic_hist.npz")
        if not z["computed"].all():
            raise ValueError("anticipation table incomplete")
        h_ant = z["h"].reshape(len(pairs), -1).astype(np.float32)
        has_ant = z["has"].astype(bool)
    base, names = baseline_matrix(pairs)
    att_idx = pairs["attempt"].to_numpy(np.int64)
    folds = np.stack([question_folds(att_q, OUTER, r)[att_idx] for r in range(REPEATS)])
    groups = pairs["question"].to_numpy()
    return Data(y=pairs["b"].to_numpy(np.int64), a=pairs["a"].to_numpy(np.int64), groups=groups,
                w=question_weights(groups), attempt=att_idx, base=base, base_names=names, counts=counts,
                h_last=h_last, h_ant=h_ant, has_ant=has_ant, table=coh.table, outer_folds=folds,
                text_scale=text_scale)


class FoldBlocks:
    """All fold-fitted design pieces for one training set and any number of evaluation row sets."""

    def __init__(self, data: Data, tr: np.ndarray, evs: list[np.ndarray]):
        self.data, self.tr, self.evs = data, tr, evs
        tt = TextTransform.fit(data.counts, tr, data.text_scale)
        self.xs_tr = tt.transform(data.counts, tr)
        self.xs_ev = [tt.transform(data.counts, e) for e in evs]
        self.n_text_cols = tt.n_active
        self._cache: dict = {}
        self._prof = None

    def profiles(self) -> F.Profiles:
        if self._prof is None:
            t = self.data.table
            mask = np.zeros(t.n_attempts, bool)
            mask[np.unique(self.data.attempt[self.tr])] = True
            rows = np.flatnonzero(mask[t.attempt])
            self._prof = F.fit_profiles(t.h[rows], t.cls[rows], t.attempt[rows], t.layers, t.experts)
        return self._prof

    def _raw(self, key: str) -> np.ndarray:
        d = self.data
        if key == "base":
            return d.base
        if key in d.extra:
            return d.extra[key]
        if key in d.perm:
            rows = np.concatenate([self.tr] + list(self.evs))
            s = np.full((d.n, K), np.nan)
            s[rows] = F.d_scores(d.h_last[d.perm[key][rows]], self.profiles())
            return s
        if key in ("last", "ant"):
            h = d.h_last if key == "last" else d.h_ant
            if h is None:
                raise ValueError("anticipation histograms are not loaded")
            rows = np.concatenate([self.tr] + list(self.evs))
            s = np.full((d.n, K), np.nan)
            s[rows] = F.d_scores(h[rows], self.profiles())
            if key == "ant":
                s[~d.has_ant] = np.nan
            return s
        raise KeyError(key)

    def block(self, key: str) -> tuple[np.ndarray, list[np.ndarray]]:
        if key not in self._cache:
            raw = self._raw(key)
            sc = Scaler.fit(raw[self.tr])
            self._cache[key] = (sc.transform(raw[self.tr]), [sc.transform(raw[e]) for e in self.evs])
        return self._cache[key]

    def dense(self, keys: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        parts = [self.block(k) for k in keys]
        return (np.column_stack([p[0] for p in parts]),
                [np.column_stack([p[1][i] for p in parts]) for i in range(len(self.evs))])


def fit_path(blocks: FoldBlocks, keys: list[str], grid_desc: np.ndarray, *, val: int = 0, stats: list | None = None
             ) -> np.ndarray:
    """Validation log-loss of the warm-started descending-penalty path: [len(grid)] in the grid's order."""
    d = blocks.data
    xd_tr, xd_ev = blocks.dense(keys)
    y_tr, w_tr = d.y[blocks.tr], d.w[blocks.tr]
    ev = blocks.evs[val]
    theta, out = None, np.zeros(len(grid_desc))
    for i, lam in enumerate(grid_desc):
        fit = fit_softmax(xd_tr, blocks.xs_tr, y_tr, w_tr, float(lam), K, theta0=theta)
        theta = fit.theta()
        if stats is not None:
            stats.append((fit.n_iter, fit.converged, fit.grad_max))
        out[i] = weighted_nll(predict_log_proba(fit, xd_ev[val], blocks.xs_ev[val]), d.y[ev], d.w[ev])
    return out


def choose_penalty(mean_loss: np.ndarray, grid: np.ndarray) -> tuple[float, int]:
    best = int(np.flatnonzero(mean_loss <= mean_loss.min() + 1e-12)[-1])       # ties -> stronger penalty
    return float(grid[best]), best


_STATE: dict = {}


def _task(args):
    r, f, chunk = args
    d: Data = _STATE["data"]
    models: dict = _STATE["models"]
    if chunk is not None:
        names = sorted(models)[chunk::_STATE["n_chunks"]]
        models = {m: models[m] for m in names}
    grid = GRID.copy()
    train = np.flatnonzero(d.outer_folds[r] != f)
    test = np.flatnonzero(d.outer_folds[r] == f)
    inner = question_folds(d.groups[train], INNER, r * 10 + f)         # seed0 = 0 as in cv._task
    losses = {m: np.zeros((INNER, len(grid))) for m in models}
    sizes = np.zeros(INNER)
    stats: list = []
    t0 = time.time()
    fixed = _STATE.get("fixed")
    for j in range(INNER if fixed is None else 0):
        tr, va = train[inner != j], train[inner == j]
        blocks = FoldBlocks(d, tr, [va])
        for m, keys in models.items():
            losses[m][j] = fit_path(blocks, keys, grid[::-1], stats=stats)[::-1]
        sizes[j] = d.w[va].sum()
    blocks = FoldBlocks(d, train, [test])
    logp, lam_of, curves = {}, {}, {}
    for m, keys in models.items():
        mean = (losses[m] * sizes[:, None]).sum(0) / sizes.sum() if fixed is None else np.full(len(grid), np.nan)
        lam = choose_penalty(mean, grid)[0] if fixed is None else float(fixed[r][f])
        xd_tr, xd_ev = blocks.dense(keys)
        fit = fit_softmax(xd_tr, blocks.xs_tr, d.y[train], d.w[train], lam, K)
        stats.append((fit.n_iter, fit.converged, fit.grad_max))
        logp[m] = predict_log_proba(fit, xd_ev[0], blocks.xs_ev[0]).astype(np.float32)
        lam_of[m], curves[m] = lam, mean
    return r, f, test, logp, lam_of, curves, stats, blocks.n_text_cols, time.time() - t0


def run_cv(data: Data, models: dict, *, workers: int, repeats: int = REPEATS, tasks=None,
           fixed_lambda: np.ndarray | None = None, n_chunks: int = 1) -> dict:
    """Cross-fitted held-out log-probabilities {model: [R, n, K]} plus penalties, inner curves and fit diagnostics.

    `fixed_lambda` [R, OUTER] skips the inner CV and uses the given penalty per (repeat, fold) for every model.
    `n_chunks` > 1 splits the models (sorted by name) over that many independent tasks per (repeat, fold).
    """
    _STATE["data"], _STATE["models"], _STATE["fixed"], _STATE["n_chunks"] = data, models, fixed_lambda, n_chunks
    chunks = [None] if n_chunks == 1 else list(range(n_chunks))
    tasks = tasks or [(r, f, c) for c in chunks for r in range(repeats) for f in range(OUTER)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = []
            for i, item in enumerate(pool.imap_unordered(_task, tasks), 1):
                res.append(item)
                log(f"cv: {i}/{len(tasks)} outer fits ({item[8]:.0f}s)")
    else:
        res = [_task(t) for t in tasks]
    logp = {m: np.full((repeats, data.n, K), np.nan, np.float32) for m in models}
    lams = {m: np.zeros((repeats, OUTER)) for m in models}
    curves = {m: np.zeros((repeats, OUTER, len(GRID))) for m in models}
    stats, seconds = [], []
    for r, f, test, lp, lam, cur, st, _, sec in res:
        for m in lp:
            logp[m][r, test] = lp[m]
            lams[m][r, f] = lam[m]
            curves[m][r, f] = cur[m]
        stats.extend(st)
        seconds.append(sec)
    for m in models:
        if np.isnan(logp[m]).any():
            raise RuntimeError("held-out predictions missing")
    st = np.asarray(stats, float)
    diag = dict(n_fits=int(len(st)), frac_converged=float(st[:, 1].mean()), mean_iter=float(st[:, 0].mean()),
                max_iter=int(st[:, 0].max()), max_grad=float(st[:, 2].max()), task_seconds_mean=float(np.mean(seconds)))
    return dict(logp=logp, lambdas=lams, curves=curves, diag=diag)
