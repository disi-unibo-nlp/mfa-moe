"""Window design and cross-fit task of the anticipation fits (`routing.anticipation`).

The saved K2 caches in `data/k2/<model>/` hold, per adjacent labelled pair (s, s+1), the pair table, the lexical counts
of sentence s and the kinematic columns of its last 16 / 32 reasoning tokens. `KData` is one response (TARGET: switch
= 1[b != a] over all pairs, 2-class softmax; dest = class of s+1 over switch pairs, 7-class). `Blocks` is the
fold-fitted design (tf-idf of the text and lexical counts, standardised dense blocks); `_task` is one
(repeat, outer fold) of the cross-fit with the inner penalty search. `routing.anticipation` subclasses `Blocks` for its
controls and runs `_task` in a pool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .common import DATA

K2DATA = DATA / "k2"
INNER = 3


def choose_penalty(mean_loss: np.ndarray, grid: np.ndarray) -> tuple[float, int]:
    best = int(np.flatnonzero(mean_loss <= mean_loss.min() + 1e-12)[-1])       # ties -> stronger penalty
    return float(grid[best]), best


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


class Blocks:
    """Fold-fitted design (tf-idf of text and lexical counts, standardised dense blocks)."""

    def __init__(self, d: KData, tr: np.ndarray, evs: list[np.ndarray]):
        from routing.dynrt.d3a_text import TextTransform
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
        from routing.dynrt.cv import Scaler
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
    from routing.dynrt.d3a_softmax import fit_softmax, predict_log_proba, weighted_nll
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


def _task(args):
    from routing.dynrt.cv import GRID, question_folds
    from routing.dynrt.d3a_softmax import fit_softmax, predict_log_proba
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


