"""Class-routing mismatch features (C1), fitted on training questions only. No outcome is used.

Definitions (ANALYSIS_PLAN.md, frozen):
- class profiles p[c,l,e]: sentence histograms averaged within attempt, then across attempts, shrunk
  toward the marginal p0 with kappa = 50 (p = (n_c p_hat_c + kappa p0) / (n_c + kappa)), floor 1e-6;
- d_c(h) = mean_l sum_e h[l,e] log(p0[l,e] / p[c,l,e])  (lower = more class-c-like);
- nuisance residualisation on class x abs-position bin x relative quintile x length tercile cell
  means, shrunk toward the class x abs-bin mean with weight n / (n + 20);
- attempt aggregation: unweighted mean over the attempt's labelled sentences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import CELL_SHRINK, EPS, EXPLORE, KAPPA, N_ABS, N_REL

NC = 7
N_LEN = 3
FEATURES = ("M_all", "M_explore", "margin")


def group_sums(x: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row sums per key: (unique keys, sums float64 [K, D], counts [K])."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sums = np.add.reduceat(x[order], starts, axis=0, dtype=np.float64)
    counts = np.diff(np.r_[starts, len(ks)])
    return ks[starts], sums, counts


@dataclass
class Profiles:
    """Marginal and class profiles, [L, E] each (rows sum to 1)."""
    p0: np.ndarray
    p: np.ndarray
    n_class: np.ndarray
    n_attempts_class: np.ndarray

    def log_ratio(self) -> np.ndarray:
        """log(p0 / p_c) flattened to [C, L*E]."""
        return (np.log(self.p0)[None] - np.log(self.p)).reshape(len(self.p), -1)


def _floor_normalise(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.maximum(p, eps)
    return p / p.sum(-1, keepdims=True)


def fit_profiles(h: np.ndarray, cls: np.ndarray, attempt: np.ndarray, layers: int, experts: int,
                 *, kappa: float = KAPPA, eps: float = EPS) -> Profiles:
    """Profiles from TRAINING sentences only: h [N, L*E] float, cls [N], attempt [N] (dense ids)."""
    d = layers * experts
    keys, sums, counts = group_sums(h, attempt.astype(np.int64))
    p0 = (sums / counts[:, None]).mean(0)
    keys_c, sums_c, counts_c = group_sums(h, attempt.astype(np.int64) * NC + cls.astype(np.int64))
    cls_of = keys_c % NC
    p_hat = np.zeros((NC, d))
    n_class = np.zeros(NC)
    n_att = np.zeros(NC, dtype=np.int64)
    means = sums_c / counts_c[:, None]
    for c in range(NC):
        sel = cls_of == c
        n_att[c] = int(sel.sum())
        n_class[c] = float(counts_c[sel].sum())
        p_hat[c] = means[sel].mean(0) if n_att[c] else p0
    p = (n_class[:, None] * p_hat + kappa * p0[None]) / (n_class[:, None] + kappa)
    p0 = _floor_normalise(p0.reshape(layers, experts), eps)
    p = _floor_normalise(p.reshape(NC, layers, experts), eps)
    return Profiles(p0=p0, p=p, n_class=n_class, n_attempts_class=n_att)


def d_scores(h: np.ndarray, profiles: Profiles) -> np.ndarray:
    """d_c(h) for all classes: [N, C] float64 (layer mean of sum_e h log(p0/p_c))."""
    layers = profiles.p0.shape[0]
    lr = profiles.log_ratio().astype(np.float32)
    return (h.astype(np.float32, copy=False) @ lr.T).astype(np.float64) / layers


@dataclass
class Residualizer:
    """Cell-mean nuisance model with hierarchical shrinkage (fit on training sentences)."""
    len_edges: np.ndarray
    cell_mean: np.ndarray
    cell_n: np.ndarray
    coarse_mean: np.ndarray
    shrink: float

    def cells(self, cls, abs_bin, rel_bin, n_tokens) -> np.ndarray:
        length = np.searchsorted(self.len_edges, n_tokens, side="left")
        return (((cls.astype(np.int64) * N_ABS + abs_bin) * N_REL + rel_bin) * N_LEN + length)

    def predict(self, cls, abs_bin, rel_bin, n_tokens) -> np.ndarray:
        cell = self.cells(cls, abs_bin, rel_bin, n_tokens)
        coarse = self.coarse_mean[cls.astype(np.int64) * N_ABS + abs_bin]
        n = self.cell_n[cell]
        w = (n / (n + self.shrink))[:, None]
        return w * self.cell_mean[cell] + (1.0 - w) * coarse

    def residual(self, x, cls, abs_bin, rel_bin, n_tokens) -> np.ndarray:
        return x - self.predict(cls, abs_bin, rel_bin, n_tokens)


def fit_residualizer(x: np.ndarray, cls, abs_bin, rel_bin, n_tokens,
                     shrink: float = CELL_SHRINK) -> Residualizer:
    """Fit on training sentences: x [N, F]."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    f = x.shape[1]
    edges = np.quantile(np.asarray(n_tokens, float), [1 / 3, 2 / 3])
    proto = Residualizer(edges, np.zeros((0, f)), np.zeros(0), np.zeros((0, f)), shrink)
    cell = proto.cells(np.asarray(cls), np.asarray(abs_bin), np.asarray(rel_bin), np.asarray(n_tokens))
    n_cells = NC * N_ABS * N_REL * N_LEN
    cell_n = np.bincount(cell, minlength=n_cells).astype(float)
    cell_sum = np.stack([np.bincount(cell, weights=x[:, j], minlength=n_cells) for j in range(f)], 1)
    cell_mean = np.divide(cell_sum, cell_n[:, None], out=np.zeros_like(cell_sum), where=cell_n[:, None] > 0)
    coarse = np.asarray(cls).astype(np.int64) * N_ABS + np.asarray(abs_bin)
    n_coarse = np.bincount(coarse, minlength=NC * N_ABS).astype(float)
    s_coarse = np.stack([np.bincount(coarse, weights=x[:, j], minlength=NC * N_ABS) for j in range(f)], 1)
    n_cls = np.bincount(np.asarray(cls).astype(np.int64), minlength=NC).astype(float)
    s_cls = np.stack([np.bincount(np.asarray(cls).astype(np.int64), weights=x[:, j], minlength=NC)
                      for j in range(f)], 1)
    global_mean = x.mean(0)
    cls_mean = np.where(n_cls[:, None] > 0, s_cls / np.maximum(n_cls, 1)[:, None], global_mean[None])
    coarse_mean = np.where(n_coarse[:, None] > 0, s_coarse / np.maximum(n_coarse, 1)[:, None],
                           np.repeat(cls_mean, N_ABS, axis=0))
    return Residualizer(edges, cell_mean, cell_n, coarse_mean, shrink)


@dataclass
class SentenceTable:
    """Sentence-level arrays for one cohort (rows sorted by attempt, then sentence)."""
    h: np.ndarray            # [N, L*E] float32
    cls: np.ndarray
    attempt: np.ndarray      # dense attempt index into the cohort's attempt table
    abs_bin: np.ndarray
    rel_bin: np.ndarray
    n_tokens: np.ndarray
    layers: int
    experts: int
    n_attempts: int


@dataclass
class Fitted:
    profiles: Profiles
    resid: Residualizer


def sentence_scores(fitted: Fitted, t: SentenceTable) -> dict[str, np.ndarray]:
    """Residual own-class score and margin per sentence (plus the raw d matrix)."""
    d = d_scores(t.h, fitted.profiles)
    own = d[np.arange(len(d)), t.cls.astype(np.int64)]
    margin = own - d.min(1)
    x = np.stack([own, margin], 1)
    r = fitted.resid.residual(x, t.cls, t.abs_bin, t.rel_bin, t.n_tokens)
    return dict(d=d, own_raw=own, margin_raw=margin, own=r[:, 0], margin=r[:, 1])


def fit(t: SentenceTable, train_attempts: np.ndarray, *, kappa: float = KAPPA) -> Fitted:
    """Fit profiles and residualiser on the sentences of the training attempts (boolean mask [A])."""
    rows = np.flatnonzero(np.asarray(train_attempts)[t.attempt])
    sub = SentenceTable(t.h[rows], t.cls[rows], t.attempt[rows], t.abs_bin[rows], t.rel_bin[rows],
                        t.n_tokens[rows], t.layers, t.experts, t.n_attempts)
    prof = fit_profiles(sub.h, sub.cls, sub.attempt, t.layers, t.experts, kappa=kappa)
    d = d_scores(sub.h, prof)
    own = d[np.arange(len(d)), sub.cls.astype(np.int64)]
    x = np.stack([own, own - d.min(1)], 1)
    resid = fit_residualizer(x, sub.cls, sub.abs_bin, sub.rel_bin, sub.n_tokens)
    return Fitted(prof, resid)


def aggregate(scores: dict[str, np.ndarray], t: SentenceTable, *, min_explore: int = 3
              ) -> dict[str, np.ndarray]:
    """Unweighted attempt means: M_all, M_explore (NaN if < min_explore Explore sentences), margin."""
    a, n = t.attempt.astype(np.int64), t.n_attempts
    cnt = np.bincount(a, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_all = np.bincount(a, weights=scores["own"], minlength=n) / cnt
        margin = np.bincount(a, weights=scores["margin"], minlength=n) / cnt
        ex = t.cls == EXPLORE
        n_ex = np.bincount(a[ex], minlength=n).astype(float)
        m_ex = np.bincount(a[ex], weights=scores["own"][ex], minlength=n) / n_ex
    m_ex = np.where(n_ex >= min_explore, m_ex, np.nan)
    m_all = np.where(cnt > 0, m_all, np.nan)
    margin = np.where(cnt > 0, margin, np.nan)
    return dict(M_all=m_all, M_explore=m_ex, margin=margin, n_sent=cnt, n_explore=n_ex)


def attempt_features(t: SentenceTable, train_attempts: np.ndarray, *, kappa: float = KAPPA
                     ) -> dict[str, np.ndarray]:
    """Fit on the training attempts, return the attempt-level features of ALL attempts."""
    fitted = fit(t, train_attempts, kappa=kappa)
    return aggregate(sentence_scores(fitted, t), t)


def apply_fitted(fitted: Fitted, t: SentenceTable) -> dict[str, np.ndarray]:
    """Attempt features of another table (e.g. cohort B) with a fitted model."""
    return aggregate(sentence_scores(fitted, t), t)


class RoutingBlock:
    """CV block: attempt features fit on the training attempts only (rows = attempts of `table`)."""

    names = list(FEATURES)

    def __init__(self, table: SentenceTable):
        self.table = table

    def fit_transform(self, train_idx: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.table.n_attempts, dtype=bool)
        mask[np.asarray(train_idx)] = True
        f = attempt_features(self.table, mask)
        return np.column_stack([f[k] for k in FEATURES])
