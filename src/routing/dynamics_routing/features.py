"""Class-routing mismatch scores, fitted on training questions only. No outcome is used.

Definitions:
- class profiles p[c,l,e]: sentence histograms averaged within attempt, then across attempts, shrunk
  toward the marginal p0 with kappa = 50 (p = (n_c p_hat_c + kappa p0) / (n_c + kappa)), floor 1e-6;
- d_c(h) = mean_l sum_e h[l,e] log(p0[l,e] / p[c,l,e])  (lower = more class-c-like).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import EPS, KAPPA

NC = 7


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
