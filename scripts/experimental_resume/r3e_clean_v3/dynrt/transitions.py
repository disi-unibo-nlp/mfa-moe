"""C2 (previous-class persistence) and C6 (Explore-onset mismatch) features, fitted on training folds only.

C2, adjacent labelled pairs a -> b (same trace segment, a != b), s = the b sentence:
- prev_aff = d_b(h_s) - d_a(h_s) (higher = routing closer to the previous class), centred on (a, b) cell
  means fitted on the training pairs (shrinkage n / (n + 20) toward the pooled mean), then residualised
  on class-b x abs-bin x relative-quintile x length-tercile cells as in D1;
- bJSD = layer-mean JSD (bits) between the last-16 histogram of s-1 and the first-16 histogram of s,
  residualised on (a, b) cell x abs-position bin (shrunk toward the (a, b) mean, then the pooled mean);
- attempt features = means over the attempt's eligible pairs (overall and by direction group), missing
  (NaN) with an indicator when there are none. Class profiles are the D1 profiles (`features.fit`).

C6: x = d_Explore(h over reasoning tokens [j-63, j]) minus the training-fold mean of the same score in j's
absolute-position bin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from .common import CELL_SHRINK, EXPLORE, N_ABS, abs_bin

NC = F.NC
GROUPS = ("entry", "exit", "other")
DYN_NAMES = (["M_all", "M_explore", "margin", "prev_aff", "bJSD", "has_trans", "n_pairs"]
             + [f"{v}_{g}" for g in GROUPS for v in ("prev_aff", "bJSD", "has")])


# ------------------------------------------------------------------------------- pairs

@dataclass
class PairTable:
    attempt: np.ndarray
    a: np.ndarray
    b: np.ndarray
    row: np.ndarray          # sentence row of s in the SentenceTable
    abs_bin: np.ndarray
    rel_bin: np.ndarray
    n_tokens: np.ndarray
    bjsd: np.ndarray         # raw layer-mean JSD (bits)
    group: np.ndarray        # 0 entry (a -> Explore), 1 exit (Explore -> b), 2 other
    n_attempts: int


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)


def layer_mean_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Layer-mean Jensen-Shannon divergence in bits; p, q [P, L, E] float (renormalised here)."""
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= p.sum(-1, keepdims=True)
    q /= q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    jsd = _entropy_bits(m) - 0.5 * (_entropy_bits(p) + _entropy_bits(q))
    return np.clip(jsd, 0.0, None).mean(-1)


def build_pairs(coh, chunk: int = 2000) -> PairTable:
    """Eligible transitions of a cohort (a != b, previous sentence labelled and in the same segment)."""
    s, hist, t = coh.sent, coh.hist, coh.table
    idx = np.flatnonzero(s["prev_same_segment"].to_numpy())
    cls = s["cls"].to_numpy(np.int64)
    a, b = cls[idx - 1], cls[idx]
    if not (t.attempt[idx - 1] == t.attempt[idx]).all():
        raise ValueError("adjacent rows of different attempts")
    keep = a != b
    idx, a, b = idx[keep], a[keep], b[keep]
    first = np.searchsorted(hist["first16_rows"], idx)
    last = np.searchsorted(hist["last16_rows"], idx - 1)
    if not ((hist["first16_rows"][first] == idx).all() and (hist["last16_rows"][last] == idx - 1).all()):
        raise ValueError("edge histograms missing for an eligible pair")
    bjsd = np.zeros(len(idx))
    for lo in range(0, len(idx), chunk):
        sl = slice(lo, lo + chunk)
        bjsd[sl] = layer_mean_jsd(hist["last16"][last[sl]], hist["first16"][first[sl]])
    group = np.where(b == EXPLORE, 0, np.where(a == EXPLORE, 1, 2))
    return PairTable(t.attempt[idx], a, b, idx, t.abs_bin[idx], t.rel_bin[idx], t.n_tokens[idx], bjsd,
                     group, t.n_attempts)


# ------------------------------------------------------------------------ pair residuals

@dataclass
class CellCentering:
    """(a, b) cell means shrunk toward the pooled mean with weight n / (n + shrink)."""
    mean: np.ndarray
    n: np.ndarray
    pooled: float
    shrink: float

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cell = a * NC + b
        w = self.n[cell] / (self.n[cell] + self.shrink)
        return w * self.mean[cell] + (1 - w) * self.pooled


def fit_centering(x: np.ndarray, a: np.ndarray, b: np.ndarray, shrink: float = CELL_SHRINK) -> CellCentering:
    cell = a * NC + b
    n = np.bincount(cell, minlength=NC * NC).astype(float)
    s = np.bincount(cell, weights=x, minlength=NC * NC)
    mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    return CellCentering(mean, n, float(x.mean()), shrink)


@dataclass
class CellPositionResidualizer:
    """(a, b) x abs-bin cell means shrunk toward the (a, b) mean, itself shrunk toward the pooled mean."""
    ab: CellCentering
    cell_mean: np.ndarray
    cell_n: np.ndarray
    shrink: float

    def predict(self, a, b, absb) -> np.ndarray:
        coarse = self.ab.predict(a, b)
        cell = (a * NC + b) * N_ABS + absb
        w = self.cell_n[cell] / (self.cell_n[cell] + self.shrink)
        return w * self.cell_mean[cell] + (1 - w) * coarse


def fit_cell_position(x, a, b, absb, shrink: float = CELL_SHRINK) -> CellPositionResidualizer:
    ab = fit_centering(x, a, b, shrink)
    cell = (a * NC + b) * N_ABS + absb
    n = np.bincount(cell, minlength=NC * NC * N_ABS).astype(float)
    s = np.bincount(cell, weights=x, minlength=NC * NC * N_ABS)
    mean = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    return CellPositionResidualizer(ab, mean, n, shrink)


@dataclass
class PairFitted:
    profiles: F.Profiles
    centering: CellCentering
    resid: F.Residualizer
    bjsd: CellPositionResidualizer


def _raw_prev_aff(pt: PairTable, table: F.SentenceTable, profiles: F.Profiles) -> np.ndarray:
    d = F.d_scores(table.h[pt.row], profiles)
    n = np.arange(len(d))
    return d[n, pt.b] - d[n, pt.a]


def fit_pairs(pt: PairTable, table: F.SentenceTable, profiles: F.Profiles, train_attempts: np.ndarray
              ) -> PairFitted:
    """Centring and residual cells from the pairs of the TRAINING attempts only."""
    m = np.asarray(train_attempts)[pt.attempt]
    raw = _raw_prev_aff(_subset(pt, m), table, profiles)
    a, b = pt.a[m], pt.b[m]
    cen = fit_centering(raw, a, b)
    resid = F.fit_residualizer((raw - cen.predict(a, b))[:, None], b, pt.abs_bin[m], pt.rel_bin[m],
                               pt.n_tokens[m])
    bj = fit_cell_position(pt.bjsd[m], a, b, pt.abs_bin[m].astype(np.int64))
    return PairFitted(profiles, cen, resid, bj)


def _subset(pt: PairTable, m: np.ndarray) -> PairTable:
    return PairTable(pt.attempt[m], pt.a[m], pt.b[m], pt.row[m], pt.abs_bin[m], pt.rel_bin[m],
                     pt.n_tokens[m], pt.bjsd[m], pt.group[m], pt.n_attempts)


def pair_features(pf: PairFitted, pt: PairTable, table: F.SentenceTable) -> dict[str, np.ndarray]:
    """Residual prev_aff and bJSD of every pair (any attempts) under a fitted model."""
    raw = _raw_prev_aff(pt, table, pf.profiles)
    centred = raw - pf.centering.predict(pt.a, pt.b)
    prev = pf.resid.residual(centred[:, None], pt.b, pt.abs_bin, pt.rel_bin, pt.n_tokens)[:, 0]
    bj = pt.bjsd - pf.bjsd.predict(pt.a, pt.b, pt.abs_bin.astype(np.int64))
    return dict(prev_aff=prev, bJSD=bj)


def aggregate_pairs(vals: dict[str, np.ndarray], pt: PairTable) -> dict[str, np.ndarray]:
    """Attempt means over eligible pairs (overall and per direction group) plus indicators."""
    n = pt.n_attempts
    out = {}
    masks = {"": np.ones(len(pt.attempt), bool)} | {g: pt.group == i for i, g in enumerate(GROUPS)}
    for g, m in masks.items():
        cnt = np.bincount(pt.attempt[m], minlength=n).astype(float)
        for k in ("prev_aff", "bJSD"):
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.bincount(pt.attempt[m], weights=vals[k][m], minlength=n) / cnt
            out[k if not g else f"{k}_{g}"] = np.where(cnt > 0, mean, np.nan)
        out["has_trans" if not g else f"has_{g}"] = (cnt > 0).astype(float)
        if not g:
            out["n_pairs"] = cnt
    return out


def dyn_features(table: F.SentenceTable, pt: PairTable, train_attempts: np.ndarray
                 ) -> tuple[dict[str, np.ndarray], PairFitted, F.Fitted]:
    """D1 features plus C2 features of ALL attempts, everything fitted on the training attempts."""
    fitted = F.fit(table, train_attempts)
    d1 = F.aggregate(F.sentence_scores(fitted, table), table)
    pf = fit_pairs(pt, table, fitted.profiles, train_attempts)
    c2 = aggregate_pairs(pair_features(pf, pt, table), pt)
    return {**d1, **c2}, pf, fitted


class DynBlock:
    """CV block: D1 routing features and C2 transition features, one profile fit per call."""

    names = list(DYN_NAMES)

    def __init__(self, table: F.SentenceTable, pairs: PairTable):
        self.table, self.pairs = table, pairs

    def fit_transform(self, train_idx: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.table.n_attempts, dtype=bool)
        mask[np.asarray(train_idx)] = True
        feats, _, _ = dyn_features(self.table, self.pairs, mask)
        return np.column_stack([feats[k] for k in DYN_NAMES])


# ------------------------------------------------------------------------------- C6

@dataclass
class OnsetData:
    """First lexicon-v1 onset decision (64 <= j <= 8192) per attempt; arrays aligned to the attempts."""
    has: np.ndarray
    j: np.ndarray
    marker: np.ndarray       # index into MARKERS, -1 if none
    h: np.ndarray            # [A, L*E] float32 (rows of attempts without a decision are 0)
    burden: np.ndarray       # Explore share of labelled sentences starting in (j, j+512], NaN if none


def load_onset(coh, onset_dir) -> OnsetData:
    from pathlib import Path

    from .onset import MARKERS
    d = Path(onset_dir)
    meta = pd.read_parquet(d / "onset_meta.parquet")
    z = np.load(d / "onset_hist.npz")
    att = coh.attempts
    pos = pd.Index(att["attempt_id"]).get_indexer(meta["attempt_id"])
    if (pos < 0).any() or len(meta) != len(att):
        raise ValueError("onset table is not aligned to the attempts")
    n = len(att)
    has = np.zeros(n, bool)
    j = np.full(n, -1, np.int64)
    marker = np.full(n, -1, np.int64)
    ok = meta["window_ok"].to_numpy(bool)
    has[pos[ok]] = True
    j[pos[ok]] = meta["j"].to_numpy()[ok]
    marker[pos[ok]] = [MARKERS.index(m) for m in meta["marker"].to_numpy()[ok]]
    h_rows = z["h"]
    L_E = h_rows.shape[1] * h_rows.shape[2]
    h = np.zeros((n, L_E), np.float32)
    rows = z["rows"]                                   # row positions in the meta table
    h[pos[rows]] = h_rows.reshape(len(rows), -1).astype(np.float32)
    burden = np.full(n, np.nan)
    s = coh.sent
    start = s["tok_start"].to_numpy()
    is_ex = (s["cls"].to_numpy() == EXPLORE)
    a_of = coh.table.attempt
    j_rank = np.full(n, -1, np.int64)
    j_rank[pos[ok]] = meta["j_rank"].to_numpy()[ok]
    for i in np.flatnonzero(has):
        m = (a_of == i) & (start > j_rank[i]) & (start <= j_rank[i] + 512)
        if m.any():
            burden[i] = is_ex[m].mean()
    return OnsetData(has, j, marker, h, burden)


class OnsetBlock:
    """CV block: x = d_Explore(window) minus the training mean of j's position bin (NaN without decision)."""

    names = ["x"]

    def __init__(self, table: F.SentenceTable, onset: OnsetData):
        self.table, self.onset = table, onset

    def scores(self, profiles: F.Profiles) -> np.ndarray:
        d = np.full(len(self.onset.has), np.nan)
        rows = np.flatnonzero(self.onset.has)
        d[rows] = F.d_scores(self.onset.h[rows], profiles)[:, EXPLORE]
        return d

    def fit_transform(self, train_idx: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.table.n_attempts, dtype=bool)
        mask[np.asarray(train_idx)] = True
        prof = F.fit(self.table, mask).profiles
        d = self.scores(prof)
        bins = np.where(self.onset.has, abs_bin(np.maximum(self.onset.j, 0)), -1)
        tr = mask & self.onset.has
        pooled = float(d[tr].mean())
        mean = np.array([d[tr & (bins == b)].mean() if (tr & (bins == b)).any() else pooled
                         for b in range(N_ABS)])
        x = np.where(self.onset.has, d - mean[np.maximum(bins, 0)], np.nan)
        return x[:, None]


class PrefixControls:
    """Prefix controls only: log j, difficulty, dataset FE, marker-identity FE (final length forbidden)."""

    def __init__(self, onset: OnsetData, difficulty, dataset, n_markers: int):
        self.log_j = np.where(onset.has, np.log(np.maximum(onset.j, 1)), np.nan)
        self.difficulty = np.asarray(difficulty, float)
        self.dataset = np.asarray(dataset)
        self.marker = onset.marker
        self.levels = sorted(set(self.dataset))
        self.n_markers = n_markers
        self.names = (["log_j", "difficulty"] + [f"ds_{lv}" for lv in self.levels[1:]]
                      + [f"marker_{i}" for i in range(1, n_markers)])

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        train = np.asarray(train)
        diff = np.where(np.isnan(self.difficulty), np.nanmean(self.difficulty[train]), self.difficulty)
        ds = np.stack([(self.dataset == lv).astype(float) for lv in self.levels[1:]], 1)
        mk = np.stack([(self.marker == i).astype(float) for i in range(1, self.n_markers)], 1)
        return np.column_stack([self.log_j, diff, ds, mk])
