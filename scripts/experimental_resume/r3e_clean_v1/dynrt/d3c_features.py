"""D3c: P-B4 features (ANALYSIS_PLAN.md, Addendum 4), fitted inside training folds. No outcome is read here.

Sentence pathway alignment (from the in-fold B1 fit of the training questions)
  PA_s = mean over the sentence's tokens and the layer pairs of the token's mean over its k x k co-selection pairs of
         delta[c_s, a, b],  delta = log q_alt - log q_null   (d3b_b1.fold_delta, lambda = 1).
Lexically controlled alignment
  PAlex_s = PA_s - E[PA_s | token ids]; the expectation is the mean over 10 class shuffles within token id x
  position bin (all universe tokens, d3b_b1.permute_classes).  Two readings are computed and frozen:
  A (primary)  the B1 model is REFIT on the shuffled training labels and the sentence is scored under its shuffled
               token classes (the lexical-only model of the B1 diagnostic);
  B (sensitivity) the real in-fold fit scores the shuffled token classes.
Attempt features: mean PA, mean PAlex, SD of PA across the attempt's sentences (PA_sd), per-class mean PA.
Marginal block: for each class c the attempt's mean over its class-c sentences of the layer-third average of
  sum_e h[l, e] log(p[c, l, e] / p0[l, e])  (in-fold D1 profiles, kappa = 50) = 7 x 3 features, missing -> 0, plus 7
  presence indicators.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from . import d3b_b1 as B
from . import features as F
from .common import CLASSES, abs_bin
from .cv import question_folds

NC = 7
N_SHUF = 10
SHUF_SEED = 20260930
PA_NAMES = ["PA", "PAlex", "PA_sd", "PAlexB", "PA_Explore", "PA_Verify", "PA_Monitor", "PA_tok"]
MARG_NAMES = [f"marg_{c}_{t}" for c in CLASSES for t in ("d1", "d2", "d3")] + [f"has_{c}" for c in CLASSES]
CLASS_FEATURES = {"PA_Explore": CLASSES.index("Explore"), "PA_Verify": CLASSES.index("Verify"),
                  "PA_Monitor": CLASSES.index("Monitor")}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- universe

@dataclass
class Tokens:
    """Labelled tokens of a set of attempts, ordered by (attempt, rank); sentences are contiguous runs."""
    ids: np.ndarray        # [N, L, k] uint8
    cls: np.ndarray        # [N] int64
    att: np.ndarray        # [N] int64 attempt index (dense, 0..n_att-1)
    sent: np.ndarray       # [N] int64 sentence index (dense, non-decreasing)
    tokid: np.ndarray      # [N]
    pos: np.ndarray        # [N] absolute-position bin
    sent_att: np.ndarray   # [S]
    sent_cls: np.ndarray   # [S]
    sent_ntok: np.ndarray  # [S]
    n_att: int
    experts: int

    @property
    def k(self) -> int:
        return int(self.ids.shape[2])

    @property
    def n_layers(self) -> int:
        return int(self.ids.shape[1])

    @property
    def n_sent(self) -> int:
        return int(len(self.sent_att))

    def sent_starts(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.sent_ntok)[:-1]].astype(np.int64)


def build_tokens(tk: dict, keep: np.ndarray, experts: int) -> Tokens:
    """Restrict a d3b token table to the attempts flagged in `keep`, renumbering attempts and sentences."""
    keep = np.asarray(keep, bool)
    att_map = np.full(len(keep), -1, np.int64)
    att_map[keep] = np.arange(int(keep.sum()))
    sents = tk["sentences"]
    s_keep = keep[sents["att"].to_numpy()]
    sent_map = np.full(len(sents), -1, np.int64)
    sent_map[s_keep] = np.arange(int(s_keep.sum()))
    row = keep[tk["att"]]
    sent = sent_map[tk["sent"][row]]
    if (np.diff(sent) < 0).any():
        raise ValueError("token rows are not ordered by sentence")
    s = sents[s_keep]
    ntok = np.bincount(sent, minlength=int(s_keep.sum()))
    if not (ntok == s["n_tokens"].to_numpy()).all():
        raise ValueError("sentence token counts differ from the token table")
    return Tokens(tk["ids"][row], tk["cls"][row], att_map[tk["att"][row]], sent, tk["tokid"][row],
                  abs_bin(tk["rank"][row]).astype(np.int64), att_map[s["att"].to_numpy()],
                  s["cls"].to_numpy().astype(np.int64), ntok.astype(np.int64), int(keep.sum()), experts)


def shuffled_labels(t: Tokens, n_shuf: int = N_SHUF, seed: int = SHUF_SEED) -> list[np.ndarray]:
    """Class labels shuffled within token id x position bin (one array per shuffle)."""
    return [B.permute_classes(t.cls, t.tokid, t.pos, seed, r) for r in range(n_shuf)]


# ----------------------------------------------------------------------------- training sets

def enumerate_sets(groups: np.ndarray, repeats: int = 5, outer: int = 5, inner: int = 3, seed0: int = 0
                   ) -> list[np.ndarray]:
    """The training-row sets C8 (cv.run_cv) fits blocks on: outer-train and its three inner-train sets."""
    groups = np.asarray(groups)
    sets = []
    for r in range(repeats):
        folds = question_folds(groups, outer, seed0 + r)
        for f in range(outer):
            train = np.flatnonzero(folds != f)
            sets.append(train)
            inner_folds = question_folds(groups[train], inner, seed0 * 1000 + r * 10 + f)
            for g in range(inner):
                sets.append(train[inner_folds != g])
    return sets


def set_key(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, np.int64).tobytes()).hexdigest()


# -------------------------------------------------------------------------------- pathway scores

def _cells(ids_i: np.ndarray, ids_j: np.ndarray, experts: int) -> np.ndarray:
    n, k = ids_i.shape
    return (ids_i.astype(np.int32)[:, :, None] * experts + ids_j.astype(np.int32)[:, None, :]).reshape(n, k * k)


def _sentence_sums(delta32: np.ndarray, cls: np.ndarray, cells: np.ndarray, e2: int, starts: np.ndarray
                   ) -> np.ndarray:
    """Per-sentence sum over tokens and k*k pairs of delta[c, cell]; delta32 is the flat [C * E^2] float32 table."""
    idx = cls.astype(np.int32)[:, None] * e2 + cells
    tok = delta32[idx].sum(1, dtype=np.float64)
    return np.add.reduceat(tok, starts)


def _fit_deltas(tables: np.ndarray, member: np.ndarray, e: int, k2: float) -> tuple[np.ndarray, np.ndarray]:
    """delta [n_sets, C*E*E] float32 and the classes with training tokens [n_sets, C] for every training set."""
    train = member @ tables / k2                                      # [n_sets, C*E*E] float64
    deltas = np.empty((len(member), NC * e * e), np.float32)
    oks = np.empty((len(member), NC), bool)
    for i in range(len(member)):
        d, ok, _, _ = B.fold_delta(train[i].reshape(NC, e, e))
        deltas[i] = d.ravel()
        oks[i] = ok
    return deltas, oks


_STATE: dict = {}


def _pair_chunk(pair_ids: list[int]) -> tuple[np.ndarray, int]:
    """Worker: accumulate sentence sums over a chunk of layer pairs -> [3, n_sets, S_score], oks of the last pair."""
    st = _STATE
    fit: Tokens = st["fit"]
    score: Tokens = st["score"]
    member = st["member"]
    e, k = fit.experts, fit.k
    e2, k2 = e * e, float(fit.k * fit.k)
    starts = score.sent_starts()
    out = np.zeros((3, len(member), score.n_sent))
    for p in pair_ids:
        i, j, _ = st["pairs"][p]
        fit_cells_needed = st["refit"]
        cells_fit = _cells(fit.ids[:, i, :], fit.ids[:, j, :], e)
        cells_sc = cells_fit if score is fit else _cells(score.ids[:, i, :], score.ids[:, j, :], e)
        t0 = B.pair_tables(fit.ids[:, i, :], fit.ids[:, j, :], fit.cls, fit.att, fit.n_att, e)
        deltas, _ = _fit_deltas(t0.reshape(fit.n_att, -1).astype(np.float64), member, e, k2)
        for s in range(len(member)):
            out[0, s] += _sentence_sums(deltas[s], score.cls, cells_sc, e2, starts)
        for r in range(len(st["fit_shuf"])):
            for s in range(len(member)):
                out[1, s] += _sentence_sums(deltas[s], st["score_shuf"][r], cells_sc, e2, starts)
            if fit_cells_needed:
                tr = B.pair_tables(fit.ids[:, i, :], fit.ids[:, j, :], st["fit_shuf"][r], fit.att, fit.n_att, e)
                dr, _ = _fit_deltas(tr.reshape(fit.n_att, -1).astype(np.float64), member, e, k2)
                for s in range(len(member)):
                    out[2, s] += _sentence_sums(dr[s], st["score_shuf"][r], cells_sc, e2, starts)
        del cells_fit, cells_sc
    return out, len(pair_ids)


def pathway_sums(fit: Tokens, score: Tokens, member: np.ndarray, fit_shuf: list, score_shuf: list, pairs: list,
                 workers: int, refit: bool = True) -> np.ndarray:
    """Sentence sums [3, n_sets, S_score] over the layer pairs: true labels; shuffled labels with the true fit
    (reading B); shuffled labels with a fit on the shuffled training labels (reading A). Shuffle sums are
    totals over the shuffles (divide by their number)."""
    _STATE.update(fit=fit, score=score, member=member, fit_shuf=fit_shuf, score_shuf=score_shuf, pairs=pairs,
                  refit=refit)
    chunks = [c.tolist() for c in np.array_split(np.arange(len(pairs)), max(1, min(workers, len(pairs))))
              if len(c)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_pair_chunk, chunks, chunksize=1)
    else:
        res = [_pair_chunk(c) for c in chunks]
    return sum(r[0] for r in res)


def membership(sets: list[np.ndarray], n: int) -> np.ndarray:
    m = np.zeros((len(sets), n))
    for i, s in enumerate(sets):
        m[i, s] = 1.0
    return m


def attempt_pa_features(sums: np.ndarray, score: Tokens, n_pairs: int, n_shuf: int) -> np.ndarray:
    """Attempt features [n_sets, n_att, len(PA_NAMES)] from the sentence sums."""
    k2 = float(score.k * score.k)
    den = score.sent_ntok * k2 * n_pairs
    pa = sums[0] / den
    lex_b = sums[1] / den / n_shuf
    lex_a = sums[2] / den / n_shuf
    n_sets, n_att = sums.shape[1], score.n_att
    out = np.full((n_sets, n_att, len(PA_NAMES)), np.nan)
    a = score.sent_att
    cnt = np.bincount(a, minlength=n_att).astype(float)
    for s in range(n_sets):
        x = pa[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.bincount(a, weights=x, minlength=n_att) / cnt
            sq = np.bincount(a, weights=(x - mean[a]) ** 2, minlength=n_att)
            sd = np.sqrt(sq / (cnt - 1))
            out[s, :, 0] = mean
            out[s, :, 1] = np.bincount(a, weights=x - lex_a[s], minlength=n_att) / cnt
            out[s, :, 2] = np.where(cnt >= 2, sd, np.nan)
            out[s, :, 3] = np.bincount(a, weights=x - lex_b[s], minlength=n_att) / cnt
            for col, c in enumerate(CLASS_FEATURES.values(), start=4):
                m = score.sent_cls == c
                n_c = np.bincount(a[m], minlength=n_att).astype(float)
                out[s, :, col] = np.where(n_c > 0, np.bincount(a[m], weights=x[m], minlength=n_att)
                                          / np.maximum(n_c, 1), np.nan)
            tok = np.bincount(a, weights=sums[0][s], minlength=n_att) / np.bincount(
                a, weights=score.sent_ntok.astype(float), minlength=n_att)
            out[s, :, 7] = tok / (k2 * n_pairs)
    return out


# --------------------------------------------------------------------------- marginal block

def marg_features(h: np.ndarray, sent_cls: np.ndarray, sent_att: np.ndarray, train_att: np.ndarray,
                  n_att: int, layers: int, experts: int) -> np.ndarray:
    """[n_att, 28]: 7 x 3 own-class depth-third marginal affinities (0 if missing) and 7 presence indicators."""
    rows = np.flatnonzero(np.asarray(train_att)[sent_att])
    prof = F.fit_profiles(h[rows], sent_cls[rows], sent_att[rows], layers, experts)
    lr = (np.log(prof.p) - np.log(prof.p0)[None]).astype(np.float32)     # [C, L, E] = log(p_c / p0)
    aff = np.einsum("sle,sle->sl", h.reshape(len(h), layers, experts), lr[sent_cls])
    thirds = np.stack([aff[:, i * layers // 3:(i + 1) * layers // 3].mean(1) for i in range(3)], 1)
    feats = np.zeros((n_att, NC, 3))
    counts = np.zeros((n_att, NC))
    for c in range(NC):
        m = sent_cls == c
        counts[:, c] = np.bincount(sent_att[m], minlength=n_att)
        for t in range(3):
            s = np.bincount(sent_att[m], weights=thirds[m, t], minlength=n_att)
            feats[:, c, t] = np.where(counts[:, c] > 0, s / np.maximum(counts[:, c], 1), 0.0)
    return np.column_stack([feats.reshape(n_att, -1), (counts > 0).astype(float)])
