"""Per-token depth-path quantities from the capture tensors (numpy only, no outcomes).

Conventions (ANALYSIS_PLAN.md):
- tensors: ids [L, T, k] selected experts, weights [L, T, k] gate weights, gaps [L, T] boundary gaps,
  reasoning = sorted completion-token indices of the reasoning span (the v3an mask);
- "top-1 expert" of a token at a layer = the selected expert with the largest gate weight (ties: first
  in stored order); "top-1 weight" is that weight;
- layer index = position in the routed-layer list (depth), 0..L-1;
- reasoning rank i = position within the reasoning tokens; the absolute windows (WIN_EDGES) act on i;
- a token pair (t-1, t) is "adjacent" when both are reasoning tokens with consecutive completion
  indices; F2 pairs are attributed to the window of token t.
"""
from __future__ import annotations

import numpy as np

from .common import EXPLORE, N_WIN, VERIFY, WIN_EDGES

CHUNK = 1 << 16
CLASS_SLOTS = (EXPLORE, VERIFY)          # class index -> slot 0 / 1 in the per-class COO keys


def top1_paths(ids: np.ndarray, weights: np.ndarray, positions: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 expert id (int32 [L, n]) and top-1 gate weight (float32 [L, n]) at `positions`."""
    sel = np.asarray(ids[:, positions, :])
    w = np.asarray(weights[:, positions, :])
    am = w.argmax(-1)[..., None]
    top1 = np.take_along_axis(sel, am, -1)[..., 0].astype(np.int32)
    w1 = np.take_along_axis(w, am, -1)[..., 0].astype(np.float32)
    return top1, w1


def window_index(n: int) -> np.ndarray:
    """Absolute-window index (0..N_WIN-1) of each reasoning rank 0..n-1."""
    return np.searchsorted(np.asarray(WIN_EDGES), np.arange(n), side="right").astype(np.int8)


def token_spearman(w1: np.ndarray) -> np.ndarray:
    """Per-token Spearman correlation of top-1 gate weight with depth: [L, n] -> [n] (NaN if constant).

    Average ranks over ties; depth has no ties, so the correlation is the Pearson of the two rank vectors.
    """
    from scipy.stats import rankdata
    L, n = w1.shape
    out = np.empty(n)
    d = np.arange(L, dtype=np.float64) - (L - 1) / 2.0
    dd = float((d ** 2).sum())
    for a in range(0, n, CHUNK):
        r = rankdata(w1[:, a:a + CHUNK].astype(np.float64), axis=0, method="average")
        rc = r - (L + 1) / 2.0
        num = (rc * d[:, None]).sum(0)
        den = np.sqrt((rc ** 2).sum(0) * dd)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:a + CHUNK] = np.where(den > 1e-12, num / den, np.nan)
    return out


def adjacent_mask(reasoning: np.ndarray) -> np.ndarray:
    """adj[i] True iff reasoning token i-1 and i have consecutive completion indices (adj[0] False)."""
    r = np.asarray(reasoning, dtype=np.int64)
    adj = np.zeros(len(r), bool)
    adj[1:] = np.diff(r) == 1
    return adj


def coherence_per_token(top1: np.ndarray) -> np.ndarray:
    """F2 per token: fraction of layers where top-1 equals the previous token's top-1; [n], [0]=NaN."""
    n = top1.shape[1]
    out = np.full(n, np.nan)
    if n > 1:
        out[1:] = (top1[:, 1:] == top1[:, :-1]).mean(0)
    return out


def coo_counts(top1: np.ndarray, num_experts: int, group: np.ndarray, n_groups: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Sparse transition counts keyed by (group, layer-pair, a, b).

    key = group*(L-1)*E*E + l*E*E + a*E + b with a = top-1 at layer l and b = top-1 at layer l+1 of
    tokens whose group id is in 0..n_groups-1 (group < 0 tokens are dropped). Returns (int32 keys sorted,
    int32 counts).
    """
    L, n = top1.shape
    E = int(num_experts)
    base = (L - 1) * E * E
    if (n_groups * base) >= 2 ** 31:
        raise ValueError("COO key overflow")
    keep = np.flatnonzero(group >= 0)
    if len(keep) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    t = top1[:, keep].astype(np.int64)
    lay = (np.arange(L - 1, dtype=np.int64) * E * E)[:, None]
    key = lay + t[:-1] * E + t[1:] + group[keep].astype(np.int64)[None, :] * base
    uniq, cnt = np.unique(key.ravel(), return_counts=True)
    return uniq.astype(np.int32), cnt.astype(np.int32)


def attempt_stats(top1: np.ndarray, w1: np.ndarray, reasoning: np.ndarray, num_experts: int,
                  cls_of_rank: np.ndarray | None = None) -> dict:
    """Sufficient statistics of one attempt for F1/F2/F3 (window-resolved) and per-class F1."""
    L, n = top1.shape
    win = window_index(n)
    adj = adjacent_mask(reasoning)
    f2 = coherence_per_token(top1)
    f3 = token_spearman(w1)
    ok2 = adj & np.isfinite(f2)
    ok3 = np.isfinite(f3)
    n_tok = np.bincount(win, minlength=N_WIN).astype(np.int64)
    out = dict(n_tok=n_tok,
               f2_sum=np.bincount(win[ok2], weights=f2[ok2], minlength=N_WIN),
               f2_n=np.bincount(win[ok2], minlength=N_WIN).astype(np.int64),
               f3_sum=np.bincount(win[ok3], weights=f3[ok3], minlength=N_WIN),
               f3_n=np.bincount(win[ok3], minlength=N_WIN).astype(np.int64))
    out["coo_key"], out["coo_cnt"] = coo_counts(top1, num_experts, win.astype(np.int64), N_WIN)
    if cls_of_rank is not None:
        grp = np.full(n, -1, np.int64)
        for slot, c in enumerate(CLASS_SLOTS):
            grp[cls_of_rank == c] = slot
        out["cls_key"], out["cls_cnt"] = coo_counts(top1, num_experts, grp, len(CLASS_SLOTS))
        out["cls_n_tok"] = np.array([(cls_of_rank == c).sum() for c in CLASS_SLOTS], np.int64)
        # per-class F2/F3 are not part of the plan; only F1 is class-resolved
    return out


# ------------------------------------------------------------------------------ bigram (F1)

def fit_log_bigram(counts_flat: np.ndarray, layers: int, experts: int, alpha: float) -> np.ndarray:
    """Add-alpha bigram log-probabilities log P(b | a) per layer pair: flat [(L-1)*E*E] float64."""
    c = counts_flat.reshape(layers - 1, experts, experts)
    den = c.sum(-1, keepdims=True) + alpha * experts
    return np.log((c + alpha) / den).reshape(-1)
