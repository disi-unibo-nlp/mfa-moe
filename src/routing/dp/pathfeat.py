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

CHUNK = 1 << 16


def top1_paths(ids: np.ndarray, weights: np.ndarray, positions: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 expert id (int32 [L, n]) and top-1 gate weight (float32 [L, n]) at `positions`."""
    sel = np.asarray(ids[:, positions, :])
    w = np.asarray(weights[:, positions, :])
    am = w.argmax(-1)[..., None]
    top1 = np.take_along_axis(sel, am, -1)[..., 0].astype(np.int32)
    w1 = np.take_along_axis(w, am, -1)[..., 0].astype(np.float32)
    return top1, w1


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
