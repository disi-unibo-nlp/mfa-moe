"""Control block C (natural spline of log reasoning tokens, difficulty, dataset FE, class shares...)."""
from __future__ import annotations

import numpy as np


def ns_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (ESL 5.2.1), K knots -> K-1 non-intercept columns.

    knots = [xi_1 ... xi_K] (boundary knots first and last); df = K - 1, so 4 knots give df = 3.
    """
    x = np.asarray(x, float)
    k = len(knots)

    def d(j):
        return (np.maximum(x - knots[j], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3) / (
            knots[-1] - knots[j])

    cols = [x] + [d(j) - d(k - 2) for j in range(k - 2)]
    return np.stack(cols, 1)


def spline_knots(x: np.ndarray, df: int = 3) -> np.ndarray:
    """Boundary knots at the extremes, interior knots at equally spaced quantiles."""
    inner = np.quantile(x, np.linspace(0, 1, df + 1)[1:-1])
    return np.r_[x.min(), inner, x.max()]


class ControlsBlock:
    """Design columns of the control model; every data-dependent quantity is fit on `train`."""

    def __init__(self, log_reasoning, difficulty, dataset, shares, n_labelled, capped=None,
                 dataset_levels=None):
        self.log_r = np.asarray(log_reasoning, float)
        self.difficulty = np.asarray(difficulty, float)
        self.dataset = np.asarray(dataset)
        self.shares = np.asarray(shares, float)
        self.log_n = np.log(np.asarray(n_labelled, float).clip(min=1))
        self.capped = np.zeros(len(self.log_r)) if capped is None else np.asarray(capped, float)
        self.levels = list(dataset_levels) if dataset_levels is not None else sorted(set(self.dataset))
        self.names = ([f"ns{i}" for i in range(3)] + ["difficulty", "capped"]
                      + [f"ds_{lv}" for lv in self.levels[1:]]
                      + [f"share_{i}" for i in range(self.shares.shape[1])] + ["log_n_labelled"])

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        train = np.asarray(train)
        knots = spline_knots(self.log_r[train], 3)
        spline = ns_basis(self.log_r, knots)
        diff = np.where(np.isnan(self.difficulty), np.nanmean(self.difficulty[train]), self.difficulty)
        dummies = np.stack([(self.dataset == lv).astype(float) for lv in self.levels[1:]], 1)
        return np.column_stack([spline, diff, self.capped, dummies, self.shares, self.log_n])
