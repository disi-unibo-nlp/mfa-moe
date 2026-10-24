"""Question-grouped folds, the ridge-penalty grid and the column scaler of the readout fits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = 10.0 ** np.arange(-3, 3)


@dataclass
class Scaler:
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        mean = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        return cls(np.where(np.isfinite(mean), mean, 0.0), np.where(sd > 1e-12, sd, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.sd
        return np.where(np.isnan(z), 0.0, z)


def question_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id (0..k-1) per row; all rows of a question share a fold; balanced by question."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of_q = np.empty(len(uniq), dtype=np.int64)
    fold_of_q[order] = np.arange(len(uniq)) % k
    return fold_of_q[inv]
