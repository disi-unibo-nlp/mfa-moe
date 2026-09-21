"""Check optimized bootstrap CIs against the original SciPy calculations."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pointbiserialr, spearmanr

from moe_exp.correlation_pipeline.analyze import _bootstrap_ci, _bootstrap_spearman_ci


def _interval(values, samples):
    values = [value for value in values if np.isfinite(value)]
    if len(values) < max(20, samples // 2):
        return None, None
    return tuple(np.quantile(values, [0.025, 0.975]))


@pytest.mark.parametrize("constant", [False, True])
def test_cluster_bootstrap_matches_scipy(constant):
    data_rng = np.random.default_rng(71)
    frame = pd.DataFrame({
        "dataset": ["a"] * 15 + ["b"] * 12,
        "source_problem_id": [0] * 3 + [1] * 5 + [2] * 7 + [0] * 4 + [1] * 8,
        "x": data_rng.normal(size=27) if not constant else np.ones(27),
        "y": data_rng.integers(0, 2, size=27).astype(float),
    }, index=np.arange(27) * 3 + 10)
    frame.loc[frame.index[2], "x"] = np.nan
    frame.loc[frame.index[8], "y"] = np.nan
    strata = [
        [group.index.to_numpy() for _, group in subset.groupby("source_problem_id", sort=False)]
        for _, subset in frame.groupby("dataset", sort=False)
    ]
    rng = np.random.default_rng(19)
    values = []
    for _ in range(100):
        indices = np.concatenate([
            groups[i] for groups in strata for i in rng.integers(0, len(groups), len(groups))
        ])
        selected = frame.loc[indices, ["x", "y"]].dropna()
        if len(selected) >= 4 and selected.x.nunique() > 1 and selected.y.nunique() > 1:
            values.append(pointbiserialr(selected.y, selected.x).statistic)
    actual = _bootstrap_ci(
        frame, feature="x", target="y", samples=100, rng=np.random.default_rng(19)
    )
    expected = _interval(values, 100)
    assert actual == pytest.approx(expected) if expected[0] is not None else actual == expected
    np.testing.assert_array_equal(rng.integers(100, size=10), _rng_after_cluster(frame))


def _rng_after_cluster(frame):
    rng = np.random.default_rng(19)
    _bootstrap_ci(frame, feature="x", target="y", samples=100, rng=rng)
    return rng.integers(100, size=10)


@pytest.mark.parametrize("size", [3, 8, 40])
@pytest.mark.parametrize("constant", [False, True])
def test_spearman_bootstrap_matches_scipy(size, constant):
    x = np.arange(size, dtype=float) % 4
    y = np.ones(size) if constant else np.arange(size, dtype=float) % 3
    x[0] = np.nan
    rng = np.random.default_rng(42)
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[valid], y[valid]
    values = []
    if len(xx) >= 4:
        for _ in range(100):
            indices = rng.integers(0, len(xx), len(xx))
            if np.unique(xx[indices]).size > 1 and np.unique(yy[indices]).size > 1:
                values.append(spearmanr(xx[indices], yy[indices]).statistic)
    actual_rng = np.random.default_rng(42)
    actual = _bootstrap_spearman_ci(x, y, samples=100, rng=actual_rng)
    expected = _interval(values, 100)
    assert actual == pytest.approx(expected) if expected[0] is not None else actual == expected
    np.testing.assert_array_equal(rng.integers(100, size=10), actual_rng.integers(100, size=10))
