"""Synthetic validation of the C8 machinery: planted effect and null calibration.

CLI: python -m dynrt.simcheck [--nsim 50]  (prints and writes results/simcheck.json)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import cv
from .common import RESULTS


def simulate(seed: int, n: int = 300, effect: float = 0.0, repeats: int = 2, boot: int = 200) -> dict:
    """One synthetic C8 comparison: 3 control columns vs controls + 2 candidate columns."""
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(size=(n, 3))
    sig = rng.normal(size=(n, 2))
    logit = -0.2 + 0.8 * ctrl[:, 0] - 0.4 * ctrl[:, 1] + effect * sig[:, 0]
    y = (rng.random(n) < cv._sigmoid(logit)).astype(float)
    blocks = {"c": cv.MatrixBlock(ctrl, ["a", "b", "c"]), "f": cv.MatrixBlock(sig, ["s1", "s2"])}
    groups = np.arange(n)
    spec = cv.CVSpec(blocks=blocks, models={"base": ["c"], "aug": ["c", "f"]}, y=y, groups=groups,
                     weights=np.ones(n), repeats=repeats)
    res = cv.run_cv(spec, workers=1, log=lambda *_: None)
    return cv.evaluate(res["preds"], "base", "aug", y, groups, np.ones(n), n_boot=boot, seed=seed)


def fresh_test_gain(seed: int, n_train: int, n_test: int = 20000, effect: float = 0.0) -> float:
    """True expected out-of-sample log-loss gain of the augmented model trained on n_train rows."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    ctrl = rng.normal(size=(n, 3))
    sig = rng.normal(size=(n, 2))
    logit = -0.2 + 0.8 * ctrl[:, 0] - 0.4 * ctrl[:, 1] + effect * sig[:, 0]
    y = (rng.random(n) < cv._sigmoid(logit)).astype(float)
    blocks = {"c": cv._Wrap(cv.MatrixBlock(ctrl, ["a", "b", "c"])),
              "f": cv._Wrap(cv.MatrixBlock(sig, ["s1", "s2"]))}
    w, groups = np.ones(n), np.arange(n)
    train, test = np.arange(n_train), np.arange(n_train, n)
    loss = {}
    for name, items in (("base", ["c"]), ("aug", ["c", "f"])):
        design = cv.Design(blocks, items, n)
        lam, _ = cv.choose_lambda(design, y, w, groups, train, seed, 3, cv.GRID, {})
        x_all = design.matrix(train, {})
        p = cv._fit_predict(x_all, y, w, train, test, lam)
        loss[name] = cv.weighted_logloss(y[test], p, w[test])
    return loss["base"] - loss["aug"]


def null_calibration(n: int, repeats: int, boot: int, nsim: int, seed0: int = 1000) -> dict:
    t0 = time.time()
    rows = [simulate(seed0 + s, n, 0.0, repeats, boot) for s in range(nsim)]
    g = np.array([r["gain"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    p = np.array([r["p_one_sided"] for r in rows])
    return dict(n=n, repeats=repeats, boot=boot, nsim=nsim, mean_gain=float(g.mean()),
                sd_gain=float(g.std()), mean_boot_sd=float(np.mean([r["boot_sd"] for r in rows])),
                covered=int(((lo <= 0) & (hi >= 0)).sum()), ci_hi_below_0=int((hi < 0).sum()),
                ci_lo_above_0=int((lo > 0).sum()), p_below_05=int((p < 0.05).sum()),
                seconds=time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=50)
    a = ap.parse_args()
    out = dict(null=[], planted=None)
    for n, rep, boot in ((300, 2, 200), (1500, 2, 300), (1500, 5, 300)):
        r = null_calibration(n, rep, boot, a.nsim)
        print(json.dumps(r), flush=True)
        out["null"].append(r)
    out["fresh_test_null_gain"] = {
        str(n_tr): float(np.mean([fresh_test_gain(7000 + s, n_tr) for s in range(a.nsim)]))
        for n_tr in (225, 1125)}
    print(json.dumps(out["fresh_test_null_gain"]), flush=True)
    pl = [simulate(500 + s, 500, 1.2, 2, 300) for s in range(10)]
    out["planted"] = dict(n=500, effect=1.2, gains=[r["gain"] for r in pl],
                          ci_lo_above_0=int(sum(r["ci_lo"] > 0 for r in pl)), nsim=10)
    print(json.dumps(out["planted"]), flush=True)
    (RESULTS / "simcheck.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
