"""C8 incremental-validity machinery (frozen in ANALYSIS_PLAN.md).

- ridge logistic (intercept unpenalised, standardised columns); the penalty is on the mean-loss scale,
  objective mean_i w_i nll_i + lam/2 |b|^2 (i.e. lam * sum(w) on the summed loss), so the grid
  10^{-3..2} spans "almost unpenalised" to "all slopes shrunk to ~0" whatever the sample size;
- 4 outer question-grouped folds x 5 repeats (seeds 0-4); inner 3 question-grouped folds choose lam
  on the grid 10^{-3..2} (question-weighted log-loss);
- every data-dependent block (spline knots, class profiles, residual cells, scaling) is re-fit on the
  training part of the fold it is used in (Block.fit_transform(train) sees training rows only);
- primary = question-weighted mean log-loss (baseline - augmented) of pooled held-out predictions,
  averaged over repeats; CI/p from paired question bootstraps of the per-attempt gains;
- secondary: DeltaAUC, Brier, calibration slope.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

GRID = 10.0 ** np.arange(-3, 3)
PMIN = 1e-9


# ------------------------------------------------------------------------------ ridge

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _objective(x1, y, w, beta, lam):
    eta = x1 @ beta
    nll = np.logaddexp(0.0, eta) - y * eta
    return float((w * nll).sum() + 0.5 * lam * (beta[1:] ** 2).sum())


def ridge_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, *, max_iter: int = 60,
                   tol: float = 1e-9) -> np.ndarray:
    """Weighted L2-logistic by damped Newton; x already standardised. Returns [b0, b...].

    `lam` is on the mean-loss scale: the summed-loss penalty is lam * sum(w).
    """
    n, p = x.shape
    lam = lam * float(w.sum())
    x1 = np.column_stack([np.ones(n), x])
    pen = np.r_[0.0, np.ones(p)]
    prior = float(np.clip((w * y).sum() / w.sum(), 1e-6, 1 - 1e-6))
    beta = np.r_[np.log(prior / (1 - prior)), np.zeros(p)]
    f = _objective(x1, y, w, beta, lam)
    for _ in range(max_iter):
        mu = _sigmoid(x1 @ beta)
        g = x1.T @ (w * (mu - y)) + lam * pen * beta
        hess = (x1 * (w * mu * (1 - mu))[:, None]).T @ x1 + lam * np.diag(pen)
        step = np.linalg.solve(hess + 1e-10 * np.eye(p + 1), g)
        t = 1.0
        while t > 1e-6:
            cand = beta - t * step
            fc = _objective(x1, y, w, cand, lam)
            if fc <= f + 1e-12 * abs(f):
                break
            t *= 0.5
        else:
            break
        done = f - fc < tol * (1 + abs(f))
        beta, f = cand, fc
        if done:
            break
    return beta


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


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(beta[0] + x @ beta[1:])


def weighted_logloss(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, PMIN, 1 - PMIN)
    return float((w * -(y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


# --------------------------------------------------------------------------------- folds

def question_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Clean R3-E frozen duplicate-family blocks, shared with the R3-D readout."""
    qf = _families()
    names = [qf[str(q)] for q in groups]
    order = sorted(set(names), key=lambda f: _json_digest(f'forum-v1|{20260929 + seed}|{f}'))
    fold_of_family = {f: i % k for i, f in enumerate(order)}
    return np.asarray([fold_of_family[f] for f in names], dtype=np.int64)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def _families():
    path = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/family-freeze.json')
    value = json.loads(path.read_text())
    if value.get('sha256') != _json_digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed frozen duplicate-family map')
    return {q: f for f, ids in value['new_parent_pools']['families'].items() for q in ids}


# ---------------------------------------------------------------------------- blocks/model

class Design:
    """Columns of one model, drawn from named blocks; blocks are fit on the training rows only."""

    def __init__(self, blocks: dict, spec: list[str], n: int):
        self.blocks, self.spec, self.n = blocks, spec, n

    def matrix(self, train: np.ndarray, cache: dict) -> np.ndarray:
        cols = []
        key_train = train.tobytes()
        for item in self.spec:
            name, _, col = item.partition(".")
            key = (name, key_train)
            if key not in cache:
                cache[key] = self.blocks[name].fit_transform(train)
            mat, names = cache[key]
            if col:
                cols.append(mat[:, [names.index(col)]])
            else:
                cols.append(mat)
        return np.column_stack(cols) if cols else np.zeros((self.n, 0))


def _fit_predict(x_all, y, w, train, test, lam):
    sc = Scaler.fit(x_all[train])
    xs = sc.transform(x_all)
    beta = ridge_logistic(xs[train], y[train], w[train], lam)
    return predict_logistic(beta, xs[test])


def choose_lambda(design: Design, y, w, groups, train, seed, inner, grid, cache_root: dict,
                  valid=None):
    """Inner question-grouped CV over the ridge grid (blocks re-fit inside every inner fold)."""
    valid = np.ones(len(y), bool) if valid is None else valid
    folds = question_folds(groups[train], inner, seed)
    losses = np.zeros((inner, len(grid)))
    sizes = np.zeros(inner)
    for f in range(inner):
        tr, va = train[folds != f], train[folds == f]
        key = ("inner", seed, f)
        cache = cache_root.setdefault(key, {})
        x_all = design.matrix(tr, cache)
        tr_r, va_r = tr[valid[tr]], va[valid[va]]
        sc = Scaler.fit(x_all[tr_r])
        xs = sc.transform(x_all)
        for gi, lam in enumerate(grid):
            beta = ridge_logistic(xs[tr_r], y[tr_r], w[tr_r], lam)
            losses[f, gi] = weighted_logloss(y[va_r], predict_logistic(beta, xs[va_r]), w[va_r])
        sizes[f] = w[va_r].sum()
    mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
    best = int(np.flatnonzero(mean <= mean.min() + 1e-12)[-1])   # ties -> stronger penalty
    return float(grid[best]), mean


@dataclass
class CVSpec:
    blocks: dict
    models: dict                     # name -> list of "block" or "block.col" items
    y: np.ndarray
    groups: np.ndarray               # question id per attempt
    weights: np.ndarray              # question weights (1 / attempts per question)
    repeats: int = 5
    outer: int = 4
    inner: int = 3
    grid: np.ndarray = field(default_factory=lambda: GRID.copy())
    seed0: int = 0
    valid: np.ndarray | None = None  # rows with an outcome; others take part in block fits only


_STATE: dict = {}


def _task(args):
    r, fold = args
    spec: CVSpec = _STATE["spec"]
    n = len(spec.y)
    valid = np.ones(n, bool) if spec.valid is None else spec.valid
    folds = question_folds(spec.groups, spec.outer, spec.seed0 + r)
    train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    out, lams = {}, {}
    cache_root: dict = {}
    for name, items in spec.models.items():
        design = Design({k: _Wrap(v) for k, v in spec.blocks.items()}, items, n)
        lam, _ = choose_lambda(design, spec.y, spec.weights, spec.groups, train,
                               spec.seed0 * 1000 + r * 10 + fold, spec.inner, spec.grid, cache_root,
                               spec.valid)
        x_all = design.matrix(train, cache_root.setdefault(("outer",), {}))
        out[name] = _fit_predict(x_all, spec.y, spec.weights, train[valid[train]], test, lam)
        lams[name] = lam
    return r, fold, test, out, lams, n


class _Wrap:
    """Adapts a block to return (matrix, names) with the names of its columns."""

    def __init__(self, block):
        self.block = block

    def fit_transform(self, train):
        return self.block.fit_transform(train), list(getattr(self.block, "names", []))


def run_cv(spec: CVSpec, workers: int = 1, log=print) -> dict:
    """Cross-fitted held-out probabilities: {model: [repeats, n]} plus chosen penalties."""
    n = len(spec.y)
    tasks = [(r, f) for r in range(spec.repeats) for f in range(spec.outer)]
    _STATE["spec"] = spec
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            res = []
            for i, item in enumerate(pool.imap_unordered(_task, tasks), 1):
                res.append(item)
                if i % 5 == 0 or i == len(tasks):
                    log(f"cv: {i}/{len(tasks)} outer fits")
    else:
        res = [_task(t) for t in tasks]
    preds = {m: np.full((spec.repeats, n), np.nan) for m in spec.models}
    lams = {m: np.zeros((spec.repeats, spec.outer)) for m in spec.models}
    for r, fold, test, out, lam, _ in res:
        for m in spec.models:
            preds[m][r, test] = out[m]
            lams[m][r, fold] = lam[m]
    for m in spec.models:
        if np.isnan(preds[m]).any():
            raise RuntimeError("held-out predictions missing")
    return dict(preds=preds, lambdas=lams)


# --------------------------------------------------------------------------- evaluation

def attempt_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PMIN, 1 - PMIN)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def weighted_auc(score: np.ndarray, y: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """AUC for each row of the weight matrix wt [B, n] (ties count 1/2)."""
    order = np.argsort(score, kind="stable")
    s, yy, ww = score[order], y[order], wt[:, order]
    pos, neg = ww * yy, ww * (1 - yy)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pg = np.add.reduceat(pos, starts, axis=1)
    ng = np.add.reduceat(neg, starts, axis=1)
    before = np.cumsum(ng, axis=1) - ng
    return (pg * (before + 0.5 * ng)).sum(1) / (pos.sum(1) * neg.sum(1))


def weighted_cal_slope(p: np.ndarray, y: np.ndarray, wt: np.ndarray, iters: int = 12) -> np.ndarray:
    """Calibration slope: coefficient of logit(p) in a weighted logistic recalibration."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    a = np.zeros(len(wt))
    b = np.ones(len(wt))
    for _ in range(iters):
        mu = _sigmoid(a[:, None] + b[:, None] * z[None])
        r = wt * (y[None] - mu)
        v = wt * mu * (1 - mu)
        ga, gb = r.sum(1), (r * z[None]).sum(1)
        haa, hab, hbb = v.sum(1), (v * z[None]).sum(1), (v * z[None] ** 2).sum(1)
        det = haa * hbb - hab ** 2 + 1e-12
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
    return b


def bootstrap_multiplicity(groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Question multiplicities [B, n_attempts] of `n_boot` question resamples."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.zeros((n_boot, len(uniq)))
    for b in range(n_boot):
        counts[b] = np.bincount(idx[b], minlength=len(uniq))
    return counts[:, inv]


def evaluate(preds: dict, base: str, aug: str, y, groups, weights, *, n_boot: int = 1000,
             seed: int = 20260929) -> dict:
    """Paired comparison of two cross-fitted models with a paired question bootstrap."""
    pb, pa = preds[base], preds[aug]
    R = pb.shape[0]
    gain_i = (attempt_loss(y[None], pb) - attempt_loss(y[None], pa)).mean(0)
    uniq, inv = np.unique(groups, return_inverse=True)
    q_w = np.bincount(inv, weights=weights)
    q_gain = np.bincount(inv, weights=weights * gain_i) / q_w.clip(min=1e-12)
    point = float(q_gain.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = q_gain[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_one = float((1 + (boot <= 0).sum()) / (n_boot + 1))
    mult = bootstrap_multiplicity(groups, n_boot, seed + 1)
    wt_b = mult * weights[None]
    wt_1 = weights[None]
    out = dict(gain=point, ci_lo=float(lo), ci_hi=float(hi), p_one_sided=p_one,
               boot_sd=float(boot.std(ddof=1)), n_boot=n_boot, n_questions=int(len(uniq)),
               n_attempts=int(len(y)),
               logloss_base=float(np.mean([weighted_logloss(y, pb[r], weights) for r in range(R)])),
               logloss_aug=float(np.mean([weighted_logloss(y, pa[r], weights) for r in range(R)])))
    for name, fn in (("auc", weighted_auc),):
        d1 = np.mean([fn(pa[r], y, wt_1)[0] - fn(pb[r], y, wt_1)[0] for r in range(R)])
        db = np.mean([fn(pa[r], y, wt_b) - fn(pb[r], y, wt_b) for r in range(R)], axis=0)
        out["d" + name] = dict(diff=float(d1), ci_lo=float(np.percentile(db, 2.5)),
                               ci_hi=float(np.percentile(db, 97.5)),
                               base=float(np.mean([fn(pb[r], y, wt_1)[0] for r in range(R)])),
                               aug=float(np.mean([fn(pa[r], y, wt_1)[0] for r in range(R)])))

    def brier(p, wt):
        return (wt * (p[None] - y[None]) ** 2).sum(1) / wt.sum(1)
    d1 = np.mean([brier(pb[r], wt_1)[0] - brier(pa[r], wt_1)[0] for r in range(R)])
    db = np.mean([brier(pb[r], wt_b) - brier(pa[r], wt_b) for r in range(R)], axis=0)
    out["dbrier_improvement"] = dict(diff=float(d1), ci_lo=float(np.percentile(db, 2.5)),
                                     ci_hi=float(np.percentile(db, 97.5)),
                                     base=float(np.mean([brier(pb[r], wt_1)[0] for r in range(R)])),
                                     aug=float(np.mean([brier(pa[r], wt_1)[0] for r in range(R)])))
    cs = {}
    for label, pr in (("base", pb), ("aug", pa)):
        s1 = np.mean([weighted_cal_slope(pr[r], y, wt_1)[0] for r in range(R)])
        sb = np.mean([weighted_cal_slope(pr[r], y, wt_b) for r in range(R)], axis=0)
        cs[label] = dict(slope=float(s1), ci_lo=float(np.percentile(sb, 2.5)),
                         ci_hi=float(np.percentile(sb, 97.5)))
    out["calibration_slope"] = cs
    out["per_repeat_gain"] = [float((np.bincount(inv, weights=weights * (
        attempt_loss(y, pb[r]) - attempt_loss(y, pa[r]))) / q_w.clip(min=1e-12)).mean())
        for r in range(R)]
    return out


def set_single_thread() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


class MatrixBlock:
    """Fixed attempt-level columns (no data-dependent fitting), e.g. static routing summaries."""

    def __init__(self, matrix: np.ndarray, names: list[str]):
        self.matrix, self.names = np.asarray(matrix, float), list(names)

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        return self.matrix
