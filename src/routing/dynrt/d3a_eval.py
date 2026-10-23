"""P-A1 evaluation statistics: paired question bootstrap of held-out multinomial log-loss and secondaries.

Every statistic is question-weighted: pair i of question q has weight w_i = 1 / (pairs of q), so each
question counts once and a question's value is the mean over its adjacent pairs. Bootstraps resample
questions with replacement (paired: baseline and augmented predictions are resampled together).
"""
from __future__ import annotations

import numpy as np

from .d3a_softmax import log_softmax

SEED = 20260929
BIN_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0000001)


def question_index(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(groups, return_inverse=True)
    return uniq, inv


def pair_nll(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pair negative log-likelihood [R, n] from log-probabilities [R, n, K]."""
    return -np.take_along_axis(logp, y[None, :, None], axis=2)[..., 0]


def pair_brier(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pair multiclass Brier score sum_k (p_k - 1[y = k])^2, [R, n]."""
    onehot = np.eye(logp.shape[2])[y][None]
    return ((np.exp(logp) - onehot) ** 2).sum(-1)


def pair_correct(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (logp.argmax(-1) == y[None]).astype(float)


def boot_indices(n_questions: int, n_boot: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_questions, size=(n_boot, n_questions))


def paired_boot(value: np.ndarray, inv: np.ndarray, w: np.ndarray, idx: np.ndarray) -> dict:
    """Question-mean of a per-pair `value` (already averaged over repeats) with its bootstrap summary."""
    q_w = np.bincount(inv, weights=w, minlength=int(inv.max()) + 1)
    q_val = np.bincount(inv, weights=w * value, minlength=len(q_w)) / q_w.clip(min=1e-12)
    boot = q_val[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(point=float(q_val.mean()), ci_lo=float(lo), ci_hi=float(hi), boot_sd=float(boot.std(ddof=1)),
                p_one_sided=float((1 + (boot <= 0).sum()) / (len(boot) + 1)), n_questions=int(len(q_w)))


def paired_boot_ratio(value: np.ndarray, inv: np.ndarray, w: np.ndarray, mask: np.ndarray,
                      idx: np.ndarray) -> dict:
    """Weighted mean of `value` over the rows of `mask` (ratio estimator), question bootstrap."""
    nq = int(inv.max()) + 1
    num = np.bincount(inv[mask], weights=(w * value)[mask], minlength=nq)
    den = np.bincount(inv[mask], weights=w[mask], minlength=nq)
    if den.sum() <= 0:
        return dict(point=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), p_one_sided=float("nan"),
                    n_pairs=0, n_questions=0)
    boot = num[idx].sum(1) / den[idx].sum(1).clip(min=1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(point=float(num.sum() / den.sum()), ci_lo=float(lo), ci_hi=float(hi),
                p_one_sided=float((1 + (boot <= 0).sum()) / (len(boot) + 1)), n_pairs=int(mask.sum()),
                n_questions=int((den > 0).sum()))


def repeat_gains(loss_base: np.ndarray, loss_aug: np.ndarray, inv: np.ndarray, w: np.ndarray) -> list[float]:
    """Question-weighted mean gain of every repeat separately."""
    q_w = np.bincount(inv, weights=w)
    return [float((np.bincount(inv, weights=w * (b - a)) / q_w.clip(min=1e-12)).mean())
            for b, a in zip(loss_base, loss_aug)]


def temperature(logp: np.ndarray, y: np.ndarray, wt: np.ndarray, iters: int = 20) -> np.ndarray:
    """Calibration slope t of `softmax(t * logp)`: weighted ML by step-clipped Newton; wt [B, n] -> t [B]."""
    lp = np.clip(logp, -30.0, 0.0)
    ly = lp[np.arange(len(y)), y]
    t = np.ones(len(wt))
    for _ in range(iters):
        s = t[:, None, None] * lp[None]
        prob = np.exp(log_softmax(s.reshape(-1, lp.shape[1])).reshape(s.shape))
        mean_lp = (prob * lp[None]).sum(-1)
        grad = ly[None] - mean_lp
        var = (prob * lp[None] ** 2).sum(-1) - mean_lp ** 2
        step = (wt * grad).sum(1) / ((wt * var).sum(1) + 1e-12)
        t = np.clip(t + np.clip(step, -0.5, 0.5), 0.02, 20.0)
    return t


def calibration_slope(logp_r: np.ndarray, y: np.ndarray, w: np.ndarray, inv: np.ndarray, n_boot: int,
                      seed: int = SEED + 7) -> dict:
    """Mean over repeats of the temperature slope, with a bootstrap over questions (chunks of 25)."""
    nq = int(inv.max()) + 1
    idx = np.random.default_rng(seed).integers(0, nq, size=(n_boot, nq))
    mult = np.stack([np.bincount(i, minlength=nq)[inv] for i in idx]) * w[None]
    point = float(np.mean([temperature(lp, y, w[None])[0] for lp in logp_r]))
    boot = np.zeros(n_boot)
    for lo in range(0, n_boot, 25):
        sl = slice(lo, lo + 25)
        boot[sl] = np.mean([temperature(lp, y, mult[sl]) for lp in logp_r], axis=0)
    return dict(slope=point, ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                n_boot=n_boot)


def reliability(logp_r: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Question-weighted reliability of all class probabilities (pairs x classes), averaged over repeats."""
    edges = np.asarray(BIN_EDGES)
    k = logp_r.shape[2]
    onehot = np.zeros((len(y), k))
    onehot[np.arange(len(y)), y] = 1.0
    mass = np.zeros(len(edges) - 1)
    pred = np.zeros_like(mass)
    obs = np.zeros_like(mass)
    for lp in logp_r:
        p = np.exp(lp)
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(mass) - 1)
        ww = np.broadcast_to(w[:, None], p.shape)
        mass += np.bincount(b.ravel(), weights=ww.ravel(), minlength=len(mass))
        pred += np.bincount(b.ravel(), weights=(ww * p).ravel(), minlength=len(mass))
        obs += np.bincount(b.ravel(), weights=(ww * onehot).ravel(), minlength=len(mass))
    ok = mass > 0
    mean_pred = np.where(ok, pred / np.maximum(mass, 1e-12), np.nan)
    mean_obs = np.where(ok, obs / np.maximum(mass, 1e-12), np.nan)
    ece = float((mass[ok] * np.abs(mean_obs[ok] - mean_pred[ok])).sum() / mass.sum())
    return dict(edges=list(BIN_EDGES[:-1]) + [1.0], mass_share=(mass / mass.sum()).tolist(),
                mean_pred=mean_pred.tolist(), mean_obs=mean_obs.tolist(), ece=ece)


def evaluate(logp_base: np.ndarray, logp_aug: np.ndarray, y: np.ndarray, groups: np.ndarray, w: np.ndarray,
             *, n_boot: int = 1000, n_boot_cal: int = 200, per_class: np.ndarray | None = None,
             n_class: int = 7, full: bool = True) -> dict:
    """Paired comparison of two cross-fitted multinomial models (log-probs [R, n, K])."""
    _, inv = question_index(groups)
    idx = boot_indices(int(inv.max()) + 1, n_boot)
    nll_b, nll_a = pair_nll(logp_base, y), pair_nll(logp_aug, y)
    gain = (nll_b - nll_a).mean(0)
    out = dict(gain=paired_boot(gain, inv, w, idx), per_repeat_gain=repeat_gains(nll_b, nll_a, inv, w),
               logloss_base=float(np.mean([(w * b).sum() / w.sum() for b in nll_b])),
               logloss_aug=float(np.mean([(w * a).sum() / w.sum() for a in nll_a])),
               n_pairs=int(len(y)), n_questions=int(inv.max()) + 1)
    out["n_positive_repeats"] = int(sum(g > 0 for g in out["per_repeat_gain"]))
    out["pair_weighted_gain"] = _pair_weighted(gain, inv, idx)
    if per_class is not None:
        rows = {}
        for c in range(n_class):
            m = per_class == c
            rows[str(c)] = paired_boot_ratio(gain, inv, w, m, idx)
            rows[str(c)]["logloss_base"] = float((w * nll_b.mean(0))[m].sum() / w[m].sum()) if m.any() else None
            rows[str(c)]["logloss_aug"] = float((w * nll_a.mean(0))[m].sum() / w[m].sum()) if m.any() else None
        out["per_source_class"] = rows
    if not full:
        return out
    out["brier_gain"] = paired_boot((pair_brier(logp_base, y) - pair_brier(logp_aug, y)).mean(0), inv, w, idx)
    out["accuracy_gain"] = paired_boot((pair_correct(logp_aug, y) - pair_correct(logp_base, y)).mean(0), inv, w,
                                       idx)
    out["calibration"] = dict(
        base=dict(**calibration_slope(logp_base, y, w, inv, n_boot_cal), **_ece(logp_base, y, w)),
        aug=dict(**calibration_slope(logp_aug, y, w, inv, n_boot_cal), **_ece(logp_aug, y, w)))
    return out


def _ece(logp_r: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    rel = reliability(logp_r, y, w)
    return dict(ece=rel["ece"], reliability=rel)


def _pair_weighted(gain: np.ndarray, inv: np.ndarray, idx: np.ndarray) -> dict:
    """Unweighted mean over pairs (long attempts count more) with the question bootstrap (sensitivity)."""
    nq = int(inv.max()) + 1
    num = np.bincount(inv, weights=gain, minlength=nq)
    den = np.bincount(inv, minlength=nq).astype(float)
    boot = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(point=float(num.sum() / den.sum()), ci_lo=float(lo), ci_hi=float(hi),
                p_one_sided=float((1 + (boot <= 0).sum()) / (len(boot) + 1)))
