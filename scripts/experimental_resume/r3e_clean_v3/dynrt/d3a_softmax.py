"""Weighted multinomial ridge (softmax) with dense + sparse blocks, on the C8 mean-loss penalty scale.

Objective:  sum_i w_i nll_i / sum_i w_i + (lam / 2) (|B_dense|^2 + |B_sparse|^2); intercepts unpenalised.
(The C8 logistic penalty `lam * sum(w)` on the summed loss is the same scale.) L-BFGS-B, warm-startable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse


@dataclass
class SoftmaxFit:
    b: np.ndarray            # [K]
    Bd: np.ndarray           # [pd, K]
    Bs: np.ndarray           # [ps, K]
    n_iter: int
    converged: bool
    objective: float
    grad_max: float

    def theta(self) -> np.ndarray:
        return np.r_[self.b, self.Bd.ravel(), self.Bs.ravel()]


def _split(theta: np.ndarray, k: int, pd_: int, ps: int):
    b = theta[:k]
    bd = theta[k:k + pd_ * k].reshape(pd_, k)
    bs = theta[k + pd_ * k:].reshape(ps, k)
    return b, bd, bs


def _logits(b, bd, bs, xd, xs):
    z = np.broadcast_to(b, (xd.shape[0], len(b))).copy()
    if xd.shape[1]:
        z += xd @ bd
    if xs is not None and xs.shape[1]:
        z += xs @ bs
    return z


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def fit_softmax(xd: np.ndarray, xs: sparse.csr_matrix | None, y: np.ndarray, w: np.ndarray, lam: float,
                k: int, *, theta0: np.ndarray | None = None, maxiter: int = 400, ftol: float = 1e-9,
                gtol: float = 1e-6) -> SoftmaxFit:
    """Fit on rows (xd [n, pd] float64, xs [n, ps] CSR float64 or None, y int labels, w weights)."""
    n, pd_ = xd.shape
    ps = 0 if xs is None else xs.shape[1]
    sw = float(w.sum())
    wn = w / sw
    xst = None if ps == 0 else xs.T.tocsr()
    rows = np.arange(n)

    def fun(theta):
        b, bd, bs = _split(theta, k, pd_, ps)
        z = _logits(b, bd, bs, xd, xs if ps else None)
        lp = log_softmax(z)
        loss = -(wn * lp[rows, y]).sum() + 0.5 * lam * ((bd ** 2).sum() + (bs ** 2).sum())
        gz = np.exp(lp)
        gz[rows, y] -= 1.0
        gz *= wn[:, None]
        g = np.empty_like(theta)
        g[:k] = gz.sum(0)
        gd = (xd.T @ gz if pd_ else np.zeros((0, k))) + lam * bd
        g[k:k + pd_ * k] = gd.ravel()
        if ps:
            g[k + pd_ * k:] = (xst @ gz + lam * bs).ravel()
        return loss, g

    if theta0 is None:
        prior = np.bincount(y, weights=wn, minlength=k).clip(1e-9)
        theta0 = np.r_[np.log(prior / prior.sum()), np.zeros((pd_ + ps) * k)]
    res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B",
                            options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=ftol, gtol=gtol, maxcor=20))
    b, bd, bs = _split(res.x, k, pd_, ps)
    return SoftmaxFit(b.copy(), bd.copy(), bs.copy(), int(res.nit), bool(res.success), float(res.fun),
                      float(np.abs(res.jac).max()))


def predict_log_proba(fit: SoftmaxFit, xd: np.ndarray, xs: sparse.csr_matrix | None) -> np.ndarray:
    """Held-out class log-probabilities [n, K]."""
    return log_softmax(_logits(fit.b, fit.Bd, fit.Bs, xd, xs if fit.Bs.shape[0] else None))


def weighted_nll(logp: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(-(w * logp[np.arange(len(y)), y]).sum() / w.sum())
