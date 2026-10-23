"""Frozen readout mathematics, with explicit model seed and family ownership."""
from __future__ import annotations
from dataclasses import dataclass
import time
import zlib
import numpy as np
import pandas as pd
from scipy import optimize, sparse
from moe_exp.io import CLASSES
from moe_exp.routing.readout_text import TextTransform

GRID = 10.0 ** np.arange(-3, 3)
ALPHA = 0.5
_STATE = {}
K = 7
N_BUCKETS = 4096
MIN_DF = 2
BAG_SCALE = 10.0
N_PCA = 64
PCA_OVERSAMPLE = 16
PCA_ITERS = 4
OUTER, INNER = 5, 3
FOLD_SEED = 20260929
N_NULL = 5
N_BOOT = 1000
BOOT_SEED = 20260929
SEED = 20260929
ABS_EDGES = (1024, 2048, 4096, 8192, 16384)
N_ABS = len(ABS_EDGES) + 1
N_REL = 10
TOKEN_CHUNK = 1 << 18
SENT_COLUMNS = ["attempt_id", "question", "sentence_index", "cls", "tok_start", "tok_end", "n_tokens", "n_reasoning",
                "abs_bin", "rel_pos"]

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


def bucket_of(tokid: np.ndarray) -> np.ndarray:
    """crc32 of the little-endian int32 token id, mod N_BUCKETS."""
    uniq, inv = np.unique(np.asarray(tokid, np.int64), return_inverse=True)
    b = np.array([zlib.crc32(int(u).to_bytes(4, "little", signed=True)) & (N_BUCKETS - 1) for u in uniq], np.int64)
    return b[inv]


def abs_bin(rank: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(ABS_EDGES), np.asarray(rank), side="right").astype(np.int64)


@dataclass
class ModelData:
    model: str
    L: int
    E: int
    sent: pd.DataFrame
    y: np.ndarray            # [S] class
    groups: np.ndarray       # [S] question
    w: np.ndarray            # [S] question weights
    base: np.ndarray         # [S, d] raw baseline dense columns
    bag: sparse.csr_matrix   # [S, N_BUCKETS] token-id counts
    top1: np.ndarray         # [L, M] uint8 labelled-token paths
    tok_sent: np.ndarray     # [M] sentence row of each token
    tokid: np.ndarray        # [M]
    tok_abs: np.ndarray      # [M] absolute-position bin of the token's reasoning rank
    adj: np.ndarray          # [M] previous token adjacent and in the same sentence
    n_before_filter: int = 0
    seed_index: int = 7
    question_family: dict | None = None

    @property
    def n(self) -> int:
        return len(self.y)


def question_weights(groups: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]




@dataclass
class PathVariant:
    """Fold-independent path quantities of one path assignment (observed or a null draw)."""
    keys: np.ndarray         # [L-1, M] int32 layer-pair keys l*E*E + a*E + b
    f2: np.ndarray           # [S] sentence mean F2 (NaN without adjacent pairs)
    hist: sparse.csr_matrix  # [S, L*E] per-layer top-1 fractions


def path_variant(md: ModelData, top1: np.ndarray) -> PathVariant:
    L, E, S = md.L, md.E, md.n
    t = top1.astype(np.int64)
    keys = ((np.arange(L - 1, dtype=np.int64) * E * E)[:, None] + t[:-1] * E + t[1:]).astype(np.int32)
    eq = np.zeros(t.shape[1])
    eq[1:] = (t[:, 1:] == t[:, :-1]).mean(0)
    a = md.adj
    num = np.bincount(md.tok_sent[a], weights=eq[a], minlength=S)
    den = np.bincount(md.tok_sent[a], minlength=S)
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = np.where(den > 0, num / den, np.nan)
    n_tok = np.bincount(md.tok_sent, minlength=S).astype(np.float64)
    rows = np.tile(md.tok_sent, L)
    cols = (np.arange(L, dtype=np.int64)[:, None] * E + t).ravel()
    vals = np.tile(1.0 / n_tok[md.tok_sent], L)
    hist = sparse.coo_matrix((vals, (rows, cols)), shape=(S, L * E)).tocsr()
    hist.sum_duplicates()
    return PathVariant(keys, f2, hist)


def null_paths(md: ModelData, draw: int) -> np.ndarray:
    """Token paths permuted among tokens of the same token id x absolute-position bin (B1 strata)."""
    n = len(md.tokid)
    stratum = md.tokid * 8 + md.tok_abs
    base = np.argsort(stratum, kind="stable")
    rng = np.random.default_rng([SEED, md.seed_index, draw, 991])
    order = np.lexsort((rng.random(n), stratum))
    perm = np.empty(n, np.int64)
    perm[base] = order
    return md.top1[:, perm]


def sentence_f1(md: ModelData, pv: PathVariant, train_rows: np.ndarray) -> np.ndarray:
    """Sentence mean path surprise under the add-alpha bigram fitted on the tokens of `train_rows`."""
    L, E = md.L, md.E
    base = (L - 1) * E * E
    in_train = np.zeros(md.n, bool)
    in_train[train_rows] = True
    tr_tok = in_train[md.tok_sent]
    m = len(md.tok_sent)
    counts = np.zeros(base)
    for a in range(0, m, TOKEN_CHUNK):
        sl = slice(a, a + TOKEN_CHUNK)
        counts += np.bincount(pv.keys[:, sl][:, tr_tok[sl]].ravel(), minlength=base)
    c = counts.reshape(L - 1, E, E)
    logp = np.log((c + ALPHA) / (c.sum(-1, keepdims=True) + ALPHA * E)).reshape(-1)
    surprise = np.empty(m)
    for a in range(0, m, TOKEN_CHUNK):
        surprise[a:a + TOKEN_CHUNK] = -logp[pv.keys[:, a:a + TOKEN_CHUNK]].mean(0)
    n_tok = np.bincount(md.tok_sent, minlength=md.n)
    return np.bincount(md.tok_sent, weights=surprise, minlength=md.n) / np.maximum(n_tok, 1)


def randomized_pca(x: sparse.csr_matrix, k: int, seed: int, n_iter: int = PCA_ITERS,
                   oversample: int = PCA_OVERSAMPLE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean [D], components [k, D], explained-variance fraction [k]) of the rows of sparse x (centred)."""
    n, dim = x.shape
    k = min(k, dim, n - 1)
    mu = np.asarray(x.mean(axis=0)).ravel()
    xt = x.T.tocsr()
    rng = np.random.default_rng(seed)

    def mv(m):
        return x @ m - np.outer(np.ones(n), mu @ m)

    def rmv(m):
        return xt @ m - np.outer(mu, m.sum(0))

    q, _ = np.linalg.qr(mv(rng.standard_normal((dim, k + oversample))))
    for _ in range(n_iter):
        z, _ = np.linalg.qr(rmv(q))
        q, _ = np.linalg.qr(mv(z))
    b = rmv(q).T
    _, s, vt = np.linalg.svd(b, full_matrices=False)
    total = float(x.multiply(x).sum() - n * (mu @ mu))
    return mu, vt[:k], (s[:k] ** 2) / total


class BagTransform:
    """In-fold tf-idf of the token-id bag (the dynamics-routing A1 TextTransform on N_BUCKETS columns)."""

    def __init__(self, counts: sparse.csr_matrix, train: np.ndarray, scale: float = BAG_SCALE):
        x = counts[train]
        df = np.bincount(x.indices, minlength=counts.shape[1]).astype(np.float64)
        self.keep = df >= MIN_DF
        self.idf = np.where(self.keep, np.log((1.0 + x.shape[0]) / (1.0 + df)) + 1.0, 0.0)
        self.cols = np.flatnonzero(self.keep)
        self.scale = scale

    def transform(self, counts: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
        x = counts[rows].tocsr(copy=True)
        x.data = (1.0 + np.log(x.data)) * self.idf[x.indices]
        x = x[:, self.cols].tocsr()
        norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
        x = sparse.diags(self.scale / np.where(norm > 0, norm, 1.0)) @ x
        return x.tocsr()


def standardise(raw: np.ndarray, tr: np.ndarray, evs: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    sc = Scaler.fit(raw[tr])
    return sc.transform(raw[tr]), [sc.transform(raw[e]) for e in evs]


def fold_design(md: ModelData, pv: PathVariant | None, tr: np.ndarray, evs: list[np.ndarray], seed: int
                ) -> tuple[np.ndarray, sparse.csr_matrix, list[np.ndarray], list[sparse.csr_matrix], dict]:
    """Dense and sparse design blocks of the training rows and every evaluation row set (fitted on tr)."""
    bt = BagTransform(md.bag, tr)
    xs_tr, xs_ev = bt.transform(md.bag, tr), [bt.transform(md.bag, e) for e in evs]
    info = dict(bag_columns=int(len(bt.cols)))
    raw = md.base
    if pv is not None:
        f1 = sentence_f1(md, pv, tr)
        mu, comp, evr = randomized_pca(pv.hist[tr], N_PCA, seed)
        rows = np.concatenate([tr] + list(evs))
        scores = np.full((md.n, comp.shape[0]), np.nan)
        scores[rows] = pv.hist[rows] @ comp.T - (mu @ comp.T)[None, :]
        raw = np.column_stack([md.base, f1, pv.f2, scores])
        info.update(pca_components=int(comp.shape[0]), pca_explained=float(evr.sum()))
    xd_tr, xd_ev = standardise(raw, tr, evs)
    if _STATE.get('mode') == 'judge_context':
        counts = _STATE['context_counts']
        transform = TextTransform.fit(counts, tr, 10.)
        xs_tr = sparse.hstack([xs_tr, transform.transform(counts, tr)], format='csr')
        xs_ev = [sparse.hstack([x, transform.transform(counts, rows)], format='csr')
                 for x, rows in zip(xs_ev, evs)]
        info['context_columns'] = transform.n_active
    return xd_tr, xs_tr, xd_ev, xs_ev, info


@dataclass
class SoftmaxFit:
    b: np.ndarray
    bd: np.ndarray
    bs: np.ndarray
    n_iter: int
    converged: bool
    grad_max: float

    def theta(self) -> np.ndarray:
        return np.r_[self.b, self.bd.ravel(), self.bs.ravel()]


def _log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def fit_softmax(xd: np.ndarray, xs: sparse.csr_matrix, y: np.ndarray, w: np.ndarray, lam: float,
                theta0: np.ndarray | None = None, maxiter: int = 400) -> SoftmaxFit:
    """Weighted multinomial ridge, objective sum w nll / sum w + lam/2 |B|^2 (dynamics-routing d3a_softmax)."""
    n, pd_ = xd.shape
    ps = xs.shape[1]
    wn = w / w.sum()
    xst = xs.T.tocsr()
    rows = np.arange(n)

    def split(theta):
        return theta[:K], theta[K:K + pd_ * K].reshape(pd_, K), theta[K + pd_ * K:].reshape(ps, K)

    def fun(theta):
        b, bd, bs = split(theta)
        lp = _log_softmax(b[None, :] + xd @ bd + xs @ bs)
        loss = -(wn * lp[rows, y]).sum() + 0.5 * lam * ((bd ** 2).sum() + (bs ** 2).sum())
        gz = np.exp(lp)
        gz[rows, y] -= 1.0
        gz *= wn[:, None]
        return loss, np.r_[gz.sum(0), (xd.T @ gz + lam * bd).ravel(), (xst @ gz + lam * bs).ravel()]

    if theta0 is None:
        prior = np.bincount(y, weights=wn, minlength=K).clip(1e-9)
        theta0 = np.r_[np.log(prior / prior.sum()), np.zeros((pd_ + ps) * K)]
    res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B",
                            options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-9, gtol=1e-6, maxcor=20))
    b, bd, bs = split(res.x)
    return SoftmaxFit(b.copy(), bd.copy(), bs.copy(), int(res.nit), bool(res.success), float(np.abs(res.jac).max()))


def predict_log_proba(fit: SoftmaxFit, xd: np.ndarray, xs: sparse.csr_matrix) -> np.ndarray:
    return _log_softmax(fit.b[None, :] + xd @ fit.bd + xs @ fit.bs)


def weighted_nll(logp: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(-(w * logp[np.arange(len(y)), y]).sum() / w.sum())


def outer_folds(groups: np.ndarray, question_family=None) -> np.ndarray:
    from moe_exp.routing.readout_analysis import family_folds
    families = question_family if question_family is not None else _STATE['md'].question_family
    return family_folds(groups, OUTER, FOLD_SEED, families)


def run_task(args: tuple) -> dict:
    """One (variant, outer fold): inner-CV penalty choice, then the held-out class log-probabilities."""
    variant, f = args
    md: ModelData = _STATE["md"]
    pv = _STATE["variants"].get(variant)
    t0 = time.time()
    folds = outer_folds(md.groups)
    train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
    from moe_exp.routing.readout_analysis import family_folds
    inner = family_folds(md.groups[train], INNER, FOLD_SEED * 1000 + f, md.question_family)
    grid = GRID.copy()
    losses = np.zeros((INNER, len(grid)))
    sizes = np.zeros(INNER)
    stats = []
    vseed = 0 if variant in ("base", "aug") else 1 + int(variant.split("_")[1])
    for j in range(INNER):
        tr, va = train[inner != j], train[inner == j]
        xd_tr, xs_tr, (xd_va,), (xs_va,), _ = fold_design(md, pv, tr, [va], seed=SEED + 100 * f + 10 * j + vseed)
        theta = None
        for gi in range(len(grid) - 1, -1, -1):                     # descending penalty, warm-started
            fit = fit_softmax(xd_tr, xs_tr, md.y[tr], md.w[tr], float(grid[gi]), theta0=theta)
            theta = fit.theta()
            stats.append((fit.n_iter, fit.converged, fit.grad_max))
            losses[j, gi] = weighted_nll(predict_log_proba(fit, xd_va, xs_va), md.y[va], md.w[va])
        sizes[j] = md.w[va].sum()
    mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
    best = int(np.flatnonzero(mean <= mean.min() + 1e-12)[-1])
    lam = float(grid[best])
    xd_tr, xs_tr, (xd_te,), (xs_te,), info = fold_design(md, pv, train, [test], seed=SEED + 100 * f + 9 + vseed)
    fit = fit_softmax(xd_tr, xs_tr, md.y[train], md.w[train], lam)
    stats.append((fit.n_iter, fit.converged, fit.grad_max))
    logp = predict_log_proba(fit, xd_te, xs_te).astype(np.float32)
    return dict(variant=variant, fold=f, test=test, logp=logp, lam=lam, inner_curve=mean, stats=stats, info=info,
                seconds=time.time() - t0)


def question_bootstrap(gain: np.ndarray, groups: np.ndarray, w: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict:
    """Weighted mean sum(w gain) / sum(w) of per-sentence gains with a paired question bootstrap.

    w = question weights gives the mean of question means; w = 1 gives the sentence-pooled mean."""
    uniq, inv = np.unique(groups, return_inverse=True)
    num, den = np.bincount(inv, weights=w * gain), np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(gain=float(num.sum() / den.sum()), ci_lo=float(lo), ci_hi=float(hi),
                p_one_sided=float((1 + (boot <= 0).sum()) / (n_boot + 1)), boot_sd=float(boot.std(ddof=1)),
                n_questions=int(len(uniq)), n_boot=n_boot)



def nll_rows(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -logp[np.arange(len(y)), y].astype(np.float64)
