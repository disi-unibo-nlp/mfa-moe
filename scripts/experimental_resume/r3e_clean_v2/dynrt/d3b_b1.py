"""D3b B1: class-dependent cross-layer co-selection (ANALYSIS_PLAN.md, Addendum 3, P-B1). No outcome is used.

For layer pairs (i, j) at depth gaps 1 and 4 and every labelled-sentence token, the k x k same-token
co-selections (weight 1/k^2) are counted in n[c, a, b] (class c, expert a in layer i, expert b in layer j).
Per training fold (question-grouped):
  null       no-3-way-interaction log-linear fit of n (margins [c,a], [c,b], [a,b]) by IPF -> mu, q_null = mu/N_c;
  alternative q_alt = (n + lambda mu) / (N_c (1 + lambda)), lambda = 1 (class-specific pair distribution
             shrunk toward the null).
Held-out score of a question = mean over its token-pair observations of log q_alt - log q_null (composite log
score; both models floored at EPS_Q so that cells unseen by both give zero gain). G_joint = question-weighted
mean over questions, averaged over the layer pairs and the fold repeats. Diagnostics: within-layer whole-top-k-set
permutation null (within class x absolute-position bin), a lexical sensitivity (the 200 most frequent token ids
dropped) and depth-gap-stratified results.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from .common import abs_bin
from .cv import question_folds

NC = 7
LAMBDA = 1.0
EPS_Q = 1e-12
GAPS = (1, 4)
THRESHOLD = 0.001
N_BOOT = 1000
N_PERM = 20
TOP_LEX = 200
IPF_TOL = 1e-6
IPF_MAX = 400
BOOT_SEED = 20260929


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def layer_pairs(n_layers: int, gaps=GAPS) -> list[tuple[int, int, int]]:
    """(i, j, gap) for every layer pair at the given depth gaps (GPT 24 layers: 43; Qwen 40 layers: 75)."""
    return [(i, i + g, g) for g in gaps for i in range(n_layers - g)]


# ------------------------------------------------------------------------------ token set

@dataclass
class TokenSet:
    """Labelled tokens of the analysed questions; `q` is the dense question index."""
    ids: np.ndarray      # [N, L, k] uint8
    cls: np.ndarray      # [N] int64
    q: np.ndarray        # [N] int64
    pos: np.ndarray      # [N] int64 absolute-position bin of the token's reasoning rank
    tokid: np.ndarray    # [N] int32 vocabulary id of the token
    n_q: int
    experts: int

    @property
    def k(self) -> int:
        return int(self.ids.shape[2])

    @property
    def n_layers(self) -> int:
        return int(self.ids.shape[1])

    def select(self, mask: np.ndarray) -> TokenSet:
        return TokenSet(self.ids[mask], self.cls[mask], self.q[mask], self.pos[mask], self.tokid[mask],
                        self.n_q, self.experts)


def make_tokenset(tk: dict, attempt_keep: np.ndarray, experts: int) -> tuple[TokenSet, list[str]]:
    """Token set of the attempts flagged in `attempt_keep` (bool over the attempt table rows)."""
    att = tk["attempts"]
    keep = np.asarray(attempt_keep, bool)
    questions = sorted(set(att["question"].to_numpy()[keep]))
    q_of_att = np.full(len(att), -1, np.int64)
    idx = {q: i for i, q in enumerate(questions)}
    for a in np.flatnonzero(keep):
        q_of_att[a] = idx[att["question"].iloc[a]]
    row_keep = keep[tk["att"]]
    ts = TokenSet(tk["ids"][row_keep], tk["cls"][row_keep], q_of_att[tk["att"][row_keep]],
                  abs_bin(tk["rank"][row_keep]).astype(np.int64), tk["tokid"][row_keep], len(questions),
                  experts)
    return ts, questions


# ---------------------------------------------------------------------------- statistics

def pair_tables(ids_i: np.ndarray, ids_j: np.ndarray, cls: np.ndarray, q: np.ndarray, n_q: int,
                experts: int) -> np.ndarray:
    """Co-selection counts [n_q, C, E*E] float32 (divide by k^2 for token weights)."""
    n, k = ids_i.shape
    e2 = experts * experts
    key = ids_i.astype(np.int64)[:, :, None] * experts + ids_j.astype(np.int64)[:, None, :]
    key += ((q * NC + cls) * e2)[:, None, None]
    counts = np.bincount(key.ravel(), minlength=n_q * NC * e2)
    return counts.reshape(n_q, NC, e2).astype(np.float32)


def ipf_null(n: np.ndarray, tol: float = IPF_TOL, max_iter: int = IPF_MAX) -> tuple[np.ndarray, int, bool]:
    """No-three-way-interaction fit of n [C, E, E] by IPF started from the class-conditional independence table.

    Returns (mu, iterations, converged); mu reproduces the [c,a], [c,b] and [a,b] margins of n.
    """
    m_ca, m_cb, m_ab = n.sum(2), n.sum(1), n.sum(0)
    n_c = m_ca.sum(1)
    mu = m_ca[:, :, None] * m_cb[:, None, :] / np.maximum(n_c, 1e-300)[:, None, None]
    positive = np.concatenate([m_ca[m_ca > 0], m_cb[m_cb > 0], m_ab[m_ab > 0]])
    floor = tol * (positive.min() if len(positive) else 1.0)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        s = mu.sum(2)
        mu *= np.divide(m_ca, s, out=np.zeros_like(s), where=s > 0)[:, :, None]
        s = mu.sum(1)
        mu *= np.divide(m_cb, s, out=np.zeros_like(s), where=s > 0)[:, None, :]
        s = mu.sum(0)
        mu *= np.divide(m_ab, s, out=np.zeros_like(s), where=s > 0)[None]
        if it % 3 == 0 or it == max_iter:
            dev = max(np.abs(mu.sum(2) - m_ca).max(), np.abs(mu.sum(1) - m_cb).max())
            if dev <= floor:
                converged = True
                break
    return mu, it, converged


def fold_delta(n_train: np.ndarray, lam: float = LAMBDA, eps: float = EPS_Q
               ) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """log q_alt - log q_null per (class, cell), the classes with training tokens, IPF iterations, converged."""
    n_c = n_train.sum((1, 2))
    ok = n_c > 0
    mu, it, conv = ipf_null(n_train)
    den = np.where(ok, n_c, 1.0)[:, None, None]
    q_null = mu / den
    q_alt = (n_train + lam * mu) / (den * (1.0 + lam))
    delta = np.log(np.maximum(q_alt, eps)) - np.log(np.maximum(q_null, eps))
    delta[~ok] = 0.0
    return delta.reshape(NC, -1), ok, it, conv


def score_questions(t_test: np.ndarray, delta: np.ndarray, ok: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-question mean gain per token-pair observation and per-class numerators/denominators.

    t_test [nq, C, X] counts. Returns (gain [nq], class_gain [nq, C]); classes without training tokens are
    dropped from both numerator and denominator.
    """
    t = t_test.astype(np.float64)
    num_c = np.einsum("qcx,cx->qc", t, delta)
    den_c = t.sum(2)
    den = (den_c * ok[None]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = (num_c * ok[None]).sum(1) / den
        class_gain = np.where((den_c > 0) & ok[None], num_c / den_c, np.nan)
    return np.where(den > 0, gain, np.nan), class_gain


_STATE: dict = {}


def run_pair(index: int) -> dict:
    """One layer pair: held-out per-question gains for every fold repeat."""
    st = _STATE
    i, j, gap = st["pairs"][index]
    ts: TokenSet = st["ts"]
    e = ts.experts
    t = pair_tables(ts.ids[:, i, :], ts.ids[:, j, :], ts.cls, ts.q, ts.n_q, e)
    total = t.sum(0, dtype=np.float64)
    k2 = float(ts.k * ts.k)
    folds = st["folds"]
    n_rep = folds.shape[0]
    gains = np.full((n_rep, ts.n_q), np.nan)
    class_gain = np.full((n_rep, ts.n_q, NC), np.nan)
    iters, unconverged = [], 0
    for r in range(n_rep):
        for f in range(st["n_folds"]):
            te = folds[r] == f
            if not te.any():
                continue
            fold_sum = t[te].sum(0, dtype=np.float64)
            n_train = ((total - fold_sum) / k2).reshape(NC, e, e)
            delta, ok, it, conv = fold_delta(n_train)
            iters.append(it)
            unconverged += int(not conv)
            g, cg = score_questions(t[te], delta, ok)
            gains[r, te] = g
            class_gain[r, te] = cg
    return dict(index=index, gains=gains, class_gain=class_gain, iters=iters, unconverged=unconverged)


def run_pairs(ts: TokenSet, folds: np.ndarray, n_folds: int, pairs: list, workers: int) -> dict:
    """Held-out gains [pair, repeat, question] (+ per class) for all layer pairs."""
    _STATE.update(ts=ts, folds=folds, n_folds=n_folds, pairs=pairs)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = list(pool.imap_unordered(run_pair, range(len(pairs)), chunksize=1))
    else:
        res = [run_pair(p) for p in range(len(pairs))]
    res.sort(key=lambda r: r["index"])
    return dict(gains=np.stack([r["gains"] for r in res]), class_gain=np.stack([r["class_gain"] for r in res]),
                iters=np.concatenate([r["iters"] for r in res]),
                unconverged=int(sum(r["unconverged"] for r in res)))


# ---------------------------------------------------------------------------- summaries

def question_gain(gains: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """[pair, repeat, question] -> per-question gain averaged over (selected) pairs and repeats."""
    g = gains if sel is None else gains[sel]
    return np.nanmean(np.nanmean(g, axis=0), axis=0)


def per_repeat(gains: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """Question-weighted G_joint of every fold repeat."""
    g = gains if sel is None else gains[sel]
    return np.nanmean(np.nanmean(g, axis=0), axis=1)


def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Question bootstrap of the mean of per-question values (paired: values are within-question gains)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = v[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_one = float((1 + (boot <= 0).sum()) / (n_boot + 1))
    return dict(mean=float(v.mean()), ci_lo=float(lo), ci_hi=float(hi), p_one_sided=p_one,
                boot_sd=float(boot.std(ddof=1)), n_questions=int(len(v)), n_boot=int(n_boot))


def summarize(res: dict, pairs: list, n_boot: int = N_BOOT) -> dict:
    """G_joint, repeats, gap strata and per-class means of one set of held-out gains."""
    g = res["gains"]
    gap = np.array([p[2] for p in pairs])
    qg = question_gain(g)
    out = dict(G_joint=boot_ci(qg, n_boot), per_repeat_G=per_repeat(g).tolist(),
               positive_repeats=int((per_repeat(g) > 0).sum()), n_repeats=int(g.shape[1]))
    out["gap"] = {str(gp): boot_ci(question_gain(g, gap == gp), n_boot) for gp in sorted(set(gap.tolist()))}
    cg = res["class_gain"]
    from .common import CLASSES
    out["per_class"] = {c: float(np.nanmean(np.nanmean(np.nanmean(cg[:, :, :, k], axis=0), axis=0)))
                        if np.isfinite(cg[:, :, :, k]).any() else None for k, c in enumerate(CLASSES)}
    out["per_pair"] = [dict(i=p[0], j=p[1], gap=p[2], G=float(np.nanmean(g[n]))) for n, p in enumerate(pairs)]
    out["ipf"] = dict(fits=int(len(res["iters"])), median_iterations=float(np.median(res["iters"])),
                      max_iterations=int(np.max(res["iters"])), unconverged=int(res["unconverged"]))
    return out


# ---------------------------------------------------------------------------- diagnostics

def permute_layers(ids: np.ndarray, stratum: np.ndarray, seed: int, perm_id: int) -> np.ndarray:
    """Independently permute whole top-k sets among the tokens of the same stratum, layer by layer.

    Per-layer (stratum) marginals and within-layer co-selection are kept; same-token cross-layer dependence
    is broken.
    """
    n, n_layers, _ = ids.shape
    base = np.argsort(stratum, kind="stable")
    out = np.empty_like(ids)
    s = stratum.astype(np.float64)
    for layer in range(n_layers):
        rng = np.random.default_rng([seed, perm_id, layer])
        drawn = np.argsort(s + rng.random(n), kind="stable")
        perm = np.empty(n, np.int64)
        perm[base] = drawn
        out[:, layer, :] = ids[perm, layer, :]
    return out


def frequent_token_mask(tokid: np.ndarray, top: int = TOP_LEX) -> tuple[np.ndarray, np.ndarray]:
    """Mask of tokens whose vocabulary id is one of the `top` most frequent ids (ties -> smaller id)."""
    ids, counts = np.unique(tokid, return_counts=True)
    order = np.lexsort((ids, -counts))[:top]
    frequent = ids[order]
    return np.isin(tokid, frequent), frequent


def permute_classes(cls: np.ndarray, tokid: np.ndarray, pos: np.ndarray, seed: int, perm_id: int) -> np.ndarray:
    """Shuffle the class labels among tokens of the same vocabulary id and position bin.

    Additional (not registered) lexical diagnostic: it keeps token identity -> routing intact and removes any
    class information that is not carried by the token itself.
    """
    n = len(cls)
    stratum = tokid.astype(np.int64) * 8 + pos
    base = np.argsort(stratum, kind="stable")
    rng = np.random.default_rng([seed, perm_id, 991])
    within = rng.random(n)
    order = np.lexsort((within, stratum))
    perm = np.empty(n, np.int64)
    perm[base] = order
    return cls[perm]


def make_folds(questions: list[str], n_folds: int, repeats: int) -> np.ndarray:
    """Question fold ids [repeat, question]: C8 question_folds, seeds 0..repeats-1."""
    g = np.asarray(questions)
    return np.stack([question_folds(g, n_folds, seed) for seed in range(repeats)])


def run_b1(ts: TokenSet, questions: list[str], pairs: list, *, n_folds: int, repeats: int, n_perm: int,
           workers: int, lexical: bool = True, n_boot: int = N_BOOT, seed: int = 0, tag: str = "",
           n_classperm: int = 0) -> dict:
    """Observed G_joint plus permutation and lexical diagnostics for one token set."""
    t0 = time.time()
    folds = make_folds(questions, n_folds, repeats)
    obs = run_pairs(ts, folds, n_folds, pairs, workers)
    out = dict(observed=summarize(obs, pairs, n_boot), n_questions=int(ts.n_q), n_tokens=int(len(ts.cls)),
               n_pairs=len(pairs), n_folds=n_folds, repeats=repeats,
               question_gain=question_gain(obs["gains"]).tolist(), questions=list(questions))
    log(f"{tag} observed G_joint {out['observed']['G_joint']['mean']:.6f} "
        f"[{out['observed']['G_joint']['ci_lo']:.6f}, {out['observed']['G_joint']['ci_hi']:.6f}] "
        f"repeats {[round(x, 6) for x in out['observed']['per_repeat_G']]} ({time.time() - t0:.0f}s)")
    if lexical:
        drop, frequent = frequent_token_mask(ts.tokid)
        lex = run_pairs(ts.select(~drop), folds, n_folds, pairs, workers)
        out["lexical"] = dict(summarize(lex, pairs, n_boot), tokens_dropped=int(drop.sum()),
                              fraction_dropped=float(drop.mean()), n_frequent_ids=int(len(frequent)))
        log(f"{tag} lexical G_joint {out['lexical']['G_joint']['mean']:.6f} "
            f"(dropped {out['lexical']['fraction_dropped']:.3f}) ({time.time() - t0:.0f}s)")
    if n_perm:
        stratum = ts.cls * 8 + ts.pos
        gains = []
        for p in range(n_perm):
            perm_ts = TokenSet(permute_layers(ts.ids, stratum, seed, p), ts.cls, ts.q, ts.pos, ts.tokid,
                               ts.n_q, ts.experts)
            r = run_pairs(perm_ts, folds[:1], n_folds, pairs, workers)
            gains.append(float(np.nanmean(question_gain(r["gains"]))))
            log(f"{tag} permutation {p + 1}/{n_perm}: G {gains[-1]:.6f} ({time.time() - t0:.0f}s)")
        g = out["observed"]["G_joint"]["mean"]
        p95 = float(np.percentile(gains, 95))
        out["permutation"] = dict(gains=gains, mean=float(np.mean(gains)), sd=float(np.std(gains, ddof=1)),
                                  p95=p95, n_perm=n_perm, observed_above_p95=bool(g > p95),
                                  rank_p=float((1 + sum(x >= g for x in gains)) / (n_perm + 1)),
                                  strata="class x absolute-position bin (whole top-k sets, per layer)")
    if n_classperm:
        gains = []
        for p in range(n_classperm):
            cp = TokenSet(ts.ids, permute_classes(ts.cls, ts.tokid, ts.pos, seed, p), ts.q, ts.pos, ts.tokid,
                          ts.n_q, ts.experts)
            r = run_pairs(cp, folds[:1], n_folds, pairs, workers)
            gains.append(float(np.nanmean(question_gain(r["gains"]))))
        obs0 = float(np.nanmean(question_gain(obs["gains"][:, :1])))
        out["class_permutation_within_token"] = dict(
            gains=gains, mean=float(np.mean(gains)), sd=float(np.std(gains, ddof=1)) if len(gains) > 1 else None,
            p95=float(np.percentile(gains, 95)), observed_repeat0=obs0,
            retained_fraction=float(np.mean(gains) / obs0) if obs0 else None, n_perm=n_classperm,
            note="extra diagnostic, not registered: class labels shuffled among tokens of the same id x position bin")
        log(f"{tag} class permutation within token id: mean G {np.mean(gains):.6f} vs observed {obs0:.6f}")
    out["seconds"] = time.time() - t0
    return out
