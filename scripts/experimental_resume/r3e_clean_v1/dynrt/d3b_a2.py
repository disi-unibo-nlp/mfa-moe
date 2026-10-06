"""D3b A2: transition-triggered routing time courses on the dense Qwen3.6 pilot (exploratory; no outcome used).

For every observed a -> b class boundary (a != b; adjacent labelled sentences of one reasoning segment) the
per-layer destination-minus-source affinity  aff_l(w) = d_a^l(h_w) - d_b^l(h_w)  (d_c^l = sum_e h log(p0/p_c),
in-fold class profiles, lower d = closer to class c, so a positive aff is closer to the destination) is read on
16-token windows ENDING at lags -64..+64 (stride 16, window = [r + lag - 16, r + lag) around the boundary token
rank r). The excess is event minus the mean of matched a -> a boundaries (same source class, absolute-position
bin and source-sentence-length tercile), each control scored with the event's destination class. A window is
censored (never padded or shortened) unless it lies inside [start of the source run, end of the destination run]
(runs = consecutive same-class labelled sentences), i.e. it contains no intervening class boundary other than
the focal one. Profiles for an event's question come from the other folds only (question-grouped).
Scalar = pooled excess over the pre-boundary lags -32, -16, 0, averaged over layers, question-weighted, with a
question-bootstrap CI. Post-boundary adaptation (lags +16..+64) is reported separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features as F
from .common import abs_bin
from .cv import question_folds

WIN = 16
LAGS = tuple(range(-64, 65, WIN))
PRE_LAGS = (-32, -16, 0)
POST_LAGS = (16, 32, 48, 64)
MIN_CTRL = 3
N_BOOT = 1000
BOOT_SEED = 20260930
NC = F.NC
N_BIN = 6
N_TERC = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------ sentence table

def sentence_histograms(ids: np.ndarray, sent: np.ndarray, n_sent: int, experts: int, chunk: int = 512
                        ) -> np.ndarray:
    """Per-sentence, per-layer expert selection frequencies [n_sent, L*E] float32 (rows sum to 1 per layer)."""
    n, n_layers, k = ids.shape
    out = np.zeros((n_sent, n_layers * experts), np.float32)
    layer = (np.arange(n_layers, dtype=np.int64) * experts)[None, :, None]
    bounds = np.searchsorted(sent, np.arange(0, n_sent + chunk, chunk))
    for a in range(0, n_sent, chunk):
        b = min(a + chunk, n_sent)
        lo, hi = bounds[a // chunk], bounds[min(a // chunk + 1, len(bounds) - 1)]
        if hi <= lo:
            continue
        sid = sent[lo:hi] - a
        sel = ids[lo:hi].astype(np.int64)
        key = (sid[:, None, None] * (n_layers * experts) + layer + sel).ravel()
        cnt = np.bincount(key, minlength=(b - a) * n_layers * experts).reshape(b - a, n_layers, experts)
        lens = np.bincount(sid, minlength=b - a).astype(np.float64)
        out[a:b] = (cnt / (np.maximum(lens, 1)[:, None, None] * k)).reshape(b - a, -1)
    return out


@dataclass
class Pilot:
    """Dense token/sentence tables of the pilot. Token rows are (attempt, rank) ordered and contiguous."""
    ids: np.ndarray
    att: np.ndarray
    rank: np.ndarray
    sent: np.ndarray
    sentences: pd.DataFrame
    attempts: pd.DataFrame
    experts: int
    tok_offset: np.ndarray       # first token row of every attempt

    @property
    def n_layers(self) -> int:
        return int(self.ids.shape[1])


def make_pilot(tk: dict, experts: int) -> Pilot:
    att = tk["att"]
    n_att = len(tk["attempts"])
    counts = np.bincount(att, minlength=n_att)
    offset = np.r_[0, np.cumsum(counts)[:-1]]
    if not (np.diff(att) >= 0).all():
        raise ValueError("token rows are not ordered by attempt")
    expect = tk["attempts"]["n_reasoning"].to_numpy()
    if not (counts == expect).all():
        raise ValueError("A2 needs every reasoning token of every attempt (dense labels)")
    if not (tk["rank"] == np.arange(len(att)) - offset[att]).all():
        raise ValueError("reasoning ranks are not contiguous")
    return Pilot(tk["ids"], att, tk["rank"], tk["sent"], tk["sentences"].reset_index(drop=True),
                 tk["attempts"], experts, offset)


# ---------------------------------------------------------------------------- boundaries

@dataclass
class Boundaries:
    """Adjacent labelled sentence pairs (m-1, m): events (a != b) and controls (a == b)."""
    row: np.ndarray         # sentence row m (destination)
    att: np.ndarray
    a: np.ndarray
    b: np.ndarray
    r: np.ndarray           # boundary token rank (first token of the destination sentence)
    r0: np.ndarray          # start rank of the source run
    r1: np.ndarray          # end rank (exclusive) of the destination run
    abs_bin: np.ndarray
    src_len: np.ndarray


def build_boundaries(sent: pd.DataFrame, labelled: np.ndarray) -> Boundaries:
    """Pairs of adjacent labelled sentences and the class-run extents (unlabelled sentences break runs)."""
    a_att = sent["att"].to_numpy()
    idx = sent["sentence_index"].to_numpy()
    seg = sent["segment"].to_numpy()
    cls = sent["cls"].to_numpy().astype(np.int64)
    ts, te = sent["tok_start"].to_numpy().astype(np.int64), sent["tok_end"].to_numpy().astype(np.int64)
    n = len(sent)
    lab = np.asarray(labelled, bool)
    adj = np.zeros(n, bool)
    adj[:-1] = (a_att[1:] == a_att[:-1]) & (idx[1:] == idx[:-1] + 1) & (seg[1:] == seg[:-1]) & lab[1:] & lab[:-1]
    same_prev = np.zeros(n, bool)
    same_prev[1:] = adj[:-1] & (cls[1:] == cls[:-1])
    start_flag = lab & ~same_prev
    run_id = np.cumsum(start_flag) - 1
    n_runs = int(start_flag.sum())
    run_start = np.full(n_runs, np.iinfo(np.int64).max, np.int64)
    run_end = np.zeros(n_runs, np.int64)
    np.minimum.at(run_start, run_id[lab], ts[lab])
    np.maximum.at(run_end, run_id[lab], te[lab])
    m = np.flatnonzero(adj) + 1
    return Boundaries(m, a_att[m], cls[m - 1], cls[m], ts[m], run_start[run_id[m - 1]], run_end[run_id[m]],
                      abs_bin(ts[m]).astype(np.int64), (te[m - 1] - ts[m - 1]))


def length_edges(sent: pd.DataFrame) -> np.ndarray:
    """Global sentence-length tercile cut points (tokens), fixed from the dense pilot."""
    return np.quantile(sent["n_tokens"].to_numpy().astype(float), [1 / 3, 2 / 3])


def match_cell(bd: Boundaries, edges: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
    """Matching cell index: source class x absolute-position bin x source-sentence-length tercile."""
    terc = np.searchsorted(edges, bd.src_len.astype(float), side="left")
    src = bd.a if source is None else source
    return (src * N_BIN + bd.abs_bin) * N_TERC + terc


def window_valid(bd: Boundaries) -> np.ndarray:
    """[n, n_lags] bool: 16-token window ending at r + lag lies inside [r0, r1)."""
    lag = np.asarray(LAGS)[None, :]
    end = bd.r[:, None] + lag
    return (end - WIN >= bd.r0[:, None]) & (end <= bd.r1[:, None])


# ------------------------------------------------------------------------------ scoring

def token_scores(ids: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """S[t, c, l] = mean over the k selected experts of lr[c, l, e]; ids [n, L, k], lr [C, L, E] -> float32."""
    n, n_layers, k = ids.shape
    out = np.empty((n, NC, n_layers), np.float32)
    for layer in range(n_layers):
        g = lr[:, layer, :][:, ids[:, layer, :]]
        out[:, :, layer] = g.mean(-1).T
    return out


def group_sum(values: np.ndarray, keys: np.ndarray, n_keys: int) -> np.ndarray:
    """Sum rows of `values` [n, ...] by integer key -> [n_keys, ...] float64."""
    out = np.zeros((n_keys,) + values.shape[1:], np.float64)
    if not len(keys):
        return out
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    out[ks[starts]] = np.add.reduceat(values[order], starts, axis=0, dtype=np.float64)
    return out


def fold_profiles(h: np.ndarray, sent_cls: np.ndarray, sent_att: np.ndarray, train_rows: np.ndarray,
                  n_layers: int, experts: int) -> np.ndarray:
    """log(p0 / p_c) [C, L, E] from the labelled training sentences (D1 profiles, kappa = 50)."""
    prof = F.fit_profiles(h[train_rows], sent_cls[train_rows], sent_att[train_rows], n_layers, experts)
    return prof.log_ratio().reshape(NC, n_layers, experts)


@dataclass
class A2Result:
    n_q: int
    events: dict            # per-question sums/counts, see `a2_run`
    meta: dict


def a2_run(p: Pilot, h: np.ndarray, attempt_keep: np.ndarray, label_mask: np.ndarray | None, *,
           n_folds: int = 8, seed: int = 0, edges: np.ndarray | None = None, folds_q: np.ndarray | None = None
           ) -> dict:
    """A2 estimates on the kept attempts; `label_mask` (bool over sentence rows) None = all sentences labelled.

    Returns per-question sums/counts of the excess (per lag x layer) plus the pre/post pooled scalars.
    """
    t0 = time.time()
    sent = p.sentences
    keep_att = np.asarray(attempt_keep, bool)
    a_of_s = sent["att"].to_numpy()
    lab = np.ones(len(sent), bool) if label_mask is None else np.asarray(label_mask, bool)
    lab = lab & keep_att[a_of_s]
    edges = length_edges(sent) if edges is None else edges
    n_layers, experts = p.n_layers, p.experts
    q_of_att = p.attempts["q"].to_numpy()
    kept_q = np.unique(q_of_att[keep_att])
    q_dense = np.full(int(q_of_att.max()) + 1, -1, np.int64)
    q_dense[kept_q] = np.arange(len(kept_q))
    n_q = len(kept_q)
    if folds_q is None:
        folds_q = question_folds(np.asarray(kept_q), n_folds, seed)
    fold_of_att = np.full(len(q_of_att), -1, np.int64)
    fold_of_att[keep_att] = folds_q[q_dense[q_of_att[keep_att]]]

    bd = build_boundaries(sent, lab)
    evt = bd.a != bd.b
    valid = window_valid(bd)
    cell = match_cell(bd, edges)
    n_cells = NC * N_BIN * N_TERC
    n_lag = len(LAGS)
    s_cls = sent["cls"].to_numpy().astype(np.int64)
    ev_aff = np.full((int(evt.sum()), n_lag, n_layers), np.nan, np.float32)
    ev_pos = -np.ones(len(bd.row), np.int64)
    ev_pos[evt] = np.arange(int(evt.sum()))
    ctrl_sum = np.zeros((n_cells, n_lag, NC, n_layers))
    ctrl_n = np.zeros((n_cells, n_lag))
    fold_att = fold_of_att[bd.att]
    for f in range(n_folds):
        att_f = np.flatnonzero(fold_of_att == f)
        if not len(att_f):
            continue
        train_att = keep_att & (fold_of_att != f)
        train_rows = np.flatnonzero(lab & train_att[a_of_s])
        lr = fold_profiles(h, s_cls, a_of_s, train_rows, n_layers, experts)
        tok_rows = np.concatenate([np.arange(p.tok_offset[a], p.tok_offset[a] + p.attempts["n_reasoning"].iloc[a])
                                   for a in att_f])
        local_off = np.zeros(len(p.tok_offset), np.int64)
        lens = p.attempts["n_reasoning"].to_numpy()[att_f]
        local_off[att_f] = np.r_[0, np.cumsum(lens)[:-1]]
        s = token_scores(p.ids[tok_rows], lr)
        cs = np.zeros((len(tok_rows) + 1, NC, n_layers))
        np.cumsum(s, axis=0, dtype=np.float64, out=cs[1:])
        del s
        sel = np.flatnonzero(fold_att == f)
        for li, lag in enumerate(LAGS):
            ok = sel[valid[sel, li]]
            if not len(ok):
                continue
            end = local_off[bd.att[ok]] + bd.r[ok] + lag
            w = (cs[end] - cs[end - WIN]) / WIN                       # [n, C, L]
            is_evt = evt[ok]
            e_ok = ok[is_evt]
            if len(e_ok):
                we = w[is_evt]
                idx = np.arange(len(e_ok))
                ev_aff[ev_pos[e_ok], li] = (we[idx, bd.a[e_ok]] - we[idx, bd.b[e_ok]]).astype(np.float32)
            c_ok = ok[~is_evt]
            if len(c_ok):
                ctrl_sum[:, li] += group_sum(w[~is_evt], cell[c_ok], n_cells)
                ctrl_n[:, li] += np.bincount(cell[c_ok], minlength=n_cells)
    # event excess
    e_idx = np.flatnonzero(evt)
    e_cell = cell[e_idx]
    e_a, e_b = bd.a[e_idx], bd.b[e_idx]
    ex = np.full_like(ev_aff, np.nan)
    ctrl_used = np.zeros((len(e_idx), n_lag), bool)
    for li in range(n_lag):
        n_c = ctrl_n[e_cell, li]
        ok = valid[e_idx, li] & (n_c >= MIN_CTRL)
        ctrl_used[:, li] = ok
        base = (ctrl_sum[e_cell, li, e_a] - ctrl_sum[e_cell, li, e_b]) / np.maximum(n_c, 1)[:, None]
        ex[ok, li] = ev_aff[ok, li] - base[ok]
    ctrl_aff = np.full_like(ev_aff, np.nan)
    for li in range(n_lag):
        ok = ctrl_used[:, li]
        n_c = np.maximum(ctrl_n[e_cell, li], 1)[:, None]
        ctrl_aff[ok, li] = ((ctrl_sum[e_cell, li, e_a] - ctrl_sum[e_cell, li, e_b]) / n_c)[ok]
    q_e = q_dense[q_of_att[bd.att[e_idx]]]
    cnt = np.zeros((n_q, n_lag))
    sums = np.zeros((n_q, n_lag, n_layers))
    aff_sums = np.zeros((n_q, n_lag, n_layers))
    ctl_sums = np.zeros((n_q, n_lag, n_layers))
    for li in range(n_lag):
        ok = ctrl_used[:, li]
        cnt[:, li] = np.bincount(q_e[ok], minlength=n_q)
        sums[:, li] = group_sum(ex[ok, li], q_e[ok], n_q)
        aff_sums[:, li] = group_sum(ev_aff[ok, li], q_e[ok], n_q)
        ctl_sums[:, li] = group_sum(ctrl_aff[ok, li], q_e[ok], n_q)
    lm = ex.mean(-1)                                                   # layer-mean excess [E, lags]

    def pooled(lags):
        cols = [LAGS.index(x) for x in lags]
        ok = ctrl_used[:, cols]
        v = np.where(ok, lm[:, cols], 0.0).sum(1)
        c = ok.sum(1)
        return group_sum(v, q_e, n_q), group_sum(c.astype(float), q_e, n_q)

    pre_sum, pre_cnt = pooled(PRE_LAGS)
    post_sum, post_cnt = pooled(POST_LAGS)
    ctrl_pool = ctrl_n.sum(0)
    return dict(n_q=n_q, questions=kept_q, cnt=cnt, sums=sums, aff_sums=aff_sums, ctl_sums=ctl_sums,
                pre_sum=pre_sum, pre_cnt=pre_cnt, post_sum=post_sum, post_cnt=post_cnt,
                n_events=int(evt.sum()), n_controls=int((~evt).sum()),
                n_events_valid=cnt.sum(0).astype(int).tolist(), ctrl_windows=ctrl_pool.astype(int).tolist(),
                seconds=time.time() - t0)


# ------------------------------------------------------------------------------ inference

def qmean(sums: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-question means and their validity weights (questions without a valid window drop out)."""
    w = (cnt > 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
    return m, w


def boot_mean(m: np.ndarray, w: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Question-bootstrap distribution of the question-weighted mean; m, w [Q, ...] -> [B, ...]."""
    n_q = m.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_q, size=(n_boot, n_q))
    mult = np.zeros((n_boot, n_q))
    for b in range(n_boot):
        mult[b] = np.bincount(idx[b], minlength=n_q)
    flat_m, flat_w = m.reshape(n_q, -1), w.reshape(n_q, -1)
    num = mult @ (flat_m * flat_w)
    den = mult @ flat_w
    with np.errstate(invalid="ignore", divide="ignore"):
        est = num / den
    return est.reshape((n_boot,) + m.shape[1:])


def scalar_ci(sums: np.ndarray, cnt: np.ndarray, n_boot: int = N_BOOT) -> dict:
    m, w = qmean(sums, cnt)
    point = float((m * w).sum() / w.sum()) if w.sum() else float("nan")
    boot = boot_mean(m, w, n_boot)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(estimate=point, ci_lo=float(lo), ci_hi=float(hi), p_one_sided_positive=float(
        (1 + np.nansum(boot <= 0)) / (np.isfinite(boot).sum() + 1)), boot_sd=float(np.nanstd(boot, ddof=1)),
        n_questions=int(w.sum()), n_windows=int(cnt.sum()), question_values=np.where(w > 0, m, np.nan))


def curves(res: dict, n_boot: int = N_BOOT) -> dict:
    """Per-layer x lag curves with pointwise and simultaneous (sup-t) question-bootstrap bands."""
    cnt = res["cnt"][:, :, None]
    m, w = qmean(res["sums"], cnt)                                       # [Q, lag, L], [Q, lag, 1]
    ma, _ = qmean(res["aff_sums"], cnt)
    mc, _ = qmean(res["ctl_sums"], cnt)
    w3 = np.broadcast_to(w, m.shape)
    tot = w3.sum(0)
    safe = np.maximum(tot, 1)
    est = np.where(tot > 0, (m * w3).sum(0) / safe, np.nan)
    aff = np.where(tot > 0, (ma * w3).sum(0) / safe, np.nan)
    ctl = np.where(tot > 0, (mc * w3).sum(0) / safe, np.nan)
    boot = boot_mean(m, w3, n_boot)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    sd = np.nanstd(boot, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.abs(boot - est[None]) / sd[None]
    tmax = np.nanmax(np.where(np.isfinite(t), t, np.nan).reshape(len(boot), -1), axis=1)
    crit = float(np.nanpercentile(tmax, 95))
    w2 = w[:, :, 0]
    lm_m = m.mean(-1)
    lm_est = (lm_m * w2).sum(0) / np.maximum(w2.sum(0), 1)
    lm_lo, lm_hi = np.nanpercentile(boot_mean(lm_m, w2, n_boot), [2.5, 97.5], axis=0)
    return dict(est=est, lo=lo, hi=hi, sd=sd, sup_lo=est - crit * sd, sup_hi=est + crit * sd, crit=crit,
                aff=aff, ctl=ctl, n_q=tot[:, 0], lm_est=lm_est, lm_lo=lm_lo, lm_hi=lm_hi)


def curves_frame(cv: dict, res: dict, subset: str) -> pd.DataFrame:
    """Long table: one row per (layer, lag), plus the layer-mean rows (layer = -1)."""
    n_lag, n_layers = cv["est"].shape
    rows = []
    for li, lag in enumerate(LAGS):
        rows.append(dict(subset=subset, layer=-1, lag=lag, n_events_valid=res["n_events_valid"][li],
                         n_questions=int(cv["n_q"][li]), excess=cv["lm_est"][li], ci_lo=cv["lm_lo"][li],
                         ci_hi=cv["lm_hi"][li], sup_lo=np.nan, sup_hi=np.nan,
                         event_aff=float(np.nanmean(cv["aff"][li])), control_aff=float(np.nanmean(cv["ctl"][li]))))
        for layer in range(n_layers):
            rows.append(dict(subset=subset, layer=layer, lag=lag, n_events_valid=res["n_events_valid"][li],
                             n_questions=int(cv["n_q"][li]), excess=cv["est"][li, layer],
                             ci_lo=cv["lo"][li, layer], ci_hi=cv["hi"][li, layer],
                             sup_lo=cv["sup_lo"][li, layer], sup_hi=cv["sup_hi"][li, layer],
                             event_aff=cv["aff"][li, layer], control_aff=cv["ctl"][li, layer]))
    return pd.DataFrame(rows)


def summarize_a2(res: dict, n_boot: int = N_BOOT) -> dict:
    """Pre- and post-boundary pooled scalars with question-bootstrap CIs."""
    pre = scalar_ci(res["pre_sum"], res["pre_cnt"], n_boot)
    post = scalar_ci(res["post_sum"], res["post_cnt"], n_boot)
    return dict(pre_boundary=pre, post_boundary=post, n_events=res["n_events"], n_controls=res["n_controls"],
                n_events_valid_by_lag=dict(zip(map(str, LAGS), res["n_events_valid"])),
                control_windows_by_lag=dict(zip(map(str, LAGS), res["ctrl_windows"])))
