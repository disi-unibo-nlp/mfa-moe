"""Part 1 (descriptive, no outcomes): depth-path structure of top-1 experts, per model.

Sample (frozen): 200 cohort-A attempts (seed 0; qwen36 = split-v1 dev+tune only), each contributing
min(4096, n_reasoning) reasoning tokens drawn uniformly without replacement (per-attempt seeded rng).
Relative position = reasoning rank / n_reasoning in 10 equal bins.

Statistics: depth NMI vs a within-relative-bin token-shuffle null (D-MI), held-out bigram path
predictability vs the unconditional mode, 3-layer top-100-path concentration vs independence,
top-1 gate-weight / boundary-gap trajectories with the per-token Spearman "commitment" slope, and the
class-conditional bigram JS distance vs a label-permutation null (sentence labels shuffled within
absolute-position bin), split by the chain rule into a layer-l usage part and a transition part.

CLI: python -m dp.part1 run [--models M ...] [--workers W] [--sub]   collect + stats for the models
     python -m dp.part1 summary                                      results/part1/summary.md
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as cm
from . import pathfeat as pf

OUT = cm.RESULTS / "part1"
CACHE = OUT / "_cache"
N_REL = 10
N_NULL = 100                # token-shuffle null draws (D-MI and concentration)
N_PERM_CLASS = 200          # label permutations (class JS distance)
TOP_PATHS = 100
CLASS_MIN_TOK = 1000        # a class enters the JS comparison with >= this many sampled labelled tokens
BOOT = 1000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- sampling / collect

def sample_attempts(model: str) -> tuple[pd.DataFrame, np.ndarray]:
    """The fixed 200-attempt sample (rows of the kept, sorted cohort-A table)."""
    att = cm.load_attempts(model, "A")
    rng = np.random.default_rng(cm.P1_SEED)
    idx = np.sort(rng.choice(len(att), size=min(cm.P1_ATTEMPTS, len(att)), replace=False))
    return att, idx


def token_sample(n_reasoning: int, attempt_row: int, m_max: int = cm.P1_TOKENS) -> np.ndarray:
    """Sorted reasoning ranks of the attempt's sampled tokens (uniform without replacement)."""
    m = min(m_max, n_reasoning)
    rng = np.random.default_rng(np.random.SeedSequence([cm.P1_SEED, int(attempt_row)]))
    return np.sort(rng.choice(n_reasoning, size=m, replace=False)).astype(np.int64)


def collect_one(task: dict) -> str:
    """Worker: one attempt -> compact npz of the sampled tokens (skipped when cached)."""
    from v3an.extract import _load_tensors
    path = Path(task["cache"])
    if path.exists():
        return str(path)
    ids, weights, gaps, reasoning = _load_tensors(task["tensor_path"])
    L, T, k = ids.shape
    if (T != int(task["completion_tokens"]) or (L, k) != (task["num_layers"], task["top_k"])
            or len(reasoning) != int(task["n_reasoning"])):
        raise ValueError(f"{task['attempt_id']}: tensor layout/tokens differ from the attempt table")
    if ids.min() < 0 or ids.max() >= task["num_experts"]:
        raise ValueError(f"{task['attempt_id']}: expert id out of range")
    n_r = len(reasoning)
    ranks = token_sample(n_r, task["row"])
    pos = reasoning[ranks]
    top1, w1 = pf.top1_paths(ids, weights, pos)
    gap = np.asarray(gaps[:, pos], dtype=np.float32)
    cls = np.full(len(ranks), -1, np.int8)
    sid = np.full(len(ranks), -1, np.int32)
    sbin = np.full(len(ranks), -1, np.int8)
    rows = task.get("sent")
    if rows is not None and len(rows):
        if not (rows["n_reasoning"] == n_r).all():
            raise ValueError(f"{task['attempt_id']}: sentence table n_reasoning differs")
        for s, e, c, si, ab in zip(rows["tok_start"], rows["tok_end"], rows["cls"],
                                   rows["sentence_index"], rows["abs_bin"]):
            sel = (ranks >= s) & (ranks < e)
            cls[sel], sid[sel], sbin[sel] = c, si, ab
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ranks=ranks.astype(np.int32), n_r=n_r, top1=top1.astype(np.int16),
             w1=w1.astype(np.float16), gap=gap, cls=cls, sid=sid, sbin=sbin)
    os.replace(tmp, path)
    return str(path)


def build_tasks(model: str) -> list[dict]:
    spec = cm.model_spec(model)
    att, idx = sample_attempts(model)
    sent = cm.load_sentence_table(model, "A")
    by = {k: g for k, g in sent.groupby("attempt_id")}
    d = CACHE / model
    d.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in idx:
        r = att.iloc[int(i)]
        tasks.append(dict(model=model, row=int(i), attempt_id=r.attempt_id, tensor_path=r.tensor_path,
                          completion_tokens=int(r.completion_tokens), n_reasoning=int(r.n_reasoning),
                          cache=str(d / f"{int(i):05d}.npz"), sent=by.get(r.attempt_id), **spec))
    return tasks


def load_pool(model: str) -> dict:
    """Pooled sample arrays of a model from the per-attempt cache."""
    tasks = build_tasks(model)
    parts = [np.load(t["cache"]) for t in tasks]
    top1 = np.concatenate([p["top1"] for p in parts], axis=1).astype(np.int32)
    w1 = np.concatenate([p["w1"] for p in parts], axis=1).astype(np.float32)
    gap = np.concatenate([p["gap"] for p in parts], axis=1)
    n_per = np.array([p["top1"].shape[1] for p in parts])
    att_id = np.repeat(np.arange(len(parts)), n_per)
    rel = np.concatenate([p["ranks"] / max(int(p["n_r"]), 1) for p in parts])
    cls = np.concatenate([p["cls"] for p in parts])
    sid_local = np.concatenate([p["sid"] for p in parts]).astype(np.int64)
    sbin = np.concatenate([p["sbin"] for p in parts])
    sid = np.where(sid_local >= 0, att_id.astype(np.int64) * 10**6 + sid_local, -1)
    return dict(top1=top1, w1=w1, gap=gap, att=att_id, n_att=len(parts), rel_bin=rel_bins(rel),
                cls=cls, sid=sid, sbin=sbin, n_per_attempt=n_per,
                n_reasoning=np.array([int(p["n_r"]) for p in parts]))


def rel_bins(rel: np.ndarray) -> np.ndarray:
    return np.minimum((np.asarray(rel) * N_REL).astype(np.int64), N_REL - 1).astype(np.int8)


# ------------------------------------------------------------------------------------ MI / NMI

def contingency(x: np.ndarray, y: np.ndarray, E: int) -> np.ndarray:
    return np.bincount(x.astype(np.int64) * E + y, minlength=E * E).reshape(E, E).astype(np.float64)


def nmi_table(t: np.ndarray) -> tuple[float, float]:
    """(NMI with arithmetic-mean normaliser, MI in nats) of a contingency table."""
    n = t.sum()
    if n <= 0:
        return np.nan, np.nan
    p = t / n
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    den = 0.5 * (hx + hy)
    return (mi / den if den > 1e-12 else np.nan), mi


def within_bin_perm(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutation of token indices that shuffles within each bin, without fixed points (bins > 1)."""
    perm = np.arange(len(bins))
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        sh = rng.permutation(idx)
        if len(idx) > 1:
            for j in np.flatnonzero(sh == idx):
                if sh[j] == idx[j]:
                    k = (j + 1) % len(idx)
                    sh[j], sh[k] = sh[k], sh[j]
        perm[idx] = sh
    return perm


def depth_mi(top1: np.ndarray, bins: np.ndarray, E: int, rng: np.random.Generator) -> dict:
    """NMI(top1_l, top1_{l+1}) per layer pair, and the within-bin token-shuffle null (N_NULL draws)."""
    L, N = top1.shape
    obs = np.array([nmi_table(contingency(top1[l], top1[l + 1], E)) for l in range(L - 1)])
    null = np.zeros((N_NULL, L - 1, 2))
    for d in range(N_NULL):
        perm = within_bin_perm(bins, rng)
        for l in range(L - 1):
            null[d, l] = nmi_table(contingency(top1[l], top1[l + 1][perm], E))
    nm, ns = null[:, :, 0].mean(0), null[:, :, 0].std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = obs[:, 0] / nm
        z = (obs[:, 0] - nm) / ns
    lm_obs, lm_null = obs[:, 0].mean(), null[:, :, 0].mean(1)
    return dict(nmi=obs[:, 0], mi_nats=obs[:, 1], null_mean=nm, null_sd=ns, ratio=ratio, z=z,
                null_mi_mean=null[:, :, 1].mean(0), layer_mean_nmi=float(lm_obs),
                layer_mean_null=float(lm_null.mean()), layer_mean_null_sd=float(lm_null.std(ddof=1)),
                layer_mean_ratio=float(lm_obs / lm_null.mean()),
                p_layer_mean=float((1 + (lm_null >= lm_obs).sum()) / (N_NULL + 1)),
                n_null=N_NULL, n_tokens=int(N))


# ---------------------------------------------------------------------------------- predictability

def bigram_predictability(top1: np.ndarray, att: np.ndarray, n_att: int, E: int, seed: int) -> dict:
    """Held-out accuracy of the training-half bigram argmax vs the training unconditional mode.

    Halves are random attempt halves (seed); both directions are reported (direction 0: half A trains,
    half B is held out, the plan's split; direction 1 swaps them)."""
    L, N = top1.shape
    rng = np.random.default_rng(seed)
    half = np.zeros(n_att, np.int8)
    half[rng.permutation(n_att)[: n_att // 2]] = 1
    tok_half = half[att]
    res = {}
    for direction, (tr_h, te_h) in enumerate(((0, 1), (1, 0))):
        tr, te = tok_half == tr_h, tok_half == te_h
        acc_b, acc_m, correct_b, correct_m = [], [], [], []
        for l in range(L - 1):
            tab = contingency(top1[l, tr], top1[l + 1, tr], E)
            mode = int(np.bincount(top1[l + 1, tr], minlength=E).argmax())
            rowsum = tab.sum(1)
            pred_of = np.where(rowsum > 0, tab.argmax(1), mode)
            pred = pred_of[top1[l, te]]
            y = top1[l + 1, te]
            cb, cm_ = (pred == y), (y == mode)
            acc_b.append(cb.mean())
            acc_m.append(cm_.mean())
            correct_b.append(np.bincount(att[te], weights=cb, minlength=n_att))
            correct_m.append(np.bincount(att[te], weights=cm_, minlength=n_att))
        n_te = np.bincount(att[te], minlength=n_att).astype(float)
        cb_, cm__ = np.array(correct_b), np.array(correct_m)          # [L-1, n_att]
        test_att = np.flatnonzero(n_te > 0)
        rb = np.random.default_rng(seed + 1 + direction)
        deltas = []
        for _ in range(BOOT):
            pick = rb.choice(test_att, size=len(test_att), replace=True)
            n = n_te[pick].sum()
            deltas.append(((cb_[:, pick].sum(1) - cm__[:, pick].sum(1)) / n).mean())
        acc_b, acc_m = np.array(acc_b), np.array(acc_m)
        res[f"direction{direction}"] = dict(
            acc_bigram=acc_b, acc_mode=acc_m, delta=acc_b - acc_m,
            layer_mean_bigram=float(acc_b.mean()), layer_mean_mode=float(acc_m.mean()),
            layer_mean_delta=float((acc_b - acc_m).mean()),
            delta_ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            n_train_tokens=int(tr.sum()), n_test_tokens=int(te.sum()),
            n_train_attempts=int((half == tr_h).sum()), n_test_attempts=int((half == te_h).sum()))
    return res


# ---------------------------------------------------------------------------------- concentration

def top_path_share(keys: np.ndarray, k: int = TOP_PATHS) -> float:
    _, cnt = np.unique(keys, return_counts=True)
    if len(cnt) <= k:
        return 1.0
    return float(np.partition(cnt, len(cnt) - k)[len(cnt) - k:].sum() / len(keys))


def path_concentration(top1: np.ndarray, bins: np.ndarray, E: int, rng: np.random.Generator) -> dict:
    """Coverage by the 100 most frequent 3-layer top-1 paths vs independent-layer permutation nulls.

    `null_indep`: each column shuffled independently over all tokens (marginals preserved);
    `null_binned`: columns l+1, l+2 shuffled independently within the relative-position bin."""
    L, N = top1.shape
    n_win = L - 2
    obs = np.zeros(n_win)
    indep = np.zeros((N_NULL, n_win))
    binned = np.zeros((N_NULL, n_win))
    t = top1.astype(np.int64)
    for l in range(n_win):
        obs[l] = top_path_share((t[l] * E + t[l + 1]) * E + t[l + 2])
    for d in range(N_NULL):
        p1, p2 = within_bin_perm(bins, rng), within_bin_perm(bins, rng)
        g1, g2 = rng.permutation(N), rng.permutation(N)
        for l in range(n_win):
            a, b, c = t[l], t[l + 1], t[l + 2]
            indep[d, l] = top_path_share((a * E + b[g1]) * E + c[g2])
            binned[d, l] = top_path_share((a * E + b[p1]) * E + c[p2])
    return dict(observed=obs, null_indep_mean=indep.mean(0), null_indep_sd=indep.std(0, ddof=1),
                null_binned_mean=binned.mean(0), null_binned_sd=binned.std(0, ddof=1),
                ratio_indep=obs / indep.mean(0), ratio_binned=obs / binned.mean(0),
                layer_mean_observed=float(obs.mean()), layer_mean_null_indep=float(indep.mean()),
                layer_mean_null_binned=float(binned.mean()),
                layer_mean_ratio_indep=float(obs.mean() / indep.mean()),
                layer_mean_ratio_binned=float(obs.mean() / binned.mean()),
                p_layer_mean_indep=float((1 + (indep.mean(1) >= obs.mean()).sum()) / (N_NULL + 1)),
                top_paths=TOP_PATHS, n_null=N_NULL, n_tokens=int(N))


# ---------------------------------------------------------------------------------- score trajectory

def score_trajectory(top1: np.ndarray, w1: np.ndarray, gap: np.ndarray, cls: np.ndarray,
                     rel_bin: np.ndarray, att: np.ndarray, n_att: int, seed: int) -> dict:
    """Per-depth mean/SD of top-1 weight and boundary gap; per-token Spearman(w1, depth) slope."""
    L, N = w1.shape
    rho = pf.token_spearman(w1)
    ok = np.isfinite(rho)
    q = np.nanpercentile(rho, [5, 25, 50, 75, 95])
    out = dict(w1_mean=w1.mean(1), w1_sd=w1.std(1), gap_mean=gap.mean(1), gap_sd=gap.std(1),
               gap_zero_frac=(gap <= 0).mean(1), slope_mean=float(np.nanmean(rho)),
               slope_sd=float(np.nanstd(rho)), slope_quantiles=dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                                                        map(float, q))),
               slope_frac_pos=float((rho[ok] > 0).mean()), slope_frac_neg=float((rho[ok] < 0).mean()),
               slope_nan_frac=float((~ok).mean()))
    per_att = np.bincount(att[ok], weights=rho[ok], minlength=n_att) / np.maximum(
        np.bincount(att[ok], minlength=n_att), 1)
    rb = np.random.default_rng(seed)
    boots = [per_att[rb.integers(0, n_att, n_att)].mean() for _ in range(BOOT)]
    out["slope_mean_attempt_weighted"] = float(per_att.mean())
    out["slope_mean_attempt_boot_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["slope_by_rel_bin"] = [float(np.nanmean(rho[rel_bin == b])) if (rel_bin == b).any() else None
                               for b in range(N_REL)]
    by_class = {}
    for c, name in enumerate(cm.CLASSES):
        m = cls == c
        if m.sum() >= 100:
            by_class[name] = dict(n_tokens=int(m.sum()), slope_mean=float(np.nanmean(rho[m])),
                                  w1_mean_by_layer=w1[:, m].mean(1), gap_mean_by_layer=gap[:, m].mean(1),
                                  w1_layer_mean=float(w1[:, m].mean()), gap_layer_mean=float(gap[:, m].mean()))
    out["by_class"] = by_class
    m = cls >= 0
    out["labelled"] = dict(n_tokens=int(m.sum()), slope_mean=float(np.nanmean(rho[m])) if m.any() else None)
    return out


# ---------------------------------------------------------------------------------- class JS distance

def _h(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)


def pairwise_jsd(P: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (bits) between the rows of P [C, D] (rows are distributions)."""
    C = len(P)
    H = _h(P)
    out = np.zeros((C, C))
    for i in range(C - 1):
        M = 0.5 * (P[i][None, :] + P[i + 1:])
        out[i, i + 1:] = np.clip(_h(M) - 0.5 * (H[i] + H[i + 1:]), 0.0, None)
    return out + out.T


def class_js_layer(a: np.ndarray, b: np.ndarray, cls_tok: np.ndarray, classes: list[int], E: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise JS distances [C, C] between class-conditional bigram matrices of one layer pair.

    Returns (joint, marginal, transition) distances in bits: `joint` = sqrt(JSD) between the class joint
    distributions Q_c(a, b); `marginal` = sqrt(JSD) between the class distributions of the layer-l expert
    Q_c(a); `transition` = sqrt(JSD_joint - JSD_marginal) (chain rule: the part of the joint divergence due
    to class differences in P(b | a), non-negative)."""
    C = len(classes)
    cidx = np.full(len(cm.CLASSES), -1)
    for i, c in enumerate(classes):
        cidx[c] = i
    ci = cidx[cls_tok]
    keep = ci >= 0
    tab = np.bincount(ci[keep] * E * E + a[keep].astype(np.int64) * E + b[keep], minlength=C * E * E)
    tab = tab.reshape(C, E, E).astype(np.float64)
    Q = tab / np.maximum(tab.sum((1, 2), keepdims=True), 1)
    jsd_joint = pairwise_jsd(Q.reshape(C, -1))
    jsd_marg = pairwise_jsd(Q.sum(2))
    trans = np.clip(jsd_joint - jsd_marg, 0.0, None)
    return np.sqrt(jsd_joint), np.sqrt(jsd_marg), np.sqrt(trans)


def permute_sentence_labels(sid: np.ndarray, sent_cls: np.ndarray, sent_bin: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """New class per sentence: labels shuffled among sentences of the same absolute-position bin."""
    new = sent_cls.copy()
    for b in np.unique(sent_bin):
        idx = np.flatnonzero(sent_bin == b)
        new[idx] = sent_cls[rng.permutation(idx)]
    return new


def class_js(top1: np.ndarray, cls: np.ndarray, sid: np.ndarray, sbin: np.ndarray, E: int,
             rng: np.random.Generator) -> dict:
    """Class differences of the depth bigram: observed pairwise JS distance vs label permutations."""
    L, N = top1.shape
    lab = cls >= 0
    counts = np.bincount(cls[lab], minlength=len(cm.CLASSES))
    classes = [c for c in range(len(cm.CLASSES)) if counts[c] >= CLASS_MIN_TOK]
    res = dict(classes=[cm.CLASSES[c] for c in classes], class_tokens={cm.CLASSES[c]: int(counts[c])
                                                                         for c in range(len(cm.CLASSES))},
               n_labelled_tokens=int(lab.sum()), n_perm=N_PERM_CLASS, min_tokens=CLASS_MIN_TOK)
    if len(classes) < 2:
        res["skipped"] = "fewer than two classes reach the token minimum"
        return res
    top1 = top1[:, lab]
    sid_l, cls_l = sid[lab], cls[lab]
    _, first, inv = np.unique(sid_l, return_index=True, return_inverse=True)
    sent_cls, sent_bin = cls_l[first], sbin[lab][first]
    C = len(classes)
    iu = np.triu_indices(C, 1)

    def stat(cls_tokens: np.ndarray):
        out = np.zeros((3, L - 1, C, C))
        for l in range(L - 1):
            out[:, l] = class_js_layer(top1[l], top1[l + 1], cls_tokens, classes, E)
        return out

    def layer_pair_means(x: np.ndarray) -> np.ndarray:            # [3, L-1, C, C] -> [3, L-1]
        return x[:, :, iu[0], iu[1]].mean(-1)

    obs_full = stat(cls_l)
    obs = layer_pair_means(obs_full)
    null = np.zeros((N_PERM_CLASS, 3, L - 1))
    pair_null = np.zeros((3, C, C))
    for d in range(N_PERM_CLASS):
        new = permute_sentence_labels(sid_l, sent_cls, sent_bin, rng)
        full = stat(new[inv])
        null[d] = layer_pair_means(full)
        pair_null += full.mean(1)
    pair_null /= N_PERM_CLASS

    def summ(o, n):
        lm_o, lm_n = float(o.mean()), n.mean(1)
        return dict(observed=o, null_mean=n.mean(0), null_sd=n.std(0, ddof=1),
                    excess=o - n.mean(0), z=(o - n.mean(0)) / n.std(0, ddof=1),
                    p=(1 + (n >= o[None]).sum(0)) / (N_PERM_CLASS + 1),
                    layer_mean_observed=lm_o, layer_mean_null=float(lm_n.mean()),
                    layer_mean_null_sd=float(lm_n.std(ddof=1)), layer_mean_excess=lm_o - float(lm_n.mean()),
                    layer_mean_ratio=lm_o / float(lm_n.mean()) if lm_n.mean() > 0 else None,
                    p_layer_mean=float((1 + (lm_n >= lm_o).sum()) / (N_PERM_CLASS + 1)))

    res.update(joint=summ(obs[0], null[:, 0]), marginal_layer_l=summ(obs[1], null[:, 1]),
               transition=summ(obs[2], null[:, 2]),
               pair_matrix_joint_layer_mean=obs_full[0].mean(0), pair_matrix_joint_null_mean=pair_null[0],
               pair_matrix_joint_excess=obs_full[0].mean(0) - pair_null[0],
               pair_matrix_transition_layer_mean=obs_full[2].mean(0),
               pair_matrix_transition_null_mean=pair_null[2],
               pair_matrix_transition_excess=obs_full[2].mean(0) - pair_null[2])
    return res


# ---------------------------------------------------------------------------------------- driver

def stats_model(model: str) -> dict:
    t0 = time.time()
    spec = cm.model_spec(model)
    E = spec["num_experts"]
    pool = load_pool(model)
    top1 = pool["top1"]
    L, N = top1.shape
    log(f"{model}: pooled {pool['n_att']} attempts, {N} tokens, L={L}, E={E}")
    rng = np.random.default_rng(np.random.SeedSequence([cm.P1_SEED, 1]))
    res = dict(model=model, spec=spec, n_attempts=pool["n_att"], n_tokens=int(N),
               n_tokens_per_attempt=dict(min=int(pool["n_per_attempt"].min()),
                                         median=float(np.median(pool["n_per_attempt"])),
                                         max=int(pool["n_per_attempt"].max())),
               n_reasoning_median=float(np.median(pool["n_reasoning"])),
               labelled_token_frac=float((pool["cls"] >= 0).mean()),
               split_restriction="split-v1 dev+tune" if model == "qwen36" else None,
               sample=dict(attempts=cm.P1_ATTEMPTS, max_tokens=cm.P1_TOKENS, seed=cm.P1_SEED,
                           rel_bins=N_REL))
    res["dmi"] = depth_mi(top1, pool["rel_bin"], E, rng)
    log(f"{model}: D-MI layer-mean NMI {res['dmi']['layer_mean_nmi']:.4f} vs null "
        f"{res['dmi']['layer_mean_null']:.4f} ({time.time() - t0:.0f}s)")
    res["predictability"] = bigram_predictability(top1, pool["att"], pool["n_att"], E, seed=cm.P1_SEED)
    log(f"{model}: bigram acc delta {res['predictability']['direction0']['layer_mean_delta']:.4f}")
    res["concentration"] = path_concentration(top1, pool["rel_bin"], E, rng)
    log(f"{model}: top-100 path share {res['concentration']['layer_mean_observed']:.4f} vs "
        f"indep {res['concentration']['layer_mean_null_indep']:.4f} ({time.time() - t0:.0f}s)")
    res["scores"] = score_trajectory(top1, pool["w1"], pool["gap"], pool["cls"], pool["rel_bin"],
                                     pool["att"], pool["n_att"], seed=cm.P1_SEED + 7)
    res["class_js"] = class_js(top1, pool["cls"], pool["sid"], pool["sbin"], E, rng)
    if "joint" in res["class_js"]:
        c = res["class_js"]
        log(f"{model}: class JS joint {c['joint']['layer_mean_observed']:.4f} vs null "
            f"{c['joint']['layer_mean_null']:.4f} p={c['joint']['p_layer_mean']:.4f}; transition "
            f"{c['transition']['layer_mean_observed']:.4f} vs {c['transition']['layer_mean_null']:.4f} "
            f"p={c['transition']['p_layer_mean']:.4f}")
    res["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}.json").write_text(json.dumps(cm.clean(res), indent=1))
    return dict(model=model, seconds=res["seconds"])


def run(models: list[str], workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [t for m in models for t in build_tasks(m)]
    log(f"collect: {len(tasks)} attempts over {len(models)} models, {workers} workers")
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    os.environ.update({v: "1" for v in thread_vars})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(collect_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 100 == 0 or i == len(tasks):
                log(f"collect: {i}/{len(tasks)} done, {time.time() - t0:.0f}s")
    log("stats: per-model pool")
    with ProcessPoolExecutor(max_workers=min(len(models), workers), mp_context=mp.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(stats_model, m) for m in models]):
            log(f"stats done: {f.result()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=list(cm.MODELS), choices=cm.MODELS)
    r.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    s = sub.add_parser("stats")
    s.add_argument("model", choices=cm.MODELS)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.models, a.workers)
    elif a.cmd == "stats":
        stats_model(a.model)
    else:
        from .part1_summary import write_summary
        write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
