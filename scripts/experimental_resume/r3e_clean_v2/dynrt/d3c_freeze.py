"""Clean B4 feature freeze with prospective training-fold-only PAlex amendment.

For every training set C8 fits blocks on (5 repeats x 4 outer folds x [outer-train + 3 inner-train] = 80 sets, GPT
cohort A minus the 37 cohort-B questions) the in-fold B1 fit produces the pathway features of ALL attempts; the
class-conditioned marginal block is fitted on the same sets. Cohort B attempts get the same scores from the B1 fit
on all universe questions. Writes results/B4/{features_A.npz, features_B.parquet, freeze_checks.json,
FEATURES_FROZEN_B4.json}. No outcome column is loaded.

  python -m dynrt.d3c_freeze [--workers W --max-pairs N --tag T]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import d3b_b1 as B
from . import d3c_features as D
from .common import CLASSES, RESULTS, V3R2, digest, sha256_file, load_split
from .d3b_tokens import OUT_GPT, load_tokens
from .d3b_util import clean
from .d3c_tokens import OUT_B
from .data import load_cohort

OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/resume-v1/r3e-b4-features-v2')
FROZEN = OUT / "FEATURES_FROZEN_B4.json"
GPT = dict(experts=32, layers=24, top_k=4)
B1_RESULTS = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json')
PREFLIGHT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/R3E_CLEAN_PREFLIGHT_v1.json')
CODE_FILES = ("d3c_features.py", "d3c_freeze.py", "d3c_tokens.py", "d3b_b1.py", "d3b_tokens.py", "d3b_util.py",
              "cv.py", "features.py", "controls.py", "data.py", "common.py", "extract.py")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {n: sha256_file(here / n) for n in CODE_FILES}


def universe_mask(attempts: pd.DataFrame) -> np.ndarray:
    """Exact-ID-clean dev/tune A attempts, with cohort B held out."""
    b_q = set(pd.read_parquet(V3R2 / "gpt/B/attempts.parquet", columns=["question"])["question"])
    split = load_split()
    if not set(attempts['question']) <= set(split):
        raise ValueError('GPT A question missing from split-v1')
    return np.asarray([split[q] in ('dev', 'tune') and q not in b_q for q in attempts['question']], bool)


def align_check(tk: dict, coh) -> dict:
    """The d3b token table must reproduce D1's attempt and sentence tables exactly."""
    ta, ca = tk["attempts"], coh.attempts
    same_att = bool((ta["attempt_id"].to_numpy() == ca["attempt_id"].to_numpy()).all())
    ts, cs = tk["sentences"], coh.sent
    same_sent = bool(len(ts) == len(cs)
                     and (ta["attempt_id"].to_numpy()[ts["att"].to_numpy()] == cs["attempt_id"].to_numpy()).all()
                     and (ts["sentence_index"].to_numpy() == cs["sentence_index"].to_numpy()).all()
                     and (ts["cls"].to_numpy().astype(int) == cs["cls"].to_numpy().astype(int)).all()
                     and (ts["n_tokens"].to_numpy() == cs["n_tokens"].to_numpy()).all())
    if not (same_att and same_sent):
        raise RuntimeError("d3b token table differs from the D1 sentence table")
    return dict(attempts_identical=same_att, sentences_identical=same_sent, sentences=int(len(ts)))


def run(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_GPT)
    coh = load_cohort("gpt", "A")
    chk = dict(alignment=align_check(tk, coh))
    keep = universe_mask(tk["attempts"])
    U = D.build_tokens(tk, keep, GPT["experts"])
    att_u = tk["attempts"][keep].reset_index(drop=True)
    groups = att_u["question"].to_numpy()
    sets = D.enumerate_sets(groups)
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight['sha256'] != digest({k: v for k, v in preflight.items() if k != 'sha256'}):
        raise ValueError('changed clean population preflight')
    expected = preflight['populations']['exact_id_clean']
    if (U.n_att != expected['attempts'] or digest(att_u['attempt_id'].astype(str).tolist()) != expected['attempt_ids_sha256'] or
        digest(att_u['question'].astype(str).tolist()) != expected['ordered_questions_sha256'] or
        digest([D.set_key(s) for s in sets]) != expected['nested_training_set_keys_sha256']):
        raise ValueError('B4 population or nested training folds differ from outcome-blind preflight')
    member = D.membership(sets, U.n_att)
    pairs = B.layer_pairs(GPT["layers"])
    if a.max_pairs:
        pairs = pairs[::max(1, len(pairs) // a.max_pairs)][:a.max_pairs]
    n_shuf = a.n_shuf
    log(f"universe: {U.n_att} attempts, {U.n_sent} sentences, {len(U.cls)} tokens; {len(sets)} training sets; "
        f"{len(pairs)} layer pairs; {n_shuf} shuffles ({time.time() - t0:.0f}s)")
    shuf = D.fold_shuffled_labels(U, U, member, n_shuf)
    sums = D.pathway_sums(U, U, member, shuf, pairs, a.workers)
    pa = D.attempt_pa_features(sums, U, len(pairs), n_shuf)
    log(f"pathway features {pa.shape} ({time.time() - t0:.0f}s)")
    # marginal block (D1 sentence histograms of the universe attempts)
    row = keep[coh.table.attempt]
    att_map = np.full(len(keep), -1, np.int64)
    att_map[keep] = np.arange(int(keep.sum()))
    h = coh.table.h[row]
    s_att, s_cls = att_map[coh.table.attempt[row]], coh.table.cls[row].astype(np.int64)
    if not (s_att == U.sent_att).all() or not (s_cls == U.sent_cls).all():
        raise RuntimeError("marginal-block sentences differ from the pathway sentences")
    marg = np.stack([D.marg_features(h, s_cls, s_att, member[i] > 0, U.n_att, GPT["layers"], GPT["experts"])
                     for i in range(len(sets))])
    log(f"marginal block {marg.shape} ({time.time() - t0:.0f}s)")
    # outcome-free diagnostics
    folds_check = _b1_consistency(pa, sets, groups, n_pairs=len(pairs))
    chk.update(folds_check)
    chk["in_vs_out_of_sample"] = _in_vs_out(pa, sets, groups)
    chk["feature_moments"] = {n: dict(mean=float(np.nanmean(pa[:, :, j])), sd=float(np.nanstd(pa[:, :, j])),
                                      missing=int(np.isnan(pa[:, :, j]).sum())) for j, n in enumerate(D.PA_NAMES)}
    chk["marg_missing_indicator_share"] = {c: float(1 - marg[:, :, 21 + i].mean()) for i, c in enumerate(CLASSES)}
    # cohort B: fit on all universe questions, scored on cohort-B tokens
    tkb = load_tokens(OUT_B)
    TB = D.build_tokens(tkb, np.ones(len(tkb["attempts"]), bool), GPT["experts"])
    full = D.membership([np.arange(U.n_att)], U.n_att)
    shuf_b = D.fold_shuffled_labels(U, TB, full, n_shuf)
    sums_b = D.pathway_sums(U, TB, full, shuf_b, pairs, a.workers)
    pa_b = D.attempt_pa_features(sums_b, TB, len(pairs), n_shuf)[0]
    dfb = pd.DataFrame(pa_b, columns=D.PA_NAMES)
    dfb.insert(0, "question", tkb["attempts"]["question"].to_numpy())
    dfb.insert(0, "attempt_id", tkb["attempts"]["attempt_id"].to_numpy())
    tag = f"_{a.tag}" if a.tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"features_A{tag}.npz", set_keys=np.array([D.set_key(s) for s in sets]),
                        pa=pa, marg=marg, attempt_id=att_u["attempt_id"].to_numpy(), pa_names=np.array(D.PA_NAMES),
                        marg_names=np.array(D.MARG_NAMES))
    dfb.to_parquet(OUT / f"features_B{tag}.parquet")
    chk["cohort_B"] = dict(attempts=int(len(dfb)), questions=int(dfb["question"].nunique()),
                           labelled_tokens=int(len(TB.cls)),
                           moments={n: float(np.nanmean(dfb[n])) for n in D.PA_NAMES})
    (OUT / f"freeze_checks{tag}.json").write_text(json.dumps(clean(chk), indent=1))
    man = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), plan="clean B4 prospective fold-training-only PAlex amendment v2",
               universe=dict(attempts=int(U.n_att), questions=int(len(set(groups))), excluded_cohort_B_questions=int(
                   (~keep).sum()), sentences=int(U.n_sent), tokens=int(len(U.cls))),
               design=dict(pairs=len(pairs), gaps=list(B.GAPS), lambda_=B.LAMBDA, eps=B.EPS_Q, n_shuffles=n_shuf,
                           shuffle_seed=D.SHUF_SEED, repeats=5, outer=5, inner=3, seed0=20260929,
                           training_sets=len(sets), sets_digest=digest([D.set_key(s) for s in sets]),
                           pa_names=D.PA_NAMES, marg_names=D.MARG_NAMES,
                           interpretation=dict(
                               PAlex="prospective reading B: real B1 in-fold fit; ten pseudo-label draws from "
                                     "training-token empirical class distribution conditional on token id x position bin; "
                                     "unseen strata use training position-bin then all-training fallback; no held-out "
                                     "class labels enter the lexical reference; PAlexB is a compatibility alias; "
                                     "historical all-universe reading A is a separately labeled transductive sensitivity",
                               PA_sd="SD (ddof 1) of the sentence PA across the attempt's labelled sentences",
                               marg="own-class mean over class-c sentences, layer thirds of 8 layers, missing -> 0 "
                                    "plus 7 presence indicators",
                               cohort_B="B1 fit on all clean universe questions; cohort-B pseudo-classes drawn "
                                        "from clean-universe training token distributions, without B labels"),
                           folds="five outer/three inner frozen duplicate-family folds, five repeats, seeds 20260929..20260933; clean GPT A"),
               files={f"features_A{tag}.npz": sha256_file(OUT / f"features_A{tag}.npz"),
                      f"features_B{tag}.parquet": sha256_file(OUT / f"features_B{tag}.parquet"),
                      f"freeze_checks{tag}.json": sha256_file(OUT / f"freeze_checks{tag}.json")},
               inputs=dict(tokens_A_sha256=tk["provenance"]["tokens_sha256"],
                           tokens_B_sha256=tkb["provenance"]["tokens_sha256"],
                           b1_results_sha256=sha256_file(B1_RESULTS),
                           preflight_sha256=sha256_file(PREFLIGHT)),
               code_sha256=code_hashes(), outcomes_read=False, slurm_job=os.environ.get("SLURM_JOB_ID"),
               seconds=time.time() - t0)
    man["sha256"] = digest({k: v for k, v in man.items() if k not in ("time", "seconds", "slurm_job")})
    path = FROZEN if not a.tag else OUT / f"FEATURES_FROZEN_B4{tag}.json"
    path.write_text(json.dumps(clean(man), indent=1))
    log(f"wrote {path} (sha256 {man['sha256'][:16]}) in {time.time() - t0:.0f}s")
    return 0


def _outer_index(sets: list, n_outer: int = 25, stride: int = 4) -> list[int]:
    return [i * stride for i in range(n_outer)]


def _b1_consistency(pa: np.ndarray, sets: list, groups: np.ndarray, n_pairs: int) -> dict:
    """Mean held-out token-weighted PA per repeat must equal the B1 per-repeat G_joint (same fits, same folds)."""
    if not B1_RESULTS.exists() or n_pairs != 43:
        return dict(b1_consistency="skipped")
    ref = json.loads(B1_RESULTS.read_text())["result"]["observed"]["per_repeat_G"]
    got = []
    for r in range(5):
        vals = np.full(len(groups), np.nan)
        for f in range(5):
            i = (r * 5 + f) * 4
            test = np.setdiff1d(np.arange(len(groups)), sets[i])
            vals[test] = pa[i, test, D.PA_NAMES.index("PA_tok")]
        got.append(float(np.nanmean(vals)))
    return dict(b1_consistency=dict(per_repeat_G_b1=ref, per_repeat_mean_heldout_PA_tok=got,
                                    max_abs_diff=float(np.max(np.abs(np.array(ref) - np.array(got))))))


def _in_vs_out(pa: np.ndarray, sets: list, groups: np.ndarray) -> dict:
    j = D.PA_NAMES.index("PA_tok")
    tr, te = [], []
    for i in _outer_index(sets):
        mask = np.zeros(len(groups), bool)
        mask[sets[i]] = True
        tr.append(float(np.nanmean(pa[i, mask, j])))
        te.append(float(np.nanmean(pa[i, ~mask, j])))
    return dict(train_mean_PA_tok=float(np.mean(tr)), heldout_mean_PA_tok=float(np.mean(te)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--max-pairs", type=int)
    ap.add_argument("--n-shuf", type=int, default=D.N_SHUF)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
