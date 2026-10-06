"""Stage 2 output: out-of-fold routing features (no outcome touched) and the freeze manifest.

For each model's cohort A (qwen36: dev+tune), features of every attempt are produced by fitting
profiles, residual cells and shrinkage on the TRAINING questions of the C8 outer folds (4 folds x
5 repeats, seeds 0-4; identical fold function to cv.py) and applying them to the held-out fold.
Cohort B attempts get features from a fit on cohort A attempts of questions absent from B.
CLI: python -m dynrt.freeze
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .common import CLASSES, RESULTS, digest, sha256_file
from .cv import question_folds
from .data import class_share_check, load_cohort

REPEATS, OUTER = 5, 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oof_features(coh) -> pd.DataFrame:
    """Cross-fitted attempt features for every repeat (question-grouped folds)."""
    q = coh.attempts["question"].to_numpy()
    n = len(q)
    out = {"attempt_id": coh.attempts["attempt_id"].to_numpy(), "question": q}
    acc = {k: np.zeros(n) for k in F.FEATURES}
    for r in range(REPEATS):
        folds = question_folds(q, OUTER, r)
        cols = {k: np.full(n, np.nan) for k in F.FEATURES}
        for f in range(OUTER):
            feats = F.attempt_features(coh.table, folds != f)
            for k in F.FEATURES:
                cols[k][folds == f] = feats[k][folds == f]
            log(f"oof repeat {r} fold {f}: fit on {int((folds != f).sum())} attempts")
        out[f"fold_r{r}"] = folds
        for k in F.FEATURES:
            out[f"{k}_r{r}"] = cols[k]
            acc[k] += np.nan_to_num(cols[k])
    for k in F.FEATURES:
        arr = np.column_stack([out[f"{k}_r{r}"] for r in range(REPEATS)])
        with np.errstate(all="ignore"):
            out[f"{k}_mean"] = np.nanmean(arr, axis=1)
    counts = F.aggregate(dict(own=np.zeros(len(coh.table.cls)), margin=np.zeros(len(coh.table.cls))),
                         coh.table)
    out["n_sent"], out["n_explore"] = counts["n_sent"], counts["n_explore"]
    return pd.DataFrame(out)


def insample_check(coh) -> dict:
    """Outcome-free diagnostic: training-attempt vs held-out-attempt feature moments (repeat 0, fold 0)."""
    q = coh.attempts["question"].to_numpy()
    folds = question_folds(q, OUTER, 0)
    train = folds != 0
    feats = F.attempt_features(coh.table, train)
    out = {}
    for k in F.FEATURES:
        v = feats[k]
        out[k] = dict(train_mean=float(np.nanmean(v[train])), train_sd=float(np.nanstd(v[train])),
                      test_mean=float(np.nanmean(v[~train])), test_sd=float(np.nanstd(v[~train])))
    return out


def b_features(model: str, coh_a, coh_b) -> tuple[pd.DataFrame, dict]:
    """Cohort B features from a fit on cohort A attempts of questions not present in B."""
    b_questions = set(coh_b.attempts["question"])
    train = ~coh_a.attempts["question"].isin(b_questions).to_numpy()
    fitted = F.fit(coh_a.table, train)
    feats = F.apply_fitted(fitted, coh_b.table)
    df = pd.DataFrame({"attempt_id": coh_b.attempts["attempt_id"].to_numpy(),
                       "question": coh_b.attempts["question"].to_numpy(), **feats})
    info = dict(train_attempts=int(train.sum()), b_questions=len(b_questions),
                b_attempts=len(coh_b.attempts), profile_n_class=fitted.profiles.n_class.tolist())
    return df, info


def _none_nan(x: float):
    return None if np.isnan(x) else x


def describe(model: str, coh) -> dict:
    a = coh.attempts
    ex = (coh.table.cls == CLASSES.index("Explore"))
    n_ex = np.bincount(coh.table.attempt[ex], minlength=len(a))
    return dict(
        attempts=int(len(a)), questions=int(a["question"].nunique()), sentences=int(len(coh.sent)),
        capped=int(a["capped"].sum()), layers=coh.table.layers, experts=coh.table.experts,
        labelled_per_attempt=dict(min=float(coh.n_labelled.min()), median=float(np.median(coh.n_labelled)),
                                  mean=float(coh.n_labelled.mean()), max=float(coh.n_labelled.max())),
        attempts_without_sentences=int((coh.n_labelled == 0).sum()),
        class_counts={c: int((coh.table.cls == i).sum()) for i, c in enumerate(CLASSES)},
        explore_per_attempt=dict(median=float(np.median(n_ex)), mean=float(n_ex.mean()),
                                 frac_ge3=float((n_ex >= 3).mean()), frac_zero=float((n_ex == 0).mean())),
        class_share_vs_v3an_cls_frac_max_abs_diff=_none_nan(class_share_check(model, coh.cohort, coh)),
        n_tokens=dict(median=float(np.median(coh.table.n_tokens)),
                      p10=float(np.percentile(coh.table.n_tokens, 10)),
                      p90=float(np.percentile(coh.table.n_tokens, 90))),
        adjacent_pairs=dict(has_prev=int(coh.sent["has_prev_labelled"].sum()),
                            has_next=int(coh.sent["has_next_labelled"].sum())))


def main() -> int:
    manifest: dict = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), repeats=REPEATS, outer=OUTER,
                          kappa=F.KAPPA if hasattr(F, "KAPPA") else 50.0, models={})
    for model in ("gpt", "qwen36"):
        t0 = time.time()
        coh_a, coh_b = load_cohort(model, "A"), load_cohort(model, "B")
        info = dict(A=describe(model, coh_a), B=describe(model, coh_b))
        log(f"{model}: loaded A {info['A']['attempts']} attempts / {info['A']['sentences']} sentences, "
            f"B {info['B']['attempts']} attempts / {info['B']['sentences']} sentences")
        oof = oof_features(coh_a)
        outdir = RESULTS / model
        oof.to_parquet(outdir / "A" / "oof_features.parquet")
        bdf, binfo = b_features(model, coh_a, coh_b)
        bdf.to_parquet(outdir / "B" / "b_features.parquet")
        info["B_fit"] = binfo
        info["insample_check"] = insample_check(coh_a)
        info["oof_summary"] = {k: dict(mean=float(np.nanmean(oof[f"{k}_mean"])),
                                       sd=float(np.nanstd(oof[f"{k}_mean"])),
                                       missing=int(np.isnan(oof[f"{k}_mean"]).sum())) for k in F.FEATURES}
        info["files"] = {p: sha256_file(outdir / p) for p in (
            "A/sentences.parquet", "A/hist.npz", "A/oof_features.parquet", "B/sentences.parquet",
            "B/hist.npz", "B/b_features.parquet")}
        manifest["models"][model] = info
        log(f"{model}: done in {time.time() - t0:.0f}s")
    code = Path(__file__).resolve().parent
    manifest["code_sha256"] = {p.name: sha256_file(p) for p in sorted(code.glob("*.py"))}
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "time"})
    (RESULTS / "FEATURES_FROZEN.json").write_text(json.dumps(manifest, indent=1, default=float))
    log("wrote FEATURES_FROZEN.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
