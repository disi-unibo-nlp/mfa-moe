"""D2 feature freeze: out-of-fold C2 features (gpt A, qwen36 A), cohort-B features, C6 x (qwen36 A).

Same folds as D1 (question-grouped, 4 folds x 5 repeats, seeds 0-4). No outcome is read.
CLI: python -m dynrt.freeze_d2  -> results/FEATURES_FROZEN_D2.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import transitions as T
from .c1 import verify_frozen
from .common import CLASSES, RESULTS, digest, sha256_file
from .cv import question_folds
from .data import load_cohort

REPEATS, OUTER = 5, 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oof_dyn(coh, pt: T.PairTable) -> pd.DataFrame:
    q = coh.attempts["question"].to_numpy()
    n = len(q)
    out = {"attempt_id": coh.attempts["attempt_id"].to_numpy(), "question": q}
    for r in range(REPEATS):
        folds = question_folds(q, OUTER, r)
        cols = {k: np.full(n, np.nan) for k in T.DYN_NAMES}
        for f in range(OUTER):
            feats, _, _ = T.dyn_features(coh.table, pt, folds != f)
            for k in T.DYN_NAMES:
                cols[k][folds == f] = feats[k][folds == f]
            log(f"dyn oof repeat {r} fold {f}")
        out[f"fold_r{r}"] = folds
        for k in T.DYN_NAMES:
            out[f"{k}_r{r}"] = cols[k]
    for k in T.DYN_NAMES:
        arr = np.column_stack([out[f"{k}_r{r}"] for r in range(REPEATS)])
        with np.errstate(all="ignore"):
            out[f"{k}_mean"] = np.nanmean(arr, axis=1)
    return pd.DataFrame(out)


def b_dyn(coh_a, coh_b, pt_a: T.PairTable, pt_b: T.PairTable) -> tuple[pd.DataFrame, dict]:
    """Cohort B features from a fit on cohort A attempts of questions absent from B."""
    train = ~coh_a.attempts["question"].isin(set(coh_b.attempts["question"])).to_numpy()
    _, pf, fitted = T.dyn_features(coh_a.table, pt_a, train)
    d1 = F.apply_fitted(fitted, coh_b.table)
    c2 = T.aggregate_pairs(T.pair_features(pf, pt_b, coh_b.table), pt_b)
    feats = {**d1, **c2}
    df = pd.DataFrame({"attempt_id": coh_b.attempts["attempt_id"].to_numpy(),
                       "question": coh_b.attempts["question"].to_numpy(),
                       **{k: feats[k] for k in T.DYN_NAMES}})
    return df, dict(train_attempts=int(train.sum()))


def describe_pairs(coh, pt: T.PairTable) -> dict:
    n_att = coh.table.n_attempts
    cnt = np.bincount(pt.attempt, minlength=n_att)
    ab = {f"{CLASSES[a]}->{CLASSES[b]}": int(((pt.a == a) & (pt.b == b)).sum())
          for a in range(7) for b in range(7) if a != b}
    top = dict(sorted(ab.items(), key=lambda kv: -kv[1])[:8])
    return dict(pairs=int(len(pt.row)), attempts=int(n_att), attempts_with_pair=int((cnt > 0).sum()),
                frac_attempts_with_pair=float((cnt > 0).mean()),
                pairs_per_attempt=dict(median=float(np.median(cnt)), mean=float(cnt.mean()),
                                       max=int(cnt.max())),
                groups={g: int((pt.group == i).sum()) for i, g in enumerate(T.GROUPS)},
                attempts_with_group={g: int((np.bincount(pt.attempt[pt.group == i], minlength=n_att) > 0).sum())
                                     for i, g in enumerate(T.GROUPS)},
                bjsd=dict(mean=float(pt.bjsd.mean()), sd=float(pt.bjsd.std())),
                top_transitions=top)


def main() -> int:
    d1 = verify_frozen()
    log(f"D1 freeze verified: {d1}")
    manifest: dict = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), d1_freeze=d1, models={})
    for model in ("gpt", "qwen36"):
        t0 = time.time()
        coh_a, coh_b = load_cohort(model, "A"), load_cohort(model, "B")
        pt_a, pt_b = T.build_pairs(coh_a), T.build_pairs(coh_b)
        info = dict(pairs_A=describe_pairs(coh_a, pt_a), pairs_B=describe_pairs(coh_b, pt_b))
        oof = oof_dyn(coh_a, pt_a)
        d1_oof = pd.read_parquet(RESULTS / model / "A" / "oof_features.parquet")
        same = all(np.array_equal(oof[f"{k}_r{r}"].to_numpy(), d1_oof[f"{k}_r{r}"].to_numpy(), equal_nan=True)
                   for k in F.FEATURES for r in range(REPEATS))
        info["dyn_reproduces_d1_frozen_features"] = bool(same)
        if not same:
            raise RuntimeError("DynBlock D1 columns differ from the D1 frozen features")
        oof.to_parquet(RESULTS / model / "A" / "oof_dyn.parquet")
        bdf, binfo = b_dyn(coh_a, coh_b, pt_a, pt_b)
        bdf.to_parquet(RESULTS / model / "B" / "b_dyn.parquet")
        info["B_fit"] = binfo
        info["oof_summary"] = {k: dict(mean=float(np.nanmean(oof[f"{k}_mean"])),
                                       sd=float(np.nanstd(oof[f"{k}_mean"])),
                                       missing=int(np.isnan(oof[f"{k}_mean"]).sum()))
                               for k in ("prev_aff", "bJSD", "prev_aff_entry", "prev_aff_exit", "prev_aff_other")}
        files = {"A/oof_dyn.parquet": sha256_file(RESULTS / model / "A" / "oof_dyn.parquet"),
                 "B/b_dyn.parquet": sha256_file(RESULTS / model / "B" / "b_dyn.parquet")}
        if model == "qwen36":
            onset_dir = RESULTS / model / "A" / "onset"
            onset = T.load_onset(coh_a, onset_dir)
            block = T.OnsetBlock(coh_a.table, onset)
            q = coh_a.attempts["question"].to_numpy()
            c6 = {"attempt_id": coh_a.attempts["attempt_id"].to_numpy(), "has_decision": onset.has,
                  "j": onset.j, "marker": onset.marker, "burden": onset.burden}
            for r in range(REPEATS):
                folds = question_folds(q, OUTER, r)
                x = np.full(len(q), np.nan)
                for f in range(OUTER):
                    x[folds == f] = block.fit_transform(np.flatnonzero(folds != f))[folds == f, 0]
                    log(f"c6 oof repeat {r} fold {f}")
                c6[f"x_r{r}"] = x
            c6df = pd.DataFrame(c6)
            with np.errstate(all="ignore"):
                c6df["x_mean"] = np.nanmean(np.column_stack([c6df[f"x_r{r}"] for r in range(REPEATS)]), axis=1)
            c6df.to_parquet(RESULTS / model / "A" / "c6_oof.parquet")
            files["A/c6_oof.parquet"] = sha256_file(RESULTS / model / "A" / "c6_oof.parquet")
            for name in ("onset_meta.parquet", "onset_hist.npz"):
                files[f"A/onset/{name}"] = sha256_file(onset_dir / name)
            marks = np.bincount(onset.marker[onset.has], minlength=6)
            info["c6"] = dict(attempts=int(len(q)), with_decision=int(onset.has.sum()),
                              j_quantiles={str(p): float(np.percentile(onset.j[onset.has], p)) for p in (10, 50, 90)},
                              marker_counts=dict(zip(["can we", "could", "is it possible", "maybe", "or maybe", "what"],
                                                     marks.tolist())),
                              burden_available=int(np.isfinite(onset.burden).sum()),
                              burden_mean=float(np.nanmean(onset.burden)),
                              x_sd=float(np.nanstd(c6df["x_mean"])))
        info["files"] = files
        manifest["models"][model] = info
        log(f"{model}: done in {time.time() - t0:.0f}s")
    code = Path(__file__).resolve().parent
    manifest["code_sha256"] = {p.name: sha256_file(p) for p in sorted(code.glob("*.py"))
                               if p.name not in ("c2.py", "c6.py", "summary.py")}
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "time"})
    (RESULTS / "FEATURES_FROZEN_D2.json").write_text(json.dumps(manifest, indent=1, default=float))
    log("wrote FEATURES_FROZEN_D2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
