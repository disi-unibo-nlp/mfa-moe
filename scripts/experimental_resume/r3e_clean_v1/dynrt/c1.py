"""Stage 4: C1 (class-routing mismatch) with the C8 incremental-validity machinery.

Run only after FEATURES_FROZEN.json exists; the frozen code hashes are re-verified first.
  python -m dynrt.c1 cv MODEL        A-cohort CV (primary gpt, replication qwen36), ablations, partial assoc.
  python -m dynrt.c1 within MODEL    B-cohort within-question contrasts (correct - wrong)
  python -m dynrt.c1 assemble        results/C1/results.json and summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv
from . import features as F
from .common import RESULTS, sha256_file
from .controls import ControlsBlock
from .data import load_cohort, load_outcomes, load_static

OUT = RESULTS / "C1"
FROZEN = RESULTS / "FEATURES_FROZEN.json"
ADVANCE_GAIN = 0.005
HOLM_M = 4
MODEL_SPECS = {
    "null": [],
    "base": ["controls"],
    "C1": ["controls", "routing.M_all", "routing.margin"],
    "dyn": ["routing.M_all", "routing.margin"],
    "static": ["controls", "static"],
    "static_dyn": ["controls", "static", "routing.M_all", "routing.margin"],
}
COMPARISONS = {
    "C1_primary": ("base", "C1"),
    "ablation_dynamics_only_vs_intercept": ("null", "dyn"),
    "ablation_static_vs_controls": ("base", "static"),
    "ablation_dynamics_beyond_static": ("static", "static_dyn"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(o):
    """JSON-safe copy (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def verify_frozen() -> dict:
    """The feature-defining code and tables must be exactly those written at the freeze."""
    fz = json.loads(FROZEN.read_text())
    here = Path(__file__).resolve().parent
    now = {p.name: sha256_file(p) for p in sorted(here.glob("*.py")) if p.name in fz["code_sha256"]}
    changed = sorted(k for k, v in fz["code_sha256"].items() if now.get(k) != v)
    if changed:
        raise RuntimeError(f"code changed since the feature freeze: {changed}")
    for model, info in fz["models"].items():
        for rel, sha in info["files"].items():
            if sha256_file(RESULTS / model / rel) != sha:
                raise RuntimeError(f"frozen table changed: {model}/{rel}")
    return dict(freeze_sha256=fz["sha256"], freeze_time=fz["time"])


class Prepared:
    """Cohort A inputs of one model: outcomes are read here for the first time."""

    def __init__(self, model: str):
        self.model = model
        self.coh = load_cohort(model, "A")
        att = self.coh.attempts
        out = load_outcomes(model, "A", att)
        y_raw = out["correct"].to_numpy(float)
        self.valid = np.isfinite(y_raw) & ~att["capped"].to_numpy(bool)
        self.y = np.where(self.valid, y_raw, 0.0)
        self.groups = att["question"].to_numpy()
        self.difficulty = out["difficulty"].to_numpy(float)
        n_valid_q = pd.Series(self.valid.astype(float)).groupby(self.groups).transform("sum").to_numpy()
        self.weights = np.where(self.valid, 1.0 / np.maximum(n_valid_q, 1.0), 0.0)
        self.controls = ControlsBlock(np.log(att["n_reasoning"].to_numpy(float)), self.difficulty,
                                      att["dataset"].to_numpy(), self.coh.shares[:, :6],
                                      self.coh.n_labelled, att["capped"].to_numpy(float))
        static = load_static(model, "A", att)
        self.static = cv.MatrixBlock(static.to_numpy(float), list(static.columns))
        self.routing = F.RoutingBlock(self.coh.table)

    def blocks(self) -> dict:
        return dict(controls=self.controls, static=self.static, routing=self.routing)


def reproduce_frozen_features(p: Prepared) -> dict:
    """Fold-fit routing features (repeat 0) must equal the frozen out-of-fold table exactly."""
    oof = pd.read_parquet(RESULTS / p.model / "A" / "oof_features.parquet")
    folds = cv.question_folds(p.groups, 4, 0)
    worst = 0.0
    for f in range(4):
        m = p.routing.fit_transform(np.flatnonzero(folds != f))
        for j, k in enumerate(F.FEATURES):
            a, b = m[folds == f, j], oof[f"{k}_r0"].to_numpy()[folds == f]
            same_nan = np.array_equal(np.isnan(a), np.isnan(b))
            worst = max(worst, float(np.nanmax(np.abs(a - b))) if same_nan else np.inf)
    return dict(max_abs_diff_vs_frozen_repeat0=worst)


def partial_associations(p: Prepared, n_boot: int = 1000) -> dict:
    """Partial association of the out-of-fold features with correctness given the controls."""
    from v3an import stats as vs
    oof = pd.read_parquet(RESULTS / p.model / "A" / "oof_features.parquet")
    if not (oof["attempt_id"].to_numpy() == p.coh.attempts["attempt_id"].to_numpy()).all():
        raise RuntimeError("frozen feature table is not aligned to the attempts")
    cmat = p.controls.fit_transform(np.flatnonzero(p.valid))
    keep = cmat[p.valid].std(0) > 1e-12
    cmat = cmat[:, keep]
    out = {}
    for k in F.FEATURES:
        x = oof[f"{k}_mean"].to_numpy(float)
        rows = p.valid & np.isfinite(x)
        res = vs.partial_corr(x[rows][:, None], p.y[rows], cmat[rows], clusters=p.groups[rows],
                              n_boot=n_boot, rng=np.random.default_rng(20260929))
        b, se, pv = vs.standardized_logit(x[rows], p.y[rows], cmat[rows])
        out[k] = dict(n=int(rows.sum()), n_wrong=int((p.y[rows] == 0).sum()), r=float(res["r"][0]),
                      ci_lo=float(res["ci_lo"][0]), ci_hi=float(res["ci_hi"][0]), p=float(res["p"][0]),
                      logit_per_sd=b, logit_se=se, logit_p=pv,
                      raw_r_with_y=float(np.corrcoef(x[rows], p.y[rows])[0, 1]))
    return out


def run_cv_model(model: str, workers: int, n_boot: int, lam_scale: str) -> dict:
    fz = verify_frozen()
    p = Prepared(model)
    log(f"{model} A: {len(p.y)} attempts, {int(p.valid.sum())} scored "
        f"({int((p.y[p.valid] == 0).sum())} wrong), {len(set(p.groups))} questions")
    repro = reproduce_frozen_features(p)
    log(f"frozen features reproduced: {repro}")
    if repro["max_abs_diff_vs_frozen_repeat0"] > 1e-9:
        raise RuntimeError("fold-fit features differ from the frozen table")
    if lam_scale == "sum":
        orig = cv.ridge_logistic
        cv.ridge_logistic = lambda x, y, w, lam, **k: orig(x, y, w, lam / float(w.sum()), **k)
    spec = cv.CVSpec(blocks=p.blocks(), models=MODEL_SPECS, y=p.y, groups=p.groups,
                     weights=p.weights, repeats=5, outer=4, inner=3, seed0=0, valid=p.valid)
    t0 = time.time()
    res = cv.run_cv(spec, workers=workers, log=log)
    log(f"cv done in {time.time() - t0:.0f}s")
    v = p.valid
    comps = {}
    for name, (b, a) in COMPARISONS.items():
        comps[name] = cv.evaluate({b: res["preds"][b][:, v], a: res["preds"][a][:, v]}, b, a, p.y[v],
                                  p.groups[v], p.weights[v], n_boot=n_boot)
        c = comps[name]
        log(f"{name}: gain {c['gain']:.5f} [{c['ci_lo']:.5f}, {c['ci_hi']:.5f}] p1={c['p_one_sided']:.4f}")
    out = dict(model=model, cohort="A", lam_scale=lam_scale, frozen=fz, reproduce=repro,
               n_attempts=int(len(p.y)), n_scored=int(v.sum()), n_wrong=int((p.y[v] == 0).sum()),
               n_questions=int(len(set(p.groups[v]))), comparisons=comps,
               lambdas={m: dict(median=float(np.median(l)), values=l.round(6).tolist())
                        for m, l in res["lambdas"].items()},
               seconds=time.time() - t0)
    if lam_scale == "mean":
        out["partial"] = partial_associations(p, n_boot)
        np.savez_compressed(OUT / f"{model}_A_preds.npz", attempt_id=p.coh.attempts["attempt_id"].to_numpy(),
                            y=p.y, valid=p.valid,
                            **{f"pred_{m}": res["preds"][m] for m in MODEL_SPECS})
    return out


def run_within(model: str, n_perm: int, n_boot: int) -> dict:
    """B: within-question correct - wrong contrasts of the frozen cohort-B features."""
    from v3an import stats as vs
    fz = verify_frozen()
    coh = load_cohort(model, "B")
    att = coh.attempts
    out = load_outcomes(model, "B", att)
    feats = pd.read_parquet(RESULTS / model / "B" / "b_features.parquet")
    if not (feats["attempt_id"].to_numpy() == att["attempt_id"].to_numpy()).all():
        raise RuntimeError("cohort-B feature table is not aligned")
    y = out["correct"].to_numpy(float)
    ok = np.isfinite(y) & ~att["capped"].to_numpy(bool)
    q = att["question"].to_numpy()
    log_len = np.log(att["n_reasoning"].to_numpy(float))
    res = dict(model=model, cohort="B", frozen=fz, n_attempts=int(len(att)), n_scored=int(ok.sum()),
               questions=int(len(set(q[ok]))), variants={})
    mixed = [g for g in sorted(set(q[ok])) if 0 < y[ok & (q == g)].sum() < (ok & (q == g)).sum()]
    res["mixed_questions"] = len(mixed)
    res["mixed_attempts"] = int(sum((ok & (q == g)).sum() for g in mixed))
    for variant, length in (("raw", None), ("length_adjusted", log_len)):
        rows = {}
        for k in F.FEATURES:
            x = feats[k].to_numpy(float)
            r_ = ok & np.isfinite(x)
            lens = None if length is None else length[r_]
            w = vs.within_question_contrast(x[r_][:, None], y[r_], q[r_], length=lens, n_perm=n_perm,
                                            n_boot=n_boot, rng=np.random.default_rng(20260929))
            rows[k] = dict(n=int(w["n"]), n_questions=int(w["n_questions"]), delta=float(w["delta"][0]),
                           d=float(w["d"][0]), ci_lo=float(w["ci_lo"][0]), ci_hi=float(w["ci_hi"][0]),
                           p=float(w["p"][0]), frac_pos=float(w["frac_pos"][0]))
            log(f"{model} B {variant} {k}: {rows[k]}")
        res["variants"][variant] = rows
    return res


def assemble() -> dict:
    """results.json + summary.md from the per-model outputs."""
    parts = {n: json.loads((OUT / f"{n}.json").read_text()) for n in (
        "gpt_A", "qwen36_A", "gpt_B", "qwen36_B") if (OUT / f"{n}.json").exists()}
    sens = {n: json.loads((OUT / f"{n}.json").read_text()) for n in (
        "gpt_A_sumscale", "qwen36_A_sumscale") if (OUT / f"{n}.json").exists()}
    g = parts["gpt_A"]["comparisons"]["C1_primary"]
    p_c1 = g["p_one_sided"]
    holm_bound = min(1.0, HOLM_M * p_c1)
    conds = dict(holm_upper_bound_lt_05=holm_bound < 0.05, ci_excludes_0=g["ci_lo"] > 0,
                 gain_ge_0_005=g["gain"] >= ADVANCE_GAIN)
    advances = all(conds.values())
    deprior = (not advances) and g["ci_hi"] < ADVANCE_GAIN
    verdict = "ADVANCES" if advances else ("DEPRIORITISED" if deprior else "INCONCLUSIVE")
    q = parts.get("qwen36_A", {}).get("comparisons", {}).get("C1_primary")
    rep = None
    if q:
        pg, pq = parts["gpt_A"]["partial"], parts["qwen36_A"]["partial"]
        rep = dict(gain=q["gain"], ci=[q["ci_lo"], q["ci_hi"]], p_one_sided=q["p_one_sided"],
                   same_direction_gain=bool(q["gain"] > 0),
                   same_sign_partial_r={k: bool(np.sign(pg[k]["r"]) == np.sign(pq[k]["r"]))
                                        for k in ("M_all", "margin")},
                   qwen_ci_excludes_0=bool(q["ci_lo"] > 0 or q["ci_hi"] < 0))
    res = dict(candidate="C1", plan="ANALYSIS_PLAN.md (frozen 2026-09-29)", primary=dict(
        cohort="gpt A", comparison="C vs C + {M_all, margin}", gain_nats_per_attempt=g["gain"],
        ci95=[g["ci_lo"], g["ci_hi"]], p_one_sided=p_c1, holm_family_size=HOLM_M,
        holm_adjusted_p_upper_bound=holm_bound,
        note="the other 3 primaries (C2, C6, C8) come in D2; Holm-adjusted p <= min(1, 4 p_C1), exact if "
             "C1 has the smallest p", conditions=conds), verdict=verdict,
        qwen36_replication=rep, parts=parts, sensitivity=sens)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(clean(res), indent=1))
    (OUT / "summary.md").write_text(summary_md(clean(res)))
    return res


def _fmt(x, nd=4):
    return "NA" if x is None else f"{x:.{nd}f}"


def summary_md(r: dict) -> str:
    P = r["primary"]
    L = [f"# C1 class-routing mismatch: results (D1)", "",
         f"Plan: {r['plan']}. Feature freeze sha256 `{r['parts']['gpt_A']['frozen']['freeze_sha256']}`.", "",
         "## Primary (gpt A)", "",
         f"- Gain of C + {{M_all, margin}} over C (question-weighted log-loss, nats/attempt): "
         f"**{_fmt(P['gain_nats_per_attempt'], 5)}**, 95% CI [{_fmt(P['ci95'][0], 5)}, {_fmt(P['ci95'][1], 5)}]",
         f"- Bootstrap one-sided p (gain > 0): {_fmt(P['p_one_sided'], 4)}; Holm-adjusted p (family of "
         f"{P['holm_family_size']}) <= {_fmt(P['holm_adjusted_p_upper_bound'], 4)} (others come in D2)",
         f"- Decision conditions: {json.dumps(P['conditions'])}", f"- **Verdict: {r['verdict']}**", ""]
    for name in ("gpt_A", "qwen36_A"):
        if name not in r["parts"]:
            continue
        d = r["parts"][name]
        L += [f"## {name} cross-validation ({d['n_scored']} scored attempts, {d['n_wrong']} wrong, "
              f"{d['n_questions']} questions; 4 folds x 5 repeats)", "",
              "| comparison | gain | 95% CI | p (one-sided) | dAUC [CI] | dBrier (improvement) [CI] | cal. slope base/aug |",
              "|---|---|---|---|---|---|---|"]
        for c, v in d["comparisons"].items():
            L.append(f"| {c} | {_fmt(v['gain'], 5)} | [{_fmt(v['ci_lo'], 5)}, {_fmt(v['ci_hi'], 5)}] | "
                     f"{_fmt(v['p_one_sided'], 4)} | {_fmt(v['dauc']['diff'], 4)} "
                     f"[{_fmt(v['dauc']['ci_lo'], 4)}, {_fmt(v['dauc']['ci_hi'], 4)}] | "
                     f"{_fmt(v['dbrier_improvement']['diff'], 5)} "
                     f"[{_fmt(v['dbrier_improvement']['ci_lo'], 5)}, {_fmt(v['dbrier_improvement']['ci_hi'], 5)}] | "
                     f"{_fmt(v['calibration_slope']['base']['slope'], 2)} / "
                     f"{_fmt(v['calibration_slope']['aug']['slope'], 2)} |")
        L += ["", "Partial association with correctness given the controls (out-of-fold features; "
              "question-clustered bootstrap 1,000):", "",
              "| feature | n | partial r | 95% CI | p | log-odds per SD |", "|---|---|---|---|---|---|"]
        for k, v in d["partial"].items():
            L.append(f"| {k} | {v['n']} | {_fmt(v['r'])} | [{_fmt(v['ci_lo'])}, {_fmt(v['ci_hi'])}] | "
                     f"{_fmt(v['p'], 4)} | {_fmt(v['logit_per_sd'], 3)} |")
        L.append("")
    for name in ("gpt_B", "qwen36_B"):
        if name not in r["parts"]:
            continue
        d = r["parts"][name]
        L += [f"## {name}: within-question contrast, correct - wrong (stratified permutation 10,000)", "",
              f"{d['n_scored']} scored attempts; {d['mixed_questions']} questions with both outcomes "
              f"({d['mixed_attempts']} attempts).", "",
              "| variant | feature | questions | delta | d | 95% CI (d) | p | frac. positive |", "|---|---|---|---|---|---|---|---|"]
        for var, rows in d["variants"].items():
            for k, v in rows.items():
                L.append(f"| {var} | {k} | {v['n_questions']} | {_fmt(v['delta'], 5)} | {_fmt(v['d'], 3)} | "
                         f"[{_fmt(v['ci_lo'], 3)}, {_fmt(v['ci_hi'], 3)}] | {_fmt(v['p'], 4)} | {_fmt(v['frac_pos'], 2)} |")
        L.append("")
    if r.get("qwen36_replication"):
        L += ["## qwen36 (dev+tune) replication of the primary", "", json.dumps(r["qwen36_replication"]), ""]
    L += ["Notes: the `null` model has a constant per-fold intercept, so its cross-fitted AUC is ~0.46 and "
          "its calibration slope is meaningless; the dAUC and slope columns of the dynamics-only ablation "
          "compare against that artefact and should not be read.", ""]
    for name, d in r.get("sensitivity", {}).items():
        c = d["comparisons"]["C1_primary"]
        L.append(f"- Sensitivity ({name}, ridge grid on the summed-loss scale): gain {_fmt(c['gain'], 5)} "
                 f"[{_fmt(c['ci_lo'], 5)}, {_fmt(c['ci_hi'], 5)}], p1 {_fmt(c['p_one_sided'], 4)}")
    L += validation_notes()
    return "\n".join(L) + "\n"


def validation_notes() -> list[str]:
    """Machinery validation (synthetic) and data notes, read from the saved simcheck / freeze files."""
    L = ["", "## Machinery validation and data notes", ""]
    sc = RESULTS / "simcheck.json"
    if sc.exists():
        d = json.loads(sc.read_text())
        for r in d["null"]:
            L.append(f"- Null simulation, N={r['n']}, {r['repeats']} repeats, {r['nsim']} sims: mean gain "
                     f"{r['mean_gain']:.5f}, CI covers 0 in {r['covered']}/{r['nsim']} (CI wholly below 0: "
                     f"{r['ci_hi_below_0']}, wholly above 0: {r['ci_lo_above_0']}), one-sided p<.05: "
                     f"{r['p_below_05']}.")
        fr = d.get("fresh_test_null_gain")
        if fr:
            L.append(f"- True expected null gain on 20,000 fresh rows (train n=225 / 1125): {fr['225']:.5f} / "
                     f"{fr['1125']:.5f}, i.e. the cross-validated estimate is unbiased for an estimand that is "
                     "slightly negative under the null (about -df/(2 n_train)). The literal 'null CI covers 0 in "
                     ">= 90%' criterion therefore cannot hold for this statistic; the decision rules use the "
                     "benefit side only, where the false-positive rate is 0/50.")
    fz = RESULTS / "FEATURES_FROZEN.json"
    if fz.exists():
        m = json.loads(fz.read_text())["models"]
        for model, i in m.items():
            e = i["A"]["explore_per_attempt"]
            L.append(f"- {model} A: {i['A']['attempts']} attempts, {i['A']['sentences']} labelled sentences "
                     f"(median {i['A']['labelled_per_attempt']['median']:.0f} per attempt); Explore sentences per "
                     f"attempt: median {e['median']:.0f}, share with >= 3: {e['frac_ge3']:.2f} "
                     f"(M_explore is missing for the rest).")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cv")
    a.add_argument("model", choices=("gpt", "qwen36"))
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--lam-scale", choices=("mean", "sum"), default="mean")
    b = sub.add_parser("within")
    b.add_argument("model", choices=("gpt", "qwen36"))
    b.add_argument("--n-perm", type=int, default=10000)
    b.add_argument("--n-boot", type=int, default=1000)
    sub.add_parser("assemble")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "cv":
        res = run_cv_model(args.model, args.workers, args.n_boot, args.lam_scale)
        suffix = "" if args.lam_scale == "mean" else "sumscale"
        name = f"{args.model}_A" + (f"_{suffix}" if suffix else "")
        (OUT / f"{name}.json").write_text(json.dumps(clean(res), indent=1))
    elif args.cmd == "within":
        res = run_within(args.model, args.n_perm, args.n_boot)
        (OUT / f"{args.model}_B.json").write_text(json.dumps(clean(res), indent=1))
    else:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
