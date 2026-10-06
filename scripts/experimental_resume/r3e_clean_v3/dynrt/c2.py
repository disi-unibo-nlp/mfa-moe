"""Stage 4 for C2 (previous-class persistence). Run only after FEATURES_FROZEN_D2.json exists.

  python -m dynrt.c2 cv MODEL       A-cohort CV (primary gpt, replication qwen36), direction groups, partial assoc.
  python -m dynrt.c2 within MODEL   B-cohort within-question contrasts of the frozen features
  python -m dynrt.c2 assemble       results/C2/results.json and summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv
from . import transitions as T
from .c1 import ADVANCE_GAIN, Prepared, clean, log, verify_frozen
from .common import RESULTS, sha256_file
from .data import load_cohort, load_outcomes

OUT = RESULTS / "C2"
FROZEN_D2 = RESULTS / "FEATURES_FROZEN_D2.json"
BASE = ["controls", "dyn.M_all"]
CORE = ["dyn.prev_aff", "dyn.bJSD", "dyn.has_trans"]
MODEL_SPECS = {
    "base": BASE,
    "aug": BASE + CORE,
    "base_ind": BASE + ["dyn.has_trans"],
    **{f"aug_{g}": BASE + [f"dyn.prev_aff_{g}", f"dyn.bJSD_{g}", f"dyn.has_{g}"] for g in T.GROUPS},
}
COMPARISONS = {
    "C2_primary": ("base", "aug"),
    "sensitivity_indicator_in_base": ("base_ind", "aug"),
    **{f"direction_{g}": ("base", f"aug_{g}") for g in T.GROUPS},
}
PARTIAL_FEATURES = ["prev_aff", "bJSD"] + [f"{v}_{g}" for g in T.GROUPS for v in ("prev_aff", "bJSD")]


def verify_frozen_d2() -> dict:
    """D1 and D2 freezes must be intact (feature-defining code and tables unchanged)."""
    d1 = verify_frozen()
    fz = json.loads(FROZEN_D2.read_text())
    here = Path(__file__).resolve().parent
    now = {p.name: sha256_file(p) for p in sorted(here.glob("*.py")) if p.name in fz["code_sha256"]}
    changed = sorted(k for k, v in fz["code_sha256"].items() if now.get(k) != v)
    if changed:
        raise RuntimeError(f"code changed since the D2 freeze: {changed}")
    for model, info in fz["models"].items():
        for rel, sha in info["files"].items():
            if sha256_file(RESULTS / model / rel) != sha:
                raise RuntimeError(f"D2 frozen table changed: {model}/{rel}")
    return dict(**d1, freeze_d2_sha256=fz["sha256"], freeze_d2_time=fz["time"])


def partial_generic(x: np.ndarray, y: np.ndarray, cmat: np.ndarray, groups: np.ndarray, rows: np.ndarray,
                    n_boot: int) -> dict:
    """Partial correlation of x with y given cmat (question-clustered bootstrap) on `rows`."""
    from v3an import stats as vs
    res = vs.partial_corr(x[rows][:, None], y[rows], cmat[rows], clusters=groups[rows], n_boot=n_boot,
                          rng=np.random.default_rng(20260929))
    b, se, pv = vs.standardized_logit(x[rows], y[rows], cmat[rows])
    return dict(n=int(rows.sum()), n_wrong=int((y[rows] == 0).sum()), r=float(res["r"][0]),
                ci_lo=float(res["ci_lo"][0]), ci_hi=float(res["ci_hi"][0]), p=float(res["p"][0]),
                logit_per_sd=b, logit_se=se, logit_p=pv,
                raw_r_with_y=float(np.corrcoef(x[rows], y[rows])[0, 1]))


def partial_assoc_c2(p: Prepared, n_boot: int) -> dict:
    oof = pd.read_parquet(RESULTS / p.model / "A" / "oof_dyn.parquet")
    if not (oof["attempt_id"].to_numpy() == p.coh.attempts["attempt_id"].to_numpy()).all():
        raise RuntimeError("frozen dyn table is not aligned")
    cmat = np.column_stack([p.controls.fit_transform(np.flatnonzero(p.valid)), oof["M_all_mean"].to_numpy()])
    cmat = np.where(np.isnan(cmat), np.nanmean(cmat, axis=0), cmat)
    cmat = cmat[:, cmat[p.valid].std(0) > 1e-12]
    out = {}
    for k in PARTIAL_FEATURES:
        x = oof[f"{k}_mean"].to_numpy(float)
        rows = p.valid & np.isfinite(x)
        out[k] = partial_generic(x, p.y, cmat, p.groups, rows, n_boot) if rows.sum() > 30 else dict(n=int(rows.sum()))
    return out


def run_cv_model(model: str, workers: int, n_boot: int, lam_scale: str) -> dict:
    fz = verify_frozen_d2()
    p = Prepared(model)
    pairs = T.build_pairs(p.coh)
    blocks = dict(controls=p.controls, dyn=T.DynBlock(p.coh.table, pairs))
    log(f"{model} A: {len(p.y)} attempts, {int(p.valid.sum())} scored ({int((p.y[p.valid] == 0).sum())} wrong); "
        f"{len(pairs.row)} eligible pairs, {int((np.bincount(pairs.attempt, minlength=len(p.y)) > 0).sum())} attempts with a pair")
    if lam_scale == "sum":
        orig = cv.ridge_logistic
        cv.ridge_logistic = lambda x, y, w, lam, **k: orig(x, y, w, lam / float(w.sum()), **k)
    spec = cv.CVSpec(blocks=blocks, models=MODEL_SPECS, y=p.y, groups=p.groups, weights=p.weights,
                     repeats=5, outer=4, inner=3, seed0=0, valid=p.valid)
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
    cnt = np.bincount(pairs.attempt, minlength=len(p.y))
    out = dict(model=model, cohort="A", lam_scale=lam_scale, frozen=fz, n_attempts=int(len(p.y)),
               n_scored=int(v.sum()), n_wrong=int((p.y[v] == 0).sum()), n_questions=int(len(set(p.groups[v]))),
               n_pairs=int(len(pairs.row)), attempts_with_pair=int((cnt > 0).sum()),
               scored_with_pair=int(((cnt > 0) & v).sum()), comparisons=comps,
               lambdas={m: dict(median=float(np.median(l)), values=l.round(6).tolist())
                        for m, l in res["lambdas"].items()}, seconds=time.time() - t0)
    if lam_scale == "mean":
        out["partial"] = partial_assoc_c2(p, n_boot)
        np.savez_compressed(OUT / f"{model}_A_preds.npz", attempt_id=p.coh.attempts["attempt_id"].to_numpy(),
                            y=p.y, valid=p.valid, **{f"pred_{m}": res["preds"][m] for m in MODEL_SPECS})
    return out


def run_within(model: str, n_perm: int, n_boot: int) -> dict:
    """B: within-question correct - wrong contrasts of the frozen cohort-B C2 features."""
    from v3an import stats as vs
    fz = verify_frozen_d2()
    coh = load_cohort(model, "B")
    att = coh.attempts
    out = load_outcomes(model, "B", att)
    feats = pd.read_parquet(RESULTS / model / "B" / "b_dyn.parquet")
    if not (feats["attempt_id"].to_numpy() == att["attempt_id"].to_numpy()).all():
        raise RuntimeError("cohort-B table is not aligned")
    y = out["correct"].to_numpy(float)
    ok = np.isfinite(y) & ~att["capped"].to_numpy(bool)
    q = att["question"].to_numpy()
    log_len = np.log(att["n_reasoning"].to_numpy(float))
    mixed = [g for g in sorted(set(q[ok])) if 0 < y[ok & (q == g)].sum() < (ok & (q == g)).sum()]
    res = dict(model=model, cohort="B", frozen=fz, n_attempts=int(len(att)), n_scored=int(ok.sum()),
               questions=int(len(set(q[ok]))), mixed_questions=len(mixed), variants={})
    for variant, length in (("raw", None), ("length_adjusted", log_len)):
        rows = {}
        for k in PARTIAL_FEATURES:
            x = feats[k].to_numpy(float)
            r_ = ok & np.isfinite(x)
            if r_.sum() < 6:
                rows[k] = dict(n=int(r_.sum()))
                continue
            lens = None if length is None else length[r_]
            w = vs.within_question_contrast(x[r_][:, None], y[r_], q[r_], length=lens, n_perm=n_perm,
                                            n_boot=n_boot, rng=np.random.default_rng(20260929))
            rows[k] = dict(n=int(w["n"]), n_questions=int(w["n_questions"]), delta=float(w["delta"][0]),
                           d=float(w["d"][0]), ci_lo=float(w["ci_lo"][0]), ci_hi=float(w["ci_hi"][0]),
                           p=float(w["p"][0]), frac_pos=float(w["frac_pos"][0]))
            log(f"{model} B {variant} {k}: {rows[k]}")
        res["variants"][variant] = rows
    return res


def _fmt(x, nd=4):
    return "NA" if x is None else f"{x:.{nd}f}"


def cv_table(d: dict) -> list[str]:
    L = ["| comparison | gain | 95% CI | p (one-sided) | dAUC [CI] | dBrier (improvement) [CI] |", "|---|---|---|---|---|---|"]
    for c, v in d["comparisons"].items():
        L.append(f"| {c} | {_fmt(v['gain'], 5)} | [{_fmt(v['ci_lo'], 5)}, {_fmt(v['ci_hi'], 5)}] | "
                 f"{_fmt(v['p_one_sided'], 4)} | {_fmt(v['dauc']['diff'], 4)} "
                 f"[{_fmt(v['dauc']['ci_lo'], 4)}, {_fmt(v['dauc']['ci_hi'], 4)}] | "
                 f"{_fmt(v['dbrier_improvement']['diff'], 5)} "
                 f"[{_fmt(v['dbrier_improvement']['ci_lo'], 5)}, {_fmt(v['dbrier_improvement']['ci_hi'], 5)}] |")
    return L


def assemble() -> dict:
    parts = {n: json.loads((OUT / f"{n}.json").read_text()) for n in ("gpt_A", "qwen36_A", "gpt_B", "qwen36_B")
             if (OUT / f"{n}.json").exists()}
    sens = {n: json.loads((OUT / f"{n}.json").read_text()) for n in ("gpt_A_sumscale", "qwen36_A_sumscale")
            if (OUT / f"{n}.json").exists()}
    g = parts["gpt_A"]["comparisons"]["C2_primary"]
    verdict_inputs = dict(gain=g["gain"], ci=[g["ci_lo"], g["ci_hi"]], p_one_sided=g["p_one_sided"])
    rep = {}
    if "qwen36_A" in parts:
        q = parts["qwen36_A"]["comparisons"]["C2_primary"]
        rep["qwen36_A"] = dict(gain=q["gain"], ci=[q["ci_lo"], q["ci_hi"]], p_one_sided=q["p_one_sided"],
                               same_direction_gain=bool(q["gain"] > 0))
    if "gpt_B" in parts:
        pa = parts["gpt_A"]["partial"]
        rep["gpt_B_same_sign_as_gpt_A_partial"] = {
            f"{k}_{var}": bool(np.sign(pa[k].get("r", np.nan)) == np.sign(parts["gpt_B"]["variants"][var][k].get("d", np.nan)))
            for k in ("prev_aff", "bJSD") for var in ("raw", "length_adjusted")
            if k in pa and "r" in pa[k] and "d" in parts["gpt_B"]["variants"][var].get(k, {})}
    res = dict(candidate="C2", primary=dict(cohort="gpt A", comparison="C + M_all + {prev_aff, bJSD} vs C + M_all",
                                            **verdict_inputs), replication=rep, parts=parts, sensitivity=sens)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(clean(res), indent=1))
    (OUT / "summary.md").write_text(summary_md(clean(res)))
    return res


def summary_md(r: dict) -> str:
    P = r["primary"]
    L = ["# C2 previous-class persistence: results (D2)", "",
         f"Primary (gpt A, {P['comparison']}): gain **{_fmt(P['gain'], 5)}** nats/attempt, 95% CI "
         f"[{_fmt(P['ci'][0], 5)}, {_fmt(P['ci'][1], 5)}], one-sided p {_fmt(P['p_one_sided'], 4)}. "
         "Verdict and Holm adjustment are in ../SUMMARY.md.", ""]
    for name in ("gpt_A", "qwen36_A"):
        if name not in r["parts"]:
            continue
        d = r["parts"][name]
        L += [f"## {name}: cross-validation ({d['n_scored']} scored, {d['n_wrong']} wrong; {d['n_pairs']} eligible pairs "
              f"in {d['attempts_with_pair']} attempts, {d['scored_with_pair']} of the scored ones)", ""] + cv_table(d)
        L += ["", "Partial association with correctness given C + M_all (out-of-fold features, question-clustered "
              "bootstrap 1,000):", "", "| feature | n | partial r | 95% CI | p |", "|---|---|---|---|---|"]
        for k, v in d["partial"].items():
            if "r" in v:
                L.append(f"| {k} | {v['n']} | {_fmt(v['r'])} | [{_fmt(v['ci_lo'])}, {_fmt(v['ci_hi'])}] | {_fmt(v['p'], 4)} |")
            else:
                L.append(f"| {k} | {v['n']} | too few | | |")
        L.append("")
    for name in ("gpt_B", "qwen36_B"):
        if name not in r["parts"]:
            continue
        d = r["parts"][name]
        L += [f"## {name}: within-question contrast, correct - wrong (stratified permutation 10,000)", "",
              f"{d['n_scored']} scored attempts, {d['mixed_questions']} questions with both outcomes.", "",
              "| variant | feature | attempts | questions | delta | d | 95% CI (d) | p |", "|---|---|---|---|---|---|---|---|"]
        for var, rows in d["variants"].items():
            for k, v in rows.items():
                if "d" in v:
                    L.append(f"| {var} | {k} | {v['n']} | {v['n_questions']} | {_fmt(v['delta'], 5)} | {_fmt(v['d'], 3)} | "
                             f"[{_fmt(v['ci_lo'], 3)}, {_fmt(v['ci_hi'], 3)}] | {_fmt(v['p'], 4)} |")
                else:
                    L.append(f"| {var} | {k} | {v['n']} | too few | | | | |")
        L.append("")
    for name, d in r.get("sensitivity", {}).items():
        c = d["comparisons"]["C2_primary"]
        L.append(f"- Sensitivity ({name}, ridge grid on the summed-loss scale): gain {_fmt(c['gain'], 5)} "
                 f"[{_fmt(c['ci_lo'], 5)}, {_fmt(c['ci_hi'], 5)}], p1 {_fmt(c['p_one_sided'], 4)}")
    L += ["", f"Replication: {json.dumps(r['replication'])}"]
    return "\n".join(L) + "\n"


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
        name = f"{args.model}_A" + ("" if args.lam_scale == "mean" else "_sumscale")
        (OUT / f"{name}.json").write_text(json.dumps(clean(res), indent=1))
    elif args.cmd == "within":
        res = run_within(args.model, args.n_perm, args.n_boot)
        (OUT / f"{args.model}_B.json").write_text(json.dumps(clean(res), indent=1))
    else:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
