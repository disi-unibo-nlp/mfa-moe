"""P-A1 driver (registered test, Addendum 3): routing-conditioned next-class forecasting, GPT cohort A.

  python -m dynrt.d3a_run calibrate     text-block scale from baseline-only inner CV of one training set
  python -m dynrt.d3a_run cv            main C8 cross-fit: base, aug (last-16 scores), aug_ant (anticipation)
  python -m dynrt.d3a_run nullsim       useless 7-dim augmentation on real features (benefit-side FP rate)
  python -m dynrt.d3a_run assemble      results/A1/results.json and summary.md
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from . import d3a_cv, d3a_eval
from .common import CLASSES, RESULTS, digest, sha256_file
from .cv import GRID, question_folds

OUT = RESULTS / "A1"
ANTIC = OUT / "antic"
ADVANCE_GAIN = 0.005
HOLM_M = 3
SCALES = (1.0, 3.0, 10.0, 30.0, 100.0)
FROZEN_FILES = ("FEATURES_FROZEN.json", "FEATURES_FROZEN_D2.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_freezes() -> dict:
    """D1 and D2 freezes must still verify (frozen code and tables untouched by this task)."""
    from .c2 import verify_frozen_d2
    return verify_frozen_d2()


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {p.name: sha256_file(p) for p in sorted(here.glob("d3a_*.py"))}


def input_hashes() -> dict:
    files = [OUT / "pairs.parquet", OUT / "text_counts.npz", RESULTS / "gpt/A/sentences.parquet",
             RESULTS / "gpt/A/hist.npz"]
    if (ANTIC / "antic_hist.npz").exists():
        files.append(ANTIC / "antic_hist.npz")
    return {str(p.relative_to(RESULTS)): sha256_file(p) for p in files}


# ------------------------------------------------------------------------------ calibration

def _scale_task(scale: float) -> dict:
    d = globals()["_D"]
    d.text_scale = scale
    grid = GRID.copy()
    train = np.flatnonzero(d.outer_folds[0] != 0)
    inner = question_folds(d.groups[train], d3a_cv.INNER, 0)
    losses, sizes, stats = np.zeros((d3a_cv.INNER, len(grid))), np.zeros(d3a_cv.INNER), []
    t0 = time.time()
    for j in range(d3a_cv.INNER):
        tr, va = train[inner != j], train[inner == j]
        blocks = d3a_cv.FoldBlocks(d, tr, [va])
        losses[j] = d3a_cv.fit_path(blocks, ["base"], grid[::-1], stats=stats)[::-1]
        sizes[j] = d.w[va].sum()
    mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
    lam, best = d3a_cv.choose_penalty(mean, grid)
    st = np.asarray(stats, float)
    return dict(scale=scale, curve=mean.tolist(), best_lambda=lam, best_loss=float(mean[best]),
                mean_iter=float(st[:, 0].mean()), frac_converged=float(st[:, 1].mean()),
                seconds=time.time() - t0)


def _dense_only_loss(d: d3a_cv.Data) -> dict:
    """Reference inner-CV losses of the baseline WITHOUT text and of the class-prior (no covariates)."""
    from .cv import Scaler
    from .d3a_softmax import fit_softmax, predict_log_proba, weighted_nll
    grid = GRID.copy()
    train = np.flatnonzero(d.outer_folds[0] != 0)
    inner = question_folds(d.groups[train], d3a_cv.INNER, 0)
    res = {}
    for name, cols in (("dense_only", slice(None)), ("current_class_only", slice(0, 7))):
        losses, sizes = np.zeros((d3a_cv.INNER, len(grid))), np.zeros(d3a_cv.INNER)
        for j in range(d3a_cv.INNER):
            tr, va = train[inner != j], train[inner == j]
            sc = Scaler.fit(d.base[tr][:, cols])
            xt, xv = sc.transform(d.base[tr][:, cols]), sc.transform(d.base[va][:, cols])
            for gi, lam in enumerate(grid):
                fit = fit_softmax(xt, None, d.y[tr], d.w[tr], float(lam), d3a_cv.K)
                losses[j, gi] = weighted_nll(predict_log_proba(fit, xv, None), d.y[va], d.w[va])
            sizes[j] = d.w[va].sum()
        mean = (losses * sizes[:, None]).sum(0) / sizes.sum()
        res[name] = dict(best_loss=float(mean.min()), best_lambda=float(grid[int(mean.argmin())]))
    return res


def calibrate(workers: int) -> dict:
    fz = verify_freezes()
    d = d3a_cv.load_data(antic_dir=None)
    globals()["_D"] = d
    with mp.get_context("fork").Pool(min(workers, len(SCALES))) as pool:
        rows = pool.map(_scale_task, SCALES)
    ref = _dense_only_loss(d)
    best = min(rows, key=lambda r: r["best_loss"])
    out = dict(rule="baseline-only inner-CV log-loss on the training set of (repeat 0, fold 0); no held-out fold, "
                    "no augmentation outcome involved", scales=rows, reference=ref, chosen_scale=best["scale"],
               n_pairs=int(d.n), frozen=fz, code=code_hashes())
    (OUT / "scale_calibration.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        log(f"scale {r['scale']:6.1f}: best inner loss {r['best_loss']:.5f} at lam {r['best_lambda']:g} "
            f"(mean iter {r['mean_iter']:.0f}, converged {r['frac_converged']:.2f}, {r['seconds']:.0f}s)")
    log(f"reference: {ref}")
    log(f"chosen scale {best['scale']}")
    return out


def chosen_scale() -> float:
    f = OUT / "scale_calibration.json"
    if not f.exists():
        raise FileNotFoundError("run `calibrate` first")
    return float(json.loads(f.read_text())["chosen_scale"])


def _clean(o):
    from .c1 import clean
    return clean(o)


# --------------------------------------------------------------------------------- main CV

def run_main_cv(workers: int, n_boot: int) -> dict:
    fz = verify_freezes()
    scale = chosen_scale()
    d = d3a_cv.load_data(antic_dir=ANTIC, text_scale=scale)
    log(f"P-A1 cv: {d.n} pairs, {len(set(d.groups))} questions, text scale {scale}, "
        f"{int((d.outer_folds[0] == 0).sum())} pairs in outer fold 0 of repeat 0")
    t0 = time.time()
    res = d3a_cv.run_cv(d, d3a_cv.MODELS, workers=workers)
    log(f"cv done in {time.time() - t0:.0f}s; diag {res['diag']}")
    np.savez_compressed(OUT / "preds.npz", y=d.y, a=d.a, w=d.w, groups=d.groups, attempt=d.attempt,
                        **{f"logp_{m}": res["logp"][m] for m in d3a_cv.MODELS})
    lp = res["logp"]
    sw = (d.a != d.y).astype(int)
    ev = {}
    for name, (b, a) in dict(primary=("base", "aug"), anticipation=("base", "aug_ant")).items():
        ev[name] = d3a_eval.evaluate(lp[b], lp[a], d.y, d.groups, d.w, n_boot=n_boot, per_class=d.a)
        ev[name]["by_transition_type"] = d3a_eval.evaluate(lp[b], lp[a], d.y, d.groups, d.w, n_boot=n_boot,
                                                            per_class=sw, n_class=2, full=False)
        g = ev[name]["gain"]
        log(f"{name}: gain {g['point']:.5f} [{g['ci_lo']:.5f}, {g['ci_hi']:.5f}] p1={g['p_one_sided']:.4f} "
            f"positive repeats {ev[name]['n_positive_repeats']}/5")
    ev["aug_ant_vs_aug"] = d3a_eval.evaluate(lp["aug"], lp["aug_ant"], d.y, d.groups, d.w, n_boot=n_boot, full=False)
    out = dict(cohort="gpt A (GPT B questions excluded)", text_scale=scale, n_pairs=int(d.n),
               n_questions=int(len(set(d.groups))), models={m: k for m, k in d3a_cv.MODELS.items()},
               dense_names=d.base_names, evaluation=ev, diag=res["diag"],
               lambdas={m: v.tolist() for m, v in res["lambdas"].items()},
               inner_curves={m: v.tolist() for m, v in res["curves"].items()},
               n_windows_missing=int((~d.has_ant).sum()), seconds=time.time() - t0)
    out["frozen_after"] = verify_freezes()
    out["frozen_before"] = fz
    out["inputs"], out["code"] = input_hashes(), code_hashes()
    (OUT / "cv.json").write_text(json.dumps(_clean(out), indent=1))
    return out


# -------------------------------------------------------------------------- null simulation

NOISE_SEED = 20260930


def run_nullsim(nsim: int, workers: int, repeats: int, n_boot: int, fixed_lambda: bool, n_chunks: int,
                kind: str = "noise") -> dict:
    """Useless augmentation on the real features. Protocol `full`: the complete inner-CV protocol, with its own
    baseline predictions (identical procedure to the main run, `repeats` repeats); `fixed`: base penalty per fold."""
    fz = verify_freezes()
    scale = chosen_scale()
    d = d3a_cv.load_data(antic_dir=None, text_scale=scale)
    for s in range(nsim):
        if kind == "noise":
            d.extra[f"noise{s}"] = np.random.default_rng(NOISE_SEED + s).standard_normal((d.n, d3a_cv.K))
        else:
            d.perm[f"lastperm{s}"] = d3a_cv.within_class_permutation(d.a, NOISE_SEED + s)
    aug_key = "noise" if kind == "noise" else "lastperm"
    models = {f"null{s}": ["base", f"{aug_key}{s}"] for s in range(nsim)}
    fixed = None
    if fixed_lambda:
        main = np.load(OUT / "preds.npz")
        cvj = json.loads((OUT / "cv.json").read_text())
        if not np.array_equal(main["y"], d.y):
            raise RuntimeError("main predictions are not aligned to the pair table")
        base = main["logp_base"][:repeats]
        fixed = np.asarray(cvj["lambdas"]["base"])
    else:
        models["base"] = ["base"]
    t0 = time.time()
    res = d3a_cv.run_cv(d, models, workers=workers, repeats=repeats, fixed_lambda=fixed, n_chunks=n_chunks)
    log(f"nullsim cv done in {time.time() - t0:.0f}s; diag {res['diag']}")
    if not fixed_lambda:
        base = res["logp"]["base"]
    rows = []
    for s in range(nsim):
        e = d3a_eval.evaluate(base, res["logp"][f"null{s}"], d.y, d.groups, d.w, n_boot=n_boot, full=False)
        g = e["gain"]
        rows.append(dict(sim=s, gain=g["point"], ci_lo=g["ci_lo"], ci_hi=g["ci_hi"], boot_sd=g["boot_sd"],
                         p_one_sided=g["p_one_sided"], n_positive_repeats=e["n_positive_repeats"],
                         mean_lambda=float(np.mean(res["lambdas"][f"null{s}"]))))
    aug_text = ("7 iid N(0,1) columns per pair (seeds %d..%d), standardised in-fold" if kind == "noise" else
                "real last-16 histograms shuffled among pairs of the same current class (seeds %d..%d), "
                "scored with the in-fold profiles") % (NOISE_SEED, NOISE_SEED + nsim - 1)
    out = dict(protocol=dict(kind=kind, nsim=nsim, repeats=repeats, n_boot=n_boot, lambda_from_base=bool(fixed_lambda),
                             n_chunks=n_chunks, augmentation=aug_text),
               summary=null_summary(rows), sims=rows, diag=res["diag"], seconds=time.time() - t0,
               base_mean_lambda=None if fixed_lambda else float(np.mean(res["lambdas"]["base"])),
               frozen=fz, inputs=input_hashes(), code=code_hashes())
    (OUT / f"nullsim_{kind}_{'fixed' if fixed_lambda else 'full'}.json").write_text(
        json.dumps(_clean(out), indent=1))
    log(json.dumps(_clean(out["summary"])))
    return out


def null_summary(rows: list[dict]) -> dict:
    g = np.array([r["gain"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    p = np.array([r["p_one_sided"] for r in rows])
    sd = np.array([r["boot_sd"] for r in rows])
    npos = np.array([r["n_positive_repeats"] for r in rows])
    advance = (p * HOLM_M < 0.05) & (lo > 0) & (npos >= 4) & (g >= ADVANCE_GAIN)
    return dict(nsim=len(rows), mean_gain=float(g.mean()), sd_gain=float(g.std(ddof=1)),
                mean_boot_sd=float(sd.mean()), sd_over_boot_sd=float(g.std(ddof=1) / sd.mean()),
                min_gain=float(g.min()), max_gain=float(g.max()),
                benefit_fp_ci_lo_above_0=int((lo > 0).sum()), benefit_fp_p_below_05=int((p < 0.05).sum()),
                benefit_fp_advance_rule=int(advance.sum()), ci_hi_below_0=int((hi < 0).sum()),
                ci_covers_0=int(((lo <= 0) & (hi >= 0)).sum()), ci_hi_below_threshold=int((hi < ADVANCE_GAIN).sum()))


# ---------------------------------------------------------------------------------- assemble

def verdict(g: dict, n_pos: int) -> dict:
    """Addendum 3 decision rule for P-A1 (Holm across P-A1, P-B1, P-B4; here the conservative bound 3p)."""
    conds = dict(holm_upper_bound_lt_05=bool(min(1.0, HOLM_M * g["p_one_sided"]) < 0.05),
                 ci_excludes_0=bool(g["ci_lo"] > 0), positive_in_ge_4_of_5_repeats=bool(n_pos >= 4),
                 gain_ge_0_005=bool(g["point"] >= ADVANCE_GAIN))
    advances = all(conds.values())
    deprior = (not advances) and g["ci_hi"] < ADVANCE_GAIN
    return dict(conditions=conds, verdict="ADVANCES" if advances else ("DEPRIORITISED" if deprior else "INCONCLUSIVE"),
                p_raw=g["p_one_sided"], holm_adjusted_upper_bound=min(1.0, HOLM_M * g["p_one_sided"]))


def _f(x, nd=5):
    return "NA" if x is None else f"{x:.{nd}f}"


def assemble() -> dict:
    cv = json.loads((OUT / "cv.json").read_text())
    pairs = json.loads((OUT / "pair_summary.json").read_text())
    scale = json.loads((OUT / "scale_calibration.json").read_text())
    nulls = {f.stem.removeprefix("nullsim_"): json.loads(f.read_text()) for f in sorted(OUT.glob("nullsim_*.json"))}
    ev = cv["evaluation"]
    prim = ev["primary"]
    vd = verdict(prim["gain"], prim["n_positive_repeats"])
    ant = ev["anticipation"]
    vd_ant = verdict(ant["gain"], ant["n_positive_repeats"])
    res = dict(test="P-A1", plan="ANALYSIS_PLAN.md Addendum 3 (frozen)", cohort=cv["cohort"],
               pairs=pairs, design=dict(text_scale=cv["text_scale"], scale_calibration=scale, models=cv["models"],
                                        dense_names=cv["dense_names"]),
               primary=dict(comparison="base vs base + 7 last-16 class-profile scores d_c", **prim["gain"],
                            logloss_base=prim["logloss_base"], logloss_aug=prim["logloss_aug"],
                            per_repeat_gain=prim["per_repeat_gain"], n_positive_repeats=prim["n_positive_repeats"],
                            pair_weighted=prim["pair_weighted_gain"], **vd),
               secondary=dict(per_source_class=prim["per_source_class"], by_transition_type=prim["by_transition_type"],
                              calibration=prim["calibration"], brier_gain=prim["brier_gain"],
                              accuracy_gain=prim["accuracy_gain"],
                              anticipation=dict(comparison="base vs base + 7 scores of the window ending 16 tokens "
                                                           "before the sentence end", **ant["gain"],
                                                per_repeat_gain=ant["per_repeat_gain"],
                                                n_positive_repeats=ant["n_positive_repeats"],
                                                pair_weighted=ant["pair_weighted_gain"],
                                                per_source_class=ant["per_source_class"],
                                                by_transition_type=ant["by_transition_type"],
                                                calibration=ant["calibration"], brier_gain=ant["brier_gain"],
                                                accuracy_gain=ant["accuracy_gain"],
                                                windows_missing=cv["n_windows_missing"], **{
                                                    "verdict_rule_applied": vd_ant["verdict"],
                                                    "holm_adjusted_upper_bound": vd_ant["holm_adjusted_upper_bound"]}),
                              aug_ant_vs_aug=ev["aug_ant_vs_aug"]["gain"]),
               null_calibration={n: dict(protocol=v["protocol"], summary=v["summary"]) for n, v in nulls.items()},
               diagnostics=dict(cv=cv["diag"], lambdas={m: np.median(np.asarray(v)).item() for m, v in
                                                        cv["lambdas"].items()},
                                frozen_before=cv["frozen_before"], frozen_after=cv["frozen_after"]),
               inputs=cv["inputs"], code=code_hashes())
    (OUT / "results.json").write_text(json.dumps(_clean(res), indent=1))
    (OUT / "summary.md").write_text(summary_md(_clean(res)))
    return res


def summary_md(r: dict) -> str:
    P, S = r["primary"], r["secondary"]
    A = S["anticipation"]
    L = ["# P-A1 routing-conditioned next-class forecasting (GPT cohort A): results (D3a)", "",
         f"Primary: gain **{_f(P['point'])}** nats per adjacent sentence (question-weighted; base minus augmented), "
         f"95% CI [{_f(P['ci_lo'])}, {_f(P['ci_hi'])}], one-sided p {_f(P['p_one_sided'], 4)} "
         f"(Holm across P-A1/P-B1/P-B4: upper bound 3p = {_f(P['holm_adjusted_upper_bound'], 4)}), positive in "
         f"{P['n_positive_repeats']}/5 repeats. Verdict (Addendum 3 rule): **{P['verdict']}**.", "",
         f"Conditions: {json.dumps(P['conditions'])}.", "",
         f"Log-loss base {_f(P['logloss_base'])} vs augmented {_f(P['logloss_aug'])}. Pair-weighted (unweighted) "
         f"sensitivity: {_f(P['pair_weighted']['point'])} [{_f(P['pair_weighted']['ci_lo'])}, "
         f"{_f(P['pair_weighted']['ci_hi'])}].", "",
         f"Design: {r['pairs']['n_pairs']} analysed pairs from {r['pairs']['n_questions']} questions "
         f"({r['pairs']['n_self']} self-transitions in all {r['pairs']['n_pairs_all']} pairs before the "
         f"{r['pairs']['n_reserved_dropped']} reserved GPT-B-question pairs were dropped); text scale "
         f"{r['design']['text_scale']}.", "", "## Pair counts per source class (analysed)", "",
         "| source class | pairs | self | self share | questions |", "|---|---|---|---|---|"]
    for c, v in r["pairs"]["per_source_class"].items():
        L.append(f"| {c} | {v['n']} | {v['n_self']} | {_f(v['self_share'], 3)} | {v['n_questions']} |")
    L += ["", "## Per-source-class gain (base - augmented, question-weighted mean over the class's pairs)", "",
          "| source class | pairs | questions | gain | 95% CI | p (one-sided) | anticipation gain | 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(CLASSES):
        v, w = S["per_source_class"][str(i)], A["per_source_class"][str(i)]
        L.append(f"| {c} | {v['n_pairs']} | {v['n_questions']} | {_f(v['point'])} | [{_f(v['ci_lo'])}, {_f(v['ci_hi'])}] | "
                 f"{_f(v['p_one_sided'], 4)} | {_f(w['point'])} | [{_f(w['ci_lo'])}, {_f(w['ci_hi'])}] |")
    L += ["", "By transition type (0 = self-transition, 1 = class switch):", ""]
    for k, v in S["by_transition_type"]["per_source_class"].items():
        w = A["by_transition_type"]["per_source_class"][k]
        L.append(f"- type {k}: n {v['n_pairs']}, gain {_f(v['point'])} [{_f(v['ci_lo'])}, {_f(v['ci_hi'])}]; "
                 f"anticipation {_f(w['point'])} [{_f(w['ci_lo'])}, {_f(w['ci_hi'])}]")
    L += ["", "## Anticipation variant (window ending 16 tokens before the sentence end)", "",
          f"Gain {_f(A['point'])} [{_f(A['ci_lo'])}, {_f(A['ci_hi'])}], one-sided p {_f(A['p_one_sided'], 4)}, positive "
          f"repeats {A['n_positive_repeats']}/5, {A['windows_missing']} pairs without a window (score set to the fold mean); "
          f"rule applied as for the primary: {A['verdict_rule_applied']}. Anticipation minus last-16 gain: "
          f"{_f(S['aug_ant_vs_aug']['point'])} [{_f(S['aug_ant_vs_aug']['ci_lo'])}, {_f(S['aug_ant_vs_aug']['ci_hi'])}] "
          "(sign: positive = the earlier window forecasts better than the last 16 tokens).", "",
          "## Calibration and secondary metrics", "",
          "| model | calibration slope [CI] | ECE (all class probabilities) |", "|---|---|---|"]
    for k in ("base", "aug"):
        c = S["calibration"][k]
        L.append(f"| {k} | {_f(c['slope'], 3)} [{_f(c['ci_lo'], 3)}, {_f(c['ci_hi'], 3)}] | {_f(c['ece'], 4)} |")
    c = A["calibration"]["aug"]
    L.append(f"| aug_ant | {_f(c['slope'], 3)} [{_f(c['ci_lo'], 3)}, {_f(c['ci_hi'], 3)}] | {_f(c['ece'], 4)} |")
    L += ["", f"Brier gain (base - aug, primary) {_f(S['brier_gain']['point'], 6)} [{_f(S['brier_gain']['ci_lo'], 6)}, "
          f"{_f(S['brier_gain']['ci_hi'], 6)}]; accuracy gain {_f(S['accuracy_gain']['point'], 5)} "
          f"[{_f(S['accuracy_gain']['ci_lo'], 5)}, {_f(S['accuracy_gain']['ci_hi'], 5)}].", "",
          "## Null calibration (useless 7-dim augmentation on the real features)", ""]
    for n, v in r["null_calibration"].items():
        s = v["summary"]
        L.append(f"- protocol {n} ({json.dumps(v['protocol'])}): mean gain (negative bias) {_f(s['mean_gain'], 5)}, "
                 f"SD of gain {_f(s['sd_gain'], 5)} vs mean bootstrap SD {_f(s['mean_boot_sd'], 5)} (ratio "
                 f"{_f(s['sd_over_boot_sd'], 2)}); benefit-side false positives: CI lower > 0 in "
                 f"{s['benefit_fp_ci_lo_above_0']}/{s['nsim']}, p < .05 in {s['benefit_fp_p_below_05']}/{s['nsim']}, "
                 f"full advance rule in {s['benefit_fp_advance_rule']}/{s['nsim']}; CI upper < 0 in {s['ci_hi_below_0']}/{s['nsim']}.")
    L += ["", f"Fit diagnostics: {json.dumps(r['diagnostics']['cv'])}; median chosen penalties "
              f"{json.dumps(r['diagnostics']['lambdas'])}.", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--workers", type=int, default=5)
    m = sub.add_parser("cv")
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--n-boot", type=int, default=1000)
    n = sub.add_parser("nullsim")
    n.add_argument("--nsim", type=int, default=50)
    n.add_argument("--workers", type=int, default=8)
    n.add_argument("--repeats", type=int, default=5)
    n.add_argument("--n-boot", type=int, default=1000)
    n.add_argument("--fixed-lambda", action="store_true")
    n.add_argument("--chunks", type=int, default=1)
    n.add_argument("--kind", choices=("noise", "perm"), default="noise")
    sub.add_parser("assemble")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "calibrate":
        calibrate(args.workers)
    elif args.cmd == "cv":
        run_main_cv(args.workers, args.n_boot)
    elif args.cmd == "nullsim":
        run_nullsim(args.nsim, args.workers, args.repeats, args.n_boot, args.fixed_lambda, args.chunks, args.kind)
    else:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
