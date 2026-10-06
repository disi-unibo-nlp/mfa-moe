"""D3c stage 3: P-B4 incremental validity (Addendum 4) with the C8 machinery. Reads outcomes ONLY after the freeze.

  python -m dynrt.d3c_cv run [--workers W --n-boot N --tag T]  -> results/B4/{results.json, summary.md}

Universe: GPT cohort A minus the 37 cohort-B questions (1,510 attempts; folds identical to B1). Models:
  base = C + class-conditioned marginal block (21 own-class depth-third affinities + 7 presence indicators)
  PRIMARY = base + {PA, PAlex, PA_sd}; threshold 0.005 nats/attempt; Holm family = P-A1, P-B1, P-B4.
Secondaries: base + PAlex; base + per-class PA (Explore, Verify, Monitor); GPT B within-question contrasts of the
frozen scores. Sensitivities: reading B of PAlex; PA alone. Every feature table comes from the frozen file and is
looked up by the exact training-row set the C8 fold used.
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

from . import cv
from . import d3c_features as D
from .common import RESULTS, sha256_file
from .controls import ControlsBlock
from .d3b_util import clean, write_json
from .d3c_freeze import CODE_FILES, FROZEN, OUT, universe_mask
from .data import load_cohort, load_outcomes

THRESHOLD = 0.005
ALPHA = 0.05
HOLM_M = 3
BASE = ["controls", "marg"]
MODEL_SPECS = {
    "controls": ["controls"],
    "base": BASE,
    "B4": BASE + ["pa.PA", "pa.PAlex", "pa.PA_sd"],
    "PAlex_only": BASE + ["pa.PAlex"],
    "PA_perclass": BASE + ["pa.PA_Explore", "pa.PA_Verify", "pa.PA_Monitor"],
    "B4_readingB": BASE + ["pa.PA", "pa.PAlexB", "pa.PA_sd"],
    "PA_only": BASE + ["pa.PA"],
}
COMPARISONS = {
    "B4_primary": ("base", "B4"),
    "secondary_PAlex_only": ("base", "PAlex_only"),
    "secondary_PA_perclass": ("base", "PA_perclass"),
    "sensitivity_PAlex_reading_B": ("base", "B4_readingB"),
    "sensitivity_PA_only": ("base", "PA_only"),
    "ablation_marginal_block_vs_controls": ("controls", "base"),
}
B_FEATURES = ["PA", "PAlex", "PA_sd", "PAlexB", "PA_Explore", "PA_Verify", "PA_Monitor"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_frozen() -> dict:
    """Feature files and feature-defining code must be exactly those written at the freeze."""
    fz = json.loads(FROZEN.read_text())
    if fz.get("outcomes_read") is not False:
        raise RuntimeError("freeze manifest does not certify an outcome-free feature stage")
    here = Path(__file__).resolve().parent
    bad = [n for n, sha in fz["code_sha256"].items() if sha256_file(here / n) != sha]
    if bad:
        raise RuntimeError(f"code changed since the P-B4 freeze: {bad}")
    for name, sha in fz["files"].items():
        if sha256_file(OUT / name) != sha:
            raise RuntimeError(f"frozen file changed: {name}")
    return dict(freeze_sha256=fz["sha256"], freeze_time=fz["time"])


class PrecomputedBlock:
    """CV block that returns the frozen feature table of the exact training-row set it is asked for."""

    def __init__(self, keys: list[str], tables: np.ndarray, names: list[str]):
        self.index = {k: i for i, k in enumerate(keys)}
        self.tables, self.names = tables, list(names)

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        key = D.set_key(train)
        if key not in self.index:
            raise KeyError("training set absent from the frozen feature file")
        return self.tables[self.index[key]]


def run_models(blocks: dict, y, groups, weights, valid, *, workers: int, n_boot: int, specs=None,
               comparisons=None) -> dict:
    """C8 cross-fitting (4 outer x 5 repeats, 3 inner) and paired question-bootstrap comparisons."""
    specs = MODEL_SPECS if specs is None else specs
    comparisons = COMPARISONS if comparisons is None else comparisons
    spec = cv.CVSpec(blocks=blocks, models=specs, y=y, groups=groups, weights=weights, repeats=5, outer=4,
                     inner=3, seed0=0, valid=valid)
    res = cv.run_cv(spec, workers=workers, log=log)
    v = valid
    out = {}
    for name, (b, a) in comparisons.items():
        out[name] = cv.evaluate({b: res["preds"][b][:, v], a: res["preds"][a][:, v]}, b, a, y[v], groups[v],
                                weights[v], n_boot=n_boot)
        c = out[name]
        log(f"{name}: gain {c['gain']:.5f} [{c['ci_lo']:.5f}, {c['ci_hi']:.5f}] p1 {c['p_one_sided']:.4f} "
            f"per-repeat {[round(x, 5) for x in c['per_repeat_gain']]}")
    return dict(comparisons=out, lambdas={m: dict(median=float(np.median(l)), values=l.round(6).tolist())
                                          for m, l in res["lambdas"].items()}, preds=res["preds"])


def verdict(c: dict, threshold: float = THRESHOLD) -> dict:
    """Addendum 3/4 rule: Holm p < .05, paired 95% CI above 0, >= 4/5 repeats positive, gain >= threshold."""
    pos = int(sum(g > 0 for g in c["per_repeat_gain"]))
    cond = dict(holm_safe_p_lt_alpha_over_3=bool(c["p_one_sided"] < ALPHA / HOLM_M),
                ci_lower_above_zero=bool(c["ci_lo"] > 0), positive_in_at_least_4_of_5_repeats=bool(pos >= 4),
                gain_at_least_threshold=bool(c["gain"] >= threshold))
    if all(cond.values()):
        label = "ADVANCE (pending the Holm adjustment over P-A1, P-B1, P-B4)"
    elif c["ci_hi"] < threshold:
        label = "DEPRIORITISE (CI upper bound below the threshold)"
    else:
        label = "INCONCLUSIVE"
    return dict(threshold=threshold, gain=c["gain"], ci=[c["ci_lo"], c["ci_hi"]], p_one_sided=c["p_one_sided"],
                p_for_holm=c["p_one_sided"], holm_m=HOLM_M, holm_adjusted_if_smallest=min(1.0, HOLM_M * c["p_one_sided"]),
                positive_repeats=pos, conditions=cond, verdict=label)


def load_universe() -> dict:
    """Cohort-A universe arrays; outcomes are read here for the first time."""
    coh = load_cohort("gpt", "A")
    keep = universe_mask(coh.attempts)
    att = coh.attempts[keep].reset_index(drop=True)
    out = load_outcomes("gpt", "A", coh.attempts)[keep].reset_index(drop=True)
    y_raw = out["correct"].to_numpy(float)
    valid = np.isfinite(y_raw) & ~att["capped"].to_numpy(bool)
    groups = att["question"].to_numpy()
    n_valid_q = pd.Series(valid.astype(float)).groupby(groups).transform("sum").to_numpy()
    controls = ControlsBlock(np.log(att["n_reasoning"].to_numpy(float)), out["difficulty"].to_numpy(float),
                             att["dataset"].to_numpy(), coh.shares[keep][:, :6], coh.n_labelled[keep],
                             att["capped"].to_numpy(float))
    return dict(att=att, y=np.where(valid, y_raw, 0.0), valid=valid, groups=groups,
                weights=np.where(valid, 1.0 / np.maximum(n_valid_q, 1.0), 0.0), controls=controls)


def within_question(n_perm: int, n_boot: int) -> dict:
    """GPT cohort B: correct - wrong contrasts of the frozen scores (fitted on A questions only)."""
    from .d3 import contrast
    from .data import load_attempts
    att = load_attempts("gpt", "B")
    out = load_outcomes("gpt", "B", att)
    feats = pd.read_parquet(OUT / "features_B.parquet")
    if not (feats["attempt_id"].to_numpy() == att["attempt_id"].to_numpy()).all():
        raise RuntimeError("cohort-B feature table is not aligned")
    y = out["correct"].to_numpy(float)
    capped = att["capped"].to_numpy(bool)
    q = att["question"].to_numpy()
    log_tokens = np.log(att["completion_tokens"].to_numpy(float).clip(min=1))
    ok_y = np.isfinite(y)
    mixed = [g for g in set(q[ok_y]) if 0 < y[ok_y & (q == g)].sum() < (ok_y & (q == g)).sum()]
    res = dict(n_attempts=int(len(att)), n_scored=int(ok_y.sum()), n_capped=int(capped.sum()),
               n_questions=int(len(set(q))), mixed_questions=len(mixed), variants={})
    spec = {"len_nocap": (~capped, log_tokens), "raw": (np.ones(len(att), bool), None),
            "len": (np.ones(len(att), bool), log_tokens)}
    for variant, (keep, length) in spec.items():
        res["variants"][variant] = {}
        for f in B_FEATURES:
            x = np.where(keep, feats[f].to_numpy(float), np.nan)
            r = contrast(x, y, q, length, n_perm, n_boot)
            res["variants"][variant][f] = r
            log(f"B {variant} {f}: n {r['n']} q {r['n_questions']} d {r['d']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] p {r['p_two_sided']:.4f}")
    return res


def _fmt(x, nd=5) -> str:
    return "NA" if x is None else f"{x:.{nd}f}"


def summary_md(r: dict) -> str:
    v = r["verdict"]
    c = r["cv"]["comparisons"]
    P = c["B4_primary"]
    L = ["# D3c P-B4: pathway alignment and correctness beyond the marginal block (GPT cohort A)", "",
         f"Feature freeze sha256 `{r['frozen']['freeze_sha256']}`. Universe: {r['data']['n_attempts']} attempts / "
         f"{r['data']['n_questions']} questions ({r['data']['n_scored']} scored, {r['data']['n_wrong']} wrong; "
         f"capped attempts excluded from the loss); 4 folds x 5 repeats, same folds as B1.", "",
         f"**Verdict (PRIMARY): {v['verdict']}**", "",
         f"- Gain of base + {{PA, PAlex, PA_sd}} over base (question-weighted log-loss, nats/attempt) = "
         f"**{_fmt(P['gain'])}**, paired 95% CI [{_fmt(P['ci_lo'])}, {_fmt(P['ci_hi'])}]; threshold {v['threshold']}.",
         f"- One-sided bootstrap p (gain > 0) for Holm = **{_fmt(v['p_for_holm'], 4)}**; Holm-adjusted if it were the "
         f"smallest of the 3 = {_fmt(v['holm_adjusted_if_smallest'], 4)}.",
         f"- Positive in {v['positive_repeats']}/5 repeats: {[round(x, 5) for x in P['per_repeat_gain']]}.",
         "- Conditions: " + ", ".join(f"{k}={val}" for k, val in v["conditions"].items()) + ".", "",
         "## All comparisons", "",
         "| comparison | gain | 95% CI | p (one-sided) | positive repeats | dAUC [CI] | dBrier (improvement) [CI] |",
         "|---|---|---|---|---|---|---|"]
    for name, x in c.items():
        pos = sum(g > 0 for g in x["per_repeat_gain"])
        L.append(f"| {name} | {_fmt(x['gain'])} | [{_fmt(x['ci_lo'])}, {_fmt(x['ci_hi'])}] | "
                 f"{_fmt(x['p_one_sided'], 4)} | {pos}/5 | {_fmt(x['dauc']['diff'], 4)} "
                 f"[{_fmt(x['dauc']['ci_lo'], 4)}, {_fmt(x['dauc']['ci_hi'], 4)}] | "
                 f"{_fmt(x['dbrier_improvement']['diff'], 5)} [{_fmt(x['dbrier_improvement']['ci_lo'], 5)}, "
                 f"{_fmt(x['dbrier_improvement']['ci_hi'], 5)}] |")
    L += ["", "Log-loss: control-only "
          f"{_fmt(c['ablation_marginal_block_vs_controls']['logloss_base'], 4)}, baseline "
          f"{_fmt(P['logloss_base'], 4)}, primary {_fmt(P['logloss_aug'], 4)}. The C8 null-bias caveat applies: "
          "held-out gains of extra columns are biased slightly negative.", ""]
    w = r.get("within_question")
    if w:
        L += ["## GPT cohort B: within-question contrast of the frozen scores, correct - wrong", "",
              f"{w['n_scored']} scored attempts ({w['n_capped']} capped), {w['mixed_questions']} questions with both "
              "outcomes; stratified permutation 10,000, question bootstrap 1,000; scores fitted on A questions "
              "only. Secondary; two-sided p, unadjusted.", "",
              "| variant | feature | questions | delta | d | 95% CI (d) | p | frac. positive |",
              "|---|---|---|---|---|---|---|---|"]
        for var, rows in w["variants"].items():
            for k, x in rows.items():
                L.append(f"| {var} | {k} | {x['n_questions']} | {_fmt(x['delta'])} | {_fmt(x['d'], 3)} | "
                         f"[{_fmt(x['ci_lo'], 3)}, {_fmt(x['ci_hi'], 3)}] | {_fmt(x['p_two_sided'], 4)} | "
                         f"{_fmt(x['frac_pos'], 2)} |")
        L.append("")
    return "\n".join(L)


def run(a) -> int:
    t0 = time.time()
    fz = verify_frozen()
    log(f"freeze verified: {fz}")
    z = np.load(OUT / "features_A.npz")
    keys = z["set_keys"].tolist()
    pa_names, marg_names = z["pa_names"].tolist(), z["marg_names"].tolist()
    u = load_universe()
    if not (z["attempt_id"] == u["att"]["attempt_id"].to_numpy()).all():
        raise RuntimeError("frozen features are not aligned to the universe attempts")
    n, n_valid = len(u["y"]), int(u["valid"].sum())
    n_wrong = int((u["y"][u["valid"]] == 0).sum())
    log(f"universe {n} attempts, {n_valid} scored ({n_wrong} wrong), {len(set(u['groups']))} questions")
    blocks = dict(controls=u["controls"], marg=PrecomputedBlock(keys, z["marg"], marg_names),
                  pa=PrecomputedBlock(keys, z["pa"], pa_names))
    res = run_models(blocks, u["y"], u["groups"], u["weights"], u["valid"], workers=a.workers, n_boot=a.n_boot)
    out = dict(frozen=fz, data=dict(n_attempts=n, n_scored=n_valid, n_wrong=n_wrong,
                                    n_questions=int(len(set(u["groups"])))),
               cv=dict(comparisons=res["comparisons"], lambdas=res["lambdas"]),
               verdict=verdict(res["comparisons"]["B4_primary"]), models=MODEL_SPECS, seconds=None)
    np.savez_compressed(OUT / f"B4_preds{('_' + a.tag) if a.tag else ''}.npz", attempt_id=u["att"]["attempt_id"].to_numpy(),
                        y=u["y"], valid=u["valid"], **{f"pred_{m}": res["preds"][m] for m in MODEL_SPECS})
    if not a.no_within:
        out["within_question"] = within_question(a.n_perm, a.n_boot)
    out["seconds"] = time.time() - t0
    tag = f"_{a.tag}" if a.tag else ""
    write_json(OUT / f"results{tag}.json", out)
    (OUT / f"summary{tag}.md").write_text(summary_md(clean(out)))
    log(f"verdict {out['verdict']['verdict']}; gain {out['verdict']['gain']:.5f}; wrote results{tag}.json in "
        f"{out['seconds']:.0f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--n-perm", type=int, default=10000)
    s.add_argument("--no-within", action="store_true")
    s.add_argument("--tag", default="")
    s.set_defaults(fn=run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
