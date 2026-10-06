"""D3: pre-registered replication of the two GPT cohort-B within-question leads (ANALYSIS_PLAN.md, Addendum 2).

L1 = bJSD (mean residual boundary JSD at class switches), L2 = margin (mean residual mismatch margin).
Per model: v3an `within_question_contrast`, correct - wrong, stratified permutation 10,000, question bootstrap 1,000.
Primary variant `len_nocap` (v3an definition: length = log completion tokens, capped attempts excluded);
`raw` is secondary. PRIMARY pooled statistic: DerSimonian-Laird over gemma, glm, nemotron, qwen330b using the
bootstrap SE of d, one-sided in the negative direction, Bonferroni alpha = .025 per lead.
A lead REPLICATES if pooled one-sided p < .025, the pooled estimate is negative and >= 3 of 4 models are negative.
Run only after FEATURES_FROZEN_D3.json exists.

  python -m dynrt.d3 run            per-model contrasts, pooling, verdicts -> results/D3/{results.json, summary.md}
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

from .c1 import clean, log
from .common import RESULTS, sha256_file
from .data import load_cohort, load_outcomes

OUT = RESULTS / "D3"
FROZEN_D3 = RESULTS / "FEATURES_FROZEN_D3.json"
REPLICATION = ("gemma", "glm", "nemotron", "qwen330b")
DESCRIPTIVE = ("qwen35", "qwen36")
LEADS = {"L1": "bJSD", "L2": "margin"}
ALPHA_LEAD = 0.025
MIN_NEGATIVE = 3
PRIMARY = "len_nocap"
VARIANTS = ("len_nocap", "raw", "len", "len_nocap_reasoning")   # primary, secondary, two sensitivities


def verify_frozen_d3() -> dict:
    """D1, D2 and D3 freezes must be intact."""
    from .c2 import verify_frozen_d2
    d12 = verify_frozen_d2()
    fz = json.loads(FROZEN_D3.read_text())
    here = Path(__file__).resolve().parent
    now = {p.name: sha256_file(p) for p in sorted(here.glob("*.py")) if p.name in fz["code_sha256"]}
    changed = sorted(k for k, v in fz["code_sha256"].items() if now.get(k) != v)
    if changed:
        raise RuntimeError(f"code changed since the D3 freeze: {changed}")
    for model, info in fz["models"].items():
        if sha256_file(RESULTS / model / "B" / "b_dyn.parquet") != info["file_sha256"]:
            raise RuntimeError(f"D3 frozen feature table changed: {model}")
    return dict(**d12, freeze_d3_sha256=fz["sha256"], freeze_d3_time=fz["time"])


# ------------------------------------------------------------------------------- statistics

def pool_dl(d: list[float], se: list[float]) -> dict:
    """DerSimonian-Laird pooling of per-model d with one-sided p in the negative direction."""
    from scipy import stats as sps

    from v3an import stats as vs
    m = vs.dersimonian_laird(d, se)
    m["p_one_sided_negative"] = float(sps.norm.cdf(m["z"])) if np.isfinite(m["z"]) else float("nan")
    return m


def lead_verdict(pooled: dict, ds: list[float], n_models: int = 4) -> dict:
    """Addendum 2 rule: pooled one-sided p < .025, pooled estimate negative, >= 3 of 4 models negative."""
    n_neg = int(sum(1 for x in ds if np.isfinite(x) and x < 0))
    cond = dict(pooled_p_lt_alpha=bool(pooled["p_one_sided_negative"] < ALPHA_LEAD),
                pooled_estimate_negative=bool(pooled["mu"] < 0),
                negative_in_at_least_3_of_4=bool(n_neg >= MIN_NEGATIVE))
    return dict(conditions=cond, n_negative=n_neg, n_models=n_models,
                verdict="REPLICATES" if all(cond.values()) else "NOT REPLICATED (GPT-specific observation)")


def contrast(x: np.ndarray, y: np.ndarray, q: np.ndarray, length: np.ndarray | None, n_perm: int, n_boot: int) -> dict:
    from v3an import stats as vs
    rows = np.isfinite(x) & np.isfinite(y)
    if length is not None:
        rows &= np.isfinite(length)
    w = vs.within_question_contrast(x[rows][:, None], y[rows], q[rows], length=None if length is None else length[rows],
                                    n_perm=n_perm, n_boot=n_boot, rng=np.random.default_rng(20260929))
    return dict(n=int(w["n"]), n_questions=int(w["n_questions"]), delta=float(w["delta"][0]), d=float(w["d"][0]),
                ci_lo=float(w["ci_lo"][0]), ci_hi=float(w["ci_hi"][0]), boot_se=float(w["boot_se"][0]),
                p_two_sided=float(w["p"][0]), frac_pos=float(w["frac_pos"][0]))


def model_contrasts(model: str, n_perm: int, n_boot: int) -> dict:
    """All variants for L1 and L2 of one model's cohort B (frozen features)."""
    coh = load_cohort(model, "B")
    att = coh.attempts
    out = load_outcomes(model, "B", att)
    feats = pd.read_parquet(RESULTS / model / "B" / "b_dyn.parquet")
    if not (feats["attempt_id"].to_numpy() == att["attempt_id"].to_numpy()).all():
        raise RuntimeError(f"{model}: frozen B table is not aligned")
    y = out["correct"].to_numpy(float)
    capped = att["capped"].to_numpy(bool)
    q = att["question"].to_numpy()
    log_tokens = np.log(att["completion_tokens"].to_numpy(float).clip(min=1))
    log_reasoning = np.log(att["n_reasoning"].to_numpy(float).clip(min=1))
    ok_y = np.isfinite(y)
    scored_q = set(q[ok_y])
    mixed = [g for g in scored_q if 0 < y[ok_y & (q == g)].sum() < (ok_y & (q == g)).sum()]
    res = dict(model=model, n_attempts=int(len(att)), n_scored=int(ok_y.sum()), n_capped=int(capped.sum()),
               n_questions=int(len(set(q))), mixed_questions=len(mixed),
               n_correct=int((y[ok_y] == 1).sum()), n_wrong=int((y[ok_y] == 0).sum()), leads={})
    spec = {"len_nocap": (~capped, log_tokens), "raw": (np.ones(len(att), bool), None),
            "len": (np.ones(len(att), bool), log_tokens), "len_nocap_reasoning": (~capped, log_reasoning)}
    for lead, col in LEADS.items():
        x = feats[col].to_numpy(float)
        res["leads"][lead] = {}
        for variant in VARIANTS:
            keep, length = spec[variant]
            xv = np.where(keep, x, np.nan)
            res["leads"][lead][variant] = contrast(xv, y, q, length, n_perm, n_boot)
            r = res["leads"][lead][variant]
            log(f"{model} {lead} {variant}: n {r['n']} q {r['n_questions']} d {r['d']:.3f} "
                f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] se {r['boot_se']:.3f} p {r['p_two_sided']:.4f}")
    return res


def run(n_perm: int, n_boot: int) -> dict:
    t0 = time.time()
    fz = verify_frozen_d3()
    per = {m: model_contrasts(m, n_perm, n_boot) for m in REPLICATION + DESCRIPTIVE}
    pooled = {}
    for lead in LEADS:
        pooled[lead] = {}
        for variant in VARIANTS:
            d = [per[m]["leads"][lead][variant]["d"] for m in REPLICATION]
            se = [per[m]["leads"][lead][variant]["boot_se"] for m in REPLICATION]
            p = pool_dl(d, se)
            p["models_negative"] = [m for m, x in zip(REPLICATION, d) if np.isfinite(x) and x < 0]
            pooled[lead][variant] = p
        pooled[lead]["verdict"] = lead_verdict(
            pooled[lead][PRIMARY], [per[m]["leads"][lead][PRIMARY]["d"] for m in REPLICATION])
    res = dict(plan="ANALYSIS_PLAN.md Addendum 2", frozen=fz, alpha_per_lead=ALPHA_LEAD, primary_variant=PRIMARY,
               replication_models=list(REPLICATION), descriptive_models=list(DESCRIPTIVE), per_model=per,
               pooled=pooled, gpt_discovery=gpt_discovery(), seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(clean(res), indent=1))
    (OUT / "summary.md").write_text(summary_md(clean(res)))
    return res


def gpt_discovery() -> dict:
    """The discovery statistics (D1 margin, D2 bJSD; length = log reasoning tokens), for reference only."""
    c1 = json.loads((RESULTS / "C1" / "gpt_B.json").read_text())["variants"]["length_adjusted"]["margin"]
    c2 = json.loads((RESULTS / "C2" / "gpt_B.json").read_text())["variants"]["length_adjusted"]["bJSD"]
    return dict(L1_bJSD=dict(d=c2["d"], ci=[c2["ci_lo"], c2["ci_hi"]], p=c2["p"], n_questions=c2["n_questions"]),
                L2_margin=dict(d=c1["d"], ci=[c1["ci_lo"], c1["ci_hi"]], p=c1["p"], n_questions=c1["n_questions"]))


def _f(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def summary_md(r: dict) -> str:
    L = ["# D3: pre-registered replication of the GPT cohort-B within-question leads", "",
         f"Plan: `ANALYSIS_PLAN.md`, Addendum 2 (frozen before any replication-model outcome was read). Feature freeze "
         f"`FEATURES_FROZEN_D3.json` sha256 `{r['frozen']['freeze_d3_sha256']}`.",
         "Statistic: length-adjusted within-question contrast d (correct - wrong; v3an `within_question_contrast`, "
         "`len_nocap` = length log completion tokens, capped attempts excluded); question bootstrap 1,000, stratified "
         "permutation 10,000. Pooled: DerSimonian-Laird over gemma, glm, nemotron, qwen330b with the bootstrap SE of d, "
         "one-sided in the negative direction (discovery direction), Bonferroni alpha = .025 per lead. A lead REPLICATES "
         "if pooled one-sided p < .025, the pooled estimate is negative and >= 3 of 4 models are negative.", ""]
    gd = r["gpt_discovery"]
    L += ["GPT discovery (length-adjusted, length = log reasoning tokens, 37 questions): "
          f"L1 bJSD d {_f(gd['L1_bJSD']['d'], 2)} [{_f(gd['L1_bJSD']['ci'][0], 2)}, {_f(gd['L1_bJSD']['ci'][1], 2)}], "
          f"p {_f(gd['L1_bJSD']['p'], 4)}; L2 margin d {_f(gd['L2_margin']['d'], 2)} "
          f"[{_f(gd['L2_margin']['ci'][0], 2)}, {_f(gd['L2_margin']['ci'][1], 2)}], p {_f(gd['L2_margin']['p'], 3)}.", ""]
    for lead, col in LEADS.items():
        pl = r["pooled"][lead]
        v = pl["verdict"]
        p = pl[PRIMARY]
        L += [f"## {lead} ({col})", "",
              f"**Verdict: {v['verdict']}.** Pooled d (len_nocap) = {_f(p['mu'])} [{_f(p['ci_lo'])}, {_f(p['ci_hi'])}], "
              f"one-sided p (negative) = {_f(p['p_one_sided_negative'], 4)} (alpha {r['alpha_per_lead']}), "
              f"I^2 = {_f(p['I2'], 2)}, tau^2 = {_f(p['tau2'], 4)}, Q = {_f(p['Q'], 2)} (p {_f(p['Q_p'], 3)}); "
              f"{v['n_negative']} of {v['n_models']} models negative. Conditions: {json.dumps(v['conditions'])}.", "",
              "| model | attempts | scored | questions with both outcomes | attempts used | d (len_nocap) | 95% CI | boot SE | p (2-sided perm.) | d (raw) | 95% CI (raw) | p (raw) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in r["replication_models"] + r["descriptive_models"]:
            mm = r["per_model"][m]
            a, b = mm["leads"][lead]["len_nocap"], mm["leads"][lead]["raw"]
            tag = m if m in r["replication_models"] else f"{m} (descriptive)"
            L.append(f"| {tag} | {mm['n_attempts']} | {mm['n_scored']} | {mm['mixed_questions']} | {a['n']} | {_f(a['d'])} | "
                     f"[{_f(a['ci_lo'])}, {_f(a['ci_hi'])}] | {_f(a['boot_se'])} | {_f(a['p_two_sided'], 4)} | {_f(b['d'])} | "
                     f"[{_f(b['ci_lo'])}, {_f(b['ci_hi'])}] | {_f(b['p_two_sided'], 4)} |")
        L += ["", "Pooled variants (DerSimonian-Laird, 4 models):", "",
              "| variant | pooled d | 95% CI | one-sided p (neg.) | I^2 | negative models |", "|---|---|---|---|---|---|"]
        for var in VARIANTS:
            q = pl[var]
            L.append(f"| {var}{' (primary)' if var == PRIMARY else ' (secondary)' if var == 'raw' else ' (sensitivity)'} | "
                     f"{_f(q['mu'])} | [{_f(q['ci_lo'])}, {_f(q['ci_hi'])}] | {_f(q['p_one_sided_negative'], 4)} | "
                     f"{_f(q['I2'], 2)} | {', '.join(q['models_negative']) or 'none'} |")
        L.append("")
    L += ["`len` keeps capped attempts; `len_nocap_reasoning` uses log reasoning tokens as the length covariate (the "
          "D1/D2 convention for GPT). Sensitivities are not used for the verdict. qwen35 and qwen36 (dev+tune only) are "
          "descriptive and are not pooled.", ""]
    return "\n".join(L) + "\n"


def replication_section(r: dict) -> list[str]:
    """Replication section for SUMMARY.md (numbers from results/D3/results.json)."""
    L = ["## D3: pre-registered replication of the two GPT cohort-B leads (Addendum 2)", "",
         "Rule: a lead REPLICATES if the DerSimonian-Laird pooled one-sided p (negative direction) over gemma, glm, "
         f"nemotron and qwen330b is < {r['alpha_per_lead']} (Bonferroni, two leads), the pooled length-adjusted d is "
         "negative and at least 3 of 4 models are negative. GPT is the discovery model and is excluded. Details: "
         "`D3/summary.md`, `D3/results.json`.", "",
         "| lead | pooled d (len_nocap) | 95% CI | one-sided p | I^2 | negative models | per-model d | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for lead, col in LEADS.items():
        pl = r["pooled"][lead]
        p = pl[PRIMARY]
        per = ", ".join(f"{m} {_f(r['per_model'][m]['leads'][lead][PRIMARY]['d'], 2)}" for m in r["replication_models"])
        L.append(f"| {lead} ({col}) | {_f(p['mu'])} | [{_f(p['ci_lo'])}, {_f(p['ci_hi'])}] | "
                 f"{_f(p['p_one_sided_negative'], 4)} | {_f(p['I2'], 2)} | {pl['verdict']['n_negative']} of 4 | {per} | "
                 f"**{pl['verdict']['verdict']}** |")
    L.append("")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    run(a.n_perm, a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
