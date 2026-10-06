"""Holm across the four primary tests and W/results/SUMMARY.md (C1 + C2 + C6).

The 4th slot is reserved for the plan's C8 ablation family; it is not separately defined, so p = 1 is used.
CLI: python -m dynrt.summary
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .c1 import ADVANCE_GAIN, clean
from .common import RESULTS

FAMILY = ("C1", "C2", "C6", "C8-ablation-slot")


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in the input order."""
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def verdict(gain: float, lo: float, hi: float, p_adj: float) -> tuple[dict, str]:
    cond = dict(holm_adjusted_p_lt_05=p_adj < 0.05, ci_excludes_0=lo > 0, gain_ge_0_005=gain >= ADVANCE_GAIN)
    if all(cond.values()):
        return cond, "ADVANCES"
    if hi < ADVANCE_GAIN:
        return cond, "DEPRIORITISED"
    return cond, "INCONCLUSIVE"


def _f(x, nd=5):
    return "NA" if x is None else f"{x:.{nd}f}"


def secondary(c2: dict, c6: dict) -> list[str]:
    """Secondary / exploratory results worth reading next to the verdicts (numbers from the result files)."""
    C1 = json.loads((RESULTS / "C1" / "results.json").read_text())
    L = ["## Secondary and exploratory results", ""]
    g1 = C1["parts"]["gpt_A"]["partial"]
    L.append("- C1 partial associations (gpt A): " + "; ".join(
        f"{k} r {_f(v['r'], 3)} [{_f(v['ci_lo'], 3)}, {_f(v['ci_hi'], 3)}]" for k, v in g1.items()) +
        f". qwen36 dev+tune A primary gain {_f(C1['parts']['qwen36_A']['comparisons']['C1_primary']['gain'])}.")
    gb = C1["parts"]["gpt_B"]["variants"]["length_adjusted"]["margin"]
    L.append(f"- C1 lead (exploratory, not in the family): gpt B within-question margin contrast d {_f(gb['d'], 2)} "
             f"[{_f(gb['ci_lo'], 2)}, {_f(gb['ci_hi'], 2)}], p {_f(gb['p'], 3)}; not replicated across questions.")
    q = c2["parts"]["qwen36_A"]
    qc = q["comparisons"]["C2_primary"]
    L.append(f"- C2 qwen36 dev+tune A replication: gain {_f(qc['gain'])} [{_f(qc['ci_lo'])}, {_f(qc['ci_hi'])}], p1 "
             f"{_f(qc['p_one_sided'], 3)} ({q['n_wrong']} wrong attempts, {q['n_pairs']} eligible pairs); too weak to "
             "confirm or refute (CI spans -0.004 to +0.009).")
    dirs = c2["parts"]["gpt_A"]["comparisons"]
    L.append("- C2 direction groups (gpt A gain): " + "; ".join(
        f"{g} {_f(dirs['direction_' + g]['gain'])} [{_f(dirs['direction_' + g]['ci_lo'])}, "
        f"{_f(dirs['direction_' + g]['ci_hi'])}]" for g in ("entry", "exit", "other")) + ".")
    pa = c2["parts"]["gpt_A"]["partial"]
    pq = q["partial"]
    L.append(f"- C2 partial associations with correctness given C + M_all: gpt A prev_aff r {_f(pa['prev_aff']['r'], 3)} "
             f"[{_f(pa['prev_aff']['ci_lo'], 3)}, {_f(pa['prev_aff']['ci_hi'], 3)}], bJSD r {_f(pa['bJSD']['r'], 3)} "
             f"[{_f(pa['bJSD']['ci_lo'], 3)}, {_f(pa['bJSD']['ci_hi'], 3)}]; qwen36 prev_aff r {_f(pq['prev_aff']['r'], 3)}, "
             f"bJSD r {_f(pq['bJSD']['r'], 3)} [{_f(pq['bJSD']['ci_lo'], 3)}, {_f(pq['bJSD']['ci_hi'], 3)}] "
             f"(p {_f(pq['bJSD']['p'], 3)}, nominal, 8 features tested, no adjustment).")
    b = c2["parts"]["gpt_B"]["variants"]
    L.append(f"- C2 gpt B within-question contrast (correct - wrong), {c2['parts']['gpt_B']['mixed_questions']} questions: "
             f"prev_aff d {_f(b['raw']['prev_aff']['d'], 2)} (p {_f(b['raw']['prev_aff']['p'], 3)}); bJSD d "
             f"{_f(b['raw']['bJSD']['d'], 2)} [{_f(b['raw']['bJSD']['ci_lo'], 2)}, {_f(b['raw']['bJSD']['ci_hi'], 2)}], "
             f"p {_f(b['raw']['bJSD']['p'], 4)} (length-adjusted d {_f(b['length_adjusted']['bJSD']['d'], 2)}, p "
             f"{_f(b['length_adjusted']['bJSD']['p'], 4)}). Correct attempts have lower boundary JSD within a question, "
             "but across questions bJSD has the opposite sign (partial r > 0 in gpt A and qwen36 A) and adds nothing "
             "predictively, so it is not a same-direction replication. Exploratory lead only; qwen36 B has 13 pairs in 8 "
             "attempts (not estimable).")
    ba = c6.get("burden_association", {})
    pc = c6.get("partial_correctness", {})
    L.append(f"- C6 secondary: x vs future Explore burden (n {ba.get('n')}): partial r {_f(ba.get('r'), 3)} "
             f"[{_f(ba.get('ci_lo'), 3)}, {_f(ba.get('ci_hi'), 3)}], p {_f(ba.get('p'), 3)}; x vs correctness given the "
             f"prefix controls: partial r {_f(pc.get('r'), 3)} [{_f(pc.get('ci_lo'), 3)}, {_f(pc.get('ci_hi'), 3)}].")
    L += ["", "Power: the qwen36 tests have 39-41 wrong attempts (C6 442 scored, C2 replication 521 scored), so a "
          "DEPRIORITISED call there rests on the CI upper bound (< 0.005) and a null-biased estimator, not on precision.", ""]
    return L


def main() -> int:
    c1 = json.loads((RESULTS / "C1" / "results.json").read_text())["primary"]
    c2 = json.loads((RESULTS / "C2" / "results.json").read_text())
    c6 = json.loads((RESULTS / "C6" / "results.json").read_text())
    rows = {
        "C1": dict(cohort="gpt A", test="C + {M_all, margin} vs C", gain=c1["gain_nats_per_attempt"],
                   lo=c1["ci95"][0], hi=c1["ci95"][1], p=c1["p_one_sided"]),
        "C2": dict(cohort="gpt A", test="C + M_all + {prev_aff, bJSD} vs C + M_all", gain=c2["primary"]["gain"],
                   lo=c2["primary"]["ci"][0], hi=c2["primary"]["ci"][1], p=c2["primary"]["p_one_sided"]),
        "C6": dict(cohort="qwen36 dev+tune A", test="prefix-C + x vs prefix-C", gain=c6["primary"]["gain"],
                   lo=c6["primary"]["ci_lo"], hi=c6["primary"]["ci_hi"], p=c6["primary"]["p_one_sided"]),
    }
    p_list = [rows["C1"]["p"], rows["C2"]["p"], rows["C6"]["p"], 1.0]
    adj = holm(p_list)
    out = {}
    for name, a in zip(FAMILY, adj):
        if name in rows:
            r = rows[name]
            cond, v = verdict(r["gain"], r["lo"], r["hi"], a)
            out[name] = dict(**r, holm_adjusted_p=a, conditions=cond, verdict=v)
    res = dict(family=list(FAMILY), p_values=p_list, holm_adjusted=adj,
               note="4th slot = plan's C8 ablation family, not separately defined: p = 1 used", candidates=out)
    (RESULTS / "SUMMARY.json").write_text(json.dumps(clean(res), indent=1))
    L = ["# Routing x reasoning-dynamics: C1, C2, C6 (D1 + D2) and the D3 replication", "",
         "Plan: `ANALYSIS_PLAN.md` (frozen; Addendum 1 accepted). Existing labels only, CPU only. Primary = question-weighted "
         "out-of-fold log-loss gain (4 folds x 5 repeats, paired question bootstrap 1,000). Holm across the four primary "
         "tests; the 4th slot is the plan's C8 ablation family, which is not separately defined, so p = 1 is used (this "
         "makes the adjustment conservative: it multiplies the smallest p by 4).", "",
         "| candidate | primary test | gain (nats/attempt) | 95% CI | one-sided p | Holm p | verdict |",
         "|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        L.append(f"| {name} | {r['cohort']}: {r['test']} | {_f(r['gain'])} | [{_f(r['lo'])}, {_f(r['hi'])}] | "
                 f"{_f(r['p'], 4)} | {_f(r['holm_adjusted_p'], 4)} | **{r['verdict']}** |")
    L += ["", "Rules: ADVANCES = Holm p < .05 and 95% CI excludes 0 and gain >= 0.005; DEPRIORITISED = CI upper bound < 0.005 "
          "(and not advancing); otherwise INCONCLUSIVE. A cross-model claim additionally needs a same-direction "
          "secondary replication. Under the null the out-of-fold gain is biased by about -df/(2 n_train) "
          "(Addendum 1), which makes DEPRIORITISED slightly easier and ADVANCES no easier.", ""]
    L += secondary(c2, c6)
    d3 = RESULTS / "D3" / "results.json"
    if d3.exists():
        from .d3 import replication_section
        L += replication_section(json.loads(d3.read_text()))
    for name in ("C1", "C2", "C6"):
        L += [f"## {name} details", "", f"See `{name}/summary.md` and `{name}/results.json`.", ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(json.dumps(clean(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
