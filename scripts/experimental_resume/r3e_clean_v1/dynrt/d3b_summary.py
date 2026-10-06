"""D3b: assemble results/pilot/summary.md from b1.json, a2.json, sparse_sim.json and overlap.json.

  python -m dynrt.d3b_summary
"""
from __future__ import annotations

import json
import sys

from .common import RESULTS

PILOT = RESULTS / "pilot"


def _f(x, nd=6) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def b1_section(b1: dict) -> list[str]:
    d = b1["design"]
    out = ["## B1 on the dense pilot (Qwen3.6, NF4 replay of FP8 generations; exploratory)", "",
           f"75 layer pairs (gaps 1 and 4), 256 experts, k = 8; {d['n_folds']} question folds x {d['repeats']} "
           f"repeats; {d['n_perm']} whole-top-k-set permutations; lambda = {d['lambda_']}.", "",
           "| subset | steering-relevant | questions | tokens | G_joint [95% CI] | positive repeats | "
           "perm null mean / p95 | above p95 | gate |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in b1["subsets"].items():
        g = s["observed"]["G_joint"]
        perm = s.get("permutation") or {}
        gate = s["gate"]
        out.append(f"| {name} | {s['steering_relevant']} | {s['n_questions']} | {s['n_tokens']:,} | "
                   f"{_f(g['mean'])} [{_f(g['ci_lo'])}, {_f(g['ci_hi'])}] | "
                   f"{gate['positive_repeats']}/{gate['n_repeats']} | {_f(perm.get('mean'))} / "
                   f"{_f(perm.get('p95'))} | {gate['above_permutation_p95']} | "
                   f"{'PASS' if gate['passes'] else 'FAIL'} |")
    for name, s in b1["subsets"].items():
        o = s["observed"]
        out += ["", f"### {name}", ""]
        out.append("- per-repeat G_joint: " + ", ".join(f"{x:.6f}" for x in o["per_repeat_G"]))
        out.append("- depth gap: " + "; ".join(f"gap {k} = {_f(v['mean'])} [{_f(v['ci_lo'])}, {_f(v['ci_hi'])}]"
                                              for k, v in o["gap"].items()))
        lex = s.get("lexical")
        if lex:
            out.append(f"- lexical sensitivity (200 most frequent token ids dropped = "
                       f"{100 * lex['fraction_dropped']:.1f}% of tokens): G_joint = {_f(lex['G_joint']['mean'])} "
                       f"[{_f(lex['G_joint']['ci_lo'])}, {_f(lex['G_joint']['ci_hi'])}], positive repeats "
                       f"{lex['positive_repeats']}/{lex['n_repeats']}")
        cp = s.get("class_permutation_within_token")
        if cp:
            out.append(f"- extra diagnostic (not registered), class labels shuffled within token id x position "
                       f"bin: mean G = {_f(cp['mean'])} (observed repeat 0 = {_f(cp['observed_repeat0'])}, "
                       f"retained fraction {_f(cp['retained_fraction'], 2)})")
        out.append("- per class: " + ", ".join(f"{k} {_f(v, 4)}" for k, v in o["per_class"].items()))
        out.append(f"- IPF: {o['ipf']}")
    return out


def a2_section(a2: dict) -> list[str]:
    out = ["## A2 transition-triggered time courses on the dense pilot (exploratory)", "",
           "Windows of 16 tokens ending at lags -64..+64 (stride 16); a->b (a != b) vs matched a->a boundaries "
           "(source class x position bin x source-sentence-length tercile); censored at intervening class "
           "boundaries; in-fold class profiles. Scalar = pooled excess over lags -32, -16, 0, layer-averaged, "
           "question-weighted, question-bootstrap CI (1,000).", "",
           "| subset | steering-relevant | questions | events | controls | pre-boundary excess [95% CI] | "
           "p (>0) | post-boundary excess (lags +16..+64) [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for name, s in a2["subsets"].items():
        m = s["summary"]
        pre, post = m["pre_boundary"], m["post_boundary"]
        out.append(f"| {name} | {s['steering_relevant']} | {s['n_questions']} | {m['n_events']} | "
                   f"{m['n_controls']} | {_f(pre['estimate'])} [{_f(pre['ci_lo'])}, {_f(pre['ci_hi'])}] | "
                   f"{_f(pre['p_one_sided_positive'], 4)} | {_f(post['estimate'])} "
                   f"[{_f(post['ci_lo'])}, {_f(post['ci_hi'])}] |")
    for name, s in a2["subsets"].items():
        m = s["summary"]
        out += ["", f"### {name}: valid a->b events by lag", "",
                "| lag | " + " | ".join(m["n_events_valid_by_lag"]) + " |",
                "|---|" + "---|" * len(m["n_events_valid_by_lag"]),
                "| events | " + " | ".join(str(v) for v in m["n_events_valid_by_lag"].values()) + " |"]
    out += ["", "Per-layer and lag curves: `a2_curves.csv` (layer = -1 is the layer mean) and `a2_curves.png`."]
    return out


def sparse_section(sp: dict) -> list[str]:
    d = sp["design"]
    out = ["## Sparse-mask simulation (campaign design applied to the dense pilot)", "",
           f"{d['draws']} draws; {d['sampling']}; ~{int(100 * d['fraction'])}% of sentences per trace (at least "
           f"{d['min_labels']}); identical question folds for dense and sparse. Gate: |bias| < "
           f"{d['gate']['abs_bias_over_sd']} between-question SD and coverage >= {d['gate']['coverage']}.", "",
           "| statistic | dense | sparse mean | bias | between-question SD | bias / SD | coverage | gate |",
           "|---|---|---|---|---|---|---|---|"]
    for key, label in (("B1_G_joint", "B1 G_joint"), ("A2_pre_boundary", "A2 pre-boundary scalar"),
                       ("A2_pre_lag0_secondary", "A2 lag-0 window only (secondary)")):
        c = sp[key]
        out.append(f"| {label} | {_f(c['dense'])} | {_f(c['sparse_mean'])} | {_f(c['bias'])} | "
                   f"{_f(c['between_question_sd'])} | {_f(c['bias_over_sd'], 3)} | {_f(c['coverage'], 3)} | "
                   f"{'PASS' if c['passes'] else 'FAIL'} |")
    out += ["", f"Registered gate (B1 and A2 pre-boundary scalar both pass): "
                f"**{'PASS' if sp['gate_passes'] else 'FAIL'}**."]
    return out


def main() -> int:
    ov = json.loads((PILOT / "overlap.json").read_text())
    b1 = json.loads((PILOT / "b1.json").read_text())
    a2 = json.loads((PILOT / "a2.json").read_text())
    sp = json.loads((PILOT / "sparse_sim.json").read_text())
    lines = ["# D3b pilot summary (dense Qwen3.6 pilot; exploratory measurement feasibility)", "",
             "Correctness is not used anywhere. The pilot replays FP8 generations through an NF4 checkpoint, so "
             "effect sizes are not native-capture effect sizes.", "",
             "## Overlap with the steering-v1 split-v1 confirm set", "",
             f"Matched on (dataset, source_problem_id): {ov['pilot_questions']} pilot questions, "
             f"{ov['in_split_v1']} found in split-v1: **{ov['confirm']} confirm**, {ov['dev']} dev, "
             f"{ov['tune']} tune. Retained for steering-relevant use: **{ov['retained_dev_tune']}** (dev+tune). "
             "Every `dev_tune` result below is fitted and evaluated on those questions only; the `all64` rows are "
             "pipeline-feasibility measurements and are not steering-relevant.", ""]
    lines += b1_section(b1) + [""] + a2_section(a2) + [""] + sparse_section(sp) + [""]
    (PILOT / "summary.md").write_text("\n".join(lines))
    print("wrote", PILOT / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
