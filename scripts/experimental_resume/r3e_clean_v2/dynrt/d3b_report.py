"""D3b: assemble results/pilot/summary.md (adds the learning curve and campaign-scale counts to d3b_summary).

  python -m dynrt.d3b_report
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from .common import RESULTS, V3R2
from .d3b_summary import _f, a2_section, b1_section, sparse_section

PILOT = RESULTS / "pilot"


def campaign_scale() -> dict:
    """Labelled-token and adjacent-pair counts of the real campaign tables (outcome columns are not read)."""
    out = {}
    for m in ("qwen36", "gpt"):
        a = pd.read_parquet(V3R2 / f"{m}/A/attempts.parquet", columns=["labelled_tokens", "labelled_units",
                                                                     "sentence_units"])
        s = pd.read_parquet(RESULTS / m / "A" / "sentences.parquet",
                            columns=["cls", "prev_same_segment", "n_tokens"])
        idx = s["prev_same_segment"].to_numpy().nonzero()[0]
        cls, nt = s["cls"].to_numpy(), s["n_tokens"].to_numpy()
        a_, b_ = cls[idx - 1], cls[idx]
        out[m] = dict(campaign_attempts_in_v3an_table=int(len(a)), labelled_tokens_all_1547=int(a["labelled_tokens"].sum()),
                      labelled_sentences_all_1547=int(a["labelled_units"].sum()),
                      analysed_questions_in_d1=int(pd.read_parquet(RESULTS / m / "A" / "sentences.parquet",
                                                                    columns=["attempt_id"])["attempt_id"].nunique()),
                      adjacent_pairs_d1=int(len(idx)), events_a_ne_b=int((a_ != b_).sum()),
                      controls_a_eq_b=int((a_ == b_).sum()),
                      events_source_ge16_tokens=int(((a_ != b_) & (nt[idx - 1] >= 16)).sum()))
    return out


def learning_section(lc: dict, scale: dict) -> list[str]:
    out = ["## Supplementary (not registered): B1 learning curve and campaign scale", "",
           "B1 needs no adjacency, so the sparse-vs-dense gap of G_joint is a training-size effect. Tables and IPF "
           "null fitted from a random fraction of the training questions' sentences (held-out questions scored on "
           "all their tokens, same folds, repeat 0):", "",
           "| fraction of sentences | G_joint (mean of draws) | draws |", "|---|---|---|"]
    for k, v in lc["curve"].items():
        out.append(f"| {k} | {_f(v['mean'])} | {len(v['draws'])} |")
    q = scale["qwen36"]
    out += ["", f"Scale: the pilot has 520,704 dense labelled tokens over 64 questions and ~37,000-39,000 tokens under the 7% "
                f"mask; the Qwen3.6 campaign table has {q['labelled_tokens_all_1547']:,} labelled tokens over "
                f"{q['campaign_attempts_in_v3an_table']} questions ({q['labelled_sentences_all_1547']:,} sentences). "
                f"A2 pairs in the D1 Qwen3.6 table (dev+tune, {q['analysed_questions_in_d1']} questions): "
                f"{q['adjacent_pairs_d1']:,} adjacent labelled pairs, {q['events_a_ne_b']:,} a->b events "
                f"({q['events_source_ge16_tokens']:,} with a source sentence of >= 16 tokens), "
                f"{q['controls_a_eq_b']:,} a->a controls; GPT A: {scale['gpt']['adjacent_pairs_d1']:,} pairs, "
                f"{scale['gpt']['events_a_ne_b']:,} events."]
    return out


def sweep_section(sw: dict) -> list[str]:
    out = ["", "A2 sparse sweep over the label fraction (supplementary, not registered; "
           f"{sw['design']['draws']} draws per fraction; the registered simulation uses 0.07):", "",
           "| label fraction | mean a->b events | contributing questions | pre-boundary bias / SD | coverage | "
           "lag-0 bias / SD | lag-0 coverage |", "|---|---|---|---|---|---|---|"]
    for k, v in sw["by_fraction"].items():
        a, b = v["pre"], v["pre_lag0"]
        out.append(f"| {k} | {v['mean_events']:.0f} | {v['mean_contributing_questions']:.1f} | "
                   f"{_f(a['bias_over_sd'], 3)} | {_f(a['coverage'], 2)} | {_f(b['bias_over_sd'], 3)} | "
                   f"{_f(b['coverage'], 2)} |")
    return out


def main() -> int:
    ov = json.loads((PILOT / "overlap.json").read_text())
    b1 = json.loads((PILOT / "b1.json").read_text())
    a2 = json.loads((PILOT / "a2.json").read_text())
    sp = json.loads((PILOT / "sparse_sim.json").read_text())
    lc = json.loads((PILOT / "learning_curve.json").read_text())
    sw = json.loads((PILOT / "sparse_sweep.json").read_text())
    scale = campaign_scale()
    (PILOT / "campaign_scale.json").write_text(json.dumps(scale, indent=1))
    lines = ["# D3b pilot summary (dense Qwen3.6 pilot; exploratory measurement feasibility)", "",
             "Correctness is not used anywhere. The pilot replays FP8 generations through an NF4 checkpoint, so "
             "effect sizes are not native-capture effect sizes.", "",
             "## Overlap with the steering-v1 split-v1 confirm set", "",
             f"Matched on (dataset, source_problem_id): {ov['pilot_questions']} pilot questions, "
             f"{ov['in_split_v1']} found in split-v1: **{ov['confirm']} confirm**, {ov['dev']} dev, "
             f"{ov['tune']} tune. Retained for steering-relevant use: **{ov['retained_dev_tune']}** (dev+tune). "
             "Every `dev_tune` result below is fitted and evaluated on those questions only; the `all64` rows are "
             "pipeline-feasibility measurements and are not steering-relevant.", ""]
    lines += b1_section(b1) + [""] + a2_section(a2) + [""] + sparse_section(sp) + sweep_section(sw) + [""]
    lines += learning_section(lc, scale) + [""]
    (PILOT / "summary.md").write_text("\n".join(lines))
    print("wrote", PILOT / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
