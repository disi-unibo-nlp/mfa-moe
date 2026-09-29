"""Bounded audit of saved v3 tables; no tensor replay, fitting, or source mutation.

Run with an existing pandas/pyarrow environment and one BLAS thread. Outputs are
descriptive diagnostics, not new significance tests. The source hashes bind the
exact tables inspected, including the latest seven-model cross-2 summary.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd


def audit(root: Path) -> dict:
    sources = {}
    code_path = Path(__file__).resolve()
    sources[str(code_path)] = {"sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
                               "bytes": code_path.stat().st_size}

    def read(path):
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed during audit: {path}")
        sources[str(path)] = {"sha256": digest, "bytes": before.st_size}
        return result

    def records(df):
        return json.loads(df.to_json(orient="records"))

    def quantiles(series):
        return {str(q): float(v) for q, v in series.quantile([0, .1, .5, .9, 1]).items()}

    cohorts, sets, ids = [], {}, {}
    for path in sorted(root.glob("*/[AB]/attempts.parquet")):
        model, cohort = path.parent.parent.name, path.parent.name
        df = read(path)
        tag = f"{model}/{cohort}"
        sets[tag] = set(df.question)
        ids[tag] = set(df.attempt_id)
        usable = df[df.correct.notna()]
        finished = usable[~usable.capped]
        mixed = lambda d: int((d.groupby("question").correct.nunique() == 2).sum())
        row = dict(model=model, cohort=cohort, attempts=len(df), questions=df.question.nunique(),
                   scored=len(usable), correct=int(usable.correct.sum()), unknown=int(df.correct.isna().sum()),
                   capped=int(df.capped.sum()), capped_correct=int(df.loc[df.capped, "correct"].sum()),
                   mixed_questions=mixed(usable), uncapped_mixed_questions=mixed(finished),
                   duplicate_trace_hashes=int(df.trace_sha256.duplicated().sum()),
                   completion_tokens=quantiles(df.completion_tokens), reasoning_tokens=quantiles(df.n_reasoning),
                   non_reasoning_token_fraction=quantiles(1 - df.n_reasoning / df.completion_tokens),
                   dataset_counts={str(k): int(v) for k, v in df.dataset.value_counts().items()})
        if cohort == "B":
            row["window_support"] = []
            for end in [1024, 2048, 4096, 8192, 16384]:
                d = finished[finished.n_reasoning >= end]
                mq = d.groupby("question").correct.nunique()
                row["window_support"].append(dict(end=end, attempts=len(d),
                    mixed_questions=int((mq == 2).sum()),
                    attempts_in_mixed_questions=int(d.question.isin(mq[mq == 2].index).sum())))
        if cohort == "A":
            row["labelled_token_fraction"] = quantiles(df.labelled_tokens / df.n_reasoning)
            row["labelled_sentence_fraction"] = quantiles(df.labelled_units / df.sentence_units)
            row["labelled_sentences"] = int(df.labelled_units.sum())
            row["sentences"] = int(df.sentence_units.sum())
            experts = read(path.parent / "E_class_experts.csv")
            per_expert = experts.groupby(["layer", "expert"]).cls.nunique()
            row["class_expert_selections"] = len(experts)
            row["unique_selected_experts"] = len(per_expert)
            row["experts_selected_for_multiple_classes"] = int((per_expert > 1).sum())
            row["selected_experts_by_class"] = {str(k): int(v) for k, v in experts.cls.value_counts().items()}
            usage = read(path.parent / "E_expert_usage_associations.csv")
            row["expert_usage_tests"] = []
            for adjustment in ["lendiff", "lendiff_frac"]:
                d = usage[(usage.scope == "pooled") & (usage.adjustment == adjustment)]
                row["expert_usage_tests"].append(dict(adjustment=adjustment,
                    n_tests=len(d), finite_tests=int(d.q.notna().sum()),
                    bh_hits=int((d.q < .05).sum()), min_q=None if not d.q.notna().any() else float(d.q.min())))
            row["expert_usage_summary_hits"] = records(usage[(usage.scope == "pooled") &
                (usage.adjustment == "lendiff_frac") & (usage.tier == "summary") & (usage.q < .05)]
                [["feature", "n", "r", "ci_lo", "ci_hi", "q"]])
            assoc = read(path.parent / "A_associations.csv")
            row["class_fraction_results"] = records(assoc[(assoc.scope == "pooled") &
                (assoc.adjustment == "lendiff") & (assoc.family.isin(["cls_frac", "cls_tokfrac"]))]
                [["feature", "n", "n_questions", "r", "ci_lo", "ci_hi", "p", "q"]])
        cohorts.append(row)

    overlap = []
    bnames = sorted(k for k in sets if k.endswith("/B"))
    for a, b in itertools.combinations(bnames, 2):
        overlap.append(dict(a=a, b=b, shared_questions=len(sets[a] & sets[b]),
                            union_questions=len(sets[a] | sets[b])))
    within_model = []
    for tag in sorted(sets):
        if not tag.endswith("/A"):
            continue
        b = tag[:-1] + "B"
        if b in sets:
            within_model.append(dict(model=tag[:-2], shared_questions=len(sets[tag] & sets[b]),
                                     shared_attempts=len(ids[tag] & ids[b])))
    question_frequency = pd.Series([q for tag in bnames for q in sets[tag]]).value_counts()
    meta = read(root / "cross-2/F_B_within_meta.csv")
    common = read(root / "common_range/pooled.csv")
    common_models = read(root / "common_range/per_model.csv")
    return dict(cohorts=cohorts, b_pairwise_question_overlap=overlap,
                b_unique_questions=len(question_frequency),
                b_questions_in_multiple_models=int((question_frequency > 1).sum()),
                b_questions_in_all_models=int((question_frequency == len(bnames)).sum()),
                a_b_overlap=within_model,
                meta_b_columns=list(meta.columns),
                meta_b_significant_all_variants=records(meta[meta.meta_q < .05]),
                meta_b_primary_significant=records(meta[(meta.meta_q < .05) & (meta.variant == "len_nocap")]),
                common_range_min_q=float(common.q.min()),
                common_range_significant=int((common.q < .05).sum()),
                common_range_support=records(common_models[["model", "start", "variant", "n", "n_questions", "median_common_tokens"]].drop_duplicates()),
                sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Read back the exact artifact just written.
    saved = json.loads(args.output.read_text())
    print(json.dumps(dict(output=str(args.output), cohorts=len(saved["cohorts"]),
                          sources=len(saved["sources"]), b_unique_questions=saved["b_unique_questions"],
                          common_range_min_q=saved["common_range_min_q"])))


if __name__ == "__main__":
    main()
