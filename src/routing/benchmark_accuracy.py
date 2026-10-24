"""Outcome-only analyses of the seven generated corpora (no routing captures needed).

Reads the outcome sidecar (pointer or parquet) and writes, under --out:
  accuracy.csv       model x dataset x corpus: attempts, capped rate, strict / adjudicated /
                     Minerva-tolerance accuracy (question-clustered bootstrap 95% CI for adjudicated),
                     accuracy among finished attempts, median completion tokens
  budget.csv         per model: accuracy at the 32k budget (corpus `historical`) vs the 131k card-sampler
                     budget (corpus `card-v3`), and the 32k accuracy after continuing its truncations to
                     131k cumulative tokens
  continuation.csv   per model: truncated parents, share that finish within 131k, their accuracy
  errors.csv         per model x corpus: share of errors that are truncations
  length_gap.csv     within-question (AIME/AMC, >=2 correct and >=2 wrong) median log length gap
  summary.md
Repeated attempts are never treated as independent: CIs resample questions.

    python -m routing.benchmark_accuracy OUT_DIR [--outcomes campaign-v3/outcomes/outcomes_v3.latest.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from store import ANALYSIS, file_sha

OUTCOMES = ANALYSIS / "campaign-v3/outcomes/outcomes_v3.latest.json"
PAIRED = ("aime24", "aime25", "amc23")


def load(path):
    path = Path(path)
    if path.suffix == ".json":
        pointer = json.loads(path.read_text())
        path = Path(pointer["path"])
        if file_sha(path) != pointer["sha256"]:
            raise ValueError("outcome sidecar does not match its pointer")
    table = pd.read_parquet(path)
    table["model"] = table["model"].replace({"qwen35gptq": "qwen35"})  # the sidecar names Qwen3.5 by its checkpoint
    for column in ("strict_correct", "adjudicated_correct", "minerva_tolerance_correct"):
        table[column] = table[column].astype("float")
    table["capped"] = table["termination"].eq("length")
    table["question"] = table["dataset"] + "|" + table["source_problem_id"].fillna(table["problem_id"])
    return table, dict(path=str(path), sha256=file_sha(path))


def clustered_ci(frame, column, reps=1000, seed=0):
    """95% bootstrap CI of the attempt-weighted mean, resampling questions."""
    groups = frame.dropna(subset=[column]).groupby("question")[column].agg(["sum", "count"])
    if len(groups) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(reps, len(groups)))
    sums, counts = groups["sum"].to_numpy()[idx].sum(1), groups["count"].to_numpy()[idx].sum(1)
    return tuple(np.quantile(sums / counts, [.025, .975]))


def accuracy_table(table):
    rows = []
    for (model, dataset, corpus), frame in table[table["corpus"] != "continuation"].groupby(
            ["model", "dataset", "corpus"]):
        low, high = clustered_ci(frame, "adjudicated_correct")
        finished = frame[~frame["capped"]]
        rows.append(dict(model=model, dataset=dataset, corpus=corpus, attempts=len(frame),
                         questions=frame["question"].nunique(), capped_rate=frame["capped"].mean(),
                         strict=frame["strict_correct"].mean(), adjudicated=frame["adjudicated_correct"].mean(),
                         adjudicated_ci_low=low, adjudicated_ci_high=high,
                         minerva_tolerance=frame["minerva_tolerance_correct"].mean() if dataset == "minerva" else np.nan,
                         adjudicated_finished=finished["adjudicated_correct"].mean(),
                         median_tokens=frame["completion_tokens"].median()))
    return pd.DataFrame(rows)


def budget_table(table):
    """Accuracy at the 32k budget, at the 131k card-sampler budget, and at 32k with truncations continued."""
    hist = table[table["corpus"] == "historical"].copy()
    cont = table[table["corpus"] == "continuation"]
    key = ["model", "dataset", "problem_id"]
    merged = hist.merge(cont[key + ["adjudicated_correct", "termination"]].rename(
        columns={"adjudicated_correct": "continued_correct", "termination": "continued_termination"}),
        on=key, how="left")
    # a continued truncation replaces its parent's outcome; unresolved continuations stay wrong
    merged["with_continuation"] = np.where(merged["capped"] & merged["continued_correct"].notna(),
                                           merged["continued_correct"], merged["adjudicated_correct"])
    rows = []
    for model, frame in merged.groupby("model"):
        card = table[(table["corpus"] == "card-v3") & (table["model"] == model)]
        rows.append(dict(model=model, historical_32k=frame["adjudicated_correct"].mean(),
                         historical_capped_rate=frame["capped"].mean(),
                         historical_continued_131k=frame["with_continuation"].mean(),
                         card_sampler_131k=card["adjudicated_correct"].mean() if len(card) else np.nan,
                         card_capped_rate=card["capped"].mean() if len(card) else np.nan,
                         card_median_tokens=card["completion_tokens"].median() if len(card) else np.nan))
    return pd.DataFrame(rows)


def continuation_table(table):
    cont = table[table["corpus"] == "continuation"]
    rows = []
    for model, frame in cont.groupby("model"):
        rows.append(dict(model=model, truncated_parents=len(frame),
                         finished_within_131k=(~frame["capped"]).mean(),
                         adjudicated=frame["adjudicated_correct"].mean(),
                         adjudicated_if_finished=frame.loc[~frame["capped"], "adjudicated_correct"].mean(),
                         unresolved=frame["adjudicated_correct"].isna().mean()))
    return pd.DataFrame(rows)


def error_table(table):
    rows = []
    for (model, corpus), frame in table[table["corpus"] != "continuation"].groupby(["model", "corpus"]):
        wrong = frame[frame["adjudicated_correct"] == 0]
        rows.append(dict(model=model, corpus=corpus, errors=len(wrong),
                         truncation_share_of_errors=wrong["capped"].mean() if len(wrong) else np.nan))
    return pd.DataFrame(rows)


def length_gap_table(table):
    """Median over questions of (median log tokens wrong - median log tokens correct), finished only."""
    rows = []
    frame = table[table["dataset"].isin(PAIRED) & (table["corpus"] != "continuation")].copy()
    frame["log_tokens"] = np.log(frame["completion_tokens"].astype(float).clip(lower=1))
    for (model, corpus), group in frame.groupby(["model", "corpus"]):
        gaps = []
        for _, question in group.groupby("question"):
            right = question.loc[question["adjudicated_correct"] == 1, "log_tokens"]
            wrong = question.loc[(question["adjudicated_correct"] == 0) & ~question["capped"], "log_tokens"]
            if len(right) >= 2 and len(wrong) >= 2:
                gaps.append(wrong.median() - right.median())
        rows.append(dict(model=model, corpus=corpus, questions=len(gaps),
                         median_log_gap=float(np.median(gaps)) if gaps else np.nan,
                         wrong_over_right_length_ratio=float(np.exp(np.median(gaps))) if gaps else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--outcomes", default=str(OUTCOMES))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, provenance = load(a.outcomes)
    tables = dict(accuracy=accuracy_table(table), budget=budget_table(table),
                  continuation=continuation_table(table), errors=error_table(table),
                  length_gap=length_gap_table(table))
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(dict(outcomes=provenance, rows=len(table),
                                                         by_corpus=table["corpus"].value_counts().to_dict()), indent=1))
    lines = ["# Outcome-only results of the seven models", "", f"Outcome sidecar `{provenance['sha256'][:16]}`, {len(table)} rows.", ""]
    for name in ("budget", "continuation", "errors", "length_gap"):
        lines += [f"## {name}", "", tables[name].round(3).to_markdown(index=False), ""]
    pivot = tables["accuracy"].pivot_table(index=["model", "dataset"], columns="corpus", values="adjudicated")
    lines += ["## adjudicated accuracy by benchmark", "", pivot.round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
