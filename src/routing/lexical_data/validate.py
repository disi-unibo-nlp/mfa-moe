"""Validate the built datasets against their sources. Writes OUT_DIR/validation_report.json.

Usage: python -m routing.lexical_data.validate OUT_DIR   (the folder the builders wrote into)
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from store import ANALYSIS

if len(sys.argv) != 2:
    sys.exit('usage: python -m routing.lexical_data.validate OUT_DIR')
OUT = Path(sys.argv[1])
RECEIPTS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/research-campaign-v3-20261014-r6")  # label index, folds
CAPTURE = ANALYSIS / "campaign-v3"
MODELS = "gpt nemotron qwen35 qwen36 gemma glm qwen330b".split()
CLASSES = ["Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor"]
report = {"native": {}, "steered": {}, "checks_failed": []}


def check(cond, msg):
    if not cond:
        report["checks_failed"].append(msg)
        print("FAILED:", msg)


for m in MODELS:
    df = pd.read_parquet(OUT / f"native_{m}.parquet")
    db = sqlite3.connect(f"file:{RECEIPTS / m / 'LABEL_INDEX.sqlite'}?mode=ro&immutable=1", uri=True)
    n_receipts = db.execute("select count(*) from receipts").fetchone()[0]
    n_contexts = db.execute("select count(*) from labels").fetchone()[0]
    rc_labels = dict(db.execute("select label,count(*) from receipts group by label").fetchall())
    li = json.load(open(RECEIPTS / m / "LABEL_INDEX.json"))
    coh = json.load(open(CAPTURE / f"cohorts/{m}/cohorts.json"))
    check(len(df) == n_receipts == coh["counts"]["labeled_sentences"] == li["records"], f"{m}: row count")
    check(df.row_id.is_unique, f"{m}: row_id unique")
    lc = df.label.value_counts(dropna=False).to_dict()
    for c in CLASSES:
        check(lc.get(c, 0) == rc_labels.get(c, 0), f"{m}: label count {c}")
    check(set(df.label.dropna()) <= set(CLASSES), f"{m}: label set")
    check(df.context_hash.nunique() == n_contexts, f"{m}: unique contexts == labels table")
    conf = df[df.context_label_conflict]
    check(conf.context_hash.nunique() == li["conflicting_original_contexts"], f"{m}: conflict contexts")
    nonconf = df[~df.context_label_conflict]
    check((nonconf.index_label.fillna("NULL") == nonconf.label.fillna("NULL")).all(), f"{m}: index_label == label")
    check(df.outcome_joined.all(), f"{m}: outcomes joined")
    check(df.family_id.notna().all() and df.fold_campaign.notna().all(), f"{m}: family/fold")
    check(df.fold_campaign.between(0, 4).all() and df.fold_readout5.between(0, 4).all(), f"{m}: fold range")
    # a family is never split across folds
    check((df.groupby("family_id").fold_campaign.nunique() == 1).all(), f"{m}: family in one campaign fold")
    check((df.groupby("family_id").fold_readout5.nunique() == 1).all(), f"{m}: family in one readout fold")
    check(df.rel_pos.dropna().between(0, 1).all(), f"{m}: rel_pos range")
    tr = df.drop_duplicates("trace_digest")
    dup = df.groupby("context_hash").size()
    report["native"][m] = dict(
        rows=int(len(df)), traces=int(df.trace_digest.nunique()), questions=int(df.question.nunique()),
        families=int(df.family_id.nunique()), unique_contexts=int(n_contexts),
        rows_in_duplicate_contexts=int((df.n_rows_with_context > 1).sum()),
        max_rows_per_context=int(dup.max()),
        label_counts={c: int(lc.get(c, 0)) for c in CLASSES}, unknown=int((df.label_status == "unknown").sum()),
        conflict_contexts=int(conf.context_hash.nunique()), conflict_rows=int(len(conf)),
        cohort_traces={k: int(v) for k, v in tr.cohort.value_counts().items()},
        truncated_traces=int(tr.truncated.sum()), truncated_rows=int(df.truncated.sum()),
        strict_correct_traces=int((tr.strict_correct == True).sum()),  # noqa: E712
        strict_incorrect_traces=int((tr.strict_correct == False).sum()),  # noqa: E712
        strict_na_traces=int(tr.strict_correct.isna().sum()),
        adjudicated_correct_traces=int((tr.adjudicated_correct == True).sum()),  # noqa: E712
        adjudicated_incorrect_traces=int((tr.adjudicated_correct == False).sum()),  # noqa: E712
        adjudicated_na_traces=int(tr.adjudicated_correct.isna().sum()),
        strict_vs_adjudicated_disagree_traces=int(((tr.strict_correct == False) & (tr.adjudicated_correct == True)).sum()),  # noqa: E712
        stored_is_correct_vs_strict_disagree_traces=int((tr.stored_is_correct.notna() & tr.strict_correct.notna() & (tr.stored_is_correct != tr.strict_correct)).sum()),
        null_n_tokens=int(df.n_tokens.isna().sum()), zero_token_rows=int((df.n_tokens == 0).sum()),
        null_rel_pos=int(df.rel_pos.isna().sum()),
        prev_start_rows=int(df.prev_is_start.sum()), next_end_rows=int(df.next_is_end.sum()),
        fold_campaign_rows=[int(x) for x in np.bincount(df.fold_campaign.to_numpy(), minlength=5)],
        fold_readout5_rows=[int(x) for x in np.bincount(df.fold_readout5.to_numpy(), minlength=5)],
        dataset_rows={k: int(v) for k, v in df.dataset.value_counts().items()},
        median_sentence_chars=float(df.n_chars.median()), median_tokens=float(df.n_tokens.median()),
    )
    print(m, "ok", len(df))

# steered
for s in ("A", "B", "C", "FRESH"):
    df = pd.read_parquet(OUT / f"steered_{s}.parquet")
    rep = dict(
        rows=int(len(df)), continuations=int(df.uid.nunique()), labelled=int(df.is_labelled.sum()),
        unknown=int((df.label_status == "unknown").sum()), not_labelled=int((df.label_status == "not_labelled").sum()),
        label_counts={c: int((df.label == c).sum()) for c in CLASSES},
        arms={k: int(v) for k, v in df.drop_duplicates("uid").arm.value_counts().items()},
        arm_group_continuations={k: int(v) for k, v in df.drop_duplicates("uid").arm_group.value_counts().items()},
        action_kind_continuations={k: int(v) for k, v in df.drop_duplicates("uid").action_kind.value_counts().items()},
        transitions={k: int(v) for k, v in df.drop_duplicates("uid").transition.value_counts().items()},
        layers={k: int(v) for k, v in df.drop_duplicates("uid").layers.value_counts().items()},
        questions=int(df.question.nunique()), families=int(df.family_id.nunique()),
        families_in_native_qwen36=int(df.drop_duplicates("family_id").family_in_native_qwen36.sum()),
        continuations_question_in_native_qwen36=int(df.drop_duplicates("uid").question_in_native_qwen36.sum()),
        continuations_prefix_trace_in_native_qwen36=(None if df.prefix_trace_in_native_qwen36.isna().all()
                                                     else int(df.drop_duplicates("uid").prefix_trace_in_native_qwen36.sum())),
        phase_rows={k: int(v) for k, v in df.phase.value_counts(dropna=False).items()},
        labelled_phase_rows={k: int(v) for k, v in df[df.is_labelled].phase.value_counts().items()},
        hit_cap_continuations=int(df.drop_duplicates("uid").hit_cap.sum()),
        virtual_pulse_continuations=int(df.drop_duplicates("uid").virtual_pulse.sum()),
    )
    check(df.groupby("uid").sentence_index.apply(lambda x: (np.diff(x.to_numpy()) == 1).all()).all(),
          f"{s}: sentences contiguous within continuation")
    check(df.blind_id.is_unique, f"{s}: blind_id unique")
    check(df.groupby("uid").size().sum() == len(df), f"{s}: row sums")
    report["steered"][s] = rep
    print(s, "ok", len(df), rep["labelled"])

par = pd.read_parquet(OUT / "judge_parity_200.parquet")
ok = par[par.dense_label.notna() & (par.dense_finish_reason == "stop")]
report["judge_parity_200"] = dict(rows=len(par), covered=len(ok),
                                  agreement=float((ok.dense_label == ok.historical_label).mean()))
u = pd.read_parquet(OUT / "steered_all_labelled.parquet")
report["steered_all_labelled"] = dict(rows=len(u), by_study={k: int(v) for k, v in u.groupby("study").size().items()})
report["files"] = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.parquet"))}
json.dump(report, open(OUT / "validation_report.json", "w"), indent=1)
print("checks failed:", report["checks_failed"])
