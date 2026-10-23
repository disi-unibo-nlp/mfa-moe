"""P-A1 stage 1: the complete adjacent labelled pair table of gpt cohort A (self-transitions included).

A pair is (s, s+1): both sentence units labelled, consecutive sentence indices, same reasoning segment.
Row = the SOURCE sentence s; target = the class of s+1. Sentences of the 37 questions that also form
GPT cohort B are dropped (reserved: no fitting, no evaluation). Text is `unit.text` of the label record
of s, joined on (dataset, problem_id, sample_id, trace_sha256, sentence_index) as in `extract.build_tasks`
(`v3an.data.load_labels` key) and verified against the D1 sentence table (class and offsets).

CLI: python -m dynrt.d3a_pairs   -> results/A1/pairs.parquet, pair_summary.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLASSES, RESULTS, V3R2, cohort_dir
from .data import load_cohort, load_outcomes

OUT = RESULTS / "A1"
LABELS_ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/labels")
NC = len(CLASSES)


@dataclass
class PairSet:
    """Pair rows of the analysed questions (arrays aligned; `src` indexes the D1 sentence table)."""
    df: pd.DataFrame          # attempt, question, dataset, src, dst, a, b, n_tokens, abs_bin, capped, difficulty, text
    n_dropped_reserved: int   # pairs dropped because the question is also in cohort B


def load_unit_texts(model: str, wanted_keys: set[tuple]) -> dict[tuple, tuple]:
    """(dataset, problem_id, sample_id, trace_sha256, sentence_index) -> (text, label, start, end)."""
    out: dict[tuple, tuple] = {}
    for part in sorted((LABELS_ROOT / model).glob("part-*")):
        f = part / "annotations.json"
        if not f.exists():
            continue
        for rec in json.loads(f.read_text()):
            ident, unit = rec["identity"], rec["unit"]
            key = (ident["dataset"], ident["problem_id"], int(ident["sample_id"]), ident["trace_sha256"])
            if key not in wanted_keys:
                continue
            k5 = key + (int(ident["sentence_index"]),)
            if k5 in out:
                raise ValueError(f"duplicate label record {k5}")
            if int(unit["index"]) != int(ident["sentence_index"]) or unit["start"] != ident["start"] \
                    or unit["end"] != ident["end"]:
                raise ValueError(f"unit/identity disagree for {k5}")
            out[k5] = (unit["text"], rec.get("label"), int(ident["start"]), int(ident["end"]))
    return out


def build_pair_table(model: str = "gpt", cohort: str = "A", *, with_text: bool = True) -> tuple[PairSet, dict]:
    """Complete pair table for the analysed questions plus a summary of counts and checks."""
    coh = load_cohort(model, cohort)
    att, sent = coh.attempts, coh.sent
    reserved = set(pd.read_parquet(V3R2 / model / "B" / "attempts.parquet")["question"])
    n = len(sent)
    idx = np.flatnonzero(sent["prev_same_segment"].to_numpy())          # successor rows
    src = idx - 1
    if len(idx) and idx.min() < 1:
        raise ValueError("first row cannot have a previous sentence")
    a_of = coh.table.attempt
    si = sent["sentence_index"].to_numpy(np.int64)
    seg = sent["segment"].to_numpy(np.int64)
    if not ((a_of[src] == a_of[idx]).all() and (si[idx] - si[src] == 1).all() and (seg[src] == seg[idx]).all()):
        raise ValueError("a flagged pair is not adjacent within one attempt and segment")
    if not (sent["next_same_segment"].to_numpy()[src]).all():
        raise ValueError("next_same_segment disagrees with prev_same_segment")
    # completeness: every adjacent same-segment labelled pair of the table is present exactly once
    adj = (a_of[1:] == a_of[:-1]) & (si[1:] - si[:-1] == 1) & (seg[1:] == seg[:-1])
    if int(adj.sum()) != len(idx) or not np.array_equal(np.flatnonzero(adj) + 1, idx):
        raise ValueError("pair table differs from the adjacency scan of the sentence table")
    cls = sent["cls"].to_numpy(np.int64)
    df = pd.DataFrame(dict(
        attempt=a_of[src].astype(np.int64), question=sent["question"].to_numpy()[src],
        dataset=sent["dataset"].to_numpy()[src], src=src.astype(np.int64), dst=idx.astype(np.int64),
        a=cls[src], b=cls[idx], n_tokens=sent["n_tokens"].to_numpy(np.int64)[src],
        abs_bin=sent["abs_bin"].to_numpy(np.int64)[src], tok_end=sent["tok_end"].to_numpy(np.int64)[src],
        sentence_index=si[src]))
    df["capped"] = att["capped"].to_numpy(bool)[df["attempt"].to_numpy()]
    out = load_outcomes(model, cohort, att)
    df["difficulty"] = out["difficulty"].to_numpy(float)[df["attempt"].to_numpy()]
    summary = dict(n_sentence_rows=int(n), n_pairs_all=int(len(df)), n_self=int((df["a"] == df["b"]).sum()),
                   n_switch=int((df["a"] != df["b"]).sum()))
    if with_text:
        keys = {(r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256) for r in att.itertuples()}
        texts = load_unit_texts(model, keys)
        by_att = att[["dataset", "problem_id", "sample_id", "trace_sha256"]].to_dict("records")
        txt, bad_label, bad_span = [], 0, 0
        for r in df.itertuples():
            rec = by_att[r.attempt]
            t = texts.get((rec["dataset"], rec["problem_id"], int(rec["sample_id"]), rec["trace_sha256"],
                           int(r.sentence_index)))
            if t is None:
                raise ValueError(f"no label record for pair source {r.attempt}/{r.sentence_index}")
            bad_label += int(t[1] != CLASSES[r.a])
            txt.append(t[0])
        df["text"] = txt
        summary.update(n_label_records_loaded=len(texts), label_class_mismatch=bad_label,
                       label_span_mismatch=bad_span)
        if bad_label:
            raise ValueError(f"{bad_label} pair sources whose label record differs from the D1 sentence table")
    reserved_mask = df["question"].isin(reserved).to_numpy()
    ps = PairSet(df[~reserved_mask].reset_index(drop=True), int(reserved_mask.sum()))
    d = ps.df
    summary.update(n_reserved_dropped=int(reserved_mask.sum()), n_pairs=int(len(d)),
                   n_questions=int(d["question"].nunique()), n_attempts=int(d["attempt"].nunique()),
                   n_reserved_questions=int(len(reserved & set(att["question"]))),
                   n_attempts_analysed=int((~att["question"].isin(reserved)).sum()),
                   n_capped_pairs=int(d["capped"].sum()), n_pairs_difficulty_nan=int(d["difficulty"].isna().sum()))
    counts = np.zeros((NC, NC), np.int64)
    np.add.at(counts, (d["a"].to_numpy(), d["b"].to_numpy()), 1)
    summary["transition_counts"] = counts.tolist()
    summary["per_source_class"] = {
        CLASSES[c]: dict(n=int(counts[c].sum()), n_self=int(counts[c, c]),
                         self_share=float(counts[c, c] / max(counts[c].sum(), 1)),
                         n_questions=int(d.loc[d["a"] == c, "question"].nunique())) for c in range(NC)}
    summary["per_target_class"] = {CLASSES[c]: int(counts[:, c].sum()) for c in range(NC)}
    return ps, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ps, summary = build_pair_table()
    ps.df.to_parquet(OUT / "pairs.parquet")
    from scipy import sparse

    from .d3a_text import hash_counts
    counts = hash_counts(ps.df["text"].tolist())
    sparse.save_npz(OUT / "text_counts.npz", counts, compressed=True)
    summary.update(text_nnz=int(counts.nnz), text_rows=int(counts.shape[0]),
                   text_nnz_per_row=float(counts.nnz / max(counts.shape[0], 1)),
                   text_buckets_used=int(len(np.unique(counts.indices))))
    (OUT / "pair_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "transition_counts"}, indent=1))
    print("transition counts (rows a, cols b):", cohort_dir("gpt", "A").name, CLASSES)
    for c, row in zip(CLASSES, summary["transition_counts"]):
        print(f"{c:10s}", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
