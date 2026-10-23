"""Build native_<model>.parquet: one row per judge-labelled sentence (campaign-v3 labels).

Read-only on repo / campaign / label dirs. Writes only under OUT.
Usage: build_native.py <model>
"""
from __future__ import annotations

import collections
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moe_exp.capture.dynamics.classes import saved_layout
from moe_exp.labeling.spans import trace_digest
from moe_exp.io import iter_jsonl
from moe_exp.schemas import TraceRecord
from moe_exp.routing.readout_analysis import family_folds

CV3 = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3")
R6 = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/research-campaign-v3-20261014-r6")
OUT = Path("/scratch_local/claude-133943/-leonardo-work-IscrC-MIOSR-lmolfett-mfa-moe-repo/"
           "88d89532-1e37-4900-bace-03c5cfadb6b4/scratchpad/lexicon/data")
OUTCOMES = CV3 / "outcomes/outcomes_v3-99bf8d59506e2893.parquet"
DATASETS = ["math500", "aime24", "aime25", "olympiad", "amc23", "minerva"]
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
ABS_EDGES = (1024, 2048, 4096, 8192, 16384)  # study/readout_fit.py
FOLD_SEED = 20260929
START, END = "<START OF RESPONSE>", "<END OF RESPONSE>"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(model: str) -> None:
    t0 = time.time()
    folds = json.load(open(R6 / "FOLDS.json"))
    qfam, fold_of = folds["question_families"], folds["folds"]
    cohorts = json.load(open(CV3 / f"cohorts/{model}/cohorts.json"))
    # A trace can be enrolled in both cohorts A and B (same selected sentences, labelled once)
    cohort_by_digest: dict[str, dict] = {}
    for r in cohorts["rows"]:
        c = cohort_by_digest.setdefault(r["trace_sha256"], dict(r, cohorts=[], p_a=None, p_b=None))
        assert c["stratum"] == r["stratum"] and c["termination"] == r["termination"]
        assert c["sentence_selection"]["indices"] == r["sentence_selection"]["indices"]
        c["cohorts"].append(r["cohort"])
        if r["cohort"] == "A":
            c["p_a"] = r["inclusion_probability"]
        else:
            c["p_b"] = r["arm_inclusion_probability"]
    outcomes = pd.read_parquet(OUTCOMES)
    outcomes = outcomes[(outcomes.corpus == "card-v3") & (outcomes.model == model)]
    out_by_key = {(r.dataset, r.problem_id): r for r in outcomes.itertuples()}

    db = sqlite3.connect(f"file:{R6 / model / 'LABEL_INDEX.sqlite'}?mode=ro&immutable=1", uri=True)
    receipts: dict[str, dict[int, tuple[str, str | None]]] = collections.defaultdict(dict)
    ctx_count: collections.Counter = collections.Counter()
    for ctx, label, source, row_index in db.execute("select context,label,source,row_index from receipts"):
        receipts[source][row_index] = (ctx, label)
        ctx_count[ctx] += 1
    index_label = dict(db.execute("select context,label from labels"))
    print(f"[{model}] receipts {sum(len(v) for v in receipts.values())} contexts {len(ctx_count)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    frames, problems = [], {}
    for p in range(4):
        part = f"part-{p:02d}"
        ann_path = CV3 / f"labels/{model}/{part}/annotations.json"
        ann = json.load(open(ann_path))
        rec = receipts[str(ann_path)]
        assert len(rec) == len(ann), (part, len(rec), len(ann))
        root = next((CV3 / f"cohorts/{model}/labels/{part}/generation").iterdir())
        traces = {}
        for ds in DATASETS:
            path = root / ds / "traces.jsonl"
            if path.is_file():
                for row in iter_jsonl(path):
                    t = TraceRecord(**row)
                    traces[trace_digest(t)] = t
        layouts, trace_info = {}, {}
        rows = []
        for i, r in enumerate(ann):
            ident, unit, inp = r["identity"], r["unit"], r["inputs"]
            tid = ident["trace_sha256"]
            t = traces[tid]
            if tid not in layouts:
                lay = saved_layout(t)
                rt = lay["reasoning_tokens"]
                rank = None
                if rt is not None:
                    rank = np.full(lay["token_count"], -1, np.int64)
                    rank[np.asarray(rt, np.int64)] = np.arange(len(rt))
                layouts[tid] = (lay, rank, len(rt) if rt is not None else None)
                co = cohort_by_digest[tid]
                o = out_by_key.get((t.dataset, t.problem_id))
                trace_info[tid] = dict(
                    cohort="+".join(sorted(co["cohorts"])), stratum=co["stratum"], termination=co["termination"],
                    truncated=co["termination"] == "length", completion_tokens=co["completion_tokens"],
                    trace_inclusion_probability=co["p_a"], arm_inclusion_probability_b=co["p_b"],
                    n_selected_in_trace=len(co["sentence_selection"]["indices"]),
                    strict_correct=None if o is None or pd.isna(o.strict_correct) else bool(o.strict_correct),
                    adjudicated_correct=None if o is None or pd.isna(o.adjudicated_correct) else bool(o.adjudicated_correct),
                    adjudication_status=None if o is None else o.adjudication_status,
                    stored_is_correct=t.is_correct,
                    outcome_joined=o is not None,
                )
            lay, rank, n_reason = layouts[tid]
            idx = ident["sentence_index"]
            toks = lay["unit_tokens"][idx]
            assert unit["text"] == inp["sentence"] == lay["units"][idx]["text"]
            assert (unit["start"], unit["end"]) == (lay["units"][idx]["start"], lay["units"][idx]["end"])
            if toks:
                r0 = int(rank[toks[0]]) if rank is not None else None
                n_tok = len(toks)
            else:
                r0, n_tok = None, 0 if toks is not None else None
            ctx, rlabel = rec[i]
            status = r.get("status", "ok")
            label = r.get("label")
            if status == "unknown":
                assert label is None
            else:
                assert label in CLASSES, label
            assert rlabel == label, (part, i, rlabel, label)
            ps = inp["problem_statement"]
            psha = sha(ps)
            problems.setdefault(psha, (ident["dataset"], ident["source_problem_id"], ps))
            question = f"{ident['dataset']}|{ident['source_problem_id']}"
            fam = qfam[question]
            rows.append(dict(
                model=model, row_id=f"{model}|{part}|{i}", annotation_part=part, annotation_row_index=i,
                context_hash=ctx, n_rows_with_context=ctx_count[ctx],
                label=label, label_status="ok" if status != "unknown" else "unknown",
                unknown_kind=(r.get("failure") or {}).get("kind") if status == "unknown" else None,
                index_label=index_label.get(ctx),
                sentence_text=inp["sentence"], prev_sentence=inp["previous_sentence"],
                next_sentence=inp["next_sentence"],
                prev_is_start=inp["previous_sentence"] == START, next_is_end=inp["next_sentence"] == END,
                problem_sha=psha, trace_digest=tid, dataset=ident["dataset"], problem_id=ident["problem_id"],
                source_problem_id=ident["source_problem_id"], question=question, sample_id=ident["sample_id"],
                family_id=fam, fold_campaign=fold_of.get(fam),
                sentence_index=idx, n_units_in_trace=len(lay["units"]),
                char_start=unit["start"], char_end=unit["end"], n_chars=unit["end"] - unit["start"],
                n_tokens=n_tok, reasoning_tok_start=r0, n_reasoning_tokens=n_reason,
                rel_pos=(r0 / n_reason) if (r0 is not None and n_reason) else None,
                rel_pos_sentence=idx / max(1, len(lay["units"]) - 1),
                abs_bin=int(np.searchsorted(np.asarray(ABS_EDGES), r0, side="right")) if r0 is not None else None,
                **trace_info[tid],
            ))
        frames.append(pd.DataFrame(rows))
        print(f"[{model}] {part}: {len(rows)} rows, {len(traces)} traces ({time.time() - t0:.0f}s)", flush=True)
        del ann, traces, layouts

    df = pd.concat(frames, ignore_index=True)
    # conflicts: a context whose index label is NULL although the row label is valid
    df["context_label_conflict"] = df.index_label.isna() & df.label.notna()
    # population-dependent readout fold (study/readout_fit.outer_folds recipe)
    df["fold_readout5"] = family_folds(df.question.to_numpy(), 5, FOLD_SEED, qfam).astype(np.int8)
    df["fold_campaign"] = df.fold_campaign.astype("int8")
    df["abs_bin"] = df.abs_bin.astype("Int8")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"native_{model}.parquet", compression="zstd", index=False)
    pd.DataFrame([dict(problem_sha=k, dataset=v[0], source_problem_id=v[1], problem_statement=v[2])
                  for k, v in problems.items()]).to_parquet(OUT / f"_problems_{model}.parquet",
                                                           compression="zstd", index=False)
    print(f"[{model}] DONE rows={len(df)} labelled={df.label.notna().sum()} "
          f"unknown={(df.label_status == 'unknown').sum()} conflicts={df.context_label_conflict.sum()} "
          f"outcome_unjoined={(~df.outcome_joined).sum()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
