"""Campaign v3 cohorts and sentence-label selections for the card-profile corpus.

Cohort A: one attempt per question, drawn uniformly at random (inclusion probability
1/attempts-for-that-question), labeled with a shared budget of sentences (every unit of
a cohort-A trace is equally likely, with a per-attempt floor).
Cohort B: AIME24/AIME25/AMC23 questions with >=2 successes and >=2 failures (finished-wrong
or length-capped; unscored attempts are neither), up to 4 successes + 4 failures each,
with a fixed number of labeled sentences per attempt.
Outcome states: correct | finished_wrong | capped | unscored (never folded together).

Traces that cannot be segmented are excluded and listed in cohorts.json (at most 0.5%).
Outputs (under --out): cohorts.json (for capture planning), label part roots
labels/part-XX/generation/<slug>/<dataset>/traces.jsonl, parts.json (exact label totals).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from moe_exp.correlation_pipeline.spans import sentence_spans, trace_digest
from moe_exp.schemas import TraceRecord

DATASETS = ("math500", "aime24", "aime25", "olympiad", "amc23", "minerva")
PAIRED = ("aime24", "aime25", "amc23")
SCHEMA_VERSION = 1


def outcome_state(row: dict) -> str:
    meta = row.get("metadata") or {}
    if (meta.get("termination") or meta.get("finish_reason")) == "length":
        return "capped"
    if row.get("is_correct") is True:
        return "correct"
    if row.get("is_correct") is False:
        return "finished_wrong"
    return "unscored"


MAX_EXCLUDED_FRACTION = 0.005


def scan(gen_root: Path) -> tuple[list[dict], list[dict], list[str]]:
    """Cohort-eligible rows, traces excluded because they cannot be segmented, and the
    sha256 of every corpus line in file order (bound into traces_sha256)."""
    rows, excluded, line_hashes = [], [], []
    for dataset in DATASETS:
        path = gen_root / dataset / "traces.jsonl"
        offset = 0
        with path.open("rb") as handle:
            for line_index, raw in enumerate(handle):
                row = json.loads(raw)
                trace = TraceRecord(**row)
                line_hashes.append(hashlib.sha256(raw).hexdigest())
                try:
                    units = sentence_spans(trace)
                except ValueError as error:
                    excluded.append(dict(dataset=dataset, problem_id=row["problem_id"],
                                         line_index=line_index, reason=str(error),
                                         state=outcome_state(row)))
                    offset += len(raw)
                    continue
                rows.append(dict(
                    dataset=dataset, problem_id=row["problem_id"],
                    source_problem_id=row.get("source_problem_id") or row["problem_id"],
                    sample_id=row.get("sample_id", 0), trace_path=str(path), line_index=line_index,
                    byte_offset=offset, line_sha256=hashlib.sha256(raw).hexdigest(),
                    trace_sha256=trace_digest(trace), units=len(units),
                    completion_tokens=len((row["metadata"].get("token_replay") or {}).get("completion_token_ids") or []),
                    termination=(row["metadata"].get("termination") or row["metadata"].get("finish_reason")),
                    state=outcome_state(row)))
                offset += len(raw)
    total = len(rows) + len(excluded)
    if total and len(excluded) / total > MAX_EXCLUDED_FRACTION:
        raise ValueError(f"{len(excluded)} of {total} traces cannot be segmented: {excluded[:3]}")
    return rows, excluded, line_hashes


def select(rows: list[dict], *, label_total: int, per_b: int, floor: int, seed: int) -> dict:
    by_question = defaultdict(list)
    for r in rows:
        by_question[(r["dataset"], r["source_problem_id"])].append(r)
    cohort_a, cohort_b = [], []
    for (dataset, question), attempts in sorted(by_question.items()):
        rng = random.Random(f"{seed}:A:{dataset}:{question}")
        pick = rng.choice(sorted(attempts, key=lambda r: r["sample_id"]))
        cohort_a.append(dict(pick, cohort="A", stratum=pick["state"], inclusion_probability=1 / len(attempts)))
        if dataset in PAIRED:
            ok = [r for r in attempts if r["state"] == "correct"]
            bad = [r for r in attempts if r["state"] in ("finished_wrong", "capped")]
            if len(ok) >= 2 and len(bad) >= 2:
                rng = random.Random(f"{seed}:B:{dataset}:{question}")
                # within an eligible question each success (failure) is drawn with
                # probability min(4, n)/n from its outcome arm (case-control design)
                for arm in (ok, bad):
                    for r in rng.sample(arm, min(4, len(arm))):
                        # population inclusion is undefined for this case-control cohort (null);
                        # the within-question arm probability is kept separately
                        cohort_b.append(dict(r, cohort="B", stratum=r["state"],
                                             inclusion_probability=None,
                                             arm_inclusion_probability=min(4, len(arm)) / len(arm)))
    selection = defaultdict(set)
    for r in cohort_b:
        rng = random.Random(f"{seed}:Bsent:{r['problem_id']}")
        selection[(r["dataset"], r["problem_id"])].update(rng.sample(range(r["units"]), min(per_b, r["units"])))
    # cohort A: per-attempt floor, then uniform over remaining units of all cohort-A traces
    pool = []
    for r in cohort_a:
        key = (r["dataset"], r["problem_id"])
        rng = random.Random(f"{seed}:Afloor:{r['problem_id']}")
        units = list(range(r["units"]))
        floor_pick = rng.sample(units, min(floor, len(units)))
        selection[key].update(floor_pick)
        pool.extend((key, u) for u in units if u not in selection[key])
    a_used = sum(min(floor, r["units"]) for r in cohort_a)
    remaining = max(0, label_total - a_used)
    random.Random(f"{seed}:Apool").shuffle(pool)
    for key, unit in pool[:remaining]:
        selection[key].add(unit)
    return dict(cohort_a=cohort_a, cohort_b=cohort_b,
                selection={k: sorted(v) for k, v in selection.items()})


def write(args, rows, chosen, excluded=(), line_hashes=None) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = Path(args.gen_root).name
    selection = chosen["selection"]
    manifest_core = dict(schema_version=SCHEMA_VERSION, model=args.model, seed=args.seed,
                         label_total=args.label_total, per_b=args.per_b, floor=args.floor,
                         traces_sha256=hashlib.sha256("".join(
                             line_hashes if line_hashes is not None else [r["line_sha256"] for r in rows]
                         ).encode()).hexdigest())
    manifest_sha = hashlib.sha256(json.dumps(manifest_core, sort_keys=True).encode()).hexdigest()
    base_sel = dict(schema_version=SCHEMA_VERSION, manifest_sha256=manifest_sha, selection="campaign-v3")
    cohort_rows = []
    for r in chosen["cohort_a"] + chosen["cohort_b"]:
        key = (r["dataset"], r["problem_id"])
        cohort_rows.append({k: r[k] for k in (
            "cohort", "dataset", "problem_id", "source_problem_id", "sample_id", "trace_path",
            "line_index", "byte_offset", "line_sha256", "trace_sha256", "completion_tokens",
            "termination", "stratum", "inclusion_probability")} |
            {"arm_inclusion_probability": r.get("arm_inclusion_probability")} |
            {"sentence_selection": dict(base_sel, stratum={"state": r["state"], "cohort": r["cohort"]},
                                        indices=selection.get(key, []))})
    (out / "cohorts.json").write_text(json.dumps(dict(
        **manifest_core, manifest_sha256=manifest_sha, corpus="card-v3",
        counts=dict(A=len(chosen["cohort_a"]), B=len(chosen["cohort_b"]),
                    labeled_sentences=sum(len(v) for v in selection.values()), excluded=len(excluded)),
        excluded=list(excluded), rows=cohort_rows)))
    # label parts: whole traces, greedy-balanced by selected sentence count
    labeled = sorted(((k, v) for k, v in selection.items() if v), key=lambda kv: -len(kv[1]))
    parts = [[] for _ in range(args.parts)]
    load = [0] * args.parts
    for key, indices in labeled:
        i = load.index(min(load))
        parts[i].append(key)
        load[i] += len(indices)
    state = {(r["dataset"], r["problem_id"]): r["state"] for r in rows}
    cohorts_of = defaultdict(set)
    for r in chosen["cohort_a"] + chosen["cohort_b"]:
        cohorts_of[(r["dataset"], r["problem_id"])].add(r["cohort"])
    part_totals = []
    for index, keys in enumerate(parts):
        wanted = set(keys)
        root = out / "labels" / f"part-{index:02d}" / "generation" / slug
        for dataset in DATASETS:
            target = root / dataset / "traces.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(".traces.jsonl.tmp")
            with temporary.open("w") as sink, (Path(args.gen_root) / dataset / "traces.jsonl").open() as source:
                for line in source:
                    row = json.loads(line)
                    key = (dataset, row["problem_id"])
                    if key not in wanted:
                        continue
                    row["metadata"]["sentence_selection"] = dict(
                        base_sel, stratum={"state": state[key], "cohorts": sorted(cohorts_of[key])},
                        indices=selection[key])
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary, target)
        part_totals.append(dict(part=index, trace_root=str(root), total=load[index], traces=len(keys)))
    (out / "parts.json").write_text(json.dumps(dict(manifest_sha256=manifest_sha, parts=part_totals), indent=1))
    print(json.dumps(dict(model=args.model, A=len(chosen["cohort_a"]), B=len(chosen["cohort_b"]),
                          labeled=sum(load), parts=load)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-root", required=True, help=".../card-v3/<model>/generation/<slug>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-total", type=int, default=100_000)
    ap.add_argument("--per-b", type=int, default=60)
    ap.add_argument("--floor", type=int, default=10)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args(argv)
    rows, excluded, line_hashes = scan(Path(args.gen_root))
    chosen = select(rows, label_total=args.label_total, per_b=args.per_b, floor=args.floor, seed=args.seed)
    write(args, rows, chosen, excluded, line_hashes)


if __name__ == "__main__":
    main()
