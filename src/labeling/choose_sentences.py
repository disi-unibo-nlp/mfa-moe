"""Pick the sentences to label for one model of the card corpus: cohorts A and B.

Input: `--traces-dir`, the model's finished generations `<dataset>/traces.jsonl`; the model is read from the traces
(`metadata.generation_config.profile`, which must be the same in every trace). Output in `--out`, a folder that does
not exist yet: `cohorts.json`, `parts.json` and four label part roots
`labels/part-0X/generation/<slug>/<dataset>/traces.jsonl` that `labeling.label_sentences` labels
(`scripts/labeling/choose_sentences.sh TRACES_DIR OUT_DIR` runs this).

Cohort A: one attempt per question, drawn uniformly at random (inclusion probability
1/attempts-for-that-question), labeled with a shared budget of sentences (every unit of
a cohort-A trace is equally likely, with a per-attempt floor).
Cohort B: AIME24/AIME25/AMC23 questions with >=2 successes and >=2 failures (finished-wrong
or length-capped; unscored attempts are neither), up to 4 successes + 4 failures each,
with a fixed number of labeled sentences per attempt.
Outcome states: correct | finished_wrong | capped | unscored (never folded together).

Traces that cannot be segmented are excluded and listed in cohorts.json (at most 0.5%).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from generation.model_profiles import single_profile, trace_profile
from labeling.spans import sentence_spans, trace_digest
from schemas import TraceRecord
from store import DATASETS

PAIRED = ("aime24", "aime25", "amc23")
SCHEMA_VERSION = 1
# sentences labelled per model (cohort A's shared budget plus cohort B's fixed shares)
LABEL_TOTAL = 100_000
PER_B = 60  # labelled sentences per cohort-B attempt
FLOOR = 10  # labelled sentences per cohort-A attempt at least
PARTS = 4
SEED = 20260925
MAX_EXCLUDED_FRACTION = 0.005


def outcome_state(row: dict) -> str:
    meta = row.get("metadata") or {}
    if (meta.get("termination") or meta.get("finish_reason")) == "length":
        return "capped"
    if row.get("is_correct") is True:
        return "correct"
    if row.get("is_correct") is False:
        return "finished_wrong"
    return "unscored"


def scan(traces_dir: Path) -> tuple[list[dict], list[dict], list[str], str]:
    """Cohort-eligible rows, traces excluded because they cannot be segmented, the
    sha256 of every corpus line in file order (bound into traces_sha256) and the model
    (the short key that every trace names in its generation contract)."""
    rows, excluded, line_hashes, profiles = [], [], [], set()
    for dataset in DATASETS:
        path = traces_dir / dataset / "traces.jsonl"
        offset = 0
        with path.open("rb") as handle:
            for line_index, raw in enumerate(handle):
                row = json.loads(raw)
                trace = TraceRecord(**row)
                profiles.add(trace_profile(row))
                line_sha = hashlib.sha256(raw).hexdigest()
                line_hashes.append(line_sha)
                try:
                    units = sentence_spans(trace)
                except ValueError as error:
                    excluded.append(dict(dataset=dataset, problem_id=row["problem_id"],
                                         line_index=line_index, reason=str(error),
                                         state=outcome_state(row)))
                else:
                    rows.append(dict(
                        dataset=dataset, problem_id=row["problem_id"],
                        source_problem_id=row.get("source_problem_id") or row["problem_id"],
                        sample_id=row.get("sample_id", 0), trace_path=str(path),
                        line_index=line_index, byte_offset=offset, line_sha256=line_sha,
                        trace_sha256=trace_digest(trace), units=len(units),
                        completion_tokens=len((row["metadata"].get("token_replay") or {})
                                              .get("completion_token_ids") or []),
                        termination=(row["metadata"].get("termination")
                                     or row["metadata"].get("finish_reason")),
                        state=outcome_state(row)))
                offset += len(raw)
    total = len(rows) + len(excluded)
    if total and len(excluded) / total > MAX_EXCLUDED_FRACTION:
        raise ValueError(f"{len(excluded)} of {total} traces cannot be segmented: {excluded[:3]}")
    return rows, excluded, line_hashes, single_profile(profiles)


def select(rows: list[dict]) -> dict:
    by_question = defaultdict(list)
    for r in rows:
        by_question[(r["dataset"], r["source_problem_id"])].append(r)
    cohort_a, cohort_b = [], []
    for (dataset, question), attempts in sorted(by_question.items()):
        rng = random.Random(f"{SEED}:A:{dataset}:{question}")
        pick = rng.choice(sorted(attempts, key=lambda r: r["sample_id"]))
        cohort_a.append(dict(pick, cohort="A", stratum=pick["state"], inclusion_probability=1 / len(attempts)))
        if dataset in PAIRED:
            ok = [r for r in attempts if r["state"] == "correct"]
            bad = [r for r in attempts if r["state"] in ("finished_wrong", "capped")]
            if len(ok) >= 2 and len(bad) >= 2:
                rng = random.Random(f"{SEED}:B:{dataset}:{question}")
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
        rng = random.Random(f"{SEED}:Bsent:{r['problem_id']}")
        picked = rng.sample(range(r["units"]), min(PER_B, r["units"]))
        selection[(r["dataset"], r["problem_id"])].update(picked)
    # cohort A: per-attempt floor, then uniform over remaining units of all cohort-A traces
    pool = []
    for r in cohort_a:
        key = (r["dataset"], r["problem_id"])
        rng = random.Random(f"{SEED}:Afloor:{r['problem_id']}")
        units = list(range(r["units"]))
        floor_pick = rng.sample(units, min(FLOOR, len(units)))
        selection[key].update(floor_pick)
        pool.extend((key, u) for u in units if u not in selection[key])
    a_used = sum(min(FLOOR, r["units"]) for r in cohort_a)
    remaining = max(0, LABEL_TOTAL - a_used)
    random.Random(f"{SEED}:Apool").shuffle(pool)
    for key, unit in pool[:remaining]:
        selection[key].add(unit)
    return dict(cohort_a=cohort_a, cohort_b=cohort_b,
                selection={k: sorted(v) for k, v in selection.items()})


def write(model: str, traces_dir: Path, out: Path, chosen, excluded, line_hashes) -> None:
    out.mkdir(parents=True)
    slug = traces_dir.name
    selection = chosen["selection"]
    manifest_core = dict(schema_version=SCHEMA_VERSION, model=model, seed=SEED,
                         label_total=LABEL_TOTAL, per_b=PER_B, floor=FLOOR,
                         traces_sha256=hashlib.sha256("".join(line_hashes).encode()).hexdigest())
    manifest_sha = hashlib.sha256(json.dumps(manifest_core, sort_keys=True).encode()).hexdigest()
    # "campaign-v3" and "card-v3" are the names stored in the sentence selections and the cohort file.
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
        excluded=excluded, rows=cohort_rows)))
    # label parts: whole traces, greedy-balanced by selected sentence count
    labeled = sorted(((k, v) for k, v in selection.items() if v), key=lambda kv: -len(kv[1]))
    parts = [[] for _ in range(PARTS)]
    load = [0] * PARTS
    for key, indices in labeled:
        i = load.index(min(load))
        parts[i].append(key)
        load[i] += len(indices)
    state, cohorts_of = {}, defaultdict(set)
    for r in chosen["cohort_a"] + chosen["cohort_b"]:
        state[(r["dataset"], r["problem_id"])] = r["state"]
        cohorts_of[(r["dataset"], r["problem_id"])].add(r["cohort"])
    part_of = {key: index for index, keys in enumerate(parts) for key in keys}
    roots = [out / "labels" / f"part-{index:02d}" / "generation" / slug for index in range(PARTS)]
    for dataset in DATASETS:
        targets = [root / dataset / "traces.jsonl" for root in roots]
        temporaries = [target.with_name(".traces.jsonl.tmp") for target in targets]
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        sinks = [temporary.open("w") for temporary in temporaries]
        try:
            with (traces_dir / dataset / "traces.jsonl").open() as source:
                for line in source:
                    row = json.loads(line)
                    key = (dataset, row["problem_id"])
                    if key not in part_of:
                        continue
                    row["metadata"]["sentence_selection"] = dict(
                        base_sel, stratum={"state": state[key], "cohorts": sorted(cohorts_of[key])},
                        indices=selection[key])
                    sinks[part_of[key]].write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            for sink in sinks:
                sink.flush()
                os.fsync(sink.fileno())
                sink.close()
        for temporary, target in zip(temporaries, targets):
            os.replace(temporary, target)
    part_totals = [dict(part=index, trace_root=str(root), total=load[index],
                        traces=len(parts[index])) for index, root in enumerate(roots)]
    (out / "parts.json").write_text(
        json.dumps(dict(manifest_sha256=manifest_sha, parts=part_totals), indent=1))
    print(json.dumps(dict(model=model, A=len(chosen["cohort_a"]), B=len(chosen["cohort_b"]),
                          labeled=sum(load), parts=load)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", type=Path, required=True,
                    help=".../card-v3/<model>/generation/<slug>, the folder holding <dataset>/traces.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="a new folder")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; the cohorts of a model are never overwritten, choose a new folder")
    rows, excluded, line_hashes, model = scan(args.traces_dir)
    write(model, args.traces_dir, args.out, select(rows), excluded, line_hashes)


if __name__ == "__main__":
    main()
