"""Discovery-only audit of v2 prefix proposals against saved native streams.

The full replay needs a CPU Slurm allocation.  --sample is a small development
check on the already rated 619 rows and cannot establish operational timing.
No validation or confirm family is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery"
RATINGS = BASE / "ratings-f5b2e28c-74c16a5d"
OUT = REPO / "report/experimental-resume-v1/TRANSITION_DETECTOR_V2_DISCOVERY_AUDIT.json"
TRANSITIONS = ("candidate_to_verify", "approach_to_commit", "failed_check_to_revise")
sys.path.insert(0, str(REPO / "src"))
from moe_exp.routing_control.transitions_v2 import StreamingTransitionDetectorV2, classify_sentence_v2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def ratings():
    frame = sealed(BASE / "TRANSITION_AUDIT_CANDIDATES.json")
    summary = sealed(RATINGS / "SUMMARY.json")
    binding = sealed(RATINGS / "BINDING.json")
    if binding["fixtures_sha256"] != frame["sha256"] or summary["binding_sha256"] != binding["sha256"]:
        raise ValueError("ratings not bound to candidate frame")
    by_uid = {}
    for start in range(0, len(frame["records"]), binding["batch_size"]):
        part = sealed(RATINGS / "batches" / f"{start:06d}.json")
        if part["start"] != start or part["binding_sha256"] != binding["sha256"]:
            raise ValueError("rating batch changed")
        by_uid.update((row["uid"], row) for row in part["records"])
    if set(by_uid) != {row["uid"] for row in frame["records"]}:
        raise ValueError("rating UID coverage differs")
    return frame, by_uid, summary


def judged(rating):
    readers = rating["readers"]
    if not all(x["finish_reason"] == "stop" and x["rating"] is not None for x in readers):
        return None
    return {"start_both": all(x["rating"]["start"] for x in readers),
            "start_either": any(x["rating"]["start"] for x in readers),
            "target_both": all(x["rating"]["target"] for x in readers)}


def sample():
    frame, by_uid, summary = ratings()
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    for row in frame["records"]:
        label = row["transition"]
        proposed = label in classify_sentence_v2(row["reader_input"]["triggering_sentence"])
        # This development approximation excludes the v1 complete-candidate
        # fallback and may differ from exact streaming prefix replay.
        vote = judged(by_uid[row["uid"]])
        if vote is None:
            counts[label]["unresolved"] += 1
            continue
        key = f"rule_{int(proposed)}_start_{int(vote['start_both'])}"
        counts[label][key] += 1
        if proposed and len(examples[label]) < 8:
            examples[label].append(row["reader_input"]["triggering_sentence"][:180])
    print(json.dumps({"schema": "transition-v2-development-sample",
                      "frame_sha256": frame["sha256"],
                      "rating_summary_sha256": summary["sha256"],
                      "counts": {k: dict(v) for k, v in counts.items()},
                      "proposed_examples": examples,
                      "warning": "in-sample, previously inspected discovery examples; not a qualification estimate"},
                     ensure_ascii=False, indent=2))


def trace_at(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location["path"]).open("rb") as stream:
        stream.seek(int(location["byte_offset"]))
        raw = stream.read(int(location["line_bytes"]))
    if hashlib.sha256(raw).hexdigest() != location["line_sha256"]:
        raise ValueError("native trace changed")
    return json.loads(raw)


def full():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("exact v2 native-prefix replay requires CPU Slurm")
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest
    units = sealed(BASE / "UNITS.json")
    frame, by_uid, rating_summary = ratings()
    source = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                             columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(row["attempt_id"]): row for row in source.to_dict("records")}
    grouped = defaultdict(list)
    for unit in units["records"]:
        grouped[unit["attempt_id"]].append(unit)
    if len(grouped) != 48:
        raise ValueError("not the frozen 48-family discovery set")
    full_counts = defaultdict(Counter)
    fire_families = defaultdict(set)
    eligible = defaultdict(list)
    sample_by_key = {(r["transition"], r["analysis_meta"]["attempt_id"],
                      r["analysis_meta"]["source_sentence_index"]): r
                     for r in frame["records"]}
    sample_counts = defaultdict(Counter)
    sample_examples = defaultdict(list)
    pair_count = 0
    for attempt in sorted(grouped):
        row = by_attempt[attempt]
        trace = trace_at(row["source_location"])
        if trace_digest(TraceRecord(**trace)) != row["trace_sha256"]:
            raise ValueError("native trace digest differs")
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = (replay["completion_token_ids"],
                              replay["completion_offsets"], trace["cot_text"])
        if len(ids) != len(offsets) or int(offsets[-1][1]) != len(text):
            raise ValueError("native token/text alignment differs")
        ordered = sorted(grouped[attempt], key=lambda x: x["sentence_index"])
        detector = StreamingTransitionDetectorV2()
        for left, right in zip(ordered, ordered[1:]):
            if right["sentence_index"] != left["sentence_index"] + 1 or right["segment"] != left["segment"]:
                continue
            prefix_tokens = left["token_end"]
            prefix_end = int(offsets[prefix_tokens - 1][1])
            if not left["char_start"] < left["char_end"] <= prefix_end:
                raise ValueError("source sentence extends beyond available prefix")
            result = detector.observe({"problem": left["inputs"]["problem_statement"],
                                       "emitted_token_ids": ids[:prefix_tokens],
                                       "emitted_text": text[:prefix_end]})
            local = {e.transition: e for e in result["events"]
                     if left["char_start"] <= e.evidence_end <= left["char_end"]}
            pair_count += 1
            for transition in TRANSITIONS:
                fired = transition in local
                full_counts[transition]["fire" if fired else "nonfire"] += 1
                if fired:
                    fire_families[transition].add(left["family"])
                    eligible[transition].append({"family": left["family"],
                                                 "attempt_id": attempt,
                                                 "sentence_index": left["sentence_index"],
                                                 "prefix_tokens": prefix_tokens,
                                                 "event_end": local[transition].evidence_end})
                sample_row = sample_by_key.get((transition, attempt, left["sentence_index"]))
                if sample_row is None:
                    continue
                vote = judged(by_uid[sample_row["uid"]])
                if vote is None:
                    sample_counts[transition]["unresolved"] += 1
                    continue
                key = f"fire_{int(fired)}_start_{int(vote['start_both'])}"
                sample_counts[transition][key] += 1
                if fired and len(sample_examples[transition]) < 12:
                    sample_examples[transition].append({
                        "text": left["inputs"]["sentence"][:200],
                        "start_both": vote["start_both"],
                        "target_both": vote["target_both"],
                        "previously_fired": sample_row["analysis_meta"]["detector_fired"]})
    if pair_count != frame["contiguous_pairs"]:
        raise ValueError("full native window count differs from frozen frame")
    body = {"schema": "transition-detector-v2-discovery-audit",
            "job_id": os.environ["SLURM_JOB_ID"],
            "units_sha256": units["sha256"],
            "rating_frame_sha256": frame["sha256"],
            "rating_summary_sha256": rating_summary["sha256"],
            "detector_sha256": hashlib.sha256((REPO / "src/moe_exp/routing_control/transitions_v2.py").read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "families": len(grouped), "contiguous_pairs": pair_count,
            "full_counts": {k: dict(v) for k, v in full_counts.items()},
            "fire_family_counts": {k: len(v) for k, v in fire_families.items()},
            "rated_sample_counts": {k: dict(v) for k, v in sample_counts.items()},
            "rated_fire_examples": sample_examples,
            "eligible_discovery_events": eligible,
            "interpretation": "Exact emitted-prefix replay in frozen discovery families; ratings are same-model LLM audits; in-sample v2 design, not independent precision, transition effect, or human truth"}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "counts": body["full_counts"],
                      "family_counts": body["fire_family_counts"],
                      "rated_sample_counts": body["rated_sample_counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    sample() if args.sample else full()
