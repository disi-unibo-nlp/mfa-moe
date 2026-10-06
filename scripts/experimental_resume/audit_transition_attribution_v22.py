"""Exact native-prefix sensitivity: lexical v1/v2.2 by source versus delimiter lag.

No detector receives a later sentence, class label, gold answer, or rating.
All 48 families are the frozen discovery pool.  Existing v1/v2 receipts remain
untouched.  This audit is descriptive and includes development adaptation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from audit_transition_detector_v2 import (
    BASE, REPO, ROOT, TRANSITIONS, digest, judged, ratings, sealed, trace_at,
)

sys.path.insert(0, str(REPO / "src"))
from moe_exp.routing_control.transitions import StreamingTransitionDetector
from moe_exp.routing_control.transitions_v22 import StreamingTransitionDetectorV2, VERSION

OUT = REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json"


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("full exact-prefix sensitivity requires CPU Slurm")
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
        raise ValueError("discovery family count differs")
    sample_by_key = {(row["transition"], row["analysis_meta"]["attempt_id"],
                      row["analysis_meta"]["source_sentence_index"]): row
                     for row in frame["records"]}
    counts = defaultdict(Counter)
    rated = defaultdict(Counter)
    families = defaultdict(set)
    lag = defaultdict(Counter)
    events = defaultdict(list)
    pair_count = 0
    for attempt in sorted(grouped):
        source_row = by_attempt[attempt]
        trace = trace_at(source_row["source_location"])
        if trace_digest(TraceRecord(**trace)) != source_row["trace_sha256"]:
            raise ValueError("native trace digest differs")
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = (replay["completion_token_ids"],
                              replay["completion_offsets"], trace["cot_text"])
        if len(ids) != len(offsets) or int(offsets[-1][1]) != len(text):
            raise ValueError("native token/text alignment differs")
        detectors = {"v1": StreamingTransitionDetector(),
                     "v2.2": StreamingTransitionDetectorV2()}
        ordered = sorted(grouped[attempt], key=lambda x: x["sentence_index"])
        for left, right in zip(ordered, ordered[1:]):
            if right["sentence_index"] != left["sentence_index"] + 1 or right["segment"] != left["segment"]:
                continue
            prefix_tokens = left["token_end"]
            prefix_end = int(offsets[prefix_tokens - 1][1])
            if not left["char_start"] < left["char_end"] <= prefix_end:
                raise ValueError("source extends beyond emitted prefix")
            if prefix_end > right["char_start"]:
                raise ValueError("prefix reaches future sentence text")
            prefix = {"problem": left["inputs"]["problem_statement"],
                      "emitted_token_ids": ids[:prefix_tokens],
                      "emitted_text": text[:prefix_end]}
            observed = {version: detector.observe(prefix)["events"]
                        for version, detector in detectors.items()}
            pair_count += 1
            for version, rows in observed.items():
                for attribution, upper in (("strict", left["char_end"]),
                                           ("delimiter_aware", prefix_end)):
                    local = {r.transition: r for r in rows
                             if left["char_start"] <= r.evidence_end <= upper}
                    for transition in TRANSITIONS:
                        key = (version, attribution, transition)
                        proposed = local.get(transition)
                        fired = proposed is not None
                        counts[key]["fire" if fired else "nonfire"] += 1
                        if fired:
                            families[key].add(left["family"])
                            offset_lag = proposed.evidence_end - left["char_end"]
                            lag[key][str(offset_lag)] += 1
                            events[key].append({"family": left["family"],
                                                "attempt_id": attempt,
                                                "sentence_index": left["sentence_index"],
                                                "prefix_tokens": prefix_tokens,
                                                "delimiter_lag_chars": offset_lag})
                        sample = sample_by_key.get((transition, attempt, left["sentence_index"]))
                        if sample is None:
                            continue
                        judgment = judged(by_uid[sample["uid"]])
                        if judgment is None:
                            rated[key]["unresolved"] += 1
                            continue
                        start = judgment["start_both"]
                        rated[key]["TP" if fired and start else
                                   "FP" if fired else
                                   "FN" if start else "TN"] += 1
                        if fired and start:
                            rated[key]["target_both_given_start_fire"] += judgment["target_both"]
    if pair_count != frame["contiguous_pairs"]:
        raise ValueError("contiguous frame count differs")
    def label(key):
        return "|".join(key)
    body = {"schema": "transition-prefix-attribution-sensitivity-v2.2",
            "job_id": os.environ["SLURM_JOB_ID"],
            "units_sha256": units["sha256"],
            "rating_frame_sha256": frame["sha256"],
            "rating_summary_sha256": rating_summary["sha256"],
            "v1_detector_sha256": hashlib.sha256((REPO / "src/moe_exp/routing_control/transitions.py").read_bytes()).hexdigest(),
            "v2_2_detector_sha256": hashlib.sha256((REPO / "src/moe_exp/routing_control/transitions_v22.py").read_bytes()).hexdigest(),
            "v2_2_version": VERSION,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "families": len(grouped), "contiguous_pairs": pair_count,
            "counts": {label(k): dict(v) for k, v in counts.items()},
            "fire_families": {label(k): len(v) for k, v in families.items()},
            "rated_sample_confusion": {label(k): dict(v) for k, v in rated.items()},
            "delimiter_lag_chars": {label(k): dict(v) for k, v in lag.items()},
            "eligible_discovery_events": {label(k): v for k, v in events.items()},
            "interpretation": "Same frozen discovery families; v2.2 fixes v2.1 lookahead spillover after inspecting v1 ratings, so its rated confusion is in-sample descriptive. Sampled v1 nonfires are family-spread, not uniform. LLM ratings are not human truth. No causal effect or validation-population performance."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "counts": body["counts"],
                      "fire_families": body["fire_families"],
                      "rated_sample_confusion": body["rated_sample_confusion"]}))


if __name__ == "__main__":
    main()
