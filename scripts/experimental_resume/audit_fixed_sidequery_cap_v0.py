"""Discovery-only fixed first-K side-query cost/coverage sensitivity.

The policy uses only the running count of already emitted candidate fires in
the same attempt. The Qwen start ratings are offline outcomes and never enter
the policy. The stratified rated sample does not estimate population recall.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
SENS = REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json"
FRAME = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
AGREEMENT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
OUT = REPO / "report/experimental-resume-v1/FIXED_SIDEQUERY_CAP_SENSITIVITY_v0.json"
HORIZON = 16384


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def main():
    sensitivity, frame, agreement = sealed(SENS), sealed(FRAME), sealed(AGREEMENT)
    if (frame["sensitivity_v22_sha256"] != sensitivity["sha256"]
            or agreement["frame_sha256"] != frame["sha256"]):
        raise ValueError("detector/rated frame binding changed")
    events = [e for e in sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|candidate_to_verify"]
              if e["prefix_tokens"] <= HORIZON]
    by_attempt = defaultdict(list)
    for event in events:
        by_attempt[event["attempt_id"]].append(event)
    ranks = {}
    for attempt, rows in by_attempt.items():
        for rank, event in enumerate(sorted(rows, key=lambda x: (x["prefix_tokens"], x["sentence_index"])), 1):
            key = attempt, event["sentence_index"]
            if key in ranks:
                raise ValueError("duplicate detector event")
            ranks[key] = rank
    by_frame = {row["uid"]: row for row in frame["records"]}
    positives = []
    for rated in agreement["records"]:
        if (rated["transition"] != "candidate_to_verify" or not rated["v22_fired"]
                or rated["qwen_status"] != "accepted"):
            continue
        source = by_frame[rated["uid"]]
        key = source["attempt_id"], source["sentence_index"]
        positives.append({"uid": rated["uid"], "family": rated["family"],
                          "event_rank": ranks[key]})
    other_calls = sum(len([e for e in sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|" + transition]
                           if e["prefix_tokens"] <= HORIZON])
                      for transition in ("approach_to_commit", "failed_check_to_revise"))
    caps = []
    for k in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
        candidate_calls = sum(min(k, len(rows)) for rows in by_attempt.values())
        caps.append({"max_candidate_sidequeries_per_attempt": k,
                     "candidate_calls": candidate_calls,
                     "all_three_transition_calls_if_other_uncapped": candidate_calls + other_calls,
                     "qwen_approved_candidate_starts_kept_in_stratified_sample":
                         sum(p["event_rank"] <= k for p in positives),
                     "qwen_approved_candidate_starts_sample_total": len(positives)})
    body = {"schema": "fixed-sidequery-cap-sensitivity-v0",
            "sensitivity_sha256": sensitivity["sha256"], "frame_sha256": frame["sha256"],
            "agreement_sha256": agreement["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "utility_output_horizon_native_tokens": HORIZON,
            "candidate_attempts": len(by_attempt), "candidate_allfire_calls": len(events),
            "other_transition_allfire_calls": other_calls,
            "caps": caps, "qwen_approved_sample_event_ranks": positives,
            "interpretation": "Counts describe discovery native attempts and a stratified Qwen-rated subsample; positives are LLM audit starts, not human truth. First-K cap is a past-only cost option, not a validated online controller. No utility outcomes or future text enter the policy."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "caps": caps}), flush=True)


if __name__ == "__main__":
    main()
