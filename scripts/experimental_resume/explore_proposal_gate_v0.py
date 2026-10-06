"""Exploratory burden and adapted-rating audit of a narrow prefix-only gate."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
OUT = REPO / "report/experimental-resume-v1/PREFIX_PROPOSAL_GATE_EXPLORATORY_v0.json"
sys.path.insert(0, str(REPO / "src"))
from moe_exp.routing_control.proposal_gate_v0 import VERSION, passes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed source seal: {path}")
    return value


def sentence_record(sentence):
    return {"problem": "", "emitted_prefix": sentence,
            "triggering_sentence": sentence}


def main():
    sensitivity = sealed(REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json")
    units = sealed(BASE / "UNITS.json")
    source = sealed(BASE / "TRANSITION_AUDIT_FIXTURES.json")
    scout = sealed(BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json")
    scout_batch = sealed(BASE / "ratings-v22-qwen-scout-d8f50b7e-06f8fabd/batches/000000.json")
    if scout_batch["records"] is None or [r["uid"] for r in scout_batch["records"]] != [r["uid"] for r in scout["records"]]:
        raise ValueError("scout order changed")
    event_rows = sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|candidate_to_verify"]
    events = {(r["attempt_id"], r["sentence_index"]) for r in event_rows}
    by_unit = {(u["attempt_id"], u["sentence_index"]): u for u in units["records"]}
    event_count = Counter()
    event_family = Counter()
    for event in event_rows:
        key = event["attempt_id"], event["sentence_index"]
        unit = by_unit[key]
        if unit["family"] != event["family"]:
            raise ValueError("event family differs from unit")
        event_family[event["family"]] += 1
        if passes(sentence_record(unit["inputs"]["sentence"])):
            event_count["passes"] += 1
            event_count["pass_families", event["family"]] += 1
        event_count["all"] += 1
    old_ratings = {}
    for path in sorted((BASE / "ratings-f5b2e28c-74c16a5d/batches").glob("[0-9]*.json")):
        old_ratings.update({row["uid"]: row for row in sealed(path)["records"]})
    old_counts = Counter()
    for row in source["records"]:
        if row["transition"] != "candidate_to_verify":
            continue
        meta = row["analysis_meta"]
        if (meta["attempt_id"], meta["source_sentence_index"]) not in events:
            continue
        rated = old_ratings[row["uid"]]
        readers = rated["readers"]
        if any(r["rating"] is None or r["finish_reason"] != "stop" for r in readers):
            old_counts["unresolved"] += 1
            continue
        positive = all(r["rating"]["start"] for r in readers)
        kept = passes(sentence_record(row["reader_input"]["triggering_sentence"]))
        old_counts["rated_v22_fires"] += 1
        old_counts["start_both"] += positive
        old_counts["kept"] += kept
        old_counts["kept_start_both"] += kept and positive
    scout_counts = Counter()
    scout_ratings = {r["uid"]: r for r in scout_batch["records"]}
    for row in scout["records"]:
        if not row["analysis_meta"]["v22_fired"]:
            continue
        readers = scout_ratings[row["uid"]]["readers"]
        if any(r["rating"] is None or r["finish_reason"] != "stop" for r in readers):
            scout_counts["unresolved"] += 1
            continue
        positive = all(r["rating"]["start"] for r in readers)
        kept = passes(row["reader_input"])
        scout_counts["rated_v22_fires"] += 1
        scout_counts["start_both"] += positive
        scout_counts["kept"] += kept
        scout_counts["kept_start_both"] += kept and positive
    burdens = [event_family[f] for f in sorted(event_family)]
    pass_families = sum(event_count["pass_families", f] > 0 for f in event_family)
    body = {"schema": "prefix-proposal-gate-exploratory-v0",
            "gate_version": VERSION,
            "gate_driver_sha256": hashlib.sha256((REPO / "src/moe_exp/routing_control/proposal_gate_v0.py").read_bytes()).hexdigest(),
            "sensitivity_sha256": sensitivity["sha256"], "units_sha256": units["sha256"],
            "old_rating_frame_sha256": source["sha256"],
            "scout_frame_sha256": scout["sha256"], "scout_batch_sha256": scout_batch["sha256"],
            "full_v22_candidate_fire_burden": {"fires": event_count["all"],
                                               "kept": event_count["passes"],
                                               "families_with_any_fire": len(event_family),
                                               "families_with_kept_fire": pass_families,
                                               "fires_per_family_min": min(burdens),
                                               "fires_per_family_median": statistics.median(burdens),
                                               "fires_per_family_max": max(burdens)},
            "adapted_old_local_context_ratings": dict(old_counts),
            "fresh_full_context_scout_ratings": dict(scout_counts),
            "limitations": "Exploratory lexical screen after v2.2 candidate fires. Old Qwen readers saw later sentences and are adapted only; fresh scout has 12 contexts and is not a precision estimate. Full independent 372-window audit and family-held-out semantic veto qualification remain necessary. Math-only candidates deliberately abstain."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "burden": body["full_v22_candidate_fire_burden"],
                      "old": dict(old_counts), "scout": dict(scout_counts)}))


if __name__ == "__main__":
    main()
