"""Freeze an arm-blind discovery diagnostic for candidate-start measurement.

The old Qwen ratings select strata offline.  Only the original problem and
already-emitted text enter the reader frame; the stratum key is stored apart.
This is a measurement diagnostic, not an independent validation sample.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
SOURCE = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
AGREEMENT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
FRAME = BASE / "CANDIDATE_START_DIAGNOSTIC_FRAME_v1.json"
KEY = REPO / "report/experimental-resume-v1/CANDIDATE_START_DIAGNOSTIC_KEY_v1.json"
ALLOWLIST = ("problem", "emitted_prefix", "triggering_sentence")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def write_new(path, body):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1,
                               ensure_ascii=False) + "\n")


def main():
    source, agreement = sealed(SOURCE), sealed(AGREEMENT)
    if (source["sha256"] != agreement["frame_sha256"]
            or source["rows"] != len(source["records"])):
        raise ValueError("source binding or row count mismatch")
    by_uid = {r["uid"]: r for r in source["records"]}
    if len(by_uid) != len(source["records"]):
        raise ValueError("duplicate source UID")
    candidate = [r for r in agreement["records"] if r["transition"] == "candidate_to_verify"]
    approved = [r for r in candidate if r["qwen_status"] == "accepted"]
    controls = [r for r in candidate if r["qwen_status"] == "rejected" and r["v22_fired"]]
    if len(approved) != 26 or len(controls) != 78:
        raise ValueError("candidate strata changed")
    selected = list(approved)
    remaining = {r["uid"]: r for r in controls}
    matches = []
    for positive in sorted(approved, key=lambda r: digest(r["uid"])):
        a = by_uid[positive["uid"]]
        options = sorted(remaining.values(), key=lambda r: (
            r["family"] != positive["family"],
            abs(by_uid[r["uid"]]["prefix_tokens"] - a["prefix_tokens"]),
            digest(r["uid"])))
        control = options[0]
        selected.append(control)
        matches.append({"approved_uid": positive["uid"], "control_uid": control["uid"],
                        "same_family": positive["family"] == control["family"],
                        "prefix_token_gap": abs(by_uid[control["uid"]]["prefix_tokens"] - a["prefix_tokens"])})
        del remaining[control["uid"]]
    selected.sort(key=lambda r: digest("candidate-diagnostic-v1:" + r["uid"]))
    records = []
    for row in selected:
        source_row = by_uid[row["uid"]]
        inp = source_row["reader_input"]
        if set(inp) != set(ALLOWLIST):
            raise ValueError("forbidden or missing reader input")
        records.append({"uid": row["uid"], "reader_input": {k: inp[k] for k in ALLOWLIST}})
    frame = {"schema": "candidate-start-diagnostic-frame-v1",
             "source_frame_sha256": source["sha256"],
             "visible_input_allowlist": list(ALLOWLIST), "rows": len(records),
             "scope": "discovery-only, arm-blind measurement diagnostic; no future labels or answers",
             "records": records}
    key = {"schema": "candidate-start-diagnostic-key-v1",
           "source_frame_sha256": source["sha256"],
           "agreement_sha256": agreement["sha256"],
           "blind_frame_sha256": digest(frame),
           "selection": "all 26 Qwen-both-true candidate rows, each paired to a distinct Qwen-both-false v2.2-fire control; same-family and nearest-prefix-token matching preferred",
           "limitations": "Earlier same-model labels choose strata. This frame cannot estimate population precision or validate a new threshold independently.",
           "counts": {"approved": len(approved), "matched_controls": len(matches),
                      "same_family_matches": sum(m["same_family"] for m in matches)},
           "matches": matches,
           "strata": [{"uid": r["uid"], "qwen_status": r["qwen_status"],
                       "native_status": r["native_status"],
                       "v22_fired": r["v22_fired"], "family": r["family"]} for r in selected]}
    # The blind file is written first; no reader should ever receive the key.
    write_new(FRAME, frame)
    write_new(KEY, key)
    print(json.dumps({"frame": str(FRAME), "key": str(KEY),
                      "rows": len(records), "same_family_matches": key["counts"]["same_family_matches"],
                      "frame_sha256": digest(frame), "key_sha256": digest(key)}))


if __name__ == "__main__":
    main()
