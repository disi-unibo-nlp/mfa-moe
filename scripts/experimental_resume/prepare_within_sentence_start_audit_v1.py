"""Freeze a separate two-reader audit of meaningful early candidate cuts.

All 31 cuts with >=4 native-token headroom are included. Twelve short-gap
controls are fixed-hash selected without reading prior Qwen start ratings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
INVENTORY = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_TIMING_INVENTORY_v0.json"
PREFIX_FRAME = BASE / "WITHIN_SENTENCE_CLOSED_MATH_PREFIX_FRAME_v0.json"
OUT = BASE / "WITHIN_SENTENCE_START_AUDIT_FRAME_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def main():
    inventory, prefixes = sealed(INVENTORY), sealed(PREFIX_FRAME)
    if (inventory["schema"] != "within-sentence-timing-inventory-v0"
            or prefixes["schema"] != "within-sentence-closed-math-prefix-frame-v0"
            or prefixes["timing_inventory_sha256"] != inventory["sha256"]):
        raise ValueError("timing inventory/frame binding changed")
    by_uid = {row["source_uid"]: row for row in prefixes["records"]}
    if len(by_uid) != prefixes["rows"]:
        raise ValueError("duplicate early prefix")
    early = [row for row in inventory["records"] if row["status"] == "early_cut"]
    meaningful = [row for row in early if row["token_headroom"] >= 4]
    controls = [row for row in early if 1 <= row["token_headroom"] <= 3]
    if len(meaningful) != 31 or len(controls) != 96:
        raise ValueError("frozen early-cut counts changed")
    chosen = sorted(meaningful, key=lambda row: digest(["within-sentence-audit-v1", "meaningful", row["uid"]]))
    used_families = {row["family"] for row in chosen}
    for row in sorted(controls, key=lambda row: digest(["within-sentence-audit-v1", "control", row["uid"]])):
        if row["family"] in used_families:
            continue
        chosen.append(row)
        used_families.add(row["family"])
        if len(chosen) == 43:
            break
    if len(chosen) != 43 or len({r["uid"] for r in chosen}) != 43:
        raise ValueError("fewer than 12 distinct-family short-gap controls")
    records = []
    for timing in chosen:
        row = by_uid[timing["uid"]]
        if (row["family"] != timing["family"] or row["prefix_tokens"] != timing["earliest_complete_token"]
                or set(row["reader_input"]) != {"problem", "emitted_prefix", "triggering_fragment"}
                or not row["reader_input"]["emitted_prefix"].rstrip().endswith(
                    row["reader_input"]["triggering_fragment"].rstrip())):
            raise ValueError("early prefix reader input changed")
        # Only these fields are passed to a model; full original sentence and
        # older Qwen statuses stay in the sealed timing inventory, offline.
        records.append({"uid": timing["uid"], "transition": "candidate_to_verify",
                        "family": timing["family"], "attempt_id": row["attempt_id"],
                        "prefix_tokens": row["prefix_tokens"],
                        "prefix_ids_sha256": row["prefix_ids_sha256"],
                        "reader_input": row["reader_input"]})
    body = {"schema": "within-sentence-start-audit-frame-v1",
            "timing_inventory_sha256": inventory["sha256"],
            "exact_prefix_frame_sha256": prefixes["sha256"],
            "source_full_sentence_frame_sha256": inventory["source_frame_sha256"],
            "selection": "all 31 closed-math candidate cuts with >=4 native-token headroom plus 12 fixed-hash short-gap controls from families outside the meaningful set; no prior Qwen ratings or future clause used",
            "meaningful_rows": 31, "short_gap_control_rows": 12,
            "rows": len(records), "families": len({r["family"] for r in records}),
            "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_fragment"],
            "records": records,
            "primary_endpoint": "original transition primary excludes triggering sentence; unchanged",
            "interpretation": "Separate discovery-only early-cut semantic-start audit; no causal outcomes or new transition definition"}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)},
                              separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "rows": body["rows"], "families": body["families"]}), flush=True)


if __name__ == "__main__":
    main()
