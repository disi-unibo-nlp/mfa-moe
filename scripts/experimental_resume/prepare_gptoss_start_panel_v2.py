"""Freeze prospective GPT-OSS measurement panel before its pilot completes.

The panel contains every selected full-prefix Qwen-approved discovery start and
the four already frozen disagreement/negative-control pilot windows.  Ratings
and selection tags stay outside the model-visible row allowlist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
POOL = BASE / "FULLPREFIX_READER_AGREED_ELIGIBLE_POOL_v0.json"
PILOT = BASE / "GPTOSS_START_PILOT_FRAME_v1.json"
SOURCE = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
OUT = BASE / "GPTOSS_START_PANEL_v2.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def model_row(row):
    return {key: row[key] for key in
            ("uid", "family", "transition", "reader_input", "prefix_tokens")}


def main():
    pool, pilot, source = sealed(POOL), sealed(PILOT), sealed(SOURCE)
    if (pool["schema"] != "fullprefix-reader-agreed-eligible-pool-v0"
            or pilot["schema"] != "gptoss-start-pilot-frame-v1"
            or source["schema"] != "transition-v22-full-prefix-start-frame-v1"
            or pool["frame_sha256"] != source["sha256"]
            or pilot["source_frame_sha256"] != source["sha256"]
            or pool["qwen_summary_sha256"] != pilot["qwen_summary_sha256"]
            or len(pool["records"]) != 23 or len(pilot["records"]) != 4):
        raise ValueError("the sealed discovery enrollment/audit sources changed")
    by_uid = {row["uid"]: row for row in source["records"]}
    if len(by_uid) != source["rows"]:
        raise ValueError("duplicate source UID")
    starts = []
    for row in pool["records"]:
        src = by_uid[row["uid"]]
        if (src["family"] != row["family"] or src["transition"] != row["transition"]
                or src["prefix_tokens"] != row["prefix_tokens"]
                or src["analysis_meta"]["v22_fired"] is not True
                or row["qwen_fullprefix_reader_status"] != "both_natural_stop_start_true"):
            raise ValueError("selected start no longer matches Qwen-rated fire")
        starts.append(model_row(src))
    controls = []
    for row in pilot["records"]:
        src = by_uid[row["uid"]]
        if model_row(src) != row:
            raise ValueError("pilot control rebound")
        controls.append(model_row(src))
    records = starts + controls
    if len(records) != 27 or len({row["uid"] for row in records}) != 27:
        raise ValueError("expanded panel size/UID changed")
    allowed = {"uid", "family", "transition", "reader_input", "prefix_tokens"}
    if any(set(row) != allowed or set(row["reader_input"]) !=
           {"problem", "emitted_prefix", "triggering_sentence"} for row in records):
        raise ValueError("GPT model input allowlist changed")
    body = {
        "schema": "gptoss-start-panel-v2",
        "source_frame_sha256": source["sha256"],
        "eligible_pool_sha256": pool["sha256"],
        "pilot_frame_sha256": pilot["sha256"],
        "qwen_summary_sha256": pool["qwen_summary_sha256"],
        "selection": "All 23 fixed-hash Qwen two-reader natural-stop approved discovery fires, plus the previously frozen two Qwen disagreements and two agreed negative controls; no GPT-OSS outcomes used",
        "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_sentence"],
        "selected_start_rows": 23,
        "frozen_control_rows": 4,
        "rows": 27,
        "selected_start_families": len({r["family"] for r in starts}),
        "total_families": len({r["family"] for r in records}),
        "records": records,
        "interpretation": "Different-model-family LLM measurement audit, not human truth; discovery-only and no causal outcomes used for selection",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)},
                              ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "rows": len(records), "families": body["total_families"]}))


if __name__ == "__main__":
    main()
