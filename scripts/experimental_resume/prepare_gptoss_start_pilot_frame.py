"""Freeze a four-prefix, Qwen-disagreement/control GPT-OSS parity frame.

Run only after the 372-window Qwen full-prefix audit is complete and verified.
Model inputs are the original problem and emitted prefix through the trigger.
Qwen ratings and detector status affect offline enrollment only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
SOURCE = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
QWEN = BASE / "ratings-v22-fullprefix-v2-6c10499b-1ef8863f"
OUT = BASE / "GPTOSS_START_PILOT_FRAME_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def main():
    source = sealed(SOURCE)
    binding = sealed(QWEN / "BINDING.json")
    summary = sealed(QWEN / "SUMMARY.json")
    batches = [sealed(path) for path in sorted((QWEN / "batches").glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))]
    rated_records = [row for batch in batches for row in batch["records"]]
    if (source["schema"] != "transition-v22-full-prefix-start-frame-v1"
            or binding["frame_sha256"] != source["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or len(rated_records) != 372
            or any(batch["binding_sha256"] != binding["sha256"] for batch in batches)
            or [r["uid"] for r in rated_records] != [r["uid"] for r in source["records"]]):
        raise ValueError("Qwen full-prefix audit is incomplete or rebound")
    row_by_uid = {r["uid"]: r for r in source["records"]}
    ambiguous, controls = [], []
    for rated in rated_records:
        readers = rated["readers"]
        valid = [r["rating"] is not None and r["finish_reason"] == "stop" for r in readers]
        if not all(valid):
            continue
        if readers[0]["rating"] != readers[1]["rating"]:
            ambiguous.append(rated["uid"])
        else:
            controls.append(rated["uid"])
    key = lambda uid: digest(["gptoss-start-pilot-v1", uid])
    ambiguous.sort(key=key)
    controls.sort(key=key)
    if len(ambiguous) < 2 or len(controls) < 2:
        raise ValueError("frozen full-prefix audit lacks two valid-reader disagreements and two agreed controls")
    chosen, chosen_families = [], set()
    for uid in ambiguous:
        if row_by_uid[uid]["family"] not in chosen_families:
            chosen.append(uid)
            chosen_families.add(row_by_uid[uid]["family"])
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise ValueError("fewer than two distinct discovery families with Qwen disagreement")
    for uid in controls:
        if row_by_uid[uid]["family"] not in chosen_families:
            chosen.append(uid)
            chosen_families.add(row_by_uid[uid]["family"])
        if len(chosen) == 4:
            break
    records = [{"uid": uid, "family": row_by_uid[uid]["family"],
                "transition": row_by_uid[uid]["transition"],
                "reader_input": row_by_uid[uid]["reader_input"],
                "prefix_tokens": row_by_uid[uid]["prefix_tokens"]} for uid in chosen]
    if len({r["family"] for r in records}) != 4:
        raise ValueError("pilot family separation failed")
    body = {"schema": "gptoss-start-pilot-frame-v1", "source_frame_sha256": source["sha256"],
            "qwen_binding_sha256": binding["sha256"], "qwen_summary_sha256": summary["sha256"],
            "qwen_batch_sha256s": [batch["sha256"] for batch in batches], "rows": 4, "families": 4,
            "selection": "two valid-reader Qwen disagreements plus two agreed controls; fixed-hash within stratum, distinct discovery families",
            "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_sentence"],
            "records": records}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "ambiguous_pool": len(ambiguous), "controls_pool": len(controls)}))


if __name__ == "__main__":
    main()
