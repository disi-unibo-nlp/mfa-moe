"""Freeze twelve independent full-prefix Qwen rating scout rows.

Four contexts in each native length bin; two v2.2 fires and two nonfires
per bin; one row per discovery family. This is a throughput/parse pilot,
not a detector precision estimate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
SOURCE = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
OUT = BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("source frame seal differs")
    return value


def length_bin(n):
    return "short" if n < 1024 else "medium" if n < 4096 else "long"


def main():
    source = sealed(SOURCE)
    if source["schema"] != "transition-v22-full-prefix-start-frame-v1" or source["rows"] != 372:
        raise ValueError("not the frozen fresh full-prefix frame")
    ordered = sorted((r for r in source["records"] if r["transition"] == "candidate_to_verify"),
                     key=lambda r: digest(["v22-qwen-scout-v1", r["uid"]]))
    chosen, families = [], set()
    for bin_name in ("short", "medium", "long"):
        for fired in (True, False):
            cell = [r for r in ordered if length_bin(r["prefix_tokens"]) == bin_name
                    and r["analysis_meta"]["v22_fired"] is fired
                    and r["family"] not in families]
            if len(cell) < 2:
                raise ValueError(f"fewer than two independent families in {bin_name}/{fired}")
            for row in cell[:2]:
                chosen.append(row)
                families.add(row["family"])
    if len(chosen) != 12 or len(families) != 12:
        raise ValueError("scout is not 12 distinct-family rows")
    chosen.sort(key=lambda r: r["uid"])
    counts = Counter((length_bin(r["prefix_tokens"]), r["analysis_meta"]["v22_fired"])
                     for r in chosen)
    body = {"schema": "transition-v22-qwen-12-scout-frame-v1",
            "source_frame_sha256": source["sha256"],
            "families": 12, "rows": 12,
            "visible_input_allowlist": source["visible_input_allowlist"],
            "selection": "fixed-hash two candidate fires and two nonfires in each <1024, 1024-4095, >=4096 native-prefix-token bin, globally distinct discovery families",
            "counts": {f"{band}|{int(fired)}": n for (band, fired), n in counts.items()},
            "records": chosen}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "counts": body["counts"],
                      "min_prefix": min(r["prefix_tokens"] for r in chosen),
                      "max_prefix": max(r["prefix_tokens"] for r in chosen)}))


if __name__ == "__main__":
    main()
