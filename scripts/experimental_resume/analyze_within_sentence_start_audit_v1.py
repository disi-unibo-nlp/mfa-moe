"""Paired descriptive audit of exact earlier-cut versus sentence-end starts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
FRAME = BASE / "WITHIN_SENTENCE_START_AUDIT_FRAME_v1.json"
EARLY = BASE / "ratings-within-sentence-v1-ed8522b7"
TIMING = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_TIMING_INVENTORY_v0.json"
FULL = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
OUT = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_QWEN_START_COMPARISON_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def pair_status(readers):
    if any(r["rating"] is None or r["finish_reason"] != "stop" for r in readers):
        return "unresolved"
    a, b = [r["rating"]["start"] for r in readers]
    return "accepted" if a and b else "rejected" if not a and not b else "split"


def main():
    frame, timing, full = sealed(FRAME), sealed(TIMING), sealed(FULL)
    binding, summary = sealed(EARLY / "BINDING.json"), sealed(EARLY / "SUMMARY.json")
    parts = [sealed(p) for p in sorted((EARLY / "batches").glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))]
    rated = [row for part in parts for row in part["records"]]
    if (frame["schema"] != "within-sentence-start-audit-frame-v1"
            or frame["timing_inventory_sha256"] != timing["sha256"]
            or full["frame_sha256"] != timing["source_frame_sha256"]
            or binding["frame_sha256"] != frame["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or summary["counts"].get("ratings") != 86
            or len(rated) != frame["rows"]
            or [r["uid"] for r in rated] != [r["uid"] for r in frame["records"]]):
        raise ValueError("early/full paired audit binding or assignments incomplete")
    by_timing = {r["uid"]: r for r in timing["records"]}
    by_full = {r["uid"]: r for r in full["records"]}
    records, counts = [], Counter()
    for i, (source, result) in enumerate(zip(frame["records"], rated, strict=True)):
        uid = source["uid"]
        previous, early = by_full[uid]["qwen_status"], pair_status(result["readers"])
        role = "meaningful_ge4_tokens" if i < frame["meaningful_rows"] else "short_gap_control"
        if (role == "meaningful_ge4_tokens") != (by_timing[uid]["token_headroom"] >= 4):
            raise ValueError("early-cut role does not match frozen token-gap rule")
        counts[role + "|rows"] += 1
        counts[role + "|full_" + previous + "|early_" + early] += 1
        records.append({"uid": uid, "family": source["family"], "role": role,
                        "token_headroom": by_timing[uid]["token_headroom"],
                        "trailing_chars_in_original_sentence": by_timing[uid]["trailing_chars_in_sentence"],
                        "full_sentence_qwen_status": previous,
                        "earlier_cut_qwen_status": early})
    body = {"schema": "within-sentence-qwen-start-comparison-v1",
            "frame_sha256": frame["sha256"],
            "early_binding_sha256": binding["sha256"],
            "early_summary_sha256": summary["sha256"],
            "early_batch_sha256s": [p["sha256"] for p in parts],
            "timing_inventory_sha256": timing["sha256"],
            "fullprefix_agreement_sha256": full["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts": dict(counts), "records": records,
            "primary_endpoint": "original triggered sentence excluded; unchanged",
            "interpretation": "Paired LLM start-rating sensitivity at a previously frozen earlier native-token cut, not causal steering or human truth. A rejected full sentence becoming an accepted fragment supports only measurement/timing feasibility."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
