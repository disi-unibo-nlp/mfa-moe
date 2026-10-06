"""Seal small paired-reader timing audit without changing the primary endpoint."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
FRAME = BASE / "WITHIN_SENTENCE_START_AUDIT_FRAME_v1.json"
INVENTORY = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_TIMING_INVENTORY_v0.json"
RATINGS = BASE / "ratings-within-sentence-v1-ed8522b7"
POOL = BASE / "JOINT_QWEN_NATIVE_EXACT_POOL_v1.json"
OUT = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_EARLYCUT_AUDIT_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed input: {path}")
    return value


def main():
    frame, inventory, binding, summary, pool = map(
        sealed, (FRAME, INVENTORY, RATINGS / "BINDING.json",
                 RATINGS / "SUMMARY.json", POOL))
    batches = [sealed(path) for path in sorted((RATINGS / "batches").glob("[0-9]" * 6 + ".json"))]
    rated = [row for batch in batches for row in batch["records"]]
    if (frame["schema"] != "within-sentence-start-audit-frame-v1"
            or frame["rows"] != 43 or frame["meaningful_rows"] != 31
            or frame["timing_inventory_sha256"] != inventory["sha256"]
            or binding["frame_sha256"] != frame["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or [x["uid"] for x in rated] != [x["uid"] for x in frame["records"]]
            or summary["counts"]["ratings"] != 86
            or summary["counts"]["parsed_stop"] != 86
            or len(batches) != 3):
        raise ValueError("early-cut assignment coverage or binding changed")
    by_timing = {row["uid"]: row for row in inventory["records"]}
    old_families = {row["family"] for row in pool["records"]}
    strata = {}
    for name, selected in (("meaningful_headroom_at_least_4", slice(0, 31)),
                           ("short_gap_control_headroom_1_to_3", slice(31, 43))):
        pairs = list(zip(frame["records"][selected], rated[selected], strict=True))
        statuses = Counter()
        positives = []
        for source, result in pairs:
            timing = by_timing[source["uid"]]
            if (source["uid"] != result["uid"] or source["family"] != timing["family"]
                    or source["prefix_tokens"] != timing["earliest_complete_token"]):
                raise ValueError("timing UID or native cut changed")
            responses = result["readers"]
            if not all(r["finish_reason"] == "stop" and r["rating"] is not None
                       for r in responses):
                raise ValueError("unresolved or capped timing reader")
            votes = [r["rating"]["start"] for r in responses]
            state = "both_true" if votes == [True, True] else (
                "both_false" if votes == [False, False] else "split")
            statuses[state] += 1
            if state == "both_true":
                positives.append({"uid": source["uid"], "family": source["family"],
                                  "prefix_tokens": source["prefix_tokens"],
                                  "token_headroom": timing["token_headroom"],
                                  "full_sentence_qwen_status": timing["qwen_fullsentence_status"],
                                  "overlaps_existing_joint_pool_family": source["family"] in old_families})
        strata[name] = {"rows": len(pairs), "statuses": dict(statuses),
                        "positive_families": len({x["family"] for x in positives}),
                        "positive_existing_pool_family_overlap": sum(x["overlaps_existing_joint_pool_family"] for x in positives),
                        "positive_records": positives}
    if (strata["meaningful_headroom_at_least_4"]["statuses"] !=
            {"both_true": 2, "both_false": 28, "split": 1}
            or strata["short_gap_control_headroom_1_to_3"]["statuses"] !=
            {"both_true": 3, "both_false": 9}):
        raise ValueError("paired rating strata changed")
    body = {"schema": "within-sentence-earlycut-audit-v1",
            "frame_sha256": frame["sha256"],
            "inventory_sha256": inventory["sha256"],
            "rating_binding_sha256": binding["sha256"],
            "rating_summary_sha256": summary["sha256"],
            "rating_batch_sha256s": [batch["sha256"] for batch in batches],
            "joint_pool_sha256": pool["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parsed_natural_stop_ratings": 86,
            "paired_agreement": 42,
            "strata": strata,
            "interpretation": "Agreement is not positive acceptance. Two meaningful earlier cuts are accepted in previously unused discovery families, but only four and seven tokens ahead of the original sentence end, and both corresponding full-sentence Qwen labels were unresolved. This is a discovery-only timing diagnostic, not human validation or a causal transition result. The original outcome excludes the full triggering sentence."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "strata": {key: value["statuses"] for key, value in strata.items()}}))


if __name__ == "__main__":
    main()
