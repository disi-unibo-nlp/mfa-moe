"""Bounded descriptive cross-check of two 12-row full-prefix model scouts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from native_prefix_semantic_veto_v0 import BASE, digest, sealed

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FRAME = BASE / "TRANSITION_V22_QWEN_12_SCOUT_FRAME.json"
QWEN = BASE / "ratings-v22-qwen-scout-d8f50b7e-06f8fabd"
NATIVE = BASE / "ratings-native-veto-scout-v0-d8f50b7e-8d2b37d2"
OUT = REPO / "report/experimental-resume-v1/NATIVE_PREFIX_VETO_12_SCOUT_COMPARISON_v0.json"


def vote(result):
    if result["rating"] is None or result["finish_reason"] != "stop":
        return None
    if set(result["rating"]) != {"start"} or type(result["rating"]["start"]) is not bool:
        raise ValueError("parsed rating structure changed")
    return result["rating"]["start"]


def main():
    frame = sealed(FRAME)
    qb, qs = sealed(QWEN / "BINDING.json"), sealed(QWEN / "SUMMARY.json")
    nb, ns = sealed(NATIVE / "BINDING.json"), sealed(NATIVE / "SUMMARY.json")
    qr = sealed(QWEN / "batches/000000.json")
    nr = [sealed(NATIVE / "batches" / f"reader{i}.json") for i in range(2)]
    if (qb["frame_sha256"] != frame["sha256"] or qs["binding_sha256"] != qb["sha256"]
            or nb["frame_sha256"] != frame["sha256"] or ns["binding_sha256"] != nb["sha256"]
            or qr["binding_sha256"] != qb["sha256"]
            or any(r["binding_sha256"] != nb["sha256"] for r in nr)
            or qs["counts"]["ratings"] != 24 or ns["counts"]["ratings"] != 24):
        raise ValueError("scout frame/binding/coverage mismatch")
    q = {r["uid"]: r for r in qr["records"]}
    n = [{r["uid"]: r for r in part["records"]} for part in nr]
    if set(q) != {r["uid"] for r in frame["records"]} or any(set(part) != set(q) for part in n):
        raise ValueError("scout UID coverage mismatch")
    rows = []
    counts = Counter()
    for source in frame["records"]:
        uid = source["uid"]
        qvotes = [vote(x) for x in q[uid]["readers"]]
        nvotes = [vote(part[uid]) for part in n]
        transition = source["transition"]
        fire = source["analysis_meta"]["v22_fired"]
        key = transition + ("|fire" if fire else "|nonfire")
        counts[key + "|rows"] += 1
        counts[key + "|qwen_joint_true"] += qvotes == [True, True]
        counts[key + "|native_joint_true"] += nvotes == [True, True]
        counts[key + "|qwen_joint_false"] += qvotes == [False, False]
        counts[key + "|native_joint_false"] += nvotes == [False, False]
        counts[key + "|qwen_disagreement"] += set(qvotes) == {True, False}
        counts[key + "|native_disagreement"] += set(nvotes) == {True, False}
        counts[key + "|either_unresolved"] += None in qvotes + nvotes
        rows.append({"uid": uid, "family": source["family"], "transition": transition,
                     "v22_fired": fire, "qwen_votes": qvotes, "native_votes": nvotes,
                     "triggering_sentence": source["reader_input"]["triggering_sentence"]})
    body = {"schema": "native-prefix-veto-12-scout-comparison-v0",
            "frame_sha256": frame["sha256"],
            "qwen_binding_sha256": qb["sha256"], "qwen_summary_sha256": qs["sha256"],
            "native_binding_sha256": nb["sha256"], "native_summary_sha256": ns["sha256"],
            "counts": dict(counts), "records": rows,
            "interpretation": "Context-stratified 12 discovery windows, not a population estimate. Same-model pairs are correlated; native and Qwen prompts/rubrics differ slightly. No behavioral effect or independent human truth."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
