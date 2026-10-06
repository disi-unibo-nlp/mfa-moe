"""Stratified, discovery-only agreement audit for full-prefix Qwen readers."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from native_prefix_semantic_veto_v0 import BASE, digest, sealed

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
FRAME = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
QWEN = BASE / "ratings-v22-fullprefix-v2-6c10499b-1ef8863f"
NATIVE = BASE / "ratings-native-veto-full-v0-6c10499b-191f97fe"
OUT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
BATCH = 16


def outcomes(records):
    result = []
    for row in records:
        votes = []
        for reader in row["readers"]:
            rating = reader["rating"]
            if reader["finish_reason"] != "stop" or rating is None:
                votes.append(None)
            elif set(rating) == {"start"} and type(rating["start"]) is bool:
                votes.append(rating["start"])
            else:
                raise ValueError("malformed parsed rating")
        if votes == [True, True]:
            status = "accepted"
        elif votes == [False, False]:
            status = "rejected"
        elif None in votes:
            status = "unresolved"
        else:
            status = "split"
        result.append((row["uid"], status, votes))
    return result


def read_audit(path, frame, expected_schema):
    binding, summary = sealed(path / "BINDING.json"), sealed(path / "SUMMARY.json")
    if (binding["frame_sha256"] != frame["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or summary["schema"] != expected_schema
            or summary["counts"].get("ratings") != 744):
        raise ValueError("full-prefix rating binding or coverage differs")
    batches = [sealed(path / "batches" / f"{start:06d}.json")
               for start in range(0, 372, BATCH)]
    if any(part["binding_sha256"] != binding["sha256"] or part["start"] != start
           for start, part in zip(range(0, 372, BATCH), batches)):
        raise ValueError("full-prefix batch binding/order differs")
    rows = [row for part in batches for row in part["records"]]
    if [r["uid"] for r in rows] != [r["uid"] for r in frame["records"]]:
        raise ValueError("full-prefix UID coverage/order differs")
    return binding, summary, batches, outcomes(rows)


def main():
    frame = sealed(FRAME)
    if frame["schema"] != "transition-v22-full-prefix-start-frame-v1" or frame["rows"] != 372:
        raise ValueError("frozen discovery rating frame changed")
    qb, qs, qparts, qr = read_audit(
        QWEN, frame, "transition-v22-fullprefix-rating-summary-v2")
    nb, ns, nparts, nr = read_audit(
        NATIVE, frame, "native-prefix-semantic-veto-full-summary-v0")
    counts = Counter()
    by_family = defaultdict(Counter)
    rows = []
    for source, q, n in zip(frame["records"], qr, nr, strict=True):
        if not (source["uid"] == q[0] == n[0]):
            raise ValueError("rating UID alignment differs")
        transition, fire = source["transition"], source["analysis_meta"]["v22_fired"]
        stratum = transition + ("|fire" if fire else "|nonfire")
        family = source["family"]
        counts[stratum + "|rows"] += 1
        counts[stratum + "|qwen_" + q[1]] += 1
        counts[stratum + "|native_" + n[1]] += 1
        counts[stratum + "|qwen_" + q[1] + "|native_" + n[1]] += 1
        by_family[family][stratum + "|rows"] += 1
        by_family[family][stratum + "|qwen_" + q[1]] += 1
        by_family[family][stratum + "|native_" + n[1]] += 1
        rows.append({"uid": source["uid"], "family": family,
                     "transition": transition, "v22_fired": fire,
                     "qwen_status": q[1], "qwen_votes": q[2],
                     "native_status": n[1], "native_votes": n[2]})
    body = {
        "schema": "full-prefix-native-veto-agreement-v1",
        "frame_sha256": frame["sha256"],
        "qwen_binding_sha256": qb["sha256"], "qwen_summary_sha256": qs["sha256"],
        "qwen_batch_sha256s": [part["sha256"] for part in qparts],
        "native_binding_sha256": nb["sha256"], "native_summary_sha256": ns["sha256"],
        "native_batch_sha256s": [part["sha256"] for part in nparts],
        "counts": dict(counts),
        "by_family": {family: dict(values) for family, values in sorted(by_family.items())},
        "records": rows,
        "interpretation": "Descriptive, independently sampled but stratified discovery windows. Two correlated draws per model; Qwen3.8 and native Qwen3.6 prompts/rubrics differ. Reader agreement is an LLM measurement audit, not human truth. This report contains no future target or intervention outcome."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
