"""Durable attempt journals and assigned-population accounting."""
from __future__ import annotations

from collections import defaultdict
import os
import time

from .contracts import digest, load, require, save, seal, verify


def begin_attempt(directory, assignment, invocation):
    directory.mkdir(parents=True, exist_ok=True)
    uid = assignment["uid"]
    require(not (directory / (uid + ".json")).exists(), "assignment already complete")
    attempts = sorted(directory.glob(uid + ".attempt-*.json"))
    require(len(attempts) < 2, "one automatic assignment retry exhausted")
    number = len(attempts)
    path = directory / f"{uid}.attempt-{number}.json"
    save(path, {"schema": "verification-attempt-v3", "uid": uid, "assignment_sha256": digest(assignment),
        "number": number, "invocation": invocation, "job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "STARTED", "started_epoch": time.time(), "resources_counted": True})
    return path


def end_attempt(path, *, receipt=None, error=None, elapsed_seconds=0):
    row = load(path)
    save(path, {**row, "status": "COMMITTED" if receipt else "FAILED",
        "receipt_sha256": receipt.get("sha256") if receipt else None,
        "error": error, "elapsed_seconds": elapsed_seconds,
        "completed_epoch": time.time()}, replace=True)


def assignment_ledger(manifest, receipts, attempts):
    verify(manifest, "verification-assignments-v3")
    roster = {r["uid"]: r for r in manifest["assignments"]}
    committed = {}
    for receipt in receipts:
        verify(receipt)
        uid = receipt["assignment_uid"]
        require(uid in roster and uid not in committed and receipt["assignment"] == roster[uid] and
                receipt["engineering_only"] is False and receipt["execution_complete"] is True,
                "foreign/duplicate/engineering scientific receipt")
        committed[uid] = receipt
    journals = defaultdict(list)
    for attempt in attempts:
        verify(attempt)
        require(attempt["uid"] in roster and attempt["assignment_sha256"] == digest(roster[attempt["uid"]]),
                "foreign attempt")
        journals[attempt["uid"]].append(attempt)
    rows = []
    for uid, assignment in roster.items():
        values = journals[uid]
        require(len(values) <= 2 and len({v["number"] for v in values}) == len(values), "retry ledger differs")
        receipt = committed.get(uid)
        rows.append({"assignment": assignment, "receipt_sha256": receipt["sha256"] if receipt else None,
            "status": "COMPLETE" if receipt else "UNKNOWN_EXECUTION", "attempts": values,
            "scientific_output_tokens": receipt["generated_tokens"] + receipt["controller_tokens"] if receipt else None,
            "research_attempt_seconds": sum(v.get("elapsed_seconds", 0) for v in values),
            "missing_cost_is_unknown": receipt is None})
    return seal({"schema": "verification-assignment-ledger-v3", "assignment_manifest_sha256": manifest["sha256"],
        "rows": rows, "assigned": len(rows), "completed": len(committed), "missing": len(rows)-len(committed),
        "all_assignments_retained": True})
