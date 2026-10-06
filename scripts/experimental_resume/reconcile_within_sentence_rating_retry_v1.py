"""Audited recovery of interrupted within-sentence Qwen reader batches.

A completed reader batch is never replayed. For an incomplete reader call,
exact execution status cannot be recovered from vLLM after process loss. This
tool preserves every original attempt and writes a failure disposition before
making that UID eligible for a deterministic retry. It never runs while the
source Slurm job is active and defaults to read-only inspection.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from rate_within_sentence_candidate_starts_v1 import BATCH, FRAME, PRICE, digest, sealed

BASE = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery")
OUT = BASE / "ratings-within-sentence-v1-ed8522b7"
FINAL_FAILURES = {"FAILED", "TIMEOUT", "CANCELLED", "PREEMPTED", "OUT_OF_MEMORY", "NODE_FAIL"}


def final_slurm_state(job_id):
    proc = subprocess.run(["sacct", "-nP", "-j", job_id,
                           "--format=JobID,State,ExitCode"],
                          text=True, capture_output=True, check=True)
    entries = [line.split("|") for line in proc.stdout.splitlines() if line]
    root = [fields for fields in entries if len(fields) == 3 and fields[0] == job_id]
    if len(root) != 1:
        raise ValueError("Slurm root job final state unavailable or ambiguous")
    state = root[0][1].split()[0].rstrip("+")
    if state not in FINAL_FAILURES:
        raise ValueError(f"reconciliation requires an ended unsuccessful job, got {state}")
    return state, root[0][2]


def atomic_receipt(path, body):
    value = {**body, "sha256": digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError("existing retry disposition differs")
        return value
    temp = path.with_name(path.name + ".part-" + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temp, path)
    return value


def main(args):
    if not args.job_id.isdigit():
        raise ValueError("failed Slurm job ID must be numeric")
    state, exit_code = final_slurm_state(args.job_id)
    frame, price, binding = sealed(FRAME), sealed(PRICE), sealed(OUT / "BINDING.json")
    if (price["frame_sha256"] != frame["sha256"]
            or binding["frame_sha256"] != frame["sha256"]
            or binding["price_sha256"] != price["sha256"]
            or binding["driver_sha256"] != price["rating_driver_sha256"]):
        raise ValueError("frame/price/driver output binding differs")
    lock = (OUT / "WRITER.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plans = []
    for start in range(0, len(frame["records"]), BATCH):
        block = frame["records"][start:start+BATCH]
        for reader in (0, 1):
            reader_path = OUT / "batches" / f"{start:06d}-reader{reader}.json"
            if reader_path.exists():
                saved = sealed(reader_path)
                if saved["binding_sha256"] != binding["sha256"]:
                    raise ValueError("completed reader batch binding differs")
                continue
            for row in block:
                uid = row["uid"]
                assignment = OUT / "assignments" / f"{uid}-reader{reader}.json"
                if not assignment.exists():
                    continue
                attempt = sealed(assignment)
                if attempt["binding_sha256"] != binding["sha256"] or attempt["job_id"] != args.job_id or attempt["uid"] != uid or attempt["reader"] != reader:
                    raise ValueError("incomplete attempt was made by a different job or binding")
                archive = OUT / "attempt_history" / f"{uid}-reader{reader}-job{args.job_id}.json"
                disposition = OUT / "reconciliations" / f"{uid}-reader{reader}-job{args.job_id}.json"
                if archive.exists():
                    raise ValueError("attempt archive already exists while live receipt remains")
                plans.append((assignment, archive, disposition, attempt, start, reader))
    result = {"job_id": args.job_id, "state": state, "exit_code": exit_code,
              "incomplete_assignment_receipts": len(plans),
              "distinct_reader_batches": len({(start, reader) for _, _, _, _, start, reader in plans}),
              "mode": "apply" if args.apply else "dry_run"}
    if not args.apply:
        print(json.dumps(result), flush=True)
        return
    (OUT / "attempt_history").mkdir(exist_ok=True)
    (OUT / "reconciliations").mkdir(exist_ok=True)
    for assignment, archive, disposition, attempt, start, reader in plans:
        atomic_receipt(disposition,
                       {"schema": "within-sentence-rating-retry-disposition-v1",
                        "binding_sha256": binding["sha256"],
                        "original_attempt_sha256": attempt["sha256"],
                        "uid": attempt["uid"], "reader": reader, "start": start,
                        "failed_job_id": args.job_id, "failed_job_state": state,
                        "failed_job_exit_code": exit_code,
                        "status": "reader_batch_result_absent; original execution uncertain; deterministic retry explicitly recorded"})
        os.replace(assignment, archive)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args())
