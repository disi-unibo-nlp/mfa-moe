"""Finite recovery for deterministic operations; stochastic Units stay unchanged."""
from pathlib import Path
import os
import time
from .contracts import Units, digest, load, locked, publish, require

DETERMINISTIC_STAGES = frozenset(("resolve", "mapping", "derive", "native", "native_contexts",
    "analysis_pilot", "supplementary", "coordination", "analysis_finalize", "score",
    "contexts", "phase_export", "snapshot", "memory_export", "memory_prepare",
    "price", "memory_price", "reader_price", "available_finalize"))


class DeterministicUnits(Units):
    def begin(self, row, job_id):
        uid = row["uid"]
        with locked(self.root, uid):
            state = self.state(uid)
            if state and state.get("status") != "started":
                return False
            starts = sorted(self.root.glob(uid + ".start-*.json"))
            attempt = 1 if state else 0
            if attempt and any(load(p).get("deterministic_recovery") for p in starts):
                return False
            publish(self.root / f"{uid}.start-{attempt}.json", {
                "uid": uid, "row_sha256": digest(row), "job_id":os.environ.get("SLURM_JOB_ID",str(job_id)),
                "started_epoch":time.time(), "status":"started", "deterministic_recovery":bool(attempt),
                "original_start":state if attempt else None, "stochastic_redraw":False})
            return True


def recoverable(stage, scope, attempt):
    return attempt == 0 and (stage in DETERMINISTIC_STAGES or scope == "deadline")


def attempt_receipt(directory, result, attempt):
    return publish(Path(directory) / f"FAILED_ATTEMPT-{attempt}.json", result)
