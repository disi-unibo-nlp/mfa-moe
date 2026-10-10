"""Resume the exact r4 transfer after Slurm reports cancelled Start as 'None'.

All four targets must already be cancelled, with zero elapsed time and zero
allocated CPUs. Preserve raw accounting and the original intent/source. This
does not reinterpret a real start or allocate another controller.
"""
from pathlib import Path
import json
import time

from moe_exp.verification_study.contracts import file_sha, load, require, save
import verification_study_v3_transfer_r4 as previous


def resume():
    campaign = load(previous.CAMPAIGN)
    intent = load(previous.CAMPAIGN.parent / "AUTOMATION_TRANSFER_INTENT-r4.json")
    require(intent["script_sha256"] == file_sha(previous.__file__), "original transfer source changed")
    read_accounting = previous.accounting
    observed = read_accounting()
    for row in observed.values():
        require(row["state"] == "CANCELLED" and row["elapsed_seconds"] == 0 and
                row["allocated_cpus"] == 0 and row["start"] == "None",
                "exact no-start cancellation evidence unavailable")
    path = previous.CAMPAIGN.parent / "AUTOMATION_TRANSFER_ACCOUNTING_RECONCILIATION-r5.json"
    if path.exists():
        proof = load(path)
        require(proof["raw_accounting"] == observed and proof["script_sha256"] == file_sha(__file__) and
                proof["intent_sha256"] == intent["sha256"], "foreign no-start reconciliation")
    else:
        proof = save(path, {"schema": "verification-autodriver-accounting-reconciliation-v3",
            "campaign_sha256": campaign["sha256"], "intent_sha256": intent["sha256"],
            "original_script_path": str(Path(previous.__file__).resolve()),
            "original_script_sha256": file_sha(previous.__file__), "raw_accounting": observed,
            "normalization": "cancelled Start=None means no recorded start; preserve raw value and represent it as Unknown for the existing gate",
            "script_path": str(Path(__file__).resolve()), "script_sha256": file_sha(__file__),
            "new_allocations": 0, "created_epoch": time.time()})

    def normalized_accounting():
        values = read_accounting()
        require(values == observed, "cancellation accounting changed during transfer resume")
        return {job: {**row, "raw_start": row["start"], "start": "Unknown",
                      "no_start_reconciliation_path": str(path), "no_start_reconciliation_sha256": proof["sha256"]}
                for job, row in values.items()}

    # Process-local adapter for an unchanged project orchestration helper. The
    # installed scheduler, scientific workers, controller and runtime stay intact.
    previous.accounting = normalized_accounting
    try:
        return previous.transfer()
    finally:
        previous.accounting = read_accounting


if __name__ == "__main__":
    result = resume()
    print(json.dumps({"replacement_job_id": result["job_id"], "release_sha256": result["sha256"]}))
