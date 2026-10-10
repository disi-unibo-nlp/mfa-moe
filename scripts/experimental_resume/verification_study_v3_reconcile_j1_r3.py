"""Seal the exact five observed pre-initialization J1 failures; no allocations.

Retain the original audit deployment, one completed verdict and all old proofs.
This script authorizes no retry outside the new r3 transport and exact roster.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
from moe_exp.verification_study.registry import REPO
import verification_study_v3_autostages_r3 as stages
import verification_study_v3_transport_r3 as transport

CAMPAIGN = REPO / "report/verification-study-v3/CAMPAIGN.json"
FAILED = {"59645451", "59645471", "59645487", "59645497", "59652408"}


def reconcile():
    campaign = load(CAMPAIGN); root = Path(campaign["result_root"])
    check_files(campaign["code_files"])
    old = load(CAMPAIGN.parent / "J1_TRANSPORT_RECONCILIATION-r2.json")
    check_files(load(CAMPAIGN.parent / "AUTOMATION_READINESS-r2.json")["program_files"])
    require({row["job_id"] for row in old["failed_jobs"]} == FAILED - {"59652408"},
            "historical failure roster changed")
    target = CAMPAIGN.parent / "J1_TRANSPORT_RECONCILIATION-r3.json"
    if target.exists():
        proof = load(target)
        require(proof["recovery_transport_executor"] == transport.binding(campaign, "extension"),
                "committed repair transport changed")
        return proof
    jobs = []
    for path in (root / "submissions").glob("*/SUBMISSION.json"):
        submission = load(path)
        if str(submission["job_id"]) in FAILED:
            card = load(path.parent / "JOB_CARD.json")
            require(submission["job_card_sha256"] == card["sha256"] and
                    card["campaign_sha256"] == campaign["sha256"] and
                    card["extension"]["kind"] == "j1-audit-slice", "foreign failed submission")
            jobs.append({"card": card, "submission": submission, "slot": path.parent})
    require({str(j["submission"]["job_id"]) for j in jobs} == FAILED and len(jobs) == 5,
            "exact failed submission set unavailable")
    observations = stages.observe_jobs(FAILED)
    previous = {row["job_id"]: row for row in old["failed_jobs"]}
    rows = []; eligible = set()
    for job in sorted(jobs, key=lambda j: int(j["submission"]["job_id"])):
        job_id = str(job["submission"]["job_id"]); state = observations[job_id]
        require(stages.terminal(state) and state["state"] == "FAILED" and state["exit_code"] == "1:0",
                "failed allocation is not terminal with exact exit")
        eligible.update(job["card"]["extension"]["selected_uids"])
        if job_id in previous:
            row = dict(previous[job_id])
            result = load(row["invocation_result_path"])
            row.update(failure_kind="site-default-cache", expected_error=result["error"])
        else:
            paths = list((root / "J1/audit/invocations").glob(job_id + "-*/RESULT.json"))
            require(len(paths) == 1, "ambiguous failed J1 invocation")
            result = load(paths[0])
            require(result["error"]["type"] == "ZMQError" and
                    "is longer than 107 characters" in result["error"]["message"], "unexpected failure")
            require(result["selected_uids"] == job["card"]["extension"]["selected_uids"] and
                    result["slice"] == [0, 1], "failed retained check identity changed")
            ts = list((root / "transport/invocations").glob(job_id + "-*/ERROR.json"))
            require(len(ts) == 1, "ambiguous failed transport invocation")
            tstart = load(ts[0].parent / "START.json"); terror = load(ts[0])
            require(tstart["job_card_sha256"] == job["card"]["sha256"] == terror["job_card_sha256"] and
                    tstart["transport_executor"] == job["card"]["transport_executor"], "foreign transport failure")
            rpc = tstart["cache_environment"]["VLLM_RPC_BASE_PATH"]
            byte_length = len(str(Path(rpc) / transport.IPC_UUID).encode())
            require(byte_length > 107, "old IPC path does not prove diagnosed limit")
            receipts = {str(p): load(p)["sha256"] for p in
                        (paths[0], paths[0].parent / "INTENT.json", ts[0], ts[0].parent / "START.json",
                         ts[0].parent / "ENTERING_STEP.json", job["slot"] / "JOB_CARD.json",
                         job["slot"] / "SUBMISSION.json")}
            logs = {}
            for field in ("StdOut", "StdErr"):
                matches = re.findall(r"(?:^|\s)" + field + r"=([^\s]+)", job["submission"]["read_back"])
                require(len(matches) == 1, "missing exact failed allocation log")
                logs[matches[0]] = file_sha(matches[0])
            row = {"job_id": job_id, "job_card_sha256": job["card"]["sha256"],
                   "submission_sha256": job["submission"]["sha256"],
                   "invocation_result_path": str(paths[0]), "source_receipts": receipts,
                   "source_logs": logs, "failure_kind": "unix-socket-path", "expected_error": result["error"],
                   "old_socket_path_bytes": byte_length}
        require(result["loaded"] is False and result["new_verdicts"] == 0 and result["status"] == "FAIL" and
                not (Path(row["invocation_result_path"]).parent / "RUNTIME.json").exists(),
                "failure cannot be reconciled as zero-draw pre-initialization")
        row["accounting"] = state; rows.append(row)
    attempts = [load(p) for p in (root / "J1/audit").glob("*.attempt-*.json")]
    require(all(str(a["job_id"]) not in FAILED for a in attempts), "failed job drew an audit item")
    verdicts = [p for p in (root / "J1/audit").glob("*.json") if re.fullmatch(r"[a-f0-9]{64}", p.stem)]
    require(len(verdicts) == 1 and load(verdicts[0])["job_id"] == "59637448", "original retained verdict changed")
    require(eligible == set(old["eligible_recovery_uids"]) and len(eligible) == 99, "original recovery roster changed")
    value = {"schema": "verification-j1-transport-reconciliation-v3", "repair_round": 3,
             "campaign_sha256": campaign["sha256"], "scientific_code_files": campaign["code_files"],
             "previous_reconciliation_path": str(CAMPAIGN.parent / "J1_TRANSPORT_RECONCILIATION-r2.json"),
             "previous_reconciliation_sha256": old["sha256"], "failed_jobs": rows,
             "item_draws_in_failed_jobs": 0, "eligible_recovery_uids": sorted(eligible),
             "retained_verdicts": {str(p): load(p)["sha256"] for p in verdicts},
             "failed_allocation_billing_core_hours": 16 * sum(s["elapsed_seconds"] for s in observations.values()) / 3600,
             "recovery_allocations_per_uid_max": 1,
             "recovery_allocations_per_uid_scope": "this sealed r3 transport repair round; old zero-draw failures retained",
             "recovery_transport_executor": transport.binding(campaign, "extension"),
             "first_recovery_count": 1, "remaining_count_after_successful_check": 98,
             "new_runtime_policy": "private short WORK paths checked before model imports; unchanged grader deployment",
             "repeated_failed_r3_allocation_policy": "STOP; explicit new reconciliation required; no blind retry",
             "script_path": str(Path(__file__).resolve()), "script_sha256": file_sha(__file__),
             "observed_epoch": time.time(), "scientific_effect": "NOT_MEASURED"}
    proof = save(target, value)
    stages.validate_j1_reconciliation(CAMPAIGN, root, jobs, observations)
    return proof


if __name__ == "__main__":
    result = reconcile()
    print(json.dumps({k: result[k] for k in ("sha256", "repair_round", "item_draws_in_failed_jobs",
                     "failed_allocation_billing_core_hours", "first_recovery_count")}, indent=2))
