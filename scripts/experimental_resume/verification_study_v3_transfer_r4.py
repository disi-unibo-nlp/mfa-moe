"""Resume this exact accepted controller handover; never cancel native science.

Prepare the r4 held primary through autodriver_r4 bootstrap --defer-release.
Then this bounded command retires three exact unstarted r2 controller jobs and
one held, unstarted reader with the proven long-path transport. Every mutation
has an immutable intent, identity checks and observed cancellation closeout.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
from moe_exp.verification_study.registry import REPO
import verification_study_v3_autodriver_r4 as driver
import verification_study_v3_autostages_r4 as stages
import verification_study_v3_transport_r3 as transport

CAMPAIGN = REPO / "report/verification-study-v3/CAMPAIGN.json"
CONTROLLERS = {"59658743", "59658761", "59658767"}
READER = "59658732"
TARGETS = CONTROLLERS | {READER}


def command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30).stdout


def accounting():
    raw = command(["sacct", "-nPX", "-j", ",".join(sorted(TARGETS, key=int)),
                   "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,Start,End"])
    result = {}
    for line in raw.splitlines():
        fields = line.split("|")
        require(fields[0] in TARGETS and fields[0] not in result, "ambiguous target accounting")
        result[fields[0]] = {"state": fields[1].split(" ", 1)[0].rstrip("+"), "exit_code": fields[2],
            "elapsed_seconds": int(fields[3]), "allocated_cpus": int(fields[4]),
            "start": fields[5], "end": fields[6]}
    require(set(result) == TARGETS, "exact target accounting unavailable")
    return result


def ensure_unstarted(value):
    require(value["state"] in {"PENDING", "CANCELLED"} and value["elapsed_seconds"] == 0 and
            value["allocated_cpus"] == 0 and value["start"] == "Unknown",
            "target has started; no cancellation or blind replacement permitted")


def transfer():
    campaign = load(CAMPAIGN); root = Path(campaign["result_root"])
    ready = load(CAMPAIGN.parent / "AUTOMATION_READINESS-r4.json")
    previous = load(CAMPAIGN.parent / "AUTOMATION_READINESS-r2.json")
    check_files(campaign["code_files"]); check_files(ready["program_files"]); check_files(previous["program_files"])
    controller = driver.controller_card(CAMPAIGN, "bootstrap")
    slot = driver.slot_for(controller); receipt = driver.accepted_receipt(controller)
    require(receipt is not None, "held replacement has not been accepted")
    prearm = load(slot / "PREARM.json")
    require(prearm["primary_job_id"] == receipt["job_id"] and len(prearm["children"]) == 1,
            "replacement startup protection missing")
    first_prearm = load(prearm["children"][0]["prearm_path"])
    require(len(first_prearm["children"]) == 1, "second replacement rescue missing")
    rproof_path = CAMPAIGN.parent / "READER_TRANSPORT_RECONCILIATION-r4.json"
    hproof_path = CAMPAIGN.parent / "HEALTH_RECONCILIATION-autodriver-r4.json"
    if not (slot / "RELEASE.json").exists():
        new_before = command(["scontrol", "show", "job", receipt["job_id"]])
        require("JobState=PENDING" in new_before and "JobHeldUser" in new_before and
                "Comment=cv3auto:" + controller["sha256"] in new_before, "replacement is not the exact held primary")
    selected = {}
    for path in (root / "automation/submissions").glob("*/SUBMISSION.json"):
        sub = load(path)
        if sub["job_id"] in CONTROLLERS:
            card = load(path.parent / "CARD.json")
            require(card["campaign_sha256"] == campaign["sha256"] and
                    card["operational_version"] == "autodriver-r2" and
                    card["program_files"] == previous["program_files"] and
                    sub["controller_card_sha256"] == card["sha256"], "foreign old controller")
            selected[sub["job_id"]] = {"card_path": str(path.parent / "CARD.json"),
                "card_sha256": card["sha256"], "submission_path": str(path), "submission_sha256": sub["sha256"],
                "comment": "cv3auto:" + card["sha256"], "partition": "lrd_all_serial"}
    reader_slot = root / "submissions/2f69b73734b4881b765cf11ab51d55d83c4eac6ece8616fe7ada05ee2d9e73a0"
    rcard = load(reader_slot / "JOB_CARD.json"); rsub = load(reader_slot / "SUBMISSION.json")
    require(rsub["job_id"] == READER and rsub["job_card_sha256"] == rcard["sha256"] and
            rcard["transport_kind"] == "reader" and rcard["selected_uids"] ==
            ["2ac63252bae97f6d0628010035a6f7fac6bb77030740447de78bc544aec9a61c"], "foreign held reader")
    check_files(rcard["transport_executor"]["code_files"])
    selected[READER] = {"card_path": str(reader_slot / "JOB_CARD.json"), "card_sha256": rcard["sha256"],
        "submission_path": str(reader_slot / "SUBMISSION.json"), "submission_sha256": rsub["sha256"],
        "comment": "cv3t:" + rcard["sha256"], "partition": "boost_usr_prod"}
    require(set(selected) == TARGETS, "exact retirement journals unavailable")
    before = accounting()
    for value in before.values():
        ensure_unstarted(value)
    intent_path = CAMPAIGN.parent / "AUTOMATION_TRANSFER_INTENT-r4.json"
    if intent_path.exists():
        intent = load(intent_path)
        require(intent["replacement_card_sha256"] == controller["sha256"] and
                intent["targets"] == selected and intent["script_sha256"] == file_sha(__file__),
                "foreign transfer intent")
    else:
        intent = save(intent_path, {"schema": "verification-autodriver-transfer-intent-v3",
            "campaign_sha256": campaign["sha256"], "replacement_card_sha256": controller["sha256"],
            "replacement_job_id": receipt["job_id"], "replacement_submission_sha256": receipt["sha256"],
            "replacement_prearm_sha256": prearm["sha256"], "targets": selected, "before_accounting": before,
            "script_sha256": file_sha(__file__), "created_epoch": time.time(), "native_science_cancellations": []})
    # Hold every exact pending target before cancelling any ancestor. Rescues
    # cannot activate in the interval after an old primary is cancelled.
    for job in sorted(TARGETS, key=int):
        if before[job]["state"] == "CANCELLED":
            continue
        observed = command(["scontrol", "show", "job", job]); row = selected[job]
        require("JobState=PENDING" in observed and "RunTime=00:00:00" in observed and
                "UserId=lmolfett(" in observed and "Account=iscrc_miosr" in observed and
                "Partition=" + row["partition"] in observed and "Comment=" + row["comment"] in observed,
                "old target changed or started before hold")
        if "JobHeldUser" not in observed:
            command(["scontrol", "hold", job])
        after_hold = command(["scontrol", "show", "job", job])
        require("JobState=PENDING" in after_hold and "JobHeldUser" in after_hold and
                "RunTime=00:00:00" in after_hold, "old target hold not observed")
    for job in sorted(TARGETS, key=int, reverse=True):
        if before[job]["state"] != "CANCELLED":
            command(["scancel", job])
    after = accounting()
    for value in after.values():
        ensure_unstarted(value); require(value["state"] == "CANCELLED", "cancellation closeout unavailable")
    args = stages.arguments(rcard); out = Path(args["--output"])
    require(not list((out / "invocations").glob(READER + "-*")) and
            not list((root / "transport/invocations").glob(READER + "-*")),
            "held reader actually entered a worker")
    for rating in out.glob("*.reader-*.json"):
        require(str(load(rating).get("job_id")) != READER, "held reader actually produced a draw")
    sources = {str(p): load(p)["sha256"] for p in (reader_slot / "JOB_CARD.json", reader_slot / "SUBMISSION.json",
               Path(args["--frame"]), CAMPAIGN.parent / "READER_TRANSPORT_HOLD-r3.json", intent_path)}
    row = {"job_id": READER, "job_card_sha256": rcard["sha256"], "submission_sha256": rsub["sha256"],
        "full_frame_sha256": rcard["full_frame_sha256"], "selected_uids": rcard["selected_uids"],
        "role": args["--role"], "output": args["--output"], "accounting": after[READER],
        "source_receipts": sources}
    if rproof_path.exists():
        rproof = load(rproof_path, "verification-reader-transport-reconciliation-v3")
        require(rproof["cancelled_unstarted_readers"] == [row] and
                rproof["transfer_intent_sha256"] == intent["sha256"], "foreign existing reader proof")
    else:
        rproof = save(rproof_path, {"schema": "verification-reader-transport-reconciliation-v3",
            "campaign_sha256": campaign["sha256"], "scientific_code_files": campaign["code_files"],
            "cancelled_unstarted_readers": [row], "replacement_transport_executor": transport.binding(campaign, "reader"),
            "replacement_allocations_per_uid_max": 1, "item_draws_in_retired_job": 0,
            "interpretation": "same original largest-input retained reader measurement, original two fixed draws; no scientific item added",
            "transfer_intent_sha256": intent["sha256"], "observed_epoch": time.time()})
    observer_rows = [{"job_id": job, **after[job], **selected[job]} for job in sorted(CONTROLLERS)]
    if hproof_path.exists():
        hproof = load(hproof_path, "verification-health-observer-reconciliation-v3")
        require(hproof["campaign_sha256"] == campaign["sha256"] and
                hproof["cancelled_unstarted_observers"] == observer_rows and
                hproof["replacement_controller_job_id"] == receipt["job_id"] and
                hproof["replacement_card_sha256"] == controller["sha256"] and
                hproof["replacement_prearm_sha256"] == prearm["sha256"] and
                hproof["transfer_intent_sha256"] == intent["sha256"] and
                hproof["reader_reconciliation_sha256"] == rproof["sha256"], "foreign existing controller proof")
    else:
        save(hproof_path, {"schema": "verification-health-observer-reconciliation-v3",
            "campaign_sha256": campaign["sha256"], "targets": sorted(CONTROLLERS, key=int),
            "cancelled_unstarted_observers": observer_rows,
            "replacement_controller_job_id": receipt["job_id"], "replacement_card_sha256": controller["sha256"],
            "replacement_prearm_sha256": prearm["sha256"], "transfer_intent_sha256": intent["sha256"],
            "reader_reconciliation_sha256": rproof["sha256"], "native_science_cancellations": [],
            "observed_epoch": time.time()})
    states = stages.observe_jobs(TARGETS)
    stages.reconciled_unstarted_readers(CAMPAIGN, root, [{"card": rcard, "submission": rsub}], states)
    result = driver.release_controller(controller)
    closeout = CAMPAIGN.parent / "AUTOMATION_TRANSFER-r4.json"
    if closeout.exists():
        final = load(closeout)
        require(final["campaign_sha256"] == campaign["sha256"] and final["intent_sha256"] == intent["sha256"] and
                final["replacement_release_sha256"] == result["sha256"], "foreign transfer closeout")
    else:
        save(closeout, {"schema": "verification-autodriver-transfer-v3", "campaign_sha256": campaign["sha256"],
            "intent_sha256": intent["sha256"], "replacement_job_id": receipt["job_id"],
            "replacement_release_sha256": result["sha256"], "old_accounting": after,
            "native_science_cancellations": [], "script_sha256": file_sha(__file__), "observed_epoch": time.time()})
    return result


if __name__ == "__main__":
    result = transfer()
    print(json.dumps({"replacement_job_id": result["job_id"], "release_sha256": result["sha256"]}))
