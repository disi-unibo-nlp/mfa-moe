"""Resume the sealed discovery cards while respecting serial callback slots.

One bounded orchestration pass; no daemon, inference or altered source closure.
Deferred callbacks use the same immutable predecessor when a slot is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
from moe_exp.verification_study.registry import REPO, MODEL_KEYS
import verification_study_v3_discovery_continue_r2 as planned


def serial_slots():
    quota = subprocess.run(["sacctmgr", "-nP", "show", "qos", "lrd_all_serial",
                            "format=Name,MaxSubmitJobsPU"], check=True, capture_output=True,
                           text=True, timeout=20).stdout
    rows = [line.split("|") for line in quota.splitlines() if line.startswith("lrd_all_serial|")]
    require(len(rows) == 1 and rows[0][1].isdigit(), "serial submit entitlement unavailable")
    limit = int(rows[0][1])
    queue = subprocess.run(["squeue", "-h", "-u", "lmolfett", "-o", "%i|%P"],
                           check=True, capture_output=True, text=True, timeout=20).stdout
    used = sum(line.split("|", 1)[1] == "lrd_all_serial" for line in queue.splitlines())
    return {"limit": limit, "used": used, "free": max(0, limit - used)}


def callback(root, card, submitted):
    candidates = []
    expected_sources = {str(p): file_sha(p) for p in (planned.health.SCRIPT, planned.health.SHELL)}
    for path in (root / "health/submissions").glob("*/HEALTH_CARD.json"):
        value = load(path)
        if value["predecessor_job_id"] == submitted["job_id"] and value["job_card_sha256"] == card["sha256"]:
            if value["observer_files"] != expected_sources:
                continue
            check_files(value["observer_files"])
            receipt_path = path.parent / "SUBMISSION.json"
            if receipt_path.exists():
                receipt = load(receipt_path)
                require(receipt["health_card_sha256"] == value["sha256"], "foreign callback receipt")
                candidates.append(receipt)
            elif (path.parent / "INTENT.json").exists():
                raise ValueError("unresolved callback intent: " + str(path.parent))
    require(len(candidates) <= 1, "multiple current-source callbacks need reconciliation")
    return candidates[0] if candidates else None


def verify_plan(path, campaign, root, wave):
    allowed = {e["job_card_sha256"] for e in wave["cards"]}
    allowed.update(e["job_card_sha256"] for e in load(path.parent / "DISCOVERY_PRICE_JOBS-r4.json")["jobs"])
    for p in (root / "submissions").glob("*/JOB_CARD.json"):
        card = load(p)
        if card["stage"] != "run" or not ((p.parent / "INTENT.json").exists() or (p.parent / "SUBMISSION.json").exists()):
            continue
        args = planned.health.arguments(card)
        if "--manifest" in args and load(args["--manifest"])["stage"] == "discovery":
            require(card["sha256"] in allowed, "foreign discovery submission requires reconciliation")
    for key in MODEL_KEYS:
        manifest = load(path.parent / key / "DISCOVERY_ASSIGNMENTS.json")
        expected = [a["uid"] for a in manifest["assignments"][1:]]
        actual = []
        for entry in wave["cards"]:
            if entry["model"] != key:
                continue
            request = load(entry["request_path"])
            rows = manifest["assignments"][entry["begin"]:entry["begin"] + entry["count"]]
            require(request["selected_uids"] == [a["uid"] for a in rows], "discovery slice identities changed")
            actual.extend(request["selected_uids"])
        require(actual == expected and len(set(actual)) == 47, "discovery slices overlap or omit a family")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(); path = args.campaign.resolve(); root = Path(load(path)["result_root"])
    with planned.lock(root):
        campaign, root, wave = planned.prepare(path)
        verify_plan(path, campaign, root, wave)
        storage = planned.exact_storage(REPO)
        require(wave["storage_bound_bytes"] < storage["available_bytes"], "live WORK capacity insufficient")
        rows = []
        for entry in wave["cards"]:
            request = load(entry["request_path"]); card = planned.native.task_card(path, request)
            require(card["sha256"] == entry["job_card_sha256"], "sealed discovery card changed")
            row = dict(entry)
            if args.submit:
                slot = root / "submissions" / card["sha256"] / "SUBMISSION.json"
                if not slot.exists():
                    planned.native.submit(path, card, root, test_only=True)
                submitted = planned.native.submit(path, card, root)
                row["job_id"] = submitted["job_id"]
                registered = callback(root, card, submitted)
                if registered is None and serial_slots()["free"]:
                    planned.health.register(path, slot, root, test_only=True)
                    registered = planned.health.register(path, slot, root)
                row["callback_job_id"] = registered["job_id"] if registered else None
                row["callback_status"] = "REGISTERED" if registered else "DEFERRED_SERIAL_SUBMIT_LIMIT"
                print(json.dumps({k: row[k] for k in ("model", "begin", "count", "job_id", "callback_status")}), flush=True)
            rows.append(row)
        result = save(path.parent / "DISCOVERY_CONTINUATION_JOBS-r2.json", {
            "schema": "verification-discovery-continuation-jobs-v3", "campaign_sha256": campaign["sha256"],
            "plan_sha256": wave["sha256"], "orchestration_file_sha256": file_sha(planned.__file__),
            "dispatcher_file_sha256": file_sha(__file__), "observed_epoch": time.time(),
            "status": "SUBMITTED_FINITE_ALL_SEVEN_DISCOVERY" if args.submit else "PRICED_FINITE_CARDS_READY",
            "cards": rows, "assignments": sum(r["count"] for r in rows), "billing_core_hours_max": wave["billing_core_hours_max"],
            "serial_slots": serial_slots(), "deferred_callbacks": [r["job_id"] for r in rows if args.submit and not r["callback_job_id"]],
            "scientific_success": "NOT_INFERRED_FROM_SUBMISSION"}, replace=True)
        print(json.dumps({k: result[k] for k in ("status", "assignments", "billing_core_hours_max", "deferred_callbacks", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
