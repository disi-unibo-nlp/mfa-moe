"""Bounded telemetry for sealed callbacks, controller journals and health events.

This reader allocates no work and writes only the replaceable HEALTH_STATUS
snapshot. Historical failures and supersession remain separate observations.
"""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import re
import socket
import subprocess
import time

from moe_exp.verification_study.contracts import check_files, digest, load, private_path, require, save
from moe_exp.verification_study.registry import REPO

SCRIPT = Path(__file__).resolve()
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def job_id(value):
    value = str(value)
    require(re.fullmatch(r"[0-9]+", value), "invalid journal job ID")
    return value


def state(value):
    return str(value or "").split(" ", 1)[0].rstrip("+")


def source_binding(card, campaign):
    require(card.get("campaign_sha256") == campaign["sha256"], "journal belongs to another campaign")
    if "code_files" in card:
        require(card["code_files"] == campaign["code_files"], "scientific source closure differs")
    if "scientific_code_files" in card:
        require(card["scientific_code_files"] == campaign["code_files"], "controller scientific source closure differs")


def journal_path(path, expected):
    path = private_path(path)
    require(path == Path(expected).resolve(), "foreign journal path")
    return path


def intent(path, key, sha, comment, unresolved):
    if not path.exists():
        return
    value = load(path)
    require(value.get(key) == sha and value.get("comment", comment) == comment,
            "journal intent card or comment differs")
    if not (path.parent / "SUBMISSION.json").exists():
        unresolved.append(str(path))


def readback(receipt, job, comment):
    text = receipt.get("read_back", "")
    if text:
        require("JobId=" + job in text and "Account=iscrc_miosr" in text, "submission readback identity differs")
        actual = re.search(r"(?:^|\s)Comment=([^\s]+)", text)
        require(not actual or actual.group(1) == comment, "submission readback comment differs")
    matching = receipt.get("matching_accounting")
    if matching:
        require(str(matching.get("job_id")) == job and matching.get("account") == "iscrc_miosr",
                "reconciled submission accounting differs")


def scheduler_snapshot(jobs):
    """Two bounded, read-only calls; only exact top-level IDs are accepted."""
    require(getpass.getuser() == "lmolfett" and "leonardo" in socket.gethostname(), "wrong LEONARDO host/user")
    jobs = sorted({job_id(j) for j in jobs}, key=int)
    accounting = {}; queue = {}
    if not jobs:
        return accounting, queue
    raw = subprocess.run(["sacct", "-nP", "-j", ",".join(jobs),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,Account,Partition,Comment%128"],
        check=True, capture_output=True, text=True, timeout=30).stdout
    for line in raw.splitlines():
        parts = line.split("|")
        if parts[0] not in jobs:
            continue
        require(len(parts) >= 7 and parts[0] not in accounting, "ambiguous or malformed top-level accounting")
        require(parts[4] == "iscrc_miosr", "foreign scheduler accounting account")
        accounting[parts[0]] = {"state": parts[1], "exit_code": parts[2], "elapsed_seconds": int(parts[3]),
            "account": parts[4], "partition": parts[5], "comment": parts[6].strip()}
    raw = subprocess.run(["squeue", "-h", "-u", "lmolfett", "-o", "%i|%T|%j|%R|%a|%P|%k"],
        check=True, capture_output=True, text=True, timeout=20).stdout
    for line in raw.splitlines():
        parts = line.split("|", 6)
        if parts[0] not in jobs:
            continue
        require(len(parts) == 7 and parts[0] not in queue and parts[4] == "iscrc_miosr",
                "ambiguous or foreign scheduler queue identity")
        queue[parts[0]] = {"state": parts[1], "name": parts[2], "reason": parts[3],
            "account": parts[4], "partition": parts[5], "comment": parts[6].strip()}
    return accounting, queue


def validate_scheduler(row, accounting, queue):
    for value in (accounting.get(row["job_id"]), queue.get(row["job_id"])):
        if value:
            require(value.get("account", "iscrc_miosr") == "iscrc_miosr" and
                    value.get("comment", row["comment"]) in {"", row["comment"]},
                    "scheduler comment/account differs from sealed journal")


def reconciliation_candidates(campaign_path, campaign):
    """Prepared intentions alone never supersede an observer."""
    candidates = {}; skipped = []
    for path in sorted(Path(campaign_path).parent.glob("HEALTH_RECONCILIATION-*.json")):
        proof = load(path)
        require(proof.get("campaign_sha256", campaign["sha256"]) == campaign["sha256"],
                "foreign health reconciliation campaign")
        require(proof.get("schema") in {"verification-health-reconciliation-v3",
                "verification-health-observer-reconciliation-v3"}, "unknown health reconciliation schema")
        if "PREPARED" in path.stem or proof.get("status") == "PREPARED":
            skipped.append({"path": str(path), "sha256": proof["sha256"]})
            continue
        targets = proof.get("targets", [])
        rows = proof.get("cancelled_unstarted_observers", [])
        require(isinstance(targets, list) and isinstance(rows, list), "invalid observer reconciliation roster")
        for value in targets:
            candidates.setdefault(job_id(value), []).append({"path": str(path), "sha256": proof["sha256"],
                "campaign_binding": "EXPLICIT" if "campaign_sha256" in proof else "EXACT_CALLBACK_JOURNAL_JOIN"})
        for row in rows:
            require(state(row.get("state")) == "CANCELLED" and row.get("elapsed_seconds") == 0,
                    "observer cancellation proof is not unstarted")
            candidates.setdefault(job_id(row["job_id"]), []).append({"path": str(path), "sha256": proof["sha256"],
                "campaign_binding": "EXPLICIT", "cancellation_accounting": row})
    return candidates, skipped


def observer_card(path, campaign, root, scientific):
    card = load(path, "verification-health-card-v3")
    source_binding(card, campaign)
    require(Path(card["campaign_path"]).resolve() == Path(campaign["_path"]).resolve() and
            Path(card["result_root"]).resolve() == root, "observer campaign/result path differs")
    predecessor = job_id(card["predecessor_job_id"])
    require(predecessor in scientific, "observer lacks exact scientific submission journal")
    parent = scientific[predecessor]
    journal_path(card["job_card_path"], parent["card_path"])
    journal_path(card["predecessor_submission_path"], parent["submission_path"])
    require(card["job_card_sha256"] == parent["card"]["sha256"] and
            card["predecessor_submission_sha256"] == parent["receipt"]["sha256"], "observer predecessor/card binding differs")
    work = card["expected_work"]
    require(work["stage"] == parent["card"]["stage"] and work["model"] == parent["card"]["model"] and
            work["expected_uids_sha256"] == digest(work["expected_uids"]), "observer expected roster or model differs")
    for input_path, sha in work.get("inputs", {}).items():
        require(load(input_path)["sha256"] == sha, "observer source input changed")
    return card


def scientific_reconciliations(campaign_path, campaign, scientific):
    """Join every sealed failure history without granting recovery or success."""
    result = {}
    paths = Path(campaign_path).parent.glob("J1_TRANSPORT_RECONCILIATION-r*.json")
    for path in sorted(paths):
        proof = load(path, "verification-j1-transport-reconciliation-v3")
        source_binding(proof, campaign)
        seen = set()
        for row in proof["failed_jobs"]:
            job = job_id(row["job_id"])
            require(job not in seen, "duplicate failed job in scientific reconciliation")
            seen.add(job)
            require(job in scientific and row["job_card_sha256"] == scientific[job]["card"]["sha256"] and
                    row["submission_sha256"] == scientific[job]["receipt"]["sha256"], "scientific reconciliation journal differs")
            parent = scientific[job]
            for source, sha in row["source_receipts"].items():
                receipt = load(private_path(source))
                require(receipt["sha256"] == sha, "scientific reconciliation receipt changed")
                if "job_id" in receipt:
                    require(job_id(receipt["job_id"]) == job, "scientific reconciliation source job differs")
                if "job_card_sha256" in receipt:
                    require(receipt["job_card_sha256"] == parent["card"]["sha256"],
                            "scientific reconciliation source card differs")
                if "campaign_sha256" in receipt:
                    source_binding(receipt, campaign)
                if receipt.get("schema") == "verification-job-card-v3":
                    journal_path(source, parent["card_path"])
                    require(receipt["sha256"] == parent["card"]["sha256"],
                            "scientific reconciliation source card differs")
                elif receipt.get("schema") == "verification-submission-v3":
                    journal_path(source, parent["submission_path"])
                    require(receipt["sha256"] == parent["receipt"]["sha256"],
                            "scientific reconciliation source submission differs")
            if "invocation_result_path" in row:
                require(row["invocation_result_path"] in row["source_receipts"],
                        "scientific reconciliation invocation lacks source receipt binding")
            result.setdefault(job, []).append({"path": str(path.resolve()), "sha256": proof["sha256"],
                "job_card_path": parent["card_path"], "job_card_sha256": parent["card"]["sha256"],
                "submission_path": parent["submission_path"], "submission_sha256": parent["receipt"]["sha256"],
                "failed_job": row,
                "interpretation": "recorded execution failure retained; reconciliation does not change execution or scientific outcome"})
    return result


def observe(campaign_path, *, write=True):
    campaign_path = Path(campaign_path).resolve()
    campaign = load(campaign_path); check_files(campaign["code_files"])
    root = private_path(campaign["result_root"])
    # Auxiliary path is never written into a sealed campaign artifact.
    campaign = {**campaign, "_path": str(campaign_path)}
    unresolved = []; errors = []; scientific = {}; callbacks = []; controllers = []; observers = {}
    checked_sources = set()

    def checked(files):
        key = digest(files)
        if key not in checked_sources:
            check_files(files); checked_sources.add(key)

    def scan(paths, function):
        for path in sorted(paths):
            try:
                function(path)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                errors.append({"path": str(path), "type": type(exc).__name__, "message": str(exc)})

    def science(path):
        card = load(path, "verification-job-card-v3"); source_binding(card, campaign)
        require(path.parent.name == card["sha256"], "scientific journal directory/card differs")
        prefix = "cv3t:" if "transport_executor" in card else "cv3x:" if "extension" in card else "cv3r:" if "reader_executor" in card else "cv3:"
        comment = prefix + card["sha256"]
        intent(path.parent / "INTENT.json", "job_card_sha256", card["sha256"], comment, unresolved)
        receipt_path = path.parent / "SUBMISSION.json"
        if not receipt_path.exists():
            return
        receipt = load(receipt_path, "verification-submission-v3")
        require(receipt["job_card_sha256"] == card["sha256"], "scientific submission/card differs")
        job = job_id(receipt["job_id"]); require(job not in scientific, "duplicate scientific job ID")
        readback(receipt, job, comment)
        scientific[job] = {"job_id": job, "card": card, "receipt": receipt, "comment": comment,
            "card_path": str(path.resolve()), "submission_path": str(receipt_path.resolve())}

    scan((root / "submissions").glob("*/JOB_CARD.json"), science)

    def callback(path):
        card = observer_card(path, campaign, root, scientific)
        require(path.parent.name == card["sha256"], "callback journal directory/card differs")
        observers[card["sha256"]] = (card, str(path.resolve()))
        comment = "cv3h:" + card["sha256"]
        intent(path.parent / "INTENT.json", "health_card_sha256", card["sha256"], comment, unresolved)
        receipt_path = path.parent / "SUBMISSION.json"
        if not receipt_path.exists():
            return
        receipt = load(receipt_path, "verification-health-submission-v3")
        require(receipt["health_card_sha256"] == card["sha256"], "callback submission/card differs")
        job = job_id(receipt["job_id"]); readback(receipt, job, comment)
        callbacks.append({"job_id": job, "comment": comment, "health_card_sha256": card["sha256"],
            "model": card["expected_work"]["model"], "stage": card["expected_work"]["stage"],
            "predecessor_job_id": job_id(card["predecessor_job_id"]), "callback_job_id": job,
            "submission_path": str(receipt_path), "health_card_path": str(path)})

    scan((root / "health/submissions").glob("*/HEALTH_CARD.json"), callback)

    def controller(path):
        card = load(path, "verification-autodriver-card-v3"); source_binding(card, campaign)
        require(path.parent.name == card["sha256"] and Path(card["result_root"]).resolve() == root and
                Path(card["campaign_path"]).resolve() == campaign_path, "controller journal path differs")
        checked(card["program_files"])
        comment = "cv3auto:" + card["sha256"]
        intent(path.parent / "INTENT.json", "controller_card_sha256", card["sha256"], comment, unresolved)
        receipt_path = path.parent / "SUBMISSION.json"
        if not receipt_path.exists():
            return
        receipt = load(receipt_path, "verification-autodriver-submission-v3")
        require(receipt["controller_card_sha256"] == card["sha256"], "controller receipt/card differs")
        job = job_id(receipt["job_id"]); readback(receipt, job, comment)
        row = {"job_id": job, "comment": comment, "controller_card_sha256": card["sha256"],
            "mode": card["mode"], "parent_job_id": card["parent_job_id"], "wait_job_ids": card["wait_job_ids"],
            "dependency": card["dependency"], "operational_version": card.get("operational_version", "autodriver-r1"),
            "card_path": str(path), "submission_path": str(receipt_path), "program_files_sha256": digest(card["program_files"])}
        for name in ("PREARM", "RELEASE"):
            proof_path = path.parent / (name + ".json")
            if proof_path.exists():
                proof = load(proof_path)
                require(proof["controller_card_sha256"] == card["sha256"] and
                        proof["submission_sha256"] == receipt["sha256"], "controller protection proof/card differs")
                row[name.lower() + "_path"] = str(proof_path); row[name.lower() + "_sha256"] = proof["sha256"]
                if name == "PREARM":
                    require(str(proof["primary_job_id"]) == job, "prearm primary job differs")
                    for child in proof["children"]:
                        child_sha = child["controller_card_sha256"]
                        child_slot = root / "automation/submissions" / child_sha
                        child_card = load(child_slot / "CARD.json", "verification-autodriver-card-v3")
                        source_binding(child_card, campaign)
                        require(child_card["sha256"] == child_sha and child_card["parent_job_id"] == job and
                                child_card["mode"] == "rescue", "prearm child card/parent differs")
                        for label, filename in (("submission", "SUBMISSION"), ("prearm", "PREARM"), ("release", "RELEASE")):
                            bound_path = journal_path(child[label + "_path"], child_slot / (filename + ".json"))
                            bound_value = load(bound_path)
                            require(bound_value["sha256"] == child[label + "_sha256"] and
                                    bound_value["controller_card_sha256"] == child_sha,
                                    "prearm child journal binding differs")
                        require(str(load(child["submission_path"])["job_id"]) == job_id(child["job_id"]),
                                "prearm child job differs")
                    row["prearmed_children"] = proof["children"]
                else:
                    require(str(proof["job_id"]) == job, "release job differs")
        result_path = root / "automation/runs" / job / "RESULT.json"
        if result_path.exists():
            result = load(result_path, "verification-autodriver-result-v3"); source_binding(result, campaign)
            require(str(result["controller_job_id"]) == job, "controller result job differs")
            row.update(result_path=str(result_path), result_sha256=result["sha256"], status=result["status"],
                successor_job_id=result.get("successor_job_id"), rescue_job_id=result.get("rescue_job_id"))
        handoff_path = root / "automation/runs" / job / "HANDOFF.json"
        if handoff_path.exists():
            handoff = load(handoff_path)
            require(str(handoff["controller_job_id"]) == job, "handoff controller differs")
            successor_sha = handoff["successor_card_sha256"]
            successor_slot = root / "automation/submissions" / successor_sha
            journal_path(handoff["submission_path"], successor_slot / "SUBMISSION.json")
            successor = load(handoff["submission_path"])
            require(successor["controller_card_sha256"] == successor_sha and
                    str(successor["job_id"]) == job_id(handoff["successor_job_id"]), "handoff successor differs")
            row.update(handoff_path=str(handoff_path), handoff_sha256=handoff["sha256"],
                       successor_job_id=handoff["successor_job_id"])
        controllers.append(row)

    scan((root / "automation/submissions").glob("*/CARD.json"), controller)

    def owned(path):
        card = observer_card(path, campaign, root, scientific)
        require(path.stem == card["sha256"] and card.get("controller_owned") is True,
                "controller-owned observer path/ownership differs")
        checked(card["observer_files"])
        observers[card["sha256"]] = (card, str(path.resolve()))

    scan((root / "automation/observer-cards").glob("*.json"), owned)
    candidates, skipped = reconciliation_candidates(campaign_path, campaign)
    reconcile = scientific_reconciliations(campaign_path, campaign, scientific)
    ids = set(scientific) | {r["job_id"] for r in callbacks + controllers}
    accounting, queue = scheduler_snapshot(ids)
    for row in [*scientific.values(), *callbacks, *controllers]:
        validate_scheduler(row, accounting, queue)
    superseded = {}; reconciliations = []
    callback_ids = {row["job_id"] for row in callbacks}
    for job, references in candidates.items():
        require(job in callback_ids, "reconciliation target lacks exact current callback journal")
        observed = accounting.get(job, {})
        evidence = observed or next((r["cancellation_accounting"] for r in references if "cancellation_accounting" in r), {})
        if job not in queue and state(evidence.get("state")) == "CANCELLED" and evidence.get("elapsed_seconds") == 0:
            superseded[job] = references
            reconciliations.append({"job_id": job, "status": "SUPERSEDED_UNSTARTED_OBSERVER", "references": references})
    events = []; by_health = {}

    def event(path):
        value = load(path, "verification-health-event-v3"); source_binding(value, campaign)
        sha = value["health_card_sha256"]
        require(path.stem == sha and sha in observers, "event lacks exact observer card journal")
        card, card_path = observers[sha]; predecessor = job_id(value["predecessor_job_id"])
        require(predecessor == str(card["predecessor_job_id"]) and value["job_card_sha256"] == card["job_card_sha256"] and
                value["expected_uids_sha256"] == card["expected_work"]["expected_uids_sha256"] and
                value["model"] == card["expected_work"]["model"] and value["stage"] == card["expected_work"]["stage"],
                "health event scientific identity differs")
        owner = job_id(value["callback_job_id"])
        if card.get("controller_owned"):
            owner_row = next((r for r in controllers if r["job_id"] == owner), None)
            require(owner_row is not None and digest(card["observer_files"]) == owner_row["program_files_sha256"],
                    "event controller lacks exact sealed observer source")
        else:
            require(any(r["callback_job_id"] == owner and r["health_card_sha256"] == sha for r in callbacks),
                    "health event callback identity differs")
        row = {"state": value["state"], "model": value["model"], "stage": value["stage"],
            "predecessor_job_id": predecessor, "callback_job_id": owner, "health_card_sha256": sha,
            "job_card_sha256": value["job_card_sha256"], "event_path": str(path), "event_sha256": value["sha256"],
            "observer_card_path": card_path, "controller_owned": bool(card.get("controller_owned")),
            "accounting": value["accounting"], "reconciliation_references": reconcile.get(predecessor, []),
            "scientific_success": "ONLY_FROM_ORIGINAL_EVENT_AND_ARTIFACT_GATES"}
        if "price_path" in value:
            require(load(value["price_path"])["sha256"] == value["price_sha256"], "event immutable price changed")
            row.update(price_path=value["price_path"], price_sha256=value["price_sha256"])
        events.append(row); by_health[sha] = row

    scan((root / "health/events").glob("*.json"), event)
    for row in callbacks:
        job = row["callback_job_id"]; predecessor = row["predecessor_job_id"]
        row.update(callback_accounting=accounting.get(job), predecessor_accounting=accounting.get(predecessor),
                   callback_queue=queue.get(job), predecessor_queue=queue.get(predecessor))
        event_row = by_health.get(row["health_card_sha256"])
        if event_row:
            row.update(signal=event_row["state"], event_path=event_row["event_path"], event_sha256=event_row["event_sha256"])
            if "price_path" in event_row:
                row.update(price_path=event_row["price_path"], price_sha256=event_row["price_sha256"])
        elif job in superseded:
            row.update(signal="SUPERSEDED_UNSTARTED_OBSERVER", reconciliation_references=superseded[job])
        elif state(accounting.get(job, {}).get("state")) in TERMINAL:
            row["signal"] = "CALLBACK_FAILED_NO_EVENT"
        elif predecessor in queue:
            row["signal"] = "WAITING_FOR_PREDECESSOR"
        elif job in queue:
            row["signal"] = "SIGNAL_PENDING"
        else:
            row["signal"] = "ACCOUNTING_OR_SIGNAL_UNAVAILABLE"
    for row in controllers:
        row.update(accounting=accounting.get(row["job_id"]), queue=queue.get(row["job_id"]))
    science_rows = [{"job_id": job, "job_card_sha256": row["card"]["sha256"],
        "model": row["card"]["model"], "stage": row["card"]["stage"], "comment": row["comment"],
        "card_path": row["card_path"], "submission_path": row["submission_path"],
        "accounting": accounting.get(job), "queue": queue.get(job), "reconciliation_references": reconcile.get(job, [])}
        for job, row in sorted(scientific.items(), key=lambda item: int(item[0]))]
    result = {"schema": "verification-campaign-health-status-v3", "campaign_sha256": campaign["sha256"],
        "observed_epoch": time.time(), "reader_version": "health-status-r3", "callbacks": callbacks,
        "controllers": controllers, "events": events, "scientific_jobs": science_rows,
        "controller_observer_cards": [{"path": path, "sha256": sha, "predecessor_job_id": card["predecessor_job_id"]}
            for sha, (card, path) in observers.items() if card.get("controller_owned")],
        "reconciliations": reconciliations, "skipped_prepared_reconciliations": skipped, "errors": errors,
        "unresolved_submission_intents": unresolved, "scientific_dispatch": "NONE: bounded telemetry only",
        "automatic_submissions": 0}
    return save(campaign_path.parent / "HEALTH_STATUS.json", result, replace=True) if write else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    args = parser.parse_args(); result = observe(args.campaign)
    print(json.dumps({"observed_epoch": result["observed_epoch"], "callbacks": len(result["callbacks"]),
        "controllers": [{"job_id": row["job_id"], "mode": row["mode"], "queue": row["queue"],
                         "status": row.get("status")} for row in result["controllers"]],
        "events": [{"job_id": row["predecessor_job_id"], "state": row["state"],
                    "controller_owned": row["controller_owned"]} for row in result["events"]],
        "errors": result["errors"], "automatic_submissions": 0}, indent=2))


if __name__ == "__main__":
    main()
