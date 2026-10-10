"""Attach the unchanged finite observer to a verified terminal predecessor.

Slurm keeps completed dependency records for only30seconds here. This bounded
registrar uses actual sacct evidence; it never allocates scientific recovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save, seal
from moe_exp.verification_study.registry import REPO
import verification_study_v3_discovery_dispatch_r1 as dispatch
import verification_study_v3_health_r2 as health


def accounting(jobs):
    require(jobs and all(re.fullmatch(r"\d+", str(j)) for j in jobs), "invalid finite predecessor IDs")
    raw = subprocess.run(["sacct", "-nP", "-j", ",".join(jobs),
                          "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"],
                         check=True, capture_output=True, text=True, timeout=20).stdout
    result = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if fields[0] in jobs:
            result[fields[0]] = {"state": fields[1].split(" ", 1)[0], "exit_code": fields[2],
                                 "elapsed_seconds": int(fields[3]), "allocated_tres": fields[4]}
    return result


def prepare(campaign_path, submission_path, root, observed):
    campaign = load(campaign_path); check_files(campaign["code_files"])
    submitted = load(submission_path, "verification-submission-v3")
    card_path = submission_path.parent / "JOB_CARD.json"
    card = load(card_path, "verification-job-card-v3")
    require(card["campaign_sha256"] == campaign["sha256"] and card["code_files"] == campaign["code_files"] and
            submitted["job_card_sha256"] == card["sha256"] and card["stage"] == "run",
            "terminal registrar requires the unchanged native discovery card")
    require(root == Path(campaign["result_root"]) and
            observed["state"] in health.TERMINAL, "terminal predecessor accounting required")
    work = health.expected_work(card, root)
    args = health.arguments(card)
    require(load(args["--manifest"])["stage"] == "discovery", "terminal registrar scope is discovery only")
    proof_path = root / "health/terminal-proofs" / (card["sha256"] + ".json")
    proof = save(proof_path, {"schema": "verification-terminal-observer-proof-v3",
        "campaign_sha256": campaign["sha256"], "predecessor_job_id": submitted["job_id"],
        "predecessor_submission_sha256": submitted["sha256"], "job_card_sha256": card["sha256"],
        "accounting": observed, "scientific_reruns": 0})
    observer = seal({"schema": "verification-health-card-v3", "campaign_path": str(campaign_path),
        "campaign_sha256": campaign["sha256"], "predecessor_job_id": submitted["job_id"],
        "predecessor_submission_path": str(submission_path), "predecessor_submission_sha256": submitted["sha256"],
        "job_card_path": str(card_path), "job_card_sha256": card["sha256"], "result_root": str(root),
        "observer_files": {str(p): file_sha(p) for p in (health.SCRIPT, health.SHELL)}, "expected_work": work,
        "claim": card["claim"], "artifact": "exact terminal predecessor completion/failure and validated actual prices",
        "failure_interpretation": "execution, artifact and observer failures stay distinct from causal effects",
        "resources": {"partition": "lrd_all_serial", "qos": "normal", "gpus": 0, "cpus": 1,
                      "memory_gb": 4, "walltime_seconds": 600, "billing_core_hours_max": 0},
        "authorization": "user: finite completion/failure health signals and autonomous approved campaign continuation",
        "scientific_dispatch": "persistent Goal only; observer allocates zero scientific recovery",
        "terminal_proof_path": str(proof_path), "terminal_proof_sha256": proof["sha256"],
        "registration_files": {str(Path(__file__).resolve()): file_sha(__file__)},
        "dependency_policy": "no afterany is needed because actual predecessor terminal accounting is already bound"})
    slot = root / "health/submissions" / observer["sha256"]
    save(slot / "HEALTH_CARD.json", observer)
    return card, observer, slot


def register(campaign_path, submission_path, root, observed, submit=False):
    submitted = load(submission_path); card = load(submission_path.parent / "JOB_CARD.json")
    old = dispatch.callback(root, card, submitted)
    if old:
        return {"status": "REGISTERED", "predecessor_job_id": submitted["job_id"], "callback_job_id": old["job_id"]}
    require(observed["state"] in health.TERMINAL, "pending or unavailable science cannot become terminal")
    if not submit or not dispatch.serial_slots()["free"]:
        return {"status": "DEFERRED_SERIAL_SUBMIT_LIMIT" if submit else "TERMINAL_CALLBACK_READY",
                "predecessor_job_id": submitted["job_id"]}
    card, observer, slot = prepare(campaign_path, submission_path, root, observed)
    receipt_path = slot / "SUBMISSION.json"; comment = "cv3h:" + observer["sha256"]
    if (slot / "INTENT.json").exists():
        queue = health.command(["squeue", "-h", "-u", "lmolfett", "-o", "%i|%k"])
        matches = [line.split("|", 1)[0] for line in queue.splitlines() if line.endswith("|" + comment)]
        require(len(matches) == 1, "terminal callback intent requires accounting reconciliation; no blind retry")
        read_back = health.command(["scontrol", "show", "job", matches[0]])
        require("JobId=" + matches[0] in read_back and "Account=iscrc_miosr" in read_back,
                "terminal callback reconciliation readback differs")
        saved = save(receipt_path, {"schema": "verification-health-submission-v3", "job_id": matches[0],
            "health_card_sha256": observer["sha256"], "predecessor_job_id": submitted["job_id"],
            "read_back": read_back, "reconciled": True})
        return {"status": "REGISTERED", "predecessor_job_id": submitted["job_id"], "callback_job_id": saved["job_id"]}
    snapshot_value = health.preflight()
    require(load(observer["terminal_proof_path"])["sha256"] == observer["terminal_proof_sha256"], "terminal proof changed")
    check_files(observer["observer_files"]); check_files(observer["registration_files"])
    logs = root / "health/logs"; logs.mkdir(parents=True, exist_ok=True)
    argv = ["sbatch", "--account=iscrc_miosr", "--partition=lrd_all_serial", "--qos=normal", "--nodes=1",
            "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:10:00", "--no-requeue",
            "--job-name=cv3-health-" + card["model"], "--comment=" + comment, "--chdir=" + str(REPO),
            "--output=" + str(logs / "%x-%j.out"), "--error=" + str(logs / "%x-%j.err"),
            str(health.SHELL), str(slot / "HEALTH_CARD.json")]
    try:
        subprocess.run([argv[0], "--test-only", *argv[1:]], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        if "QOSMaxSubmitJobPerUserLimit" in (error.stderr or ""):
            return {"status": "DEFERRED_SERIAL_SUBMIT_LIMIT", "predecessor_job_id": submitted["job_id"]}
        raise
    snapshot = save(slot / "PREFLIGHT.json", snapshot_value)
    intent = save(slot / "INTENT.json", {"schema": "verification-health-intent-v3", "health_card_sha256": observer["sha256"],
        "preflight_sha256": snapshot["sha256"], "epoch": time.time(), "terminal_proof_sha256": observer["terminal_proof_sha256"]})
    try:
        response = health.command([argv[0], "--parsable", *argv[1:]])
    except subprocess.CalledProcessError as error:
        save(slot / "SUBMISSION_ERROR.json", {"schema": "verification-health-submission-error-v3",
            "intent_sha256": intent["sha256"], "returncode": error.returncode,
            "stdout": (error.stdout or "")[:1000], "stderr": (error.stderr or "")[:2000],
            "resolution": "reconcile exact comment and accounting before any new allocation"})
        raise
    job = response.split(";", 1)[0]; require(re.fullmatch(r"\d+", job), "invalid terminal callback job ID")
    read_back = health.command(["scontrol", "show", "job", job])
    require("JobId=" + job in read_back and "Account=iscrc_miosr" in read_back, "terminal callback readback failed")
    save(receipt_path, {"schema": "verification-health-submission-v3", "health_card_sha256": observer["sha256"],
        "job_id": job, "predecessor_job_id": submitted["job_id"], "read_back": read_back,
        "preflight_sha256": snapshot["sha256"], "terminal_proof_sha256": observer["terminal_proof_sha256"]})
    return {"status": "REGISTERED", "predecessor_job_id": submitted["job_id"], "callback_job_id": job}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(); path = args.campaign.resolve(); campaign = load(path); root = Path(campaign["result_root"])
    with dispatch.planned.lock(root):
        check_files(campaign["code_files"])
        wave = load(path.parent / "DISCOVERY_CONTINUATION_JOBS-r2.json")
        require(wave["campaign_sha256"] == campaign["sha256"], "foreign finite discovery wave")
        ids = [e["job_id"] for e in wave["cards"]]; states = accounting(ids); rows = []
        for entry in wave["cards"]:
            job = entry["job_id"]; observed = states.get(job)
            if observed is None or observed["state"] not in health.TERMINAL:
                continue
            submission = root / "submissions" / entry["job_card_sha256"] / "SUBMISSION.json"
            require(load(submission)["job_id"] == job, "terminal predecessor receipt changed")
            result = register(path, submission, root, observed, args.submit)
            rows.append(result); print(json.dumps(result), flush=True)
        save(path.parent / "TERMINAL_HEALTH_STATUS-r1.json", {"schema": "verification-terminal-health-status-v3",
            "campaign_sha256": campaign["sha256"], "observed_epoch": time.time(), "source_file_sha256": file_sha(__file__),
            "terminal_predecessors": rows, "scientific_reruns": 0, "serial_slots": dispatch.serial_slots()}, replace=True)
        print(json.dumps({"terminal_predecessors": len(rows), "scientific_reruns": 0}), flush=True)


if __name__ == "__main__":
    main()
