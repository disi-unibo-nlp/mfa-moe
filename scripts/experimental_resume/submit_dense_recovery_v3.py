"""Finite qualified existing-label recovery dispatch with exact native accounting.

This additive correction parses sacct fields instead of assuming a trailing
separator. It subtracts all queued project commitments and prices only jobs
that do not already have a verified submission receipt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from moe_exp.routing_control.study_v1 import file_sha, require, save, sealed
import dispatch_overnight_readers_v1 as shared
from dispatch_utility_production_v1 import remaining_commitments
from submit_routing_study_v1 import live, OUT, REPO

DIRECTORY = OUT / "dense-recovery-v3"
RECEIPTS = OUT / "submissions-v5"
WRAPPER = Path(__file__).with_name("routing_study_v2.sbatch")


def completed_zero(output, identifier):
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    exact = [row for row in rows if row[0] == str(identifier)]
    return len(exact) == 1 and len(exact[0]) >= 3 and exact[0][1:3] == ["COMPLETED", "0:0"]


def main():
    observations, _remaining = live()
    manifest, price = sealed(DIRECTORY / "MANIFEST.json"), sealed(DIRECTORY / "PRICE.json")
    require(manifest == sealed(OUT / "dense-recovery-v2/MANIFEST.json") and
            price == sealed(OUT / "dense-recovery-v2/PRICE.json"), "scientific measurement changed")
    require(all(file_sha(p) == h for p, h in manifest["code_files"].items()), "frozen recovery source changed")
    qual = sealed(DIRECTORY / "PACKAGING_QUALIFICATION.json")
    require(qual["status"] == "PASS" and qual["manifest_sha256"] == manifest["sha256"], "packaging unqualified")
    command = ["sacct", "-X", "-nP", "-j", qual["job_id"], "--format=JobIDRaw,State,ExitCode"]
    final = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    require(completed_zero(final, qual["job_id"]), "qualification allocation not successfully completed")
    retry = sealed(OUT / "submissions-v4/qualification.json")
    require(retry["job_id"] == qual["job_id"] and retry["binding"]["code_files"][str(WRAPPER)] == file_sha(WRAPPER),
            "qualified runtime wrapper changed")
    queue = subprocess.run(["squeue", "-h", "-A", "iscrc_miosr", "-o", "%100i|%T|%l|%L|%C|%b|%D"],
        check=True, capture_output=True, text=True).stdout
    budget = remaining_commitments(observations["balance"]["output"], queue)
    required = sum(g["wall_seconds"] * 16 / 3600 for i, g in enumerate(price["groups"])
                   if not (RECEIPTS / f"group-{i}.json").exists())
    if not (RECEIPTS / "dense-analysis.json").exists():
        required += 2
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    save(RECEIPTS / ("preflight-" + str(os.getpid()) + ".json"), {"schema": "routing-study-recovery-project-preflight-v3",
        "observations": observations, "queue_commitments": queue, "budget": budget,
        "required_additional_billing_core_hours": required, "qualification_accounting": {"command": command, "output": final}})
    require(required <= budget["uncommitted_reported_billing_core_hours"], "complete recovery cannot fit uncommitted project budget")
    authorization = REPO / "report/experimental-resume-v1/RESOURCE_AUTHORIZATION_EXPANDED.json"
    binding = {"schema": "routing-study-qualified-dense-dispatch-v3", "manifest_sha256": manifest["sha256"],
        "price_sha256": price["sha256"], "qualification_sha256": qual["sha256"],
        "qualified_retry_receipt_sha256": retry["sha256"],
        "authorization_file_sha256": file_sha(authorization), "code_files": {str(p): file_sha(p) for p in (WRAPPER, Path(__file__))},
        "scope": "Existing 187,524 assigned sentence labels only; immutable historical artifacts; no new family outcomes."}
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    jobs = []
    for i, group in enumerate(price["groups"]):
        wall = group["wall_seconds"]
        args = ["--partition=boost_usr_prod", "--gres=gpu:2", "--cpus-per-task=16", "--mem=120G",
            f'--time={wall//3600:02d}:{wall//60%60:02d}:{wall%60:02d}', "--job-name=routing-dense-packed-" + str(i),
            str(WRAPPER), "run", "--directory", str(DIRECTORY), "--group-index", str(i)]
        jobs.append(shared.submit(RECEIPTS, "group-" + str(i), args, os.environ.copy(), binding))
    analysis = shared.submit(RECEIPTS, "dense-analysis", ["--dependency=afterok:" + ":".join(jobs),
        "--job-name=routing-dense-recovered-analysis", str(WRAPPER), "analyze", "--directory", str(DIRECTORY)], os.environ.copy(), binding)
    save(RECEIPTS / "RECOVERY_CHAIN.json", {"schema": "routing-study-complete-dense-recovery-chain-v3", "binding": binding,
        "jobs": jobs, "analysis_job": analysis, "assigned_labels": manifest["max_labels"],
        "complete_maximum_GPU_hours": price["maximum_GPU_hours"]})
    print(json.dumps({"status": "VERIFIED_SUBMITTED_EXISTING_LABEL_RECOVERY", "jobs": jobs, "analysis_job": analysis}), flush=True)


if __name__ == "__main__":
    main()
