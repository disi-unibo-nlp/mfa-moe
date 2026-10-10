"""Execution-only IPC repair and historical integration dispatch.

The failed qualification remains immutable in v2. A fresh v3 overlay uses the
same frozen batches, sampler, model profile and measurement implementation.
No new target screen or independent evaluation workload is authorized here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from moe_exp.routing_control.study_v1 import file_sha, require, save, sealed
import dispatch_overnight_readers_v1 as shared
from submit_routing_study_v1 import live, OUT, REPO

WRAPPER = Path(__file__).with_name("routing_study_v2.sbatch")
DIRECTORY = OUT / "dense-recovery-v3"
RECEIPTS = OUT / "submissions-v4"


def dispatch(mode):
    observations, remaining = live()
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    save(RECEIPTS / (mode + "-preflight-" + str(os.getpid()) + ".json"), {
        "schema": "routing-study-execution-repair-preflight-v1", "observations": observations,
        "remaining_billing_core_hours": remaining})
    original = OUT / "dense-recovery-v2"
    manifest, price = sealed(original / "MANIFEST.json"), sealed(original / "PRICE.json")
    require(sealed(OUT / "STUDY_INVENTORY_v2.json")["ambiguous_historical_calls_terminal"],
            "historical label calls not reconciled")
    binding = {"schema": "routing-study-execution-repair-binding-v1", "mode": mode,
        "frozen_dense_manifest_sha256": manifest["sha256"], "frozen_price_sha256": price["sha256"],
        "code_files": {str(p): file_sha(p) for p in (WRAPPER, Path(__file__))},
        "recorded_authorization": {"path": str(REPO / "report/experimental-resume-v1/RESOURCE_AUTHORIZATION_EXPANDED.json"),
            "file_sha256": file_sha(REPO / "report/experimental-resume-v1/RESOURCE_AUTHORIZATION_EXPANDED.json")},
        "scope": "Existing dense-label completion and historical routing analysis; no new scientific workload."}
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    env = os.environ.copy()
    if mode == "integration":
        script = Path(__file__).with_name("integrate_historical_routing_v1.sbatch")
        binding["code_files"].update({str(p): file_sha(p) for p in (script,
            script.with_suffix(".py"), REPO / "src/moe_exp/routing_control/historical_analysis_v1.py")})
        require(remaining >= 2, "insufficient remaining balance for historical CPU integration")
        subprocess.run(["bash", "-n", str(script)], check=True)
        job = shared.submit(RECEIPTS, "integration", [str(script)], env, binding)
        print(json.dumps({"integration_job": job}), flush=True)
    elif mode == "qualification":
        # Reconcile the exact failed allocation and its pre-model attempt.
        result = subprocess.run(["sacct", "-X", "-nP", "-j", "59490750",
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"], check=True, capture_output=True, text=True).stdout
        require("59490750|FAILED|1:0|" in result, "previous qualification is not terminal failed")
        err = OUT / "logs/routing-dense-packaging-qual-59490750.err"
        require("longer than 107 characters" in err.read_text() and
                not list(original.glob("LOAD-*.json")) and not (original / "PACKAGING_QUALIFICATION.json").exists(),
                "failure cannot be classified as pre-model IPC construction")
        attempt = sealed(original / "QUALIFICATION_ATTEMPT.json")
        require(attempt["job_id"] == "59490750" and attempt["manifest_sha256"] == manifest["sha256"],
                "failed attempt binding differs")
        require(len("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/rd-9999999999/tmp/" + "x"*36) <= 107,
                "private runtime path still exceeds IPC limit")
        require(remaining >= 10, "insufficient remaining balance for bounded execution repair")
        save(RECEIPTS / "EXECUTION_AMENDMENT.json", {"schema": "routing-study-dense-execution-amendment-v1",
            "binding": binding, "failed_job": "59490750", "failed_accounting": result,
            "failed_attempt_sha256": attempt["sha256"], "failed_error_file_sha256": file_sha(err),
            "repair": "Only shorten private runtime path; preserve frozen measurement code and use a fresh overlay.",
            "new_directory": str(DIRECTORY), "maximum_additional_GPU_hours": 1,
            "maximum_additional_CPU_core_hours": 2,
            "scientific_outputs_from_failed_job": 0})
        prepare = shared.submit(RECEIPTS, "dense-prepare", ["--job-name=routing-dense-repair-prepare", str(WRAPPER),
            "prepare", "--directory", str(DIRECTORY)], env, binding)
        args = ["--dependency=afterok:" + prepare, "--partition=boost_usr_prod", "--gres=gpu:2",
            "--cpus-per-task=16", "--mem=120G", "--time=00:30:00", "--job-name=routing-dense-shortpath-qual",
            str(WRAPPER), "qualify", "--directory", str(DIRECTORY)]
        job = shared.submit(RECEIPTS, "qualification", args, env, binding)
        print(json.dumps({"prepare_job": prepare, "qualification_job": job}), flush=True)
    else:
        actual, actual_price = sealed(DIRECTORY / "MANIFEST.json"), sealed(DIRECTORY / "PRICE.json")
        require(actual == manifest and actual_price == price, "fresh recovery scientific contract differs")
        qual = sealed(DIRECTORY / "PACKAGING_QUALIFICATION.json")
        require(qual["status"] == "PASS" and qual["manifest_sha256"] == manifest["sha256"], "packaging not qualified")
        final = subprocess.run(["sacct", "-X", "-nP", "-j", qual["job_id"],
            "--format=JobIDRaw,State,ExitCode"], check=True, capture_output=True, text=True).stdout
        require(f'{qual["job_id"]}|COMPLETED|0:0|' in final, "qualification allocation not successfully completed")
        require(remaining >= price["maximum_GPU_hours"] * 8, "insufficient balance for complete recovery")
        binding["qualification_sha256"] = qual["sha256"]
        jobs = []
        for index, group in enumerate(price["groups"]):
            wall = group["wall_seconds"]
            args = ["--partition=boost_usr_prod", "--gres=gpu:2", "--cpus-per-task=16", "--mem=120G",
                f'--time={wall//3600:02d}:{wall//60%60:02d}:{wall%60:02d}',
                "--job-name=routing-dense-packed-" + str(index), str(WRAPPER), "run",
                "--directory", str(DIRECTORY), "--group-index", str(index)]
            jobs.append(shared.submit(RECEIPTS, "group-" + str(index), args, env, binding))
        analysis = shared.submit(RECEIPTS, "dense-analysis", ["--dependency=afterok:" + ":".join(jobs),
            "--job-name=routing-dense-recovered-analysis", str(WRAPPER), "analyze", "--directory", str(DIRECTORY)], env, binding)
        save(RECEIPTS / "RECOVERY_CHAIN.json", {"schema": "routing-study-dense-recovery-chain-v3", "binding": binding,
            "jobs": jobs, "analysis_job": analysis, "maximum_GPU_hours": price["maximum_GPU_hours"]})
        print(json.dumps({"packed_jobs": jobs, "analysis_job": analysis}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("integration", "qualification", "recovery"))
    dispatch(parser.parse_args().mode)
