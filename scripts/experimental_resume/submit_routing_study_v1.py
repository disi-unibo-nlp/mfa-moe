"""Finite native-login dispatch for routing study CPU work and dense qualification.

This dispatcher reuses exact receipts. Expanded historical authorization covers
completion of existing labels. New target and final-evaluation stages have their
own prospective price/authorization gate and are not submitted here.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import socket
import subprocess

from moe_exp.routing_control.study_v1 import file_sha, require, save, sealed
import dispatch_overnight_readers_v1 as shared

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report/routing-study-v1"
WRAPPER = Path(__file__).with_name("routing_study_v1.sbatch")


def live():
    require(getpass.getuser() == "lmolfett" and socket.gethostname().startswith("login") and
            socket.gethostname().endswith("leonardo.local"), "dispatch requires lmolfett native LEONARDO login")
    observations = {}
    commands = {"balance": ["saldo", "-b"], "queue": ["squeue", "-u", "lmolfett"],
        "association": ["sacctmgr", "-nP", "show", "assoc", "where", "user=lmolfett", "format=Account,Partition,QOS,DefaultQOS"],
        "gpu_partition": ["scontrol", "show", "partition", "boost_usr_prod"],
        "cpu_partition": ["scontrol", "show", "partition", "lrd_all_viz"]}
    for key, command in commands.items():
        observations[key] = {"command": command,
            "output": subprocess.run(command, check=True, capture_output=True, text=True).stdout}
    require("iscrc_miosr|" in observations["association"]["output"] and
            "normal" in observations["association"]["output"], "account/QoS association changed")
    require("MaxTime=1-00:00:00" in observations["gpu_partition"]["output"] and
            "State=UP" in observations["gpu_partition"]["output"], "GPU partition limit changed")
    balance_lines = [line.split() for line in observations["balance"]["output"].splitlines()
                     if line.startswith("IscrC_MIOSR")]
    require(len(balance_lines) == 1 and len(balance_lines[0]) >= 9, "unrecognized project balance")
    row = balance_lines[0]
    remaining = float(row[3]) - float(row[5])
    return observations, remaining


def submit(mode):
    observations, remaining = live()
    OUT.mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    directory = OUT / "submissions-v3"
    directory.mkdir(exist_ok=True)
    authorization = REPO / "report/experimental-resume-v1/RESOURCE_AUTHORIZATION_EXPANDED.json"
    binding = {"schema": "routing-study-dispatch-binding-v1", "mode": mode,
        "code_files": {str(p): file_sha(p) for p in (WRAPPER, Path(__file__).resolve(),
            Path(__file__).with_name("routing_study_v1.py"), Path(__file__).with_name("dense_recovery_v2.py"),
            REPO / "src/moe_exp/routing_control/study_v1.py", REPO / "src/moe_exp/routing_control/state_analysis_v1.py")},
        "recorded_authorization": {"path": str(authorization), "file_sha256": file_sha(authorization)},
        "scope": "Existing studies: immutable reproduction, saved discovery analysis, qualified dense-label recovery."}
    save(directory / (mode + "-live-" + str(os.getpid()) + ".json"), {
        "schema": "routing-study-native-preflight-v1", "observations": observations,
        "remaining_billing_core_hours": remaining})
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    env = os.environ.copy()
    if mode == "cpu":
        require(remaining >= 6, "insufficient balance for three two-CPU analyses")
        previous = None
        jobs = []
        for stage in ("inventory", "discovery", "report"):
            arguments = (["--dependency=afterok:" + previous] if previous else [])
            arguments.extend(["--job-name=routing-study-" + stage, str(WRAPPER), stage])
            previous = shared.submit(directory, stage, arguments, env, binding)
            jobs.append(previous)
        save(directory / "CPU_CHAIN.json", {"schema": "routing-study-cpu-chain-v1", "binding": binding,
            "jobs": jobs, "maximum_CPU_core_hours": 6, "GPU_hours": 0})
    elif mode == "qualification":
        recovery = OUT / "dense-recovery-v2"
        manifest, price = sealed(recovery / "MANIFEST.json"), sealed(recovery / "PRICE.json")
        inv = sealed(OUT / "STUDY_INVENTORY_v2.json")
        require(inv["ambiguous_historical_calls_terminal"], "ambiguous historical calls still active")
        require(remaining >= 8, "insufficient balance for bounded qualification")
        binding.update(manifest_sha256=manifest["sha256"], price_sha256=price["sha256"])
        args = ["--partition=boost_usr_prod", "--gres=gpu:2", "--cpus-per-task=16", "--mem=120G",
                "--time=00:30:00", "--job-name=routing-dense-packaging-qual", str(WRAPPER),
                "qualify", "--directory", str(recovery)]
        shared.submit(directory, "qualification", args, env, binding)
    else:
        recovery = OUT / "dense-recovery-v2"
        manifest, price = sealed(recovery / "MANIFEST.json"), sealed(recovery / "PRICE.json")
        qual = sealed(recovery / "PACKAGING_QUALIFICATION.json")
        require(qual["status"] == "PASS" and qual["manifest_sha256"] == manifest["sha256"], "packaging not qualified")
        require(remaining >= price["maximum_GPU_hours"] * 8, "insufficient live balance for complete packed recovery")
        binding.update(manifest_sha256=manifest["sha256"], price_sha256=price["sha256"], qualification_sha256=qual["sha256"])
        jobs = []
        for index, group in enumerate(price["groups"]):
            wall = group["wall_seconds"]
            clock = f"{wall // 3600:02d}:{wall // 60 % 60:02d}:{wall % 60:02d}"
            args = ["--partition=boost_usr_prod", "--gres=gpu:2", "--cpus-per-task=16", "--mem=120G",
                    "--time=" + clock, "--job-name=routing-dense-packed-" + str(index), str(WRAPPER),
                    "run", "--directory", str(recovery), "--group-index", str(index)]
            jobs.append(shared.submit(directory, "group-" + str(index), args, env, binding))
        args = ["--dependency=afterok:" + ":".join(jobs), "--job-name=routing-dense-recovered-analysis",
                str(WRAPPER), "analyze", "--directory", str(recovery)]
        analysis_job = shared.submit(directory, "dense-analysis", args, env, binding)
        save(directory / "RECOVERY_CHAIN.json", {"schema": "routing-study-dense-recovery-chain-v2", "binding": binding,
            "jobs": jobs, "analysis_job": analysis_job, "complete_maximum_GPU_hours": price["maximum_GPU_hours"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "qualification", "recovery"))
    submit(parser.parse_args().mode)
