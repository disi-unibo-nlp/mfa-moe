"""Finite claim-linked cards and authorization-bound Slurm handoff."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

from .contracts import REPO, check_files, digest, load, private_path, require, save, seal
from .journal import Journal


def reconcile_submission(card_path, root):
    """Read exact Slurm comments/accounting; never infer rejection from no match."""
    root = private_path(root)
    value = load(card_path)
    uid = digest(["submission", value["sha256"]])
    directory = root / "submissions"
    intent = load(directory / (uid + ".intent.json"))
    require(intent["card"] == value["sha256"], "foreign submission intent")
    identity = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    host = subprocess.run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    require(identity == "lmolfett" and "leonardo" in host, "wrong accounting target")
    start = datetime.fromtimestamp(intent["created_epoch"] - 60, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    queue = subprocess.run(["squeue", "-u", "lmolfett", "-h", "-o", "%i|%u|%a|%j|%T|%k"],
                           capture_output=True, text=True, check=True, timeout=30).stdout
    accounting = subprocess.run(["sacct", "-S", start, "-u", "lmolfett", "-X", "-n", "-P", "-o",
                                 "JobIDRaw,User,Account,JobName%80,State,ExitCode,Start,ElapsedRaw,Comment%80"],
                                capture_output=True, text=True, check=True, timeout=30).stdout
    matches = {}
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    for source, raw in (("queue", queue), ("accounting", accounting)):
        for line in raw.splitlines():
            fields = line.rstrip("|").split("|")
            if not fields or fields[-1].strip() != "em1-" + uid:
                continue
            require(len(fields) >= 6 and fields[0].isdigit() and fields[1] == "lmolfett" and
                    fields[2].lower() == "iscrc_miosr" and fields[3] == "em1-" + value["stage"],
                    "Slurm comment/account identity collision")
            prior = matches.get(fields[0], {})
            matches[fields[0]] = {**prior, source: fields}
    require(len(matches) <= 1, "duplicate owned jobs observed; manual reconciliation required")
    status, job = "UNKNOWN_NO_MATCH", next(iter(matches), None)
    if job:
        status = "ACCEPTED"
        accepted = directory / (uid + ".accepted.json")
        if accepted.exists():
            require(load(accepted)["job_id"] == job, "accepted job conflicts with live accounting")
        else:
            save(accepted, {"schema": "expert-memory-submission-v1", "intent_sha256": intent["sha256"],
                           "card_sha256": value["sha256"], "job_id": job, "reconciled": True,
                           "readback": matches[job], "observed_epoch": time.time()})
        record = matches[job].get("accounting")
        if record and record[4].split()[0].rstrip("+") in terminal:
            status = "TERMINAL_ACCOUNTING"
            target = directory / (uid + ".terminal.json")
            if not target.exists():
                save(target, {"intent_sha256": intent["sha256"], "job_id": job, "terminal": True,
                              "accounting": record, "scientific_success_inferred": False})
    return save(directory / (uid + f".reconciliation-{time.time_ns()}.json"),
                {"schema": "expert-memory-submission-reconciliation-v1", "status": status,
                 "intent_sha256": intent["sha256"], "job_id": job, "queue": queue, "accounting": accounting,
                 "no_match_is_not_rejection": True, "performs_dispatch": False})


def card(campaign, stage, *, model=None, seconds=None, price=None, request=None):
    require(stage in ("measure-cpu", "selection", "transitions", "qualification", "pricing", "pilot", "final", "grade", "analyze"),
            "unknown finite stage")
    gpu = stage in ("qualification", "pricing", "pilot", "final")
    initial = stage in ("measure-cpu", "qualification", "pricing")
    require(initial or price is not None, "bulk work requires measured role/context price")
    ceiling = 1200 if stage == "measure-cpu" else 7200 if stage in ("qualification", "pricing") else 28800 if gpu else 14400
    seconds = ceiling if seconds is None else seconds
    require(0 < seconds <= ceiling, "card walltime exceeds stage limit")
    if stage in ("selection", "qualification", "pricing", "pilot", "final"):
        from .protocol import MODELS
        require(model in MODELS, "finite model-specific card requires a model")
    if price is not None:
        require(price["campaign_sha256"] == campaign["sha256"], "foreign stage price")
    exhibit = "Figure: held-out expert coverage and stability" if stage in ("measure-cpu", "selection") else \
              "Figure: all-seven consecutive-sentence transitions" if stage == "transitions" else \
              "Table: all-seven loader capability and realized memory" if stage == "qualification" else \
              "Table: seven-model paired accuracy-memory tradeoff"
    return seal({"schema": "expert-memory-job-card-v1", "stage": stage, "model": model,
                 "campaign_sha256": campaign["sha256"], "paper_artifact": exhibit,
                 "claim": "C1 all-seven consecutive-sentence dynamics" if stage == "transitions" else
                          "C2 retrospective training-family coverage" if stage in ("selection", "measure-cpu") else
                          "C3 physically reduced expert storage with jointly measured accuracy",
                 "reused_sources": campaign["saved_source_files"],
                 "missing_measurement": stage + ": existing aggregate coverage/static bytes do not supply this cell",
                 "independent_units": "duplicate families; qualification is engineering evidence only",
                 "uncertainty": "engineering evidence only; numerical errors and storage recorded, no accuracy inference" if stage == "qualification" else
                                "509 shared families; label coverage and censoring reported" if stage == "transitions" else
                                "five common training-family folds" if stage in ("selection", "measure-cpu") else
                                "21 simultaneous family contrasts; pilot-blinded final-size rule; no retention margin",
                 "controls": "native/state-balanced/pooled labelled-token frequency/fixed matched random; identical cache specs",
                 "unfavorable_interpretation": "report failed capability, missing cell or observed tradeoff without redraw or gate relaxation",
                 "price_basis": "bounded first measurement exception" if initial else price,
                 "resources": {"account": "iscrc_miosr", "partition": "boost_usr_prod" if gpu else "lrd_all_serial",
                               "qos": "normal", "nodes": 1, "tasks": 1, "cpus": 16 if gpu else 4,
                               "memory_gib": 120 if gpu else 16, "gpus": 2 if gpu else 0,
                               "seconds": seconds, "billing_core_hours_upper": seconds * (16 if gpu else 4) / 3600},
                 "qualification_total_ceiling_core_hours": 224, "qualification_total_ceiling_gpu_hours": 28,
                 "maximum_infrastructure_retries": 0,
                 "request": request, "authorization": None, "submission_allowed": False})


def validate_allocation(card):
    resource = card["resources"]
    require(os.environ.get("SLURM_JOB_ACCOUNT") == resource["account"] and
            os.environ.get("SLURM_JOB_PARTITION") == resource["partition"] and
            int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) == resource["cpus"],
            "live allocation differs from bound card")
    require(int(os.environ.get("SLURM_MEM_PER_NODE", "0")) == resource["memory_gib"] * 1024 and
            int(os.environ.get("SLURM_NTASKS", "0")) == 1 and
            int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) == 1,
            "live host memory/task/node shape differs from card")
    if resource["gpus"]:
        require(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == resource["gpus"],
                "allocated GPU count differs")


def submit(card_path, authorization_path, campaign_path, root):
    root = private_path(root)
    card_value, authorization, campaign = load(card_path), load(authorization_path), load(campaign_path)
    require(authorization.get("approved_card_sha256") == card_value["sha256"] and
            authorization.get("user_quote") and authorization.get("resources") == card_value["resources"],
            "explicit workload and exact resource approval required")
    check_files(campaign["code_files"])
    require(card_value["campaign_sha256"] == campaign["sha256"] and card_value["request"] is not None,
            "card request is absent or foreign")
    # The CLI does not infer authorization from this preparatory request.
    identity = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    host = subprocess.run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    require(identity == "lmolfett" and "leonardo" in host, "wrong submission target")
    preflight = {}
    for name, cmd in {
        "balance": ["saldo", "-b"], "association": ["sacctmgr", "-nP", "show", "assoc", "where", "user=lmolfett", "format=Account,Partition,QOS,DefaultQOS"],
        "partition": ["scontrol", "show", "partition", card_value["resources"]["partition"]],
        "quota": ["cinQuota"], "jobs": ["squeue", "-u", "lmolfett", "-h", "-o", "%i|%P|%j|%T"]}.items():
        preflight[name] = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout
    require("iscrc_miosr" in preflight["association"] and "normal" in preflight["association"] and
            "State=UP" in preflight["partition"], "live entitlement/partition unavailable")
    if card_value["resources"]["gpus"] == 0:
        require(not any("|em1-" in line and "|lrd_all_serial|" in line for line in preflight["jobs"].splitlines()),
                "another expert-memory CPU card is active or pending; one at a time")
    # Queue contents can change between review and submission. Bind the review's
    # finite resource bounds instead of demanding replay of a volatile queue.
    review = authorization.get("preflight_review", {})
    require(review.get("account") == "iscrc_miosr" and review.get("reviewed_epoch", 0) > time.time() - 3600 and
            review.get("reserved_core_hours", -1) >= card_value["resources"]["billing_core_hours_upper"] and
            review.get("storage_bytes_available", -1) >= review.get("storage_bytes_required", 0),
            "recent explicit entitlement/storage/deadline review required")
    journal = Journal(Path(root) / "submissions")
    intent = journal.submission_intent(card_value)
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    r = card_value["resources"]
    seconds = r["seconds"]
    wall = f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"
    script = REPO / "scripts/expert_memory_study" / ("gpu.sbatch" if r["gpus"] else "cpu.sbatch")
    command = ["sbatch", "--parsable", "--no-requeue", f"--account={r['account']}",
               f"--partition={r['partition']}", f"--qos={r['qos']}", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_gib']}G", f"--time={wall}",
               "--job-name=em1-" + card_value["stage"], "--comment=em1-" + intent["uid"],
               f"--chdir={REPO}", f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
    if r["gpus"]:
        command.append(f"--gres=gpu:{r['gpus']}")
    command += [str(script), str(campaign_path), str(card_path), str(authorization_path)]
    raw = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    job = raw.split(";")[0]
    require(job.isdigit(), "ambiguous submission; reconcile immutable intent")
    observed = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True, check=True).stdout
    require("JobId=" + job in observed and "UserId=lmolfett" in observed, "submitted job readback differs")
    return save(Path(root) / "submissions" / (intent["uid"] + ".accepted.json"),
                {"schema": "expert-memory-submission-v1", "intent_sha256": intent["sha256"],
                 "card_sha256": card_value["sha256"], "raw_sbatch": raw, "job_id": job,
                 "readback": observed, "preflight": preflight, "observed_epoch": time.time()})
