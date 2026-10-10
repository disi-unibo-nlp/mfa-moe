"""Finite, separately sealed continuation for the authorized expert-memory study.

This controller never alters the v3 campaign, retries a scientific assignment,
relaxes a qualification, or releases an unpriced final allocation. It resumes
immutable submission intents. Each primary is held until two failure guards
are accepted. There is no login-node process after installation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from moe_exp.expert_memory_study.contracts import (
    REPO, allocated, check_files, digest, file_sha, load, require, save, seal)
from moe_exp.expert_memory_study.protocol import ARMS, MODELS

SCRIPT = Path(__file__).resolve()
SHELL = SCRIPT.with_suffix(".sbatch")
ROOT = REPO / "report/expert-memory-study-v1"
CUTOFF = datetime(2026, 10, 13, 23, 59, 59, tzinfo=timezone.utc).timestamp()
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def command(args, timeout=30):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          timeout=timeout).stdout.strip()


def clean_environment(mapping=None):
    source = os.environ if mapping is None else mapping
    return {k: v for k, v in source.items()
            if not k.startswith(("SLURM_", "SRUN_", "SBATCH_", "PMI_", "PMIX_"))}


@contextmanager
def transport():
    original = subprocess.run

    def run(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and args and args[0] == "sbatch":
            kwargs["env"] = clean_environment(kwargs.get("env"))
        return original(args, *positional, **kwargs)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original


def once(path, value):
    path = Path(path)
    if path.exists():
        existing = load(path)
        require(existing == seal(value), "immutable continuation artifact differs: " + str(path))
        return existing
    return save(path, value)


def live_review(required_hours=0., storage_bytes=0):
    identity, hostname = command(["id", "-un"]), command(["hostname", "-f"])
    require(identity == "lmolfett" and "leonardo" in hostname, "wrong cluster/user")
    balance, quota = command(["saldo", "-b"]), command(["cinQuota"])
    account = next(line.split() for line in balance.splitlines() if line.lower().startswith("iscrc_miosr"))
    remaining = float(account[3]) - float(account[5])
    project = command(["lfs", "project", "-d", "/leonardo_work/IscrC_MIOSR"])
    project_id = project.split()[0]
    require(project_id.isdigit(), "unrecognized WORK project quota identity")
    exact_quota = command(["lfs", "quota", "-p", project_id, "/leonardo_work"])
    fields = next(line.split() for line in exact_quota.splitlines()
                  if line.strip().startswith("/leonardo_work "))
    used, soft, hard = (int(x.rstrip("*")) for x in fields[1:4])
    limits = [v for v in (soft, hard) if v > 0]
    require(limits, "WORK project quota is not measured")
    available = max(0, (min(limits)-used)*1024)
    # Pending reservations do not appear in saldo. Account for every outstanding
    # owned scientific card as well as the protected predecessor campaign.
    queue = command(["squeue", "-u", "lmolfett", "-h", "-o", "%i|%j|%T|%C|%l"])
    require(all(line.split("|")[1].startswith(("cv3", "em1-", "ems1-")) for line in queue.splitlines()),
            "unreviewed workload requires fresh exposure reconciliation")
    pending_hours = 0.
    for line in queue.splitlines():
        fields = line.split("|")
        text = fields[4]
        days, clock = (text.split("-", 1) if "-" in text else ("0", text))
        parts = [int(x) for x in clock.split(":")]
        seconds = int(days) * 86400 + (parts[0] * 3600 + parts[1] * 60 + parts[2]
                                      if len(parts) == 3 else parts[0] * 60 + parts[1])
        pending_hours += int(fields[3]) * seconds / 3600
    remaining = max(0., remaining - pending_hours)
    require(required_hours <= remaining and storage_bytes <= available,
            "finite allocation exceeds observed remaining entitlement/storage")
    return {"account": "iscrc_miosr", "reviewed_epoch": time.time(),
            "reserved_core_hours": required_hours, "storage_bytes_available": available,
            "storage_bytes_required": storage_bytes, "remaining_core_hours": remaining,
            "balance": balance, "quota": quota, "exact_project_quota": exact_quota,
            "project_quota_id": project_id, "queue": queue,
            "deadline_utc": "2026-10-13T23:59:59Z", "hostname": hostname,
            "monthly_quota_is_priority_not_hard_limit": True}


def submit_card(path):
    from moe_exp.expert_memory_study.cards import reconcile_submission, submit
    path, value = Path(path), load(path)
    uid = digest(["submission", value["sha256"]])
    directory = ROOT / "submissions"
    accepted = directory / (uid + ".accepted.json")
    if (directory / (uid + ".intent.json")).exists():
        reconciled = reconcile_submission(path, ROOT)
        require(reconciled["status"] != "UNKNOWN_NO_MATCH", "ambiguous scientific submission; no retry")
        return load(accepted)
    approval = load(ROOT / "WORKLOAD_AUTHORIZATION-r1.json")
    review = live_review(value["resources"]["billing_core_hours_upper"],
                         value["request"].get("maximum_output_storage_bytes", 1024**2))
    authorization_path = ROOT / "cards" / (path.stem + f"-authorization-{time.time_ns()}.json")
    save(authorization_path, {"schema": "expert-memory-card-authorization-v1",
        "approved_card_sha256": value["sha256"], "resources": value["resources"],
        "campaign_sha256": value["campaign_sha256"], "user_quote": approval["user_quote"],
        "workload_authorization_sha256": approval["sha256"], "preflight_review": review})
    with transport():
        return submit(path, authorization_path, ROOT / "CAMPAIGN.json", ROOT)


def make_card(stage, label, *, model=None, price=None, seconds=None, request=None):
    from moe_exp.expert_memory_study.cards import card
    path = ROOT / "cards" / (label + ".json")
    once(path, card(load(ROOT / "CAMPAIGN.json"), stage, model=model, price=price,
                    seconds=seconds, request=request))
    return path


def terminal_jobs(wave):
    observed = []
    for row in wave["jobs"]:
        card = load(row["card_path"])
        require(card["sha256"] == row["card_sha256"], "predecessor scientific card changed")
        uid = digest(["submission", card["sha256"]])
        accepted = load(ROOT / "submissions" / (uid + ".accepted.json"))
        require(accepted["sha256"] == row["accepted_sha256"] and accepted["job_id"] == row["job_id"],
                "predecessor scientific job receipt changed")
        raw = command(["sacct", "-X", "-n", "-P", "-j", row["job_id"], "-o",
                       "JobIDRaw,User,Account,State,ExitCode,ElapsedRaw,AllocTRES"])
        records = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == row["job_id"]]
        require(len(records) == 1, "owned scientific job accounting unresolved")
        r = records[0]
        require(r[1] == "lmolfett" and r[2] == "iscrc_miosr" and
                r[3].split()[0].rstrip("+") in TERMINAL, "owned job is not terminal")
        observed.append({**row, "accounting": raw, "scheduler_success": r[3] == "COMPLETED" and r[4] == "0:0"})
    return observed


def controller_accept(card_path):
    """Exact-comment reconciliation precedes every attempt to submit a controller."""
    value = load(card_path)
    uid = value["sha256"]
    directory = ROOT / "continuation/submissions"
    intent_path, accepted_path = directory / (uid + ".intent.json"), directory / (uid + ".accepted.json")
    if accepted_path.exists():
        return load(accepted_path)
    if intent_path.exists():
        intent = load(intent_path)
        start = datetime.fromtimestamp(intent["created_epoch"] - 60, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        queue = command(["squeue", "-u", "lmolfett", "-h", "-o", "%i|%u|%a|%k"])
        accounting = command(["sacct", "-S", start, "-u", "lmolfett", "-X", "-n", "-P", "-o",
                              "JobIDRaw,User,Account,Comment%80"])
        matches = {line.split("|")[0] for raw in (queue, accounting) for line in raw.splitlines()
                   if line.rstrip("|").split("|")[-1] == "ems1-" + uid and
                   line.split("|")[1:3] == ["lmolfett", "iscrc_miosr"]}
        require(len(matches) == 1, "ambiguous controller intent; never blindly resubmit")
        job = matches.pop()
    else:
        live_review(value["resources"]["billing_core_hours_upper"], 1024**2)
        save(intent_path, {"card_sha256": uid, "created_epoch": time.time(), "comment": "ems1-" + uid})
        r = value["resources"]
        args = ["sbatch", "--parsable", "--hold", "--no-requeue", "--kill-on-invalid-dep=yes", "--account=iscrc_miosr",
                "--partition=lrd_all_serial", "--qos=normal", "--nodes=1", "--ntasks=1",
                "--cpus-per-task=4", "--mem=16G", f"--time=00:{r['seconds']//60:02}:00",
                "--job-name=ems1-handoff", "--comment=ems1-" + uid,
                "--dependency=" + value["dependency"], "--chdir=" + str(REPO),
                "--output=" + str(ROOT / "logs/ems1-handoff-%j.out"),
                "--error=" + str(ROOT / "logs/ems1-handoff-%j.err"), str(SHELL), str(card_path)]
        with transport():
            job = command(args, timeout=60).split(";")[0]
        require(job.isdigit(), "ambiguous controller sbatch response")
    readback = command(["scontrol", "show", "job", job])
    require("JobId=" + job in readback and "UserId=lmolfett" in readback and
            "Comment=ems1-" + uid in readback, "controller ownership readback differs")
    return save(accepted_path, {"card_sha256": uid, "job_id": job, "readback": readback})


def install(phase, wave_path):
    """Prearm two bounded failure guards before releasing a new primary."""
    binding = load(ROOT / "CONTINUATION_READINESS-r1.json")
    check_files(binding["code_files"])
    wave = load(wave_path)
    jobs = [r["job_id"] for r in wave["jobs"]]
    parent_controller = os.environ.get("SLURM_JOB_ID")
    if parent_controller:
        require(parent_controller.isdigit(), "invalid predecessor controller job ID")
        jobs = list(dict.fromkeys(jobs + [parent_controller]))
    require(jobs and all(j.isdigit() for j in jobs), "finite predecessor job IDs required")
    base = {"schema": "expert-memory-continuation-card-v1", "phase": phase,
            "wave_path": str(Path(wave_path).resolve()), "wave_sha256": wave["sha256"],
            "campaign_sha256": load(ROOT / "CAMPAIGN.json")["sha256"],
            "code_files": binding["code_files"], "readiness_sha256": binding["sha256"],
            "resources": {"cpus": 4, "memory_gib": 16, "seconds": 600,
                          "billing_core_hours_upper": 2/3},
            "user_quote": load(ROOT / "WORKLOAD_AUTHORIZATION-r1.json")["user_quote"],
            "paper_artifact": "all-seven expert-memory capability and paired accuracy-memory exhibits",
            "missing_measurement": "reconcile scientific gates; release only finite authorized measured stages",
            "price_basis": "bounded first orchestration/retained CPU grading measurement; ten-minute ceiling",
            "maximum_scientific_retries": 0, "maximum_controller_failure_guards": 2}
    token = digest([phase, wave["sha256"]])
    paths, accepted, ancestors = [], [], []
    for attempt in range(3):
        dependency = ("afterany:" + ":".join(jobs)) if not attempt else ("afternotok:" + ":".join(ancestors))
        path = ROOT / "continuation/cards" / (token + f"-{attempt}.json")
        once(path, {**base, "dependency": dependency, "attempt": attempt,
                    "failure_ancestor_job_ids": list(ancestors)})
        receipt = controller_accept(path)
        paths.append(str(path)); accepted.append(receipt); ancestors.append(receipt["job_id"])
    proof_path = ROOT / "continuation" / (token + ".protected.json")
    proof = once(proof_path, {"schema": "expert-memory-protected-continuation-v1",
        "phase": phase, "wave_sha256": wave["sha256"], "cards": paths,
        "job_ids": ancestors, "accepted_sha256": [r["sha256"] for r in accepted],
        "all_failure_guards_accepted_before_release": True, "billing_core_hours_upper": 2.})
    live_jobs = {line.split("|")[0] for line in command(
        ["squeue", "-u", "lmolfett", "-h", "-o", "%i|%T"]).splitlines()}
    for job in reversed(ancestors):
        if job not in live_jobs:
            continue
        observed = command(["scontrol", "show", "job", job])
        if "JobState=PENDING" in observed and ("Reason=JobHeldUser" in observed or "Reason=JobHeldAdmin" in observed):
            command(["scontrol", "release", job])
    return proof


def wave_dispatch(phase, paths):
    jobs = []
    for path in paths:
        value, accepted = load(path), submit_card(path)
        jobs.append({"job_id": accepted["job_id"], "card_path": str(path),
                     "card_sha256": value["sha256"], "model": value["model"],
                     "accepted_sha256": accepted["sha256"]})
    wave_path = ROOT / "continuation" / (phase + "-" + digest([str(p) for p in paths]) + ".wave.json")
    once(wave_path, {"schema": "expert-memory-stage-wave-v1", "jobs": jobs,
                     "campaign_sha256": load(ROOT / "CAMPAIGN.json")["sha256"]})
    return install(phase, wave_path)


def report(status, evidence):
    path = ROOT / "continuation" / (status + "-" + digest(evidence) + ".json")
    return once(path, {"status": status, "evidence": evidence,
                      "campaign_sha256": load(ROOT / "CAMPAIGN.json")["sha256"],
                      "user_authorization_sha256": load(ROOT / "WORKLOAD_AUTHORIZATION-r1.json")["sha256"]})


def qualification_phase(observed):
    campaign, freeze = load(ROOT / "CAMPAIGN.json"), load(ROOT / "FROZEN_MASKS.json")
    failures = []
    for model in MODELS:
        path = ROOT / model / "qualification/QUALIFICATION.json"
        if not path.exists():
            failures.append({"model": model, "status": "MISSING_QUALIFICATION_RECEIPT"})
            continue
        q = load(path)
        require(q["campaign_sha256"] == campaign["sha256"] and
                q["selection_sha256"] == freeze["models"][model]["sha256"], "foreign qualification")
        if q["status"] != "PASS":
            failures.append({"model": model, "status": q["status"], "failures": q["failures"]})
    if failures or not all(r["scheduler_success"] for r in observed):
        return report("QUALIFICATION_LIMIT", {"models": failures, "jobs": observed,
                      "accuracy_unmeasured": True, "no_model_scope_reduction": True})
    require(time.time() + 7200 < CUTOFF, "insufficient time for retained pricing before cutoff")
    from transformers import AutoTokenizer
    from moe_exp.verification_study.readers import normalize_prompt_ids
    roster, population = load(ROOT / "PILOT_ASSIGNMENTS.json"), load(ROOT / "POPULATION.json")
    catalog = {r["question_id"]: r["problem"] for r in population["pilot"]}
    paths, lengths, retained_uids = [], {}, {}
    for model in campaign["engineering_order"]:
        contract = campaign["models"][model]
        tokenizer = AutoTokenizer.from_pretrained(contract["local_path"], local_files_only=True,
                                                  trust_remote_code=False)
        kwargs = dict(contract["template_kwargs"])
        if contract["reasoning_effort"] is not None:
            kwargs["reasoning_effort"] = contract["reasoning_effort"]
        model_lengths = {q: len(normalize_prompt_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=True, add_generation_prompt=True, **kwargs)))
                         for q, problem in catalog.items()}
        lengths[model] = model_lengths
        retained_uids[model] = {}
        for arm in ARMS:
            records = [r for r in roster["records"] if r["model"] == model and r["arm"] == arm]
            if records[0]["alias_of_arm"] != arm:
                source_uid = retained_uids[model][records[0]["alias_of_arm"]]
                source = next(r for r in roster["records"] if r["uid"] == source_uid)
                retained_uids[model][arm] = next(r["uid"] for r in records if r["family"] == source["family"])
                continue
            row = min(records, key=lambda r: (-model_lengths[r["question_id"]], r["uid"]))
            retained_uids[model][arm] = row["uid"]
            request = generation_request(model, arm, [row["uid"]], "pilot")
            paths.append(make_card("pricing", "pricing-" + model + "-" + arm,
                                   model=model, request=request))
    once(ROOT / "PRICING_INPUTS-r1.json", {"campaign_sha256": campaign["sha256"],
         "assignments_sha256": roster["sha256"], "prompt_lengths": lengths, "retained_uids": retained_uids,
         "selection_rule": "original largest native-templated pilot input per model/arm, UID tie break"})
    live_review(len(paths) * 32., len(paths) * 256 * 1024**2)
    return wave_dispatch("pricing", paths)


def generation_request(model, arm, uids, stage, price_path=None):
    result = {"arm": arm, "frozen_masks": str(ROOT / "FROZEN_MASKS.json"),
              "qualification": str(ROOT / model / "qualification/QUALIFICATION.json"),
              "assignments": str(ROOT / ("PILOT_ASSIGNMENTS.json" if stage == "pilot" else "FINAL_ASSIGNMENTS.json")),
              "population": str(ROOT / "POPULATION.json"), "root": str(ROOT),
              "assignment_uids": uids, "maximum_output_storage_bytes": len(uids)*1024**2 + 256*1024**2}
    if price_path:
        result["price_path"] = str(price_path)
    return result


def pricing_phase(observed):
    from moe_exp.expert_memory_study.cli import main as cli
    from moe_exp.expert_memory_study.stages import resolve_result
    from moe_exp.expert_memory_study.pricing import shard_price
    campaign, roster = load(ROOT / "CAMPAIGN.json"), load(ROOT / "PILOT_ASSIGNMENTS.json")
    retained = load(ROOT / "PRICING_INPUTS-r1.json")["retained_uids"]
    require(all(r["scheduler_success"] for r in observed), "retained pricing accounting failed; no redraw")
    price_paths, paths = {}, []
    for model in MODELS:
        price_paths[model] = {}
        for arm in ARMS:
            rows = [r for r in roster["records"] if r["model"] == model and r["arm"] == arm]
            row = next(r for r in rows if r["uid"] == retained[model][arm])
            require((ROOT / "results" / (row["uid"] + ".result.json")).exists(),
                    "original retained role price is missing")
            receipt_path = ROOT / "results" / (row["uid"] + ".result.json")
            original = resolve_result(ROOT, row, roster)
            path = ROOT / "prices" / (model + "-" + arm + ".json")
            if not path.exists():
                args = ["price", "--campaign", str(ROOT / "CAMPAIGN.json"), "--result", str(receipt_path),
                        "--closeout", str(ROOT / "closeout" / (original["assignment"]["uid"] + ".json")),
                        "--output", str(path)]
                if row["alias_of_arm"] != arm:
                    args += ["--source-result", str(ROOT / "results" / (original["assignment"]["uid"] + ".result.json"))]
                cli(args)
            price_paths[model][arm] = str(path)
            if row["alias_of_arm"] != arm:
                continue
            price = load(path)
            pending = [r["uid"] for r in rows if r["uid"] != retained[model][arm]]
            fit = math.floor((28500/1.25-price["load_seconds"])/price["worst_cap_assignment_seconds"])
            require(fit > 0, "one measured cap-bound assignment exceeds eight hours")
            for offset in range(0, len(pending), fit):
                chosen = pending[offset:offset+fit]
                priced = shard_price(price, len(chosen))
                # The sealed launcher keeps five minutes for closeout.
                seconds = priced["reservation_seconds"] + 300
                require(seconds <= 28800, "launcher closeout reserve exceeds finite shard")
                paths.append(make_card("pilot", f"pilot-{model}-{arm}-{offset}", model=model,
                    price=price, seconds=seconds,
                    request=generation_request(model, arm, chosen, "pilot", path)))
    once(ROOT / "ROLE_PRICES.json", {"campaign_sha256": campaign["sha256"], "models": price_paths})
    # CPU pricing uses seven already generated native assignments, retained in
    # the pilot. No inference, outcome ranking, new judge or discarded draw.
    grading_price()
    live_review(sum(load(p)["resources"]["billing_core_hours_upper"] for p in paths),
                sum(load(p)["request"]["maximum_output_storage_bytes"] for p in paths))
    require(time.time() + max(load(p)["resources"]["seconds"] for p in paths) < CUTOFF,
            "remaining pilot cannot fit computation cutoff")
    return wave_dispatch("pilot", paths)


def grading_price():
    from moe_exp.expert_memory_study.grading import score
    from moe_exp.expert_memory_study.stages import resolve_result
    path = ROOT / "CPU_GRADING_PRICE.json"
    if path.exists():
        return load(path)
    campaign, roster = load(ROOT / "CAMPAIGN.json"), load(ROOT / "PILOT_ASSIGNMENTS.json")
    source = load(REPO / "report/routing-study-v1/FULL_MATH_QUESTION_TABLE.json")
    questions = {r["id"]: r for r in source["records"]}
    samples = []
    for model in MODELS:
        row = next(r for r in roster["records"] if r["model"] == model and r["arm"] == "native" and
                   (ROOT / "results" / (r["uid"] + ".result.json")).exists())
        receipt = resolve_result(ROOT, row, roster)
        started = time.monotonic()
        value = score(receipt, questions[row["question_id"]]["solution"])
        samples.append({"assignment": row, "seconds": time.monotonic()-started,
                        "result_sha256": receipt["sha256"], "grade": value})
    return save(path, {"schema": "expert-memory-cpu-grading-price-v1", "samples": samples,
                      "campaign_sha256": campaign["sha256"], "retained_original_pilot_items": True,
                      "seconds_per_assignment_upper": max(30., max(r["seconds"] for r in samples)),
                      "basis": "observed seven retained grades plus conservative frozen scorer timeout bound"})


def grade_next(stage):
    from moe_exp.expert_memory_study.cli import main as cli
    roster = load(ROOT / ("PILOT_ASSIGNMENTS.json" if stage == "pilot" else "FINAL_ASSIGNMENTS.json"))
    price = load(ROOT / "CPU_GRADING_PRICE.json")
    rows, grades = roster["records"], []
    for offset in range(0, len(rows), 128):
        target = ROOT / "grades" / f"{stage}-{offset}.json"
        subset_path = ROOT / "grades" / f"{stage}-{offset}-assignments.json"
        once(subset_path, {**roster, "records": rows[offset:offset+128],
                          "parent_assignments_sha256": roster["sha256"]})
        if not target.exists():
            seconds = math.ceil(1.25 * len(rows[offset:offset+128]) * price["seconds_per_assignment_upper"] + 120)
            require(seconds <= 14400, "graded shard exceeds serial limit")
            request = {"assignments": str(subset_path), "root": str(ROOT),
                       "questions": str(REPO / "report/routing-study-v1/FULL_MATH_QUESTION_TABLE.json"),
                       "output": str(target), "maximum_output_storage_bytes": 4*1024**2}
            path = make_card("grade", f"grade-{stage}-{offset}", price=price,
                             seconds=seconds, request=request)
            return wave_dispatch("grade-" + stage, [path])
        artifact = load(target)
        require(artifact["assignments_sha256"] == load(subset_path)["sha256"], "foreign grade shard")
        grades.extend(artifact["records"])
    from moe_exp.expert_memory_study.stages import score_grid
    aggregate = seal({"schema": "expert-memory-grades-v1", "records": grades,
                      "assignments_sha256": roster["sha256"], "campaign_sha256": roster["campaign_sha256"],
                      "source_files": {str(p): file_sha(p) for p in (ROOT / "grades").glob(stage+"-*.json")}})
    score_grid(aggregate, roster)
    once(ROOT / ("PILOT_GRADES.json" if stage == "pilot" else "FINAL_GRADES.json"), aggregate)
    if stage == "final":
        request = {"assignments": str(ROOT / "FINAL_ASSIGNMENTS.json"), "grades": str(ROOT / "FINAL_GRADES.json"),
                   "output": str(ROOT / "ANALYSIS.json"), "maximum_output_storage_bytes": 1024**2}
        path = make_card("analyze", "final-analysis", price=load(ROOT / "CPU_PRICE.json"),
                         seconds=1200, request=request)
        return wave_dispatch("analysis", [path])
    size_path = ROOT / "SAMPLE_SIZE.json"
    if not size_path.exists():
        cli(["size-final", "--campaign", str(ROOT / "CAMPAIGN.json"), "--grades", str(ROOT / "PILOT_GRADES.json"),
             "--assignments", str(ROOT / "PILOT_ASSIGNMENTS.json"), "--output", str(size_path)])
    if not (ROOT / "FINAL_ASSIGNMENTS.json").exists():
        cli(["freeze-assignments", "--population", str(ROOT / "POPULATION.json"), "--frozen-masks", str(ROOT / "FROZEN_MASKS.json"),
             "--stage", "final", "--sample-size", str(size_path), "--output", str(ROOT / "FINAL_ASSIGNMENTS.json")])
    review = live_review()
    observation_path = ROOT / "continuation" / ("availability-" + str(time.time_ns()) + ".json")
    save(observation_path, review)
    availability_path = observation_path.with_name(observation_path.stem + "-bound.json")
    # One original GPU job is the guaranteed conservative planning concurrency;
    # greater effective capacity needs an actual complete scheduler assessment.
    save(availability_path, {"complete": True, "reviewed_epoch": review["reviewed_epoch"],
         "remaining_core_hours": review["remaining_core_hours"],
         "available_storage_bytes": review["storage_bytes_available"],
         "remaining_seconds": max(0, CUTOFF-time.time()), "parallel_gpu_jobs": 1,
         "source_files": {str(observation_path): file_sha(observation_path)},
         "capacity_scope": "conservative single-job screen; deadline failure is conditional on this concurrency"})
    plan_path = ROOT / "FINAL_PLAN.json"
    if not plan_path.exists():
        cli(["plan-final", "--campaign", str(ROOT / "CAMPAIGN.json"), "--assignments", str(ROOT / "FINAL_ASSIGNMENTS.json"),
             "--frozen-masks", str(ROOT / "FROZEN_MASKS.json"), "--prices", str(ROOT / "ROLE_PRICES.json"),
             "--availability", str(availability_path), "--root", str(ROOT), "--output", str(plan_path)])
    plan = load(plan_path)
    return report("FINAL_ALLOCATION_ASSESSMENT_REQUIRED", {"plan_sha256": plan["sha256"], "N": plan["N"],
        "reasons": plan["reasons"], "single_job_deadline_screen_is_not_precision_infeasibility": True,
        "hard_infeasibility": [r for r in plan["reasons"] if r != "DEADLINE_INFEASIBLE"],
        "authorization_persists": True, "remaining_gate": "complete measured scheduler/deadline allocation assessment",
        "no_unpriced_final_submission": True, "no_sample_shrink": True})


def run(card_path):
    allocated()
    value = load(card_path)
    check_files(value["code_files"])
    campaign = load(ROOT / "CAMPAIGN.json")
    check_files(campaign["code_files"])
    check_files(campaign["scientific_files_preserved"])
    check_files(campaign["readiness_files_preserved"])
    require(value["campaign_sha256"] == campaign["sha256"] and value["user_quote"] ==
            load(ROOT / "WORKLOAD_AUTHORIZATION-r1.json")["user_quote"], "foreign continuation authorization")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "4" and
            os.environ.get("SLURM_MEM_PER_NODE") == str(16*1024), "wrong controller allocation")
    wave = load(value["wave_path"])
    require(wave["sha256"] == value["wave_sha256"], "continuation predecessor changed")
    with (ROOT / "continuation/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = terminal_jobs(wave)
        phase = value["phase"]
        if phase in ("qualification", "pricing", "pilot") and time.time() >= CUTOFF:
            result = report("COMPUTATION_CUTOFF", {"phase": phase, "jobs": observed})
        elif phase == "qualification":
            result = qualification_phase(observed)
        elif phase == "pricing":
            result = pricing_phase(observed)
        elif phase in ("pilot", "grade-pilot", "grade-final"):
            require(all(r["scheduler_success"] for r in observed), "scientific job failed; preserve receipts and reconcile")
            result = grade_next("final" if phase == "grade-final" else "pilot")
        elif phase == "analysis":
            require(all(r["scheduler_success"] for r in observed) and (ROOT / "ANALYSIS.json").exists(),
                    "final analysis is unobserved")
            result = report("SCIENTIFIC_CLOSEOUT", {"analysis_sha256": load(ROOT / "ANALYSIS.json")["sha256"]})
        else:
            raise ValueError("unknown bounded continuation phase")
        from moe_exp.expert_memory_study.paper import export
        output = ROOT / "continuation" / ("exhibits-" + os.environ["SLURM_JOB_ID"])
        if not output.exists():
            export(campaign, ROOT, output)
        once(ROOT / "continuation" / ("closeout-" + os.environ["SLURM_JOB_ID"] + ".json"),
             {"controller_card_sha256": value["sha256"], "result_sha256": result["sha256"],
              "result_status": result.get("status", "PROTECTED_SUCCESSOR"), "phase": phase,
              "jobs": observed, "preserved_verification_ownership": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    install_parser = sub.add_parser("install")
    install_parser.add_argument("--phase", required=True)
    install_parser.add_argument("--wave", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--card", required=True)
    args = parser.parse_args()
    try:
        result = install(args.phase, args.wave) if args.command == "install" else run(args.card)
    except Exception as error:
        if args.command == "run":
            report("CONTINUATION_FAILURE", {"job_id": os.environ.get("SLURM_JOB_ID"),
                   "card_path": args.card, "type": type(error).__name__, "message": str(error)[:2000],
                   "no_scientific_retry": True})
        raise
    print(json.dumps({"sha256": result["sha256"], "status": result.get("status", "PROTECTED_SUCCESSOR"),
                      "jobs": result.get("job_ids", [])}))
