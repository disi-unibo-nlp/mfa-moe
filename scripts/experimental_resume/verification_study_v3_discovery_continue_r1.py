"""One bounded continuation of the seven frozen discovery manifests.

Scientific pilots, archive hashes and completion events qualify prices. This
orchestrator runs no inference and dispatches no later scientific stage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import time

from moe_exp.verification_study import campaign as native
from moe_exp.verification_study.contracts import (
    check_files, file_sha, load, private_path, require, require_generation_binding, save,
)
from moe_exp.verification_study.pricing import shard_budget
from moe_exp.verification_study.registry import MODEL_KEYS, REPO
import verification_study_v3_health_r2 as health


def work(count, pilot_tokens):
    require(type(count) is int and count > 0 and pilot_tokens > 0, "invalid discovery slice")
    scale = max(1, 16384 / pilot_tokens)
    # Reserve one complete capped retry, a second cold load and capture scaling.
    return {"generator_load": 2, "prefill": count + 1,
            "generation": 16384 * (count + 1),
            "capture": scale * (count + 1), "closeout": scale * (count + 1)}


def sized(prices, remaining, pilot_tokens):
    best = None
    for count in range(1, remaining + 1):
        try:
            budget = shard_budget(prices, work(count, pilot_tokens), hours=24)
        except ValueError:
            break
        best = count, budget
    require(best is not None, "one capped discovery assignment and recovery exceed partition walltime")
    count, budget = best
    seconds = 60 * math.ceil(budget["reservation_seconds"] / 60)
    return count, seconds, shard_budget(prices, work(count, pilot_tokens), hours=seconds / 3600)


@contextmanager
def lock(root):
    path = root / "discovery-orchestration.lock"
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def receipt(path, assignment, campaign, model):
    value = load(path, "verification-generation-receipt-v3")
    require_generation_binding(value, campaign, model)
    require(value["assignment"] == assignment and value["assignment_uid"] == assignment["uid"] and
            value["engineering_only"] is False and value["execution_complete"] is True,
            "incomplete or foreign discovery generation")
    archive = private_path(value["capture"]["array_path"])
    require(archive == path.resolve().with_suffix(".npz") and archive.is_file() and
            archive.stat().st_size == value["capture"]["array_bytes"] and
            file_sha(archive) == value["capture"]["array_sha256"], "discovery capture changed")
    require(len(value["capture"]["geometry"]) == model["expected_routed_layers"] and
            all(g["native_k"] == model["native_k"] for g in value["capture"]["geometry"]),
            "discovery capture geometry differs")
    return value


def exact_storage(repo):
    project = subprocess.run(["lfs", "project", "-d", str(repo)], check=True,
                             capture_output=True, text=True, timeout=20).stdout
    pid = project.split()[0]
    require(pid == "20149688", "unexpected WORK project quota identity")
    result = subprocess.run(["lfs", "quota", "-p", pid, str(repo)], check=True,
                            capture_output=True, text=True, timeout=20).stdout
    rows = [line.split() for line in result.splitlines() if re.match(r"^\s*\d+\s+\d+", line)]
    require(len(rows) == 1, "exact project storage quota unavailable")
    used, quota = (int(x.rstrip("*")) for x in rows[0][:2])
    require(quota > used, "WORK project quota exhausted")
    return {"project_id": pid, "used_kib": used, "quota_kib": quota,
            "available_bytes": (quota - used) * 1024, "raw": result}


def prepare(path):
    campaign = load(path); check_files(campaign["code_files"])
    require(set(campaign["models"]) == set(MODEL_KEYS), "all seven models required")
    root = private_path(campaign["result_root"])
    pilots = load(path.parent / "DISCOVERY_PRICE_JOBS-r4.json")
    require(pilots["campaign_sha256"] == campaign["sha256"] and
            {r["model"] for r in pilots["jobs"]} == set(MODEL_KEYS), "foreign pilot roster")
    # Resume this exact finite wave rather than create competing requests.
    wave_path = path.parent / "DISCOVERY_CONTINUATION_PLAN-r1.json"
    if wave_path.exists():
        wave = load(wave_path)
        require(wave["campaign_sha256"] == campaign["sha256"] and
                wave["orchestration_file_sha256"] == file_sha(__file__), "continuation source changed")
        return campaign, root, wave
    # A production intent from another orchestrator must be reconciled first.
    pilot_ids = {r["job_card_sha256"] for r in pilots["jobs"]}
    for p in (root / "submissions").glob("*/JOB_CARD.json"):
        c = load(p)
        if c.get("stage") != "run" or c["sha256"] in pilot_ids:
            continue
        if (p.parent / "INTENT.json").exists() or (p.parent / "SUBMISSION.json").exists():
            args = health.arguments(c)
            if "--manifest" in args and load(args["--manifest"])["stage"] == "discovery":
                raise ValueError("existing discovery production intent requires reconciliation: " + str(p))
    entries = []; closeout = []; projected_bytes = 0
    for pilot in pilots["jobs"]:
        key = pilot["model"]; model = load(campaign["models"][key])
        manifest_path = path.parent / key / "DISCOVERY_ASSIGNMENTS.json"
        manifest = load(manifest_path, "verification-assignments-v3")
        require(manifest["assigned"] == 48 and len(manifest["assignments"]) == 48 and
                manifest["model_sha256"] == model["sha256"] and manifest["stage"] == "discovery",
                "discovery manifest differs")
        qualification_path = root / key / "qualification/QUALIFICATION.json"
        qualified = load(qualification_path)
        require(qualified["status"] == "PASS" and qualified["campaign_sha256"] == campaign["sha256"] and
                qualified["model_sha256"] == model["sha256"] and qualified["code_files"] == campaign["code_files"],
                "backend not qualified for this exact campaign")
        submission = load(pilot["receipt_path"])
        require(submission["job_id"] == pilot["job_id"] and
                submission["job_card_sha256"] == pilot["job_card_sha256"], "foreign pilot submission")
        matching = []
        for p in (root / "health/events").glob("*.json"):
            event = load(p)
            if event["predecessor_job_id"] == pilot["job_id"] and event["job_card_sha256"] == pilot["job_card_sha256"]:
                matching.append((p, event))
        require(len(matching) == 1, "pilot completion event missing or ambiguous")
        event_path, event = matching[0]
        hslot = root / "health/submissions" / event["health_card_sha256"]
        hcard = load(hslot / "HEALTH_CARD.json"); check_files(hcard["observer_files"])
        hsub = load(hslot / "SUBMISSION.json")
        require(event["campaign_sha256"] == campaign["sha256"] and event["model"] == key and
                event["state"] == "STEP_COMPLETE" and event["accounting"]["state"] == "COMPLETED" and
                event["accounting"]["exit_code"] == "0:0" and
                hsub["job_id"] == event["callback_job_id"] and hcard["sha256"] == event["health_card_sha256"] and
                hcard["job_card_sha256"] == pilot["job_card_sha256"], "unqualified pilot health evidence")
        first = manifest["assignments"][0]
        generation_path = root / key / "discovery" / (first["uid"] + ".json")
        generated = receipt(generation_path, first, campaign, model)
        require(generated["job_id"] == pilot["job_id"] and event["committed_artifacts"] ==
                [{"path": str(generation_path), "schema": generated["schema"],
                  "sha256": generated["sha256"], "status": None}], "pilot receipt not joined to health event")
        prices = load(event["price_path"], "verification-prices-v3")
        require(prices["sha256"] == event["price_sha256"] and prices["model_sha256"] == model["sha256"] and
                prices["code_files"] == campaign["code_files"], "pilot price binding differs")
        for role in ("generator_load", "prefill", "generation", "capture", "closeout", "recovery"):
            require(prices["components"][role]["status"] == "MEASURED" and prices["components"][role]["samples"],
                    "unmeasured discovery role: " + role)
        require(prices["components"]["generation"]["max_total_context_tokens"] >= 32768,
                "full native context price coverage missing")
        for a in manifest["assignments"][1:]:
            out = root / key / "discovery" / (a["uid"] + ".json")
            require(not out.exists() and not list(out.parent.glob(a["uid"] + ".attempt-*.json")),
                    "remaining discovery identity already attempted; reconcile before allocating")
        count_left = 47; begin = 1; model_entries = []
        while count_left:
            count, seconds, budget = sized(prices, count_left, generated["generated_tokens"])
            selected = manifest["assignments"][begin:begin + count]
            # Uncompressed native arrays plus generous JSON/header allowance,
            # one temporary archive and one orphan recovery capture per shard.
            per_assignment = (12 * model["native_k"] + 5) * model["expected_routed_layers"] * 16384 + 4 * 1024**2
            storage_bytes = (count + 2) * per_assignment
            request = save(path.parent / f"job-requests-r4/{key}/DISCOVERY_{begin:03d}_{count:03d}_{prices['sha256'][:12]}.json", {
                "schema": "verification-task-request-v3", "stage": "run", "model": key, "population": "discovery",
                "arguments": ["--model", key, "--qualification", str(qualification_path), "--manifest", str(manifest_path),
                              "--output", str(root / key / "discovery"), "--root", str(root),
                              "--begin", str(begin), "--count", str(count), "--prices", event["price_path"]],
                "claim": "C2: qualify substantive-verification support and nominate a frozen singleton for this required model",
                "artifact": f"{key}: complete original48-family native discovery receipts and all-layer capture for audit and nomination",
                "reuse": "retained first discovery assignment, frozen original48 manifest and source-qualified native executor",
                "missing_cell": f"{key} remaining discovery family support; no audit freeze or target qualification from the single pilot",
                "decision_if_failure": "retain all committed assignments; reconcile exact failed identity and bounded attempts; report unsupported or imprecise cells, never a causal null",
                "measurement_pilot": False, "finite_assignments": count, "selected_uids": [a["uid"] for a in selected],
                "price_path": event["price_path"], "price_event_sha256": event["sha256"],
                "priced_work": work(count, generated["generated_tokens"]), "storage_bound_bytes": storage_bytes,
                "orchestration_file_sha256": file_sha(__file__), "prerequisites": [str(qualification_path), str(event_path)],
                "resources": {"partition": "boost_usr_prod", "qos": "normal", "gpus": 2, "cpus": 16, "memory_gb": 120,
                              "walltime_seconds": seconds, "billing_core_hours_max": 16 * seconds / 3600,
                              "price_basis": "actual source-compatible role maxima; every assignment cap16384, one full capped retry and second load, cap-scaled capture/closeout,1.25 factor plus600seconds"}})
            request_path = path.parent / f"job-requests-r4/{key}/DISCOVERY_{begin:03d}_{count:03d}_{prices['sha256'][:12]}.json"
            card = native.task_card(path, request)
            entry = {"model": key, "begin": begin, "count": count, "request_path": str(request_path),
                     "job_card_sha256": card["sha256"], "budget": budget, "storage_bound_bytes": storage_bytes}
            entries.append(entry); model_entries.append(entry); projected_bytes += storage_bytes
            begin += count; count_left -= count
        closeout.append({"model": key, "pilot_job_id": pilot["job_id"], "pilot_callback_job_id": event["callback_job_id"],
                         "generation_sha256": generated["sha256"], "generated_tokens": generated["generated_tokens"],
                         "capture_bytes": generated["capture"]["array_bytes"], "event_path": str(event_path),
                         "event_sha256": event["sha256"], "price_path": event["price_path"], "price_sha256": prices["sha256"],
                         "actual_billing_core_hours": 16 * event["accounting"]["elapsed_seconds"] / 3600,
                         "remaining": 47, "production_slices": len(model_entries)})
    storage = exact_storage(REPO)
    require(projected_bytes < storage["available_bytes"], "bounded discovery outputs exceed live WORK capacity")
    total = sum(e["budget"]["billing_core_hours"] for e in entries)
    snapshot = native.preflight()
    row = next((line.split() for line in snapshot["balance"].splitlines() if line.startswith("IscrC_MIOSR")), None)
    require(row and total < int(row[3]) - int(row[5]), "discovery wave exceeds observed remaining allocation")
    wave = save(wave_path, {"schema": "verification-discovery-continuation-plan-v3", "campaign_sha256": campaign["sha256"],
                           "orchestration_file_sha256": file_sha(__file__), "pilot_closeout": closeout, "cards": entries,
                           "all_seven_remaining_assignments": 329, "storage": storage, "storage_bound_bytes": projected_bytes,
                           "billing_core_hours_max": total, "preflight": snapshot,
                           "monthly_quota": "monthly exhaustion reduces priority; total observed allocation still bounds dispatch",
                           "scientific_effect": "NOT_MEASURED"})
    return campaign, root, wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(); path = args.campaign.resolve(); root = private_path(load(path)["result_root"])
    with lock(root):
        campaign, root, wave = prepare(path)
        storage = exact_storage(REPO)
        require(wave["storage_bound_bytes"] < storage["available_bytes"], "rechecked WORK capacity insufficient")
        rows = []
        for entry in wave["cards"]:
            request = load(entry["request_path"]); card = native.task_card(path, request)
            require(card["sha256"] == entry["job_card_sha256"], "finite discovery card changed")
            row = dict(entry)
            if args.submit:
                native.submit(path, card, root, test_only=True)
                submitted = native.submit(path, card, root)
                slot = root / "submissions" / card["sha256"] / "SUBMISSION.json"
                health.register(path, slot, root, test_only=True)
                callback = health.register(path, slot, root)
                row.update(job_id=submitted["job_id"], callback_job_id=callback["job_id"])
                print(json.dumps({k: row[k] for k in ("model", "begin", "count", "job_id", "callback_job_id")}), flush=True)
            rows.append(row)
        result = save(path.parent / "DISCOVERY_CONTINUATION_JOBS-r1.json", {
            "schema": "verification-discovery-continuation-jobs-v3", "campaign_sha256": campaign["sha256"],
            "plan_sha256": wave["sha256"], "orchestration_file_sha256": file_sha(__file__), "observed_epoch": time.time(),
            "status": "SUBMITTED_FINITE_ALL_SEVEN_DISCOVERY" if args.submit else "PRICED_FINITE_CARDS_READY",
            "cards": rows, "assignments": sum(r["count"] for r in rows), "billing_core_hours_max": wave["billing_core_hours_max"],
            "scientific_success": "NOT_INFERRED_FROM_SUBMISSION"}, replace=True)
        print(json.dumps({k: result[k] for k in ("status", "assignments", "billing_core_hours_max", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
