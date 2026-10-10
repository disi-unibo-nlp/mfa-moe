"""Advance the approved frozen all-seven study from one allocated CPU step.

The durable caller owns the global lock and its finite Slurm continuation.
This companion owns no timer, callback, daemon or scientific implementation.
Every GPU request is immutable, priced or a retained first measurement, and
submitted through an existing source-bound executor. Failed scientific gates
remain model-specific limits; incomplete artifacts cannot become a null effect.
"""
from __future__ import annotations

import json
import math
import re
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from moe_exp.verification_study import campaign as native
from moe_exp.verification_study.contracts import (
    check_files, digest, file_sha, load, private_path, require,
    require_generation_binding, save,
)
from moe_exp.verification_study.pricing import shard_budget
from moe_exp.verification_study.registry import MODEL_KEYS, REPO
import verification_study_v3_reader_r1 as reader
import verification_study_v3_j1_r1 as j1
import verification_study_v3_extension_r1 as extension
import verification_study_v3_health_r2 as health
import verification_study_v3_transport_r3 as transport

SCRIPT = Path(__file__).resolve()


def reconciled_unstarted_readers(campaign_path, root, jobs, observations):
    """Only a source-bound, genuinely unstarted transport cancellation qualifies."""
    path = Path(campaign_path).parent / "READER_TRANSPORT_RECONCILIATION-r4.json"
    if not path.exists():
        return set()
    campaign = load(campaign_path)
    proof = load(path, "verification-reader-transport-reconciliation-v3")
    require(proof["campaign_sha256"] == campaign["sha256"] and
            proof["scientific_code_files"] == campaign["code_files"] and
            proof["replacement_transport_executor"] == transport.binding(campaign, "reader") and
            proof["replacement_allocations_per_uid_max"] == 1,
            "reader reconciliation campaign or transport differs")
    by_job = {str(j["submission"]["job_id"]): j for j in jobs}
    result = set()
    for row in proof["cancelled_unstarted_readers"]:
        job_id = row["job_id"]; require(job_id in by_job and job_id not in result, "foreign/duplicate reader reconciliation")
        job = by_job[job_id]; card = job["card"]; args = arguments(card)
        state = observations.get(job_id)
        require(terminal(state) and state["state"] == "CANCELLED" and state["elapsed_seconds"] == 0 and
                row["accounting"]["start"] == "Unknown" and row["accounting"]["allocated_cpus"] == 0,
                "reader cancellation was not genuinely unstarted")
        require(row["job_card_sha256"] == card["sha256"] and
                row["submission_sha256"] == job["submission"]["sha256"] and card["stage"] == "rate" and
                row["full_frame_sha256"] == card["full_frame_sha256"] and row["selected_uids"] == card["selected_uids"] and
                row["role"] == args["--role"] and row["output"] == args["--output"],
                "reader cancellation identity or role changed")
        require(load(args["--frame"])["sha256"] == card["full_frame_sha256"], "reader frame changed")
        for source, sha in row["source_receipts"].items():
            require(load(source)["sha256"] == sha, "reader cancellation source receipt changed")
        output = Path(args["--output"])
        require(not list((output / "invocations").glob(job_id + "-*")) and
                not list((Path(root) / "transport/invocations").glob(job_id + "-*")),
                "unstarted reader actually entered a worker")
        for rating in output.glob("*.reader-*.json"):
            require(str(load(rating).get("job_id")) != job_id, "unstarted reader actually produced a draw")
        result.add(job_id)
    return result

TERMINAL = health.TERMINAL
SUCCESS = {"COMPLETED"}


def submit_transport(campaign_path, card, *, test_only=False):
    """Submit one separately sealed launch without editing its scientific worker."""
    from moe_exp.verification_study.contracts import verify
    verify(card, "verification-job-card-v3")
    campaign, parent = transport.verify_card(campaign_path, card, card["transport_kind"])
    check_files(campaign["code_files"])
    bound = transport.binding(campaign, card["transport_kind"])
    require(card["transport_executor"] == bound and card["code_files"] == campaign["code_files"] and
            card["campaign_sha256"] == campaign["sha256"], "transport/scientific card binding differs")
    check_files(bound["code_files"])
    root = private_path(campaign["result_root"])
    slot = root / "submissions" / card["sha256"]
    save(slot / "JOB_CARD.json", card)
    receipt_path = slot / "SUBMISSION.json"; comment = "cv3t:" + card["sha256"]
    if receipt_path.exists() and not test_only:
        receipt = load(receipt_path)
        require(receipt["job_card_sha256"] == card["sha256"], "foreign transport submission")
        return receipt
    if (slot / "INTENT.json").exists() and not test_only:
        from verification_study_v3_autodriver_r4 import comment_matches
        matched = comment_matches(comment)
        require(matched["partition"] == "boost_usr_prod", "transport intent partition differs")
        return save(receipt_path, {"schema": "verification-submission-v3",
                    "job_card_sha256": card["sha256"], "job_id": matched["job_id"],
                    "reconciled": True, "matching_accounting": matched,
                    "transport_executor_sha256": bound["sha256"]})
    r = card["resources"]
    require(r["partition"] == "boost_usr_prod" and r["gpus"] == 2 and
            r["cpus"] == 16 and r["memory_gb"] == 120, "transport allocation differs")
    logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    argv = ["sbatch", "--test-only" if test_only else "--parsable", "--account=iscrc_miosr",
            "--partition=boost_usr_prod", "--qos=normal", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=16", "--gres=gpu:2", "--mem=120G",
            "--time=" + str(r["walltime_seconds"] // 60), "--no-requeue",
            "--job-name=cv3t-" + card["stage"] + "-" + card["model"],
            "--comment=" + comment, "--chdir=" + str(REPO),
            "--output=" + str(logs / "%x-%j.out"), "--error=" + str(logs / "%x-%j.err"),
            bound["shell"], str(slot / "JOB_CARD.json")]
    env = transport.clean_submission_environment()
    if test_only:
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30, env=env)
        return {"test_only": True, "scheduler_output": result.stdout + result.stderr,
                "job_card": str(slot / "JOB_CARD.json")}
    snapshot = save(slot / "PREFLIGHT.json", native.preflight())
    intent = save(slot / "INTENT.json", {"schema": "verification-submission-intent-v3",
                  "job_card_sha256": card["sha256"], "preflight_sha256": snapshot["sha256"],
                  "comment": comment, "epoch": time.time(), "transport_executor_sha256": bound["sha256"]})
    response = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30, env=env).stdout.strip()
    job = response.split(";", 1)[0]; require(re.fullmatch(r"\d+", job), "invalid transport job ID")
    observed = native.command(["scontrol", "show", "job", job])
    require("JobId=" + job in observed and "Account=iscrc_miosr" in observed and
            "Partition=boost_usr_prod" in observed, "transport submission readback differs")
    return save(receipt_path, {"schema": "verification-submission-v3", "job_card_sha256": card["sha256"],
                "job_id": job, "sbatch_response": response, "read_back": observed,
                "preflight_sha256": snapshot["sha256"], "intent_sha256": intent["sha256"],
                "transport_executor_sha256": bound["sha256"], "scientific_success": "NOT_YET_MEASURED"})


def validate_j1_reconciliation(campaign_path, root, failed, observations):
    """Only the five explicitly proven pre-initialization failures qualify."""
    campaign = load(campaign_path)
    path = campaign_path.parent / "J1_TRANSPORT_RECONCILIATION-r3.json"
    proof = load(path, "verification-j1-transport-reconciliation-v3")
    require(proof["campaign_sha256"] == campaign["sha256"] and
            proof["scientific_code_files"] == campaign["code_files"] and
            proof["item_draws_in_failed_jobs"] == 0 and proof["recovery_allocations_per_uid_max"] == 1 and
            proof["repair_round"] == 3 and proof["recovery_transport_executor"] == transport.binding(campaign, "extension"),
            "J1 transport reconciliation source or recovery bound differs")
    expected = {row["job_id"]: row for row in proof["failed_jobs"]}
    require(set(expected) == {str(j["submission"]["job_id"]) for j in failed},
            "J1 failure set differs; no unseen failed allocation may be retried")
    selected = set()
    for job in failed:
        job_id = str(job["submission"]["job_id"]); row = expected[job_id]
        state = observations.get(job_id)
        require(terminal(state) and state["state"] == "FAILED" and state["exit_code"] == "1:0" and
                row["job_card_sha256"] == job["card"]["sha256"] and
                row["submission_sha256"] == job["submission"]["sha256"],
                "J1 failed allocation accounting or immutable receipt changed")
        for source, sha in row["source_receipts"].items():
            require(load(source)["sha256"] == sha, "J1 failure proof receipt changed")
        for source, sha in row["source_logs"].items():
            require(file_sha(source) == sha, "J1 failure proof log changed")
        invocation = load(row["invocation_result_path"])
        require(invocation["status"] == "FAIL" and invocation.get("loaded") is False and
                invocation.get("new_verdicts") == 0 and invocation["error"] == row["expected_error"] and
                ((row["failure_kind"] == "site-default-cache" and row["expected_error"] ==
                  {"type": "PermissionError", "message": "[Errno 13] Permission denied: '/scratch_local'"}) or
                 (row["failure_kind"] == "unix-socket-path" and row["expected_error"]["type"] == "ZMQError" and
                  'is longer than 107 characters' in row["expected_error"]["message"])) and
                not (Path(row["invocation_result_path"]).parent / "RUNTIME.json").exists(),
                "reconciled J1 failure was not before reader initialization")
        selected.update(job["card"]["extension"]["selected_uids"])
    require(selected == set(proof["eligible_recovery_uids"]), "J1 recovery roster differs")
    for path in (root / "J1/audit").glob("*.attempt-*.json"):
        attempt = load(path)
        require(str(attempt["job_id"]) not in expected, "failed J1 job actually attempted an audit draw")
    return proof


def arguments(card):
    return health.arguments(card)


def observe_jobs(jobs):
    """Read actual top-level accounting; unavailable observations stay unknown."""
    ids = sorted(set(str(j) for j in jobs), key=int)
    if not ids:
        return {}
    output = subprocess.run(
        ["sacct", "-nP", "-j", ",".join(ids),
         "--format=JobIDRaw,State,ExitCode,ElapsedRaw"],
        check=True, capture_output=True, text=True, timeout=30).stdout
    result = {}
    for line in output.splitlines():
        fields = line.split("|")
        if fields[0] in ids:
            require(fields[0] not in result, "ambiguous scientific accounting")
            result[fields[0]] = {"state": fields[1].split(" ", 1)[0].rstrip("+"),
                                 "exit_code": fields[2], "elapsed_seconds": int(fields[3])}
    output = subprocess.run(["squeue", "-h", "-u", "lmolfett", "-o", "%i|%T"],
                            check=True, capture_output=True, text=True, timeout=20).stdout
    for line in output.splitlines():
        job, state = line.split("|", 1)
        if job in ids:
            result[job] = {"state": state, "exit_code": None, "live": True}
    return result


def terminal(state):
    return bool(state and not state.get("live") and state.get("state") in TERMINAL)


def successful(state):
    return terminal(state) and state["state"] == "COMPLETED" and state.get("exit_code") == "0:0"


def uncovered_blocks(records, completed, covered):
    """Contiguous original-frame slices; never create replacement subframes."""
    uids = [r["uid"] for r in records]
    require(set(completed) | set(covered) <= set(uids), "foreign completed/submitted identity")
    result = []; begin = None
    for index, uid in enumerate(uids):
        available = uid not in completed and uid not in covered
        if available and begin is None:
            begin = index
        if begin is not None and (not available or index == len(uids) - 1):
            stop = index if not available else index + 1
            result.append((begin, stop - begin)); begin = None
    return result


def fit_size(prices, work, remaining):
    require(type(remaining) is int and remaining > 0, "finite nonempty production slice required")
    best = None
    for count in range(1, remaining + 1):
        try:
            budget = shard_budget(prices, work(count), hours=24)
        except ValueError:
            break
        best = count, budget
    require(best is not None, "one full capped assignment plus recovery exceeds 24-hour partition")
    count, budget = best
    seconds = 60 * math.ceil(budget["reservation_seconds"] / 60)
    return count, seconds, shard_budget(prices, work(count), hours=seconds / 3600)


def native_work(count, cap, pilot_tokens, *, controller=False):
    require(count > 0 and cap in (1024, 16384) and pilot_tokens > 0, "invalid native work shape")
    units = count + 1
    work = {"generator_load": 2, "prefill": units, "generation": cap * units,
            "capture": max(1, cap / pilot_tokens) * units,
            "closeout": max(1, cap / pilot_tokens) * units}
    if controller:
        # The frozen price unit is an actual call. At most one call/emitted
        # token occurs; do not scale this per-call unit by request length again.
        work["controller"] = units * cap
    return work


def reader_work(prices, role, count):
    component = prices["components"][role]
    require(component["status"] == "MEASURED" and component["samples"], "reader role has no actual price")
    # Prices retain per-draw units. Account for short draws at the full1024 cap.
    scale = max(max(1, 1024 / max(1, row.get("output_tokens", 0))) for row in component["samples"])
    return {"reader_load": 2, role: 2 * (count + 1) * scale}


def resources(seconds, basis):
    return {"partition": "boost_usr_prod", "qos": "normal", "gpus": 2,
            "cpus": 16, "memory_gb": 120, "walltime_seconds": seconds,
            "billing_core_hours_max": 16 * seconds / 3600, "price_basis": basis}


def probe_native_inputs(campaign_path, key, rows):
    """Call the exact frozen prompt formatter with a tokenizer, without an engine."""
    from transformers import AutoTokenizer
    from moe_exp.verification_study.backend import Backend
    from moe_exp.verification_study.registry import engine_kwargs
    campaign = load(campaign_path); model = load(campaign["models"][key])
    check_files(model["config_files"])
    public = load(campaign_path.parent / "PUBLIC_QUESTIONS.json")
    tokenizer = AutoTokenizer.from_pretrained(model["local_path"], local_files_only=True)
    formatter = SimpleNamespace(contract=model, tokenizer=tokenizer)
    sizes = [len(Backend.prompt_ids(formatter, public["questions"][a["question_id"]]["problem"])) +
             len((a.get("prefix") or {}).get("prefix_token_ids", [])) for a in rows]
    require(all(n + a["cap"] <= engine_kwargs(model)["max_model_len"] for n, a in zip(sizes, rows)),
            "planned native context exceeds pinned model deployment")
    return {"max_input_tokens": max(sizes), "max_total_tokens": max(n + a["cap"] for n, a in zip(sizes, rows)),
            "uids": [a["uid"] for a in rows]}


def storage_bound(card, campaign):
    """Conservative persistent/temporary outputs of this exact finite card."""
    args = arguments(card); model = load(campaign["models"][card["model"]])
    if card["stage"] == "run":
        manifest = load(args["--manifest"]); begin = int(args.get("--begin", 0)); count = int(args["--count"])
        rows = manifest["assignments"][begin:begin + count]
        require(len(rows) == count and rows, "storage card lies outside original roster")
        cap = max(a["cap"] for a in rows)
        layers = model["expected_routed_layers"] if manifest["stage"] == "discovery" else 1
        return (count + 2) * ((12 * model["native_k"] + 5) * layers * cap + 4 * 1024**2)
    if card["stage"] == "calibrate":
        # The frozen bank contains at most16 families/64 predictions, one
        # captured layer, cached expert logits and a single pinned target/pool.
        return 18 * ((12 * model["native_k"] + 5) * 64 + 16 * model["experts"] * 64 + 4 * 1024**2)
    count = len(card.get("selected_uids", card.get("extension", {}).get("selected_uids", [])))
    if card["stage"] in {"rate", "j1-audit", "grade-j1"}:
        require(count > 0, "reader/grader storage bound needs the exact selected identities")
        return (count + 2) * 3 * 4 * 1024**2
    return 0


class Advance:
    def __init__(self, campaign_path, root, *, submit, observations):
        self.path = Path(campaign_path).resolve(); self.campaign = load(self.path)
        check_files(self.campaign["code_files"])
        require(set(self.campaign["models"]) == set(MODEL_KEYS), "all seven model cells are required")
        self.root = private_path(root)
        require(self.root == Path(self.campaign["result_root"]), "foreign autonomous result root")
        self.submit = submit; self.actions = []; self.errors = []; self.limits = []
        self.active_stage = {}
        self.jobs = []; self.intents = []
        self.home = self.root / "autostages-r4"
        for path in (self.root / "submissions").glob("*/JOB_CARD.json"):
            card = load(path)
            require(card["campaign_sha256"] == self.campaign["sha256"], "foreign scientific campaign card")
            if card.get("stage") == "qualify" or card.get("resources", {}).get("gpus", 0) == 0:
                continue
            sub = path.parent / "SUBMISSION.json"
            if sub.exists():
                receipt = load(sub, "verification-submission-v3")
                require(receipt["job_card_sha256"] == card["sha256"], "foreign scientific submission receipt")
                self.jobs.append({"card": card, "submission": receipt, "slot": path.parent})
            elif (path.parent / "INTENT.json").exists():
                self.intents.append(str(path.parent / "INTENT.json"))
        self.observations = observations if observations is not None else observe_jobs(
            [j["submission"]["job_id"] for j in self.jobs])

    def state(self, job):
        return self.observations.get(str(job))

    def pending(self, jobs):
        return [j["submission"]["job_id"] for j in jobs if not terminal(self.state(j["submission"]["job_id"]))]

    def stage_jobs(self, key, *, output=None, stage=None):
        result = []
        for job in self.jobs:
            card = job["card"]
            if card["model"] != key or (stage and card["stage"] != stage):
                continue
            args = arguments(card)
            if output is not None and args.get("--output") != str(output):
                continue
            result.append(job)
        return result

    def action(self, key, stage, status, **details):
        row = {"model": key, "stage": stage, "status": status, **details}
        self.actions.append(row)
        return row

    def limit(self, key, stage, reason, **details):
        value = save(self.home / "limits" / key / (stage + ".json"), {
            "schema": "verification-autonomous-limit-v3", "campaign_sha256": self.campaign["sha256"],
            "model": key, "stage": stage, "reason": reason,
            "interpretation": "required model cell remains a scoped qualification/execution limit; no causal-null inference",
            **details})
        self.limits.append(value); self.action(key, stage, "LIMIT", reason=reason)
        return "LIMIT"

    def cpu(self, key, stage, function, outputs, inputs):
        from moe_exp.verification_study.closeout import require_cpu
        require_cpu()
        identity = {"schema": "verification-autonomous-cpu-action-v3",
                    "campaign_sha256": self.campaign["sha256"], "model": key, "stage": stage,
                    "inputs": {str(p): load(p)["sha256"] for p in inputs},
                    "source_file_sha256": file_sha(SCRIPT), "outputs": [str(p) for p in outputs]}
        slot = self.home / "cpu" / digest(identity)
        intent = save(slot / "INTENT.json", identity)
        if not all(p.exists() for p in outputs):
            function()
        result = save(slot / "RESULT.json", {"schema": "verification-autonomous-cpu-result-v3",
                     "intent_sha256": intent["sha256"], "outputs": {str(p): load(p)["sha256"] for p in outputs},
                     "status": "COMPLETE"})
        self.action(key, stage, "CPU_COMPLETE", receipt_sha256=result["sha256"])
        return result

    def request(self, key, stage, arguments_, *, label, population, count,
                prices=None, work=None, seconds=None, extension_=None, claim="C2/C3/C4/C5", extra=None):
        value = {"schema": "verification-task-request-v3", "stage": stage, "model": key,
                 "population": population, "arguments": arguments_, "finite_assignments": count,
                 "claim": claim + ": advance the approved required model cell",
                 "artifact": label + ": canonical immutable stage receipts and all-assignment paper accounting",
                 "reuse": "frozen campaign, existing full manifests/frames, pinned deployment and committed receipts",
                 "missing_cell": key + "/" + label,
                 "decision_if_failure": "preserve receipts and at most two identity attempts; retain missing/scoped limits without changing targets or gates",
                 "measurement_pilot": prices is None, "orchestration_file_sha256": file_sha(SCRIPT),
                 "resources": resources(seconds or 8 * 3600,
                     "retained first-role measurement; finite original enrollment" if prices is None else
                     "actual role maxima; full caps, one complete retry, two loads,1.25 factor plus600 seconds")}
        if prices:
            value.update(price_path=str(prices), priced_work=work)
        if extension_:
            value["extension"] = extension_
        if extra:
            value.update(extra)
        # The canonical file is reloaded before compilation (float sum ordering).
        path = self.home / "requests" / key / (label + "-" + digest(value) + ".json")
        save(path, value)
        return path, load(path)

    def dispatch(self, key, label, request_path, request, kind="native"):
        if kind == "reader":
            parent_card = reader.reader_card(self.path, request)
        elif kind == "extension":
            parent_card = extension.compile_card(self.path, request)
        else:
            parent_card = native.task_card(self.path, request)
        card = transport.wrap_card(self.path, parent_card, kind)
        submit = lambda test: submit_transport(self.path, card, test_only=test)
        slot = self.root / "submissions" / card["sha256"]
        if not self.submit:
            self.action(key, label, "CARD_READY", request_path=str(request_path), card_sha256=card["sha256"])
            return
        if (slot / "SUBMISSION.json").exists():
            existing = load(slot / "SUBMISSION.json")
            require(not terminal(self.state(existing["job_id"])),
                    "existing terminal card has missing required artifacts; exact reconciliation required")
        if not (slot / "SUBMISSION.json").exists() and not (slot / "INTENT.json").exists():
            self.entitlement(card)
            submit(True)
        receipt = submit(False)
        require(receipt["job_card_sha256"] == card["sha256"], "dispatch receipt identity changed")
        if not any(j["submission"]["job_id"] == receipt["job_id"] for j in self.jobs):
            self.jobs.append({"card": card, "submission": receipt, "slot": slot})
        self.action(key, label, "SUBMITTED", job_id=receipt["job_id"], request_path=str(request_path))

    def entitlement(self, card):
        """Pending reservations consume entitlement even before saldo charges them."""
        snapshot = native.preflight()
        row = next((line.split() for line in snapshot["balance"].splitlines()
                    if line.startswith("IscrC_MIOSR")), None)
        require(row is not None, "live project allocation unavailable")
        available = int(row[3]) - int(row[5])
        reserved = sum(j["card"]["resources"]["billing_core_hours_max"] for j in self.jobs
                       if not terminal(self.state(j["submission"]["job_id"])))
        require(card["resources"]["billing_core_hours_max"] + reserved < available,
                "insufficient live allocation after existing pending reservations")
        from verification_study_v3_discovery_continue_r2 import exact_storage
        capacity = exact_storage(REPO)["available_bytes"]
        outstanding = sum(storage_bound(j["card"], self.campaign) for j in self.jobs
                          if not terminal(self.state(j["submission"]["job_id"])))
        require(storage_bound(card, self.campaign) + outstanding < capacity,
                "finite outputs plus outstanding GPU output reservations exceed live WORK capacity")

    def price(self, key, kind="native"):
        """Snapshots are named by committed evidence, never reused as mutable paths."""
        base = self.root / key; directories = [base / "qualification/fixtures"]
        directories += [base / s for s in ("discovery", "validation", "final_native", "conditional", "utility")]
        if kind in {"reader", "j1"}:
            for name in ("eligibility-ratings", "outcome-price-ratings", "outcome-ratings"):
                directory = base / name; directories += [directory]
                directories += list((directory / "invocations").glob("*"))
        if kind == "j1":
            directories += [self.root / "J1/audit", base / "final_native-grades/j1", base / "utility-grades/j1"]
            for directory in directories[-3:]:
                directories += list((directory / "invocations").glob("*"))
        evidence = {}
        for directory in directories:
            for path in directory.glob("*.json"):
                if ".attempt-" in path.name or path.name in {"LEDGER.json", "INTENT.json", "RESULT.json"} or path.name.startswith("SHARD-"):
                    continue
                value = load(path)
                if value.get("schema") in {"verification-generation-receipt-v3", "verification-reader-runtime-v3",
                                            "verification-reader-receipt-v3", "verification-j1-runtime-v3", "verification-j1-verdict-v3"}:
                    evidence[str(path)] = value["sha256"]
        path = self.home / "prices" / key / (kind + "-" + digest(evidence) + ".json")
        if not path.exists():
            if kind == "reader":
                reader.collect_reader_prices(self.path, key, self.root, path)
            elif kind == "j1":
                j1.collect_prices(self.path, key, self.root, path)
            else:
                from moe_exp.verification_study.pricing import collect_prices
                collect_prices(self.path, key, self.root, path)
        value = load(path, "verification-prices-v3")
        require(value["model_sha256"] == load(self.campaign["models"][key])["sha256"] and
                value["code_files"] == self.campaign["code_files"], "foreign immutable stage prices")
        return path, value

    def generation(self, key, manifest_path, source, *, strict_success=False, produce=True):
        """Validate all receipts/archives and wait for their actual allocations."""
        model = load(self.campaign["models"][key]); manifest = load(manifest_path)
        self.active_stage[key] = manifest["stage"]
        require(manifest["model_sha256"] == model["sha256"], "foreign stage manifest")
        jobs = self.stage_jobs(key, output=source, stage="run"); covered = set(); completed = set(); exhausted = set()
        ownership = {}; exposures = {}
        for job in jobs:
            args = arguments(job["card"])
            require(load(args["--manifest"])["sha256"] == manifest["sha256"], "changed submitted stage manifest")
            begin = int(args.get("--begin", 0)); count = int(args.get("--count", len(manifest["assignments"])))
            selected = {a["uid"] for a in manifest["assignments"][begin:begin + count]}
            ownership[job["submission"]["job_id"]] = selected
            for uid in selected:
                exposures.setdefault(uid, []).append(job["submission"]["job_id"])
            if not terminal(self.state(job["submission"]["job_id"])):
                require(not (covered & selected), "overlapping live native slices")
                covered |= selected
        for assignment in manifest["assignments"]:
            path = source / (assignment["uid"] + ".json")
            attempts = [load(p) for p in source.glob(assignment["uid"] + ".attempt-*.json")]
            require(len(attempts) <= 2, "third native identity attempt is forbidden")
            if path.exists():
                value = load(path); require_generation_binding(value, self.campaign, model)
                require(value["assignment"] == assignment and value["execution_complete"] is True and
                        value["engineering_only"] is False, "incomplete/foreign native receipt")
                require(assignment["uid"] in ownership.get(value["job_id"], set()), "native receipt lacks exact submitted slice ownership")
                if manifest["stage"] != "discovery":
                    recognizer = load(self.root / key / "RECOGNIZER.json")
                    policy = load(self.root / key / "calibration/POLICY.json")
                    require(value["recognizer_sha256"] == recognizer["sha256"], "native receipt detector differs")
                    if manifest["stage"] in {"validation", "final_native"}:
                        expected_action = {"layer": policy["layer"], "expert": policy["target"], "bias": 0.}
                    else:
                        from moe_exp.verification_study.policy import action_for
                        expected_action = action_for(policy, assignment)
                    require(value["action"] == expected_action, "native receipt action differs from frozen policy")
                archive = private_path(value["capture"]["array_path"])
                require(archive == path.resolve().with_suffix(".npz") and archive.exists() and
                        archive.stat().st_size == value["capture"]["array_bytes"] and
                        file_sha(archive) == value["capture"]["array_sha256"], "native archive differs")
                state = self.state(value["job_id"])
                if terminal(state):
                    if strict_success:
                        require(successful(state), "full discovery needs successful completion accounting")
                    completed.add(assignment["uid"])
            elif attempts:
                require(all(a["uid"] == assignment["uid"] and a["assignment_sha256"] == digest(assignment)
                            for a in attempts), "foreign native attempt journal")
                require(all(any(j["submission"]["job_id"] == str(a["job_id"]) for j in jobs) for a in attempts),
                        "untracked native attempt requires reconciliation")
                require(len({a["number"] for a in attempts}) == len(attempts), "duplicate native attempt number")
                if len(attempts) == 2 and all(terminal(self.state(a["job_id"])) for a in attempts):
                    exhausted.add(assignment["uid"])
        if covered:
            return {"status": "WAITING", "completed": completed, "covered": covered, "exhausted": exhausted}
        all_uids = {a["uid"] for a in manifest["assignments"]}
        if completed == all_uids:
            return {"status": "COMPLETE", "completed": completed, "covered": covered, "exhausted": exhausted}
        if strict_success and exhausted:
            return {"status": "LIMIT", "completed": completed, "covered": covered, "exhausted": exhausted}
        if completed | exhausted == all_uids:
            return {"status": "TERMINAL_MISSING", "completed": completed, "covered": covered, "exhausted": exhausted}
        if not produce:
            return {"status": "MISSING", "completed": completed, "covered": covered, "exhausted": exhausted}
        for uid in all_uids - completed - exhausted:
            require(len(exposures.get(uid, [])) < 2,
                    "two native allocations already covered a missing identity; reconcile before any further allocation")
        self.native_slices(key, manifest_path, source, completed | exhausted)
        return {"status": "DISPATCHED", "completed": completed, "covered": covered, "exhausted": exhausted}

    def native_slices(self, key, manifest_path, source, completed):
        manifest = load(manifest_path); stage = manifest["stage"]; base = self.root / key
        prices_path, prices = self.price(key)
        needs_controller = stage in {"validation", "final_native", "utility"}
        measured = all(prices["components"][r]["status"] == "MEASURED" for r in
                       ("generator_load", "prefill", "generation", "capture", "closeout", "recovery") +
                       (("controller",) if needs_controller else ()))
        if not measured:
            require(stage == "validation" and not completed and not self.stage_jobs(key, output=source, stage="run"),
                    "unpriced final/controller work cannot become another measurement")
            blocks = [(0, 1)]
        else:
            blocks = uncovered_blocks(manifest["assignments"], completed, set())
        pilot = load(base / "discovery" / (load(self.path.parent / key / "DISCOVERY_ASSIGNMENTS.json")["assignments"][0]["uid"] + ".json"))
        for begin, length in blocks:
            while length:
                cap = manifest["assignments"][begin]["cap"]
                work = lambda n: native_work(n, cap, pilot["generated_tokens"], controller=needs_controller)
                count, seconds = (1, 8 * 3600) if not measured else fit_size(prices, work, length)[:2]
                args = ["--model", key, "--qualification", str(base / "qualification/QUALIFICATION.json"),
                        "--manifest", str(manifest_path), "--output", str(source), "--root", str(self.root),
                        "--begin", str(begin), "--count", str(count)]
                if stage != "discovery":
                    args += ["--policy", str(base / "calibration/POLICY.json"), "--recognizer", str(base / "RECOGNIZER.json")]
                if stage in {"final_native", "conditional", "utility"}:
                    args += ["--validation", str(base / "VALIDATION.json")]
                if measured:
                    args += ["--prices", str(prices_path)]
                selected = manifest["assignments"][begin:begin + count]
                shape = probe_native_inputs(self.path, key, selected)
                if measured:
                    require(shape["max_input_tokens"] <= prices["components"]["prefill"]["max_context_input_tokens"] and
                            shape["max_total_tokens"] <= prices["components"]["generation"]["max_total_context_tokens"],
                            "missing native prefill/generation price context cell: " + json.dumps(shape))
                retries = {a["uid"]: len(list(source.glob(a["uid"] + ".attempt-*.json"))) for a in selected}
                prior_cards = sorted(j["card"]["sha256"] for j in self.stage_jobs(key, output=source, stage="run")
                                     if terminal(self.state(j["submission"]["job_id"])))
                request_path, request = self.request(key, "run", args, label=stage + f"-{begin}-{count}-" + digest(retries)[:12],
                    population=stage if stage in {"discovery", "validation"} else "final",
                    count=count, prices=prices_path if measured else None, work=work(count) if measured else None, seconds=seconds,
                    extra={"reconciled_terminal_predecessor_cards": prior_cards, "existing_identity_attempts": retries})
                self.dispatch(key, stage, request_path, request)
                begin += count; length -= count

    def ratings(self, key, frame_path, output, role, *, measurement=False):
        self.active_stage[key] = output.name
        campaign, contract, frame, executor = reader.deployment(self.path, frame_path)
        reader.validate_directory(output, campaign, contract, frame, executor, role)
        completed = {r["uid"] for r in frame["records"] if all((output / (r["uid"] + f".reader-{d}.json")).exists() for d in (0, 1))}
        reconciled = reconciled_unstarted_readers(self.path, self.root, self.jobs, self.observations)
        jobs = [job for job in self.stage_jobs(key, output=output, stage="rate")
                if str(job["submission"]["job_id"]) not in reconciled]; covered = set()
        for job in jobs:
            card = job["card"]; require(card["reader_executor"] == executor and card["full_frame_sha256"] == frame["sha256"],
                                       "reader submitted deployment/frame changed")
            if not terminal(self.state(job["submission"]["job_id"])):
                require(not (covered & set(card["selected_uids"])), "overlapping live reader slices")
                covered.update(card["selected_uids"])
            elif not successful(self.state(job["submission"]["job_id"])):
                raise ValueError("failed reader allocation requires exact invocation reconciliation")
        if covered:
            return "WAITING"
        if len(completed) == len(frame["records"]):
            return "COMPLETE"
        if measurement and completed:
            return "COMPLETE"
        prices_path, prices = self.price(key, "reader")
        price_role = "eligibility_reader" if role == "eligibility" else "outcome_reader"
        priced = prices["components"][price_role]["status"] == "MEASURED"
        shape = reader.probe_inputs(self.path, frame_path, role)
        if measurement:
            require(not jobs, "retained reader measurement already attempted; reconcile it")
            blocks = [(shape["largest_record_index"], 1)]; priced = False
        elif not priced:
            require(role == "eligibility" and not jobs and not completed, "unpriced final outcome reader forbidden")
            blocks = [(shape["largest_record_index"], 1)]
        else:
            require(shape["max_input_tokens"] <= prices["components"][price_role]["max_context_input_tokens"],
                    "missing " + price_role + " context price cell: planned=" + str(shape["max_input_tokens"]) +
                    ", measured=" + str(prices["components"][price_role]["max_context_input_tokens"]))
            blocks = uncovered_blocks(frame["records"], completed, set())
        for begin, length in blocks:
            while length:
                work = lambda n: reader_work(prices, price_role, n)
                count, seconds = (1, 8 * 3600) if not priced else fit_size(prices, work, length)[:2]
                args = ["--model", key, "--frame", str(frame_path), "--output", str(output), "--role", role,
                        "--begin", str(begin), "--count", str(count)]
                request_path, request = self.request(key, "rate", args, label=output.name + f"-{begin}-{count}",
                    population="discovery" if not priced else "final" if role == "outcome" else "discovery",
                    count=count, prices=prices_path if priced else None, work=work(count) if priced else None, seconds=seconds)
                self.dispatch(key, output.name, request_path, request, "reader")
                begin += count; length -= count
        return "DISPATCHED"

    def shared_j1(self):
        frame_path = self.root / "J1/AUDIT_FRAME.json"
        output = self.root / "J1/J1_QUALIFICATION.json"
        if not frame_path.exists():
            return "WAITING"
        campaign, contract, frame, executor = j1.deployment(self.path, frame_path)
        jobs = [j for j in self.jobs if j["card"].get("extension", {}).get("kind") == "j1-audit-slice"]
        if self.pending(jobs):
            return "WAITING"
        completed = set()
        for item in frame["records"]:
            path = self.root / "J1/audit" / (item["uid"] + ".json")
            if path.exists():
                j1.validate_verdict(path, item, campaign, contract, frame, executor)
                completed.add(item["uid"])
        failed = [j for j in jobs if not successful(self.state(j["submission"]["job_id"]))]
        proof = validate_j1_reconciliation(self.path, self.root, failed, self.observations) if failed else None
        reconciled_ids = {row["job_id"] for row in proof["failed_jobs"]} if proof else set()
        repaired = [j for j in jobs if j["card"].get("transport_executor") and
                    str(j["submission"]["job_id"]) not in reconciled_ids]
        # Reconciled failures produced no draws; every prospective repair must
        # match this new sealed transport and succeed before further allocation.
        for job in repaired:
            transport.verify_card(self.path, job["card"], "extension")
        require(all(successful(self.state(j["submission"]["job_id"])) for j in repaired),
                "transport recovery allocation failed; explicit diagnosis required, no repeated retry")
        for job in repaired:
            require(set(job["card"]["extension"]["selected_uids"]) <= completed,
                    "successful repair allocation is missing its exact audit verdicts")
        if len(completed) == 100:
            self.cpu("all-seven", "shared-j1-qualification", lambda: j1.qualification(
                self.path, frame_path, self.root / "J1/audit", output), [output], [frame_path])
            return load(output)["status"]
        require(proof is not None, "incomplete shared J1 audit lacks explicit terminal execution reconciliation")
        # One retained original missing item must qualify the new launch before
        # allocating the remaining audit. The scientific deployment is unchanged.
        if not repaired:
            begin = next(i for i, row in enumerate(frame["records"]) if row["uid"] not in completed)
            blocks = [(begin, 1)]
        else:
            require(any(len(job["card"]["extension"]["selected_uids"]) == 1 for job in repaired),
                    "transport recovery lacks the required retained one-item check")
            blocks = uncovered_blocks(frame["records"], completed, set())
        prices_path, prices = self.price("gpt", "j1")
        require(prices.get("j1_executor") == executor, "recovery J1 price deployment changed")
        shape = j1.probe(self.path, frame_path)
        require(shape["max_input_tokens"] <= prices["components"]["j1_generation"]["max_context_input_tokens"],
                "audit recovery exceeds existing measured context price")
        for begin, length in blocks:
            while length:
                work = lambda n: {"j1_load": 2, "j1_generation": 3 * 8192 * (n + 1)}
                count, seconds, budget = fit_size(prices, work, length)
                selected = frame["records"][begin:begin + count]
                require({r["uid"] for r in selected} <= set(proof["eligible_recovery_uids"]),
                        "recovery slice contains an item outside proven pre-initialization failures")
                params = {"frame": str(frame_path), "output": str(self.root / "J1/audit"),
                          "begin": begin, "count": count}
                args = [v for key, value in params.items() for v in ("--" + key, str(value))]
                request_path, request = self.request("gpt", "j1-audit", args,
                    label=f"shared-j1-transport-{begin}-{count}", population="frozen_J1_audit", count=count,
                    prices=prices_path, work=work(count), seconds=seconds,
                    extension_={"kind": "j1-audit-slice", "parameters": params},
                    extra={"transport_reconciliation_sha256": proof["sha256"],
                           "transport_reconciliation_path": str(self.path.parent / "J1_TRANSPORT_RECONCILIATION-r3.json"),
                           "recovery_phase": "one-retained-item" if not repaired else "remaining-original-items"})
                self.dispatch("gpt", "shared-j1-transport", request_path, request, "extension")
                begin += count; length -= count
        return "DISPATCHED_TRANSPORT_CHECK" if not repaired else "DISPATCHED_RECONCILED_AUDIT"

    def grading(self, key, stage, manifest_path):
        self.active_stage[key] = stage + "-grading"
        from moe_exp.verification_study.closeout import grade_strict
        base = self.root / key; source = base / stage; grades = base / (stage + "-grades")
        frame_path = grades / "J1_FROZEN_FRAME.json"
        if not frame_path.exists():
            grade_strict(self.path, key, manifest_path, source, grades, begin=0, count=None)
            self.cpu(key, stage + "-j1-freeze", lambda: j1.freeze_residuals(
                self.path, key, manifest_path, source, grades, frame_path), [frame_path], [manifest_path])
        frame = load(frame_path)
        if not frame["records"]:
            return "COMPLETE"
        qualification_path = self.root / "J1/J1_QUALIFICATION.json"
        if not qualification_path.exists():
            return "WAITING_FOR_SHARED_J1"
        qualified = load(qualification_path)
        if qualified["status"] != "PASS":
            self.action(key, stage + "-j1", "LIMIT", reason="COMMON_GRADER_UNQUALIFIED; residual accuracy stays unknown")
            return "COMPLETE_WITH_UNKNOWN_J1"
        campaign, contract, frame, executor = j1.deployment(self.path, frame_path)
        output = grades / "j1"; jobs = self.stage_jobs(key, output=output, stage="grade-j1")
        if self.pending(jobs):
            return "WAITING"
        require(all(successful(self.state(j["submission"]["job_id"])) for j in jobs), "J1 production allocation requires reconciliation")
        completed = set()
        for item in frame["records"]:
            path = output / (item["uid"] + ".json")
            if path.exists():
                j1.validate_verdict(path, item, campaign, contract, frame, executor, qualified); completed.add(item["uid"])
        if len(completed) == len(frame["records"]):
            return "COMPLETE"
        prices_path, prices = self.price(key, "j1"); shape = j1.probe(self.path, frame_path)
        require(prices.get("j1_executor") == executor and shape["max_input_tokens"] <= prices["components"]["j1_generation"]["max_context_input_tokens"],
                "missing j1_generation context price cell: planned=" + str(shape["max_input_tokens"]) +
                ", measured=" + str(prices["components"]["j1_generation"]["max_context_input_tokens"]))
        for begin, length in uncovered_blocks(frame["records"], completed, set()):
            while length:
                work = lambda n: {"j1_load": 2, "j1_generation": 3 * 8192 * (n + 1)}
                count, seconds, budget = fit_size(prices, work, length)
                params = {"frame": str(frame_path), "output": str(output), "begin": begin, "count": count,
                          "qualification": str(qualification_path)}
                args = [v for k, value in params.items() for v in ("--" + k, str(value))]
                request_path, request = self.request(key, "grade-j1", args, label=stage + f"-j1-{begin}-{count}",
                    population="final", count=count, prices=prices_path, work=work(count), seconds=seconds,
                    extension_={"kind": "grade-j1-slice", "parameters": params})
                self.dispatch(key, stage + "-j1", request_path, request, "extension")
                begin += count; length -= count
        return "DISPATCHED"

    def model(self, key):
        from moe_exp.verification_study.execution import make_audit, fit_nomination
        from moe_exp.verification_study import stages
        base = self.root / key
        discovery = self.path.parent / key / "DISCOVERY_ASSIGNMENTS.json"
        ready = self.generation(key, discovery, base / "discovery", strict_success=True)
        if ready["status"] == "LIMIT":
            return self.limit(key, "discovery", "EXHAUSTED_NATIVE_DISCOVERY; no partial audit freeze")
        if ready["status"] != "COMPLETE":
            return ready["status"]
        audit = base / "AUDIT_FRAME.json"
        self.cpu(key, "audit-frame", lambda: make_audit(self.path, key, base / "discovery", audit), [audit], [discovery])
        frame = load(audit)
        require(not frame["missing_discovery_assignments"], "partial discovery audit cannot advance")
        if not frame["records"]:
            return self.limit(key, "recognition", "NO_AUDIT_PREFIX_SUPPORT")
        if self.ratings(key, audit, base / "eligibility-ratings", "eligibility") != "COMPLETE":
            return "WAITING"
        recognizer = base / "RECOGNIZER.json"; nomination = base / "NOMINATION.json"
        self.cpu(key, "recognition", lambda: stages.recognition(self.path, key, audit, base / "eligibility-ratings", recognizer),
                 [recognizer], [audit])
        if load(recognizer)["status"] != "PASS":
            return self.limit(key, "recognition", load(recognizer)["status"], receipt_sha256=load(recognizer)["sha256"])
        self.cpu(key, "nominate", lambda: fit_nomination(self.path, key, audit, base / "eligibility-ratings", base / "discovery", nomination),
                 [nomination], [audit, recognizer])
        if load(nomination)["status"] != "NOMINATED":
            return self.limit(key, "nomination", load(nomination)["status"], receipt_sha256=load(nomination)["sha256"])
        policy = base / "calibration/POLICY.json"
        self.active_stage[key] = "calibration"
        if not policy.exists():
            jobs = self.stage_jobs(key, stage="calibrate")
            if jobs:
                require(self.pending(jobs), "partial/failed unsliced calibration requires compatible reconciliation")
                return "WAITING"
            args = ["--model", key, "--qualification", str(base / "qualification/QUALIFICATION.json"),
                    "--nomination", str(nomination), "--frame", str(audit), "--source", str(base / "discovery"),
                    "--output", str(base / "calibration")]
            # Match the frozen calibrate() enrollment exactly before allocating.
            first = {}
            for record in sorted(frame["records"], key=lambda r: (r["position"], r["uid"])):
                if record["split"] == "apparent" and record["proposed"]["eligible"]:
                    first.setdefault(record["family"], record)
            chosen = sorted(first, key=lambda family: digest(["calibration-families", family]))[:16]
            if len(chosen) < 10:
                return self.limit(key, "calibration", "INSUFFICIENT_DISCOVERY_CALIBRATION_FAMILIES", families=len(chosen))
            allowed = {a["family"]: a for a in load(discovery)["assignments"]}
            bank_rows = [{**allowed[family], "cap": 64, "prefix": first[family]} for family in chosen]
            shape = probe_native_inputs(self.path, key, bank_rows)
            prices_path, prices = self.price(key)
            pilot_tokens = load(base / "discovery" / (load(discovery)["assignments"][0]["uid"] + ".json"))["generated_tokens"]
            bank_work = {"generator_load": 2, "prefill": len(chosen) + 1, "generation": 64 * (len(chosen) + 1),
                         "capture": max(1, 64 / pilot_tokens) * (len(chosen) + 1),
                         "closeout": max(1, 64 / pilot_tokens) * (len(chosen) + 1)}
            bank_budget = shard_budget(prices, bank_work, hours=8)
            # First calibration also measures the still-unknown bounded cached
            # policy computation: one target/pool,9 brackets and12 bisections.
            request_path, request = self.request(key, "calibrate", args, label="calibration-measurement",
                population="discovery", count=len(chosen), seconds=8 * 3600,
                extra={"calibration_enrollment": {"families": chosen, "native_bank_cap": 64,
                       "bank_shape": shape, "measured_native_bank_budget": bank_budget,
                       "native_price_path": str(prices_path), "native_price_sha256": prices["sha256"],
                       "unknown_first_measurement_role": "discovery cached-dose/control policy calibration"}})
            self.dispatch(key, "calibration", request_path, request)
            return "WAITING"
        if self.pending(self.stage_jobs(key, stage="calibrate")):
            return "WAITING"
        calibration_jobs = self.stage_jobs(key, stage="calibrate")
        require(len(calibration_jobs) == 1 and successful(self.state(calibration_jobs[0]["submission"]["job_id"])),
                "policy file lacks exact successful calibration allocation accounting")
        calibrated = load(policy); model = load(self.campaign["models"][key]); nominated = load(nomination)
        enrollment = load(base / "calibration/ENROLLMENT.json")
        require(calibrated["campaign_sha256"] == self.campaign["sha256"] and calibrated["code_files"] == self.campaign["code_files"] and
                calibrated["model_sha256"] == model["sha256"] and calibrated["nomination_sha256"] == nominated["sha256"] and
                calibrated["layer"] == nominated["selected"]["layer"] and calibrated["target"] == nominated["selected"]["expert"] and
                calibrated["enrollment_sha256"] == enrollment["sha256"] and enrollment["frame_sha256"] == frame["sha256"],
                "policy/calibration enrollment/source binding differs")
        require({row["family"] for row in calibrated["calibration_families"]} == set(enrollment["families"]),
                "policy does not cover exact retained calibration families")
        for row in calibrated["calibration_families"]:
            bank = load(base / "calibration/native-predictions" / (row["uid"] + ".json"))
            require_generation_binding(bank, self.campaign, model)
            require(bank["job_id"] == calibration_jobs[0]["submission"]["job_id"] and
                    digest(bank["assignment"]) == row["source_sha256"] and bank["assignment"]["family"] == row["family"] and
                    bank["assignment"]["cap"] == 64 and bank["execution_complete"] is True,
                    "policy native bank is foreign/incomplete")
            archive = private_path(bank["capture"]["array_path"])
            require(archive.exists() and file_sha(archive) == bank["capture"]["array_sha256"], "calibration bank archive changed")
        if load(policy)["status"] != "PASS":
            return self.limit(key, "policy", load(policy)["status"], receipt_sha256=load(policy)["sha256"])
        validation_manifest = self.path.parent / key / "VALIDATION_ASSIGNMENTS.json"
        ready = self.generation(key, validation_manifest, base / "validation")
        if ready["status"] == "TERMINAL_MISSING":
            return self.limit(key, "validation", "EXHAUSTED_NATIVE_VALIDATION")
        if ready["status"] != "COMPLETE":
            return ready["status"]
        validation = base / "VALIDATION.json"
        self.cpu(key, "validate", lambda: stages.validation(self.path, key, policy, recognizer, base / "validation", validation),
                 [validation], [validation_manifest, policy, recognizer])
        if load(validation)["status"] != "PASS":
            return self.limit(key, "validation", load(validation)["status"], receipt_sha256=load(validation)["sha256"])
        final_manifest = self.path.parent / key / "FINAL_NATIVE_ASSIGNMENTS.json"
        ready = self.generation(key, final_manifest, base / "final_native")
        if ready["status"] not in {"COMPLETE", "TERMINAL_MISSING"}:
            return ready["status"]
        if ready["status"] == "TERMINAL_MISSING":
            from moe_exp.verification_study.closeout import terminal_source
            output = base / "final_native/TERMINAL.json"
            self.cpu(key, "terminal-final-source", lambda: terminal_source(final_manifest, base / "final_native", output),
                     [output], [final_manifest])
        conditional = base / "final/CONDITIONAL_ASSIGNMENTS.json"; utility = base / "final/UTILITY_ASSIGNMENTS.json"
        self.cpu(key, "freeze-final", lambda: stages.freeze_final(self.path, key, base / "final_native", policy, recognizer, base / "final"),
                 [base / "final/PREFIXES.json", conditional, utility], [final_manifest, policy, recognizer])
        ready_c = self.generation(key, conditional, base / "conditional")
        ready_u = self.generation(key, utility, base / "utility")
        settled = {"COMPLETE", "TERMINAL_MISSING"}
        if ready_c["status"] not in settled or ready_u["status"] not in settled:
            return "WAITING"
        if load(conditional)["assigned"]:
            outcome_frame = base / "OUTCOME_FRAME.json"
            self.cpu(key, "outcome-frame", lambda: stages.outcome_frame(self.path, key, conditional, base / "conditional", outcome_frame),
                     [outcome_frame, outcome_frame.with_name("OUTCOME_FRAME-PRIVATE-BINDING.json")], [conditional])
            if load(outcome_frame)["records"]:
                price_frame = base / "OUTCOME_PRICE_FRAME.json"
                self.cpu(key, "outcome-price-frame", lambda: reader.outcome_price_frame(self.path, key, self.root, price_frame),
                         [price_frame], [audit])
                if self.ratings(key, price_frame, base / "outcome-price-ratings", "outcome", measurement=True) != "COMPLETE":
                    return "WAITING"
                if self.ratings(key, outcome_frame, base / "outcome-ratings", "outcome") != "COMPLETE":
                    return "WAITING"
        grade_n = self.grading(key, "final_native", final_manifest)
        grade_u = self.grading(key, "utility", utility)
        if grade_n.startswith("COMPLETE") and grade_u.startswith("COMPLETE"):
            self.action(key, "scientific-chain", "COMPLETE")
            return "COMPLETE"
        return "WAITING"


def next_actions(campaign_path, root, *, submit=True, observations=None):
    """One finite all-seven pass. Call only inside the driver's allocated step.

    Optional observations map job IDs to state/exit_code/live dictionaries. The
    caller must hold the global orchestration lock; this function creates no
    callbacks. Every error stops that cell visibly without changing old sources.
    """
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu()
    run = Advance(campaign_path, root, submit=submit, observations=observations)
    if run.intents:
        return {"status": "RECONCILE_SUBMISSION_INTENTS", "actions": [],
                "live_scientific_job_ids": run.pending(run.jobs), "errors": run.intents,
                "ready_cpu_actions": False}
    outcomes = {}
    try:
        j1_status = run.shared_j1()
    except Exception as error:
        j1_status = "ERROR"; run.errors.append({"model": "all-seven", "stage": "J1", "error": str(error)[:2000]})
    for key in MODEL_KEYS:
        try:
            outcomes[key] = run.model(key)
        except Exception as error:
            outcomes[key] = "ERROR"
            stage = run.active_stage.get(key, "readiness")
            run.errors.append({"model": key, "stage": stage, "missing_cell": key + "/" + stage,
                               "error": str(error)[:2000]})
    closed = all(v in {"COMPLETE", "LIMIT"} for v in outcomes.values()) and not run.pending(run.jobs) and j1_status in {"PASS", "FAIL"}
    if closed and not run.errors:
        try:
            from moe_exp.verification_study.closeout import assemble
            from moe_exp.verification_study.analysis import analyze
            from moe_exp.verification_study.paper import export
            analysis_input = run.root / "ANALYSIS_INPUT.json"; analysis = run.root / "ANALYSIS.json"
            if not analysis_input.exists():
                assemble(run.path, run.root, analysis_input)
            data = load(analysis_input)
            if not analysis.exists():
                save(analysis, {**analyze(data["records"], load(run.path.parent / "FAMILIES.json")["final"],
                                        conditional_enrollment=data.get("conditional_enrollment")),
                                "campaign_sha256": run.campaign["sha256"], "input_sha256": data["sha256"]})
            export(run.path, run.root); run.action("all-seven", "assemble-analyze-export", "COMPLETE")
        except Exception as error:
            run.errors.append({"model": "all-seven", "stage": "closeout", "error": str(error)[:2000]})
    live = run.pending(run.jobs)
    return {"status": "ERROR" if run.errors else "COMPLETE" if closed else "ADVANCED" if run.actions else "WAITING",
            "campaign_sha256": run.campaign["sha256"], "source_file_sha256": file_sha(SCRIPT),
            "models": outcomes, "shared_j1": j1_status, "actions": run.actions,
            "live_scientific_job_ids": live, "errors": run.errors, "ready_cpu_actions": False}
