#!/usr/bin/env python3
"""Seal observed discovery advancement; never dispatch or alter scientific files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/experimental_resume")]
from moe_exp.verification_study.contracts import (
    check_files, digest, file_sha, load, require, require_generation_binding, save, seal,
)
import verification_study_v3_health_status_r4 as health
import verification_study_v3_reader_r1 as reader


def accounting(ids):
    result = subprocess.run([
        "sacct", "-X", "-nP", "-j", ",".join(sorted(ids, key=int)),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,Partition,Account",
    ], check=True, capture_output=True, text=True, timeout=20)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] in ids:
            rows[fields[0]] = dict(state=fields[1], exit_code=fields[2],
                elapsed_seconds=int(fields[3]), allocated_cpus=int(fields[4]),
                partition=fields[5], account=fields[6])
    require(set(rows) == set(ids), "required accounting rows missing")
    for row in rows.values():
        require(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" and
            row["account"] == "iscrc_miosr", "successful own-account accounting required")
        row["actual_allocated_billing_core_hours"] = (
            row["elapsed_seconds"] * row["allocated_cpus"] / 3600)
    return rows


def closeout(campaign_path, key, controller_id, output):
    campaign_path = campaign_path.resolve()
    c = load(campaign_path)
    check_files(c["code_files"])
    model = load(c["models"][key])
    root = Path(c["result_root"])
    manifest = load(campaign_path.parent / key / "DISCOVERY_ASSIGNMENTS.json")
    require(len(manifest["assignments"]) == 48, "original discovery must contain48 assignments")
    frame_path = root / key / "AUDIT_FRAME.json"
    frame = load(frame_path)
    require(frame["model_sha256"] == model["sha256"] and
        not frame["missing_discovery_assignments"] and frame["splits_frozen_before_ratings"],
        "full source-bound discovery frame required")
    source_receipts = {}
    source_archives = {}
    native = {}
    jobs = Counter()
    for a in manifest["assignments"]:
        p = root / key / "discovery" / (a["uid"] + ".json")
        r = load(p)
        require_generation_binding(r, c, model)
        require(r["assignment"] == a and r["assignment_uid"] == a["uid"] and
            r["execution_complete"] and not r["engineering_only"] and
            a["stage"] == "discovery" and a["arm"] == "native", "original native source differs")
        archive = root / key / "discovery" / (a["uid"] + ".npz")
        capture = r["capture"]
        require(Path(capture["array_path"]).resolve() == archive.resolve() and
            archive.stat().st_size == capture["array_bytes"] and
            file_sha(archive) == capture["array_sha256"], "source capture archive differs")
        source_receipts[str(p)] = r["sha256"]
        source_archives[str(archive)] = capture["array_sha256"]
        native[a["uid"]] = r
        jobs[str(r["job_id"])] += 1
    require(frame["source_receipts"] == list(source_receipts.values()), "frame source closure differs")
    require(frame["records"] and len({r["uid"] for r in frame["records"]}) == len(frame["records"]),
        "empty or duplicate audit frame")
    for row in frame["records"]:
        source = native[row["source_uid"]]
        require(row["population"] == "discovery" and row["prefix_only"] and
            row["family"] == source["assignment"]["family"] and
            row["split"] == frame["family_split"][row["family"]] and
            row["position"] == len(row["prefix_token_ids"]) and
            row["prefix_token_ids"] == source["emitted_token_ids"][:row["position"]] and
            row["prefix_ids_sha256"] == digest(row["prefix_token_ids"]), "emitted-prefix source join differs")

    run = root / "automation/runs" / controller_id
    result = load(run / "RESULT.json")
    handoff = load(run / "HANDOFF.json")
    require(result["campaign_sha256"] == c["sha256"] and
        result["operational_version"] == "autodriver-r5" and not result["frontier"]["errors"],
        "successful current controller frontier required")
    h = seal(health.observe(campaign_path, write=False))
    require(not h["errors"] and not h["unresolved_submission_intents"], "health requires reconciliation")
    science = {r["job_id"]: r for r in h["scientific_jobs"]}
    native_card_proofs = {}
    for jid in jobs:
        require(jid in science, "native receipt lacks an accepted scientific submission")
        native_row = science[jid]
        native_card = load(native_row["card_path"])
        native_submission = load(native_row["submission_path"])
        args = native_card["payload_arguments"]
        require(args[0] == "run" and len(args[1:]) % 2 == 0, "native execution arguments differ")
        params = dict(zip(args[1::2], args[2::2]))
        require(native_card["campaign_sha256"] == c["sha256"] and
            native_row["model"] == key and native_card["model"] == key and
            native_card["stage"] == "run" and params["--model"] == key and
            Path(params["--campaign"]).resolve() == campaign_path and
            Path(params["--manifest"]).resolve() ==
                (campaign_path.parent / key / "DISCOVERY_ASSIGNMENTS.json").resolve() and
            Path(params["--output"]).resolve() == (root / key / "discovery").resolve() and
            native_submission["job_card_sha256"] == native_card["sha256"] and
            str(native_submission["job_id"]) == jid, "native accepted card/submission differs")
        begin, count = int(params["--begin"]), int(params["--count"])
        require(0 <= begin < 48 and 0 < count <= 48 - begin, "native slice outside original manifest")
        expected = [a["uid"] for a in manifest["assignments"][begin:begin + count]]
        observed = [a["uid"] for a in manifest["assignments"]
            if str(native[a["uid"]]["job_id"]) == jid]
        require(expected == observed, "original receipts differ from their accepted job slice")
        native_card_proofs[jid] = dict(card_path=native_row["card_path"],
            card_sha256=native_card["sha256"], submission_path=native_row["submission_path"],
            submission_sha256=native_submission["sha256"], begin=begin, count=count,
            resources=native_card["resources"])
    retained = []
    for row in science.values():
        if row["model"] == key and row["stage"] == "rate":
            card = load(row["card_path"])
            if card["full_frame_sha256"] == frame["sha256"]:
                retained.append((row, card))
    require(len(retained) == 1, "exact retained reader required before bulk advancement")
    row, card = retained[0]
    shape = card["reader_input_shape"]
    full_shape = reader.probe_inputs(campaign_path, frame_path, "eligibility")
    i = full_shape["largest_record_index"]
    expected_shape = reader.probe_inputs(campaign_path, frame_path, "eligibility", i, 1)
    require(shape == expected_shape and shape["max_input_tokens"] == full_shape["max_input_tokens"] and
        card["selected_uids"] == [frame["records"][i]["uid"]], "largest-input retained slice differs")
    submission = load(row["submission_path"])
    require(submission["job_card_sha256"] == card["sha256"] and
        str(submission["job_id"]) == row["job_id"] and
        card["reader_executor"] == reader.deployment(campaign_path, frame_path)[3],
        "reader submission/deployment differs")
    successor = str(handoff["successor_job_id"])
    live = {r["job_id"]: r for r in h["controllers"] if r.get("queue")}
    require(successor in live and live[successor]["operational_version"] == "autodriver-r5",
        "current successor not live")
    guards = sorted([jid for jid, r in live.items() if r["mode"] == "rescue" and
        successor in r["dependency"].split(":")[1:]], key=int)
    require(len(guards) == 2, "both successor failure guards required")
    protection = {}
    for jid in [successor, *guards]:
        r = live[jid]
        require(r["queue"]["reason"] != "(JobHeldUser)" and
            load(r["prearm_path"])["sha256"] == r["prearm_sha256"] and
            load(r["release_path"])["sha256"] == r["release_sha256"], "unreleased or unbound protection")
        protection[jid] = {k: r[k] for k in ["dependency", "queue", "prearm_sha256", "release_sha256"]}
    actual = accounting(set(jobs) | {controller_id})
    for jid, proof in native_card_proofs.items():
        require(actual[jid]["allocated_cpus"] == proof["resources"]["cpus"] and
            actual[jid]["partition"] == proof["resources"]["partition"], "native allocated shape differs")
    allocations = {jid: {**actual[jid], "assignments": n} for jid, n in jobs.items()}
    readiness = load(campaign_path.parent / "AUTOMATION_READINESS-r5.json")
    check_files(c["code_files"])
    check_files(readiness["code_files"])
    value = dict(schema="verification-discovery-advancement-v3", campaign_sha256=c["sha256"], model=key,
        reporting_helper_path=str(Path(__file__).resolve()), reporting_helper_sha256=file_sha(__file__),
        original_assignments=48, discovery_manifest_sha256=manifest["sha256"],
        source_receipts=source_receipts, source_archive_bindings=source_archives,
        source_job_card_bindings=native_card_proofs,
        audit_frame_path=str(frame_path), audit_frame_sha256=frame["sha256"],
        audit_prefixes=len(frame["records"]), missing_discovery_assignments=[],
        all_stratum_quotas_filled=all(x["available"] == x["requested"] for x in frame["shortages"]),
        successful_discovery_allocations=allocations,
        actual_allocated_billing_core_hours=sum(r["actual_allocated_billing_core_hours"] for r in allocations.values()),
        controller_job_id=controller_id, controller_elapsed_seconds=actual[controller_id]["elapsed_seconds"],
        controller_result_sha256=result["sha256"], handoff_sha256=handoff["sha256"],
        successor_job_id=successor, successor_wait_job_ids=handoff["wait_job_ids"],
        released_rescue_job_ids=guards, controller_protection=protection, observed_health_sha256=h["sha256"],
        observed_health_epoch=h["observed_epoch"], observed_health_reader_version=h["reader_version"],
        retained_reader_job_id=row["job_id"], retained_reader_card_path=row["card_path"],
        retained_reader_card_sha256=card["sha256"], retained_reader_submission_sha256=submission["sha256"],
        retained_reader_input_shape=shape, full_reader_input_shape_sha256=full_shape["sha256"],
        retained_reader_resources=card["resources"], retained_reader_observed_queue=row.get("queue"),
        recognizer_status="NOT_MEASURED", new_causal_and_utility_effects="NOT_MEASURED",
        scientific_files_verified=len(c["code_files"]), operational_readiness_files_verified=len(readiness["code_files"]),
        additional_manual_submissions=0, supersedes_status=key + " discovery/audit wait; previous execution evidence preserved")
    proof = save(output, value)
    require(load(output) == proof, "closeout readback differs")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    p.add_argument("--model", required=True)
    p.add_argument("--controller-job-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = closeout(a.campaign, a.model, a.controller_job_id, a.output)
    print(json.dumps({k: r[k] for k in ["sha256", "model", "original_assignments", "audit_prefixes",
        "actual_allocated_billing_core_hours", "retained_reader_job_id", "successor_job_id"]}))


if __name__ == "__main__":
    main()
