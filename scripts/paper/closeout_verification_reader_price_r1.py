#!/usr/bin/env python3
"""Report retained reader pricing and original coverage without dispatching work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/experimental_resume")]
from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save, seal
import verification_study_v3_health_status_r4 as health
import verification_study_v3_reader_r1 as reader
from closeout_verification_discovery_r1 import accounting


def closeout(campaign_path, key, controller_id, output):
    campaign_path = campaign_path.resolve()
    base = campaign_path.parent
    previous = load(base / (key.upper() + "_DISCOVERY_ADVANCEMENT-r1.json"))
    c = load(campaign_path)
    require(previous["campaign_sha256"] == c["sha256"] and previous["model"] == key,
        "discovery advancement parent differs")
    root = Path(c["result_root"])
    frame_path = root / key / "AUDIT_FRAME.json"
    c, contract, frame, executor = reader.deployment(campaign_path, frame_path)
    require(frame["sha256"] == previous["audit_frame_sha256"], "frozen frame differs")
    index = previous["retained_reader_input_shape"]["begin"]
    record = frame["records"][index]
    uid = record["uid"]
    jid = previous["retained_reader_job_id"]
    draws = []
    runtime = None
    for draw in (0, 1):
        path = root / key / "eligibility-ratings" / (uid + f".reader-{draw}.json")
        r = reader.validate_rating(path, record, draw, "eligibility", c, contract, frame, executor)
        require(str(r["job_id"]) == jid and r["input_tokens"] ==
            previous["retained_reader_input_shape"]["max_input_tokens"], "retained input/job differs")
        rt = load(r["invocation_runtime_path"])
        require(runtime is None or runtime == rt, "fixed draws belong to different invocations")
        runtime = rt
        draws.append({"path": str(path), **{k: r[k] for k in ["sha256", "reader", "finish_reason",
            "input_tokens", "output_tokens", "seconds", "invocation_runtime_path", "invocation_runtime_sha256"]}})
    require(runtime["slice"] == [index, 1] and runtime["selected_uids"] == [uid], "retained invocation differs")
    invocation = load(Path(draws[0]["invocation_runtime_path"]).parent / "RESULT.json")
    require(invocation["status"] == "COMPLETE" and invocation["error"] is None and
        invocation["loaded"] and invocation["new_ratings"] == 2 and
        invocation["selected_uids"] == [uid] and invocation["slice"] == [index, 1] and
        invocation["reader_executor"] == executor and invocation["campaign_sha256"] == c["sha256"] and
        invocation["frame_sha256"] == frame["sha256"] and str(invocation["job_id"]) == jid,
        "retained invocation not complete/source-bound")
    actual = accounting({jid, controller_id})
    run = root / "automation/runs" / controller_id
    result = load(run / "RESULT.json")
    handoff = load(run / "HANDOFF.json")
    require(result["campaign_sha256"] == c["sha256"] and
        result["operational_version"] == "autodriver-r5" and not result["frontier"]["errors"],
        "price controller requires successful original frontier")
    h = seal(health.observe(campaign_path, write=False))
    require(not h["errors"] and not h["unresolved_submission_intents"], "health requires reconciliation")
    covered = [uid]
    bulk = []
    original_hashes = {r["sha256"] for r in draws}
    model = load(c["models"][key])
    for row in h["scientific_jobs"]:
        if row["model"] != key or row["stage"] != "rate" or row["job_id"] == jid:
            continue
        card = load(row["card_path"])
        require(card["campaign_sha256"] == c["sha256"] and card["full_frame_sha256"] == frame["sha256"] and
            card["reader_executor"] == executor and Path(card["output_directory"]).resolve() ==
                (root / key / "eligibility-ratings").resolve(), "bulk deployment/frame/output differs")
        submission = load(row["submission_path"])
        require(str(submission["job_id"]) == row["job_id"] and
            submission["job_card_sha256"] == card["sha256"], "bulk accepted submission differs")
        shape = card["reader_input_shape"]
        expected = reader.probe_inputs(campaign_path, frame_path, "eligibility", shape["begin"], shape["count"])
        require(shape == expected and card["selected_uids"] == shape["uids"], "bulk original input shape differs")
        price = load(card["price_path"])
        component = price["components"]["eligibility_reader"]
        require(price["sha256"] == card["price_sha256"] and price["model_sha256"] == model["sha256"] and
            price["reader_executor"] == executor and price["code_files"] == c["code_files"] and
            component["status"] == "MEASURED" and component["max_context_input_tokens"] ==
                previous["retained_reader_input_shape"]["max_input_tokens"] and
            shape["max_input_tokens"] <= component["max_context_input_tokens"], "unqualified bulk role/context price")
        require({s["receipt_sha256"] for s in component["samples"]} == original_hashes and
            all(load(s["receipt_path"])["sha256"] == s["receipt_sha256"] and
                s["execution_sha256"] == c["runtime_sha256"] and
                s["reader_executor_sha256"] == executor["sha256"] for s in component["samples"]),
            "role price samples differ from original retained draws")
        require(row.get("queue") and row["queue"]["state"] in {"RUNNING", "PENDING"},
            "fresh live accepted bulk job required before full-audit closeout")
        covered.extend(card["selected_uids"])
        bulk.append(dict(job_id=row["job_id"], card_path=row["card_path"], card_sha256=card["sha256"],
            submission_path=row["submission_path"], submission_sha256=submission["sha256"],
            begin=shape["begin"], count=shape["count"], selected_uids=card["selected_uids"],
            max_input_tokens=shape["max_input_tokens"], resources=card["resources"],
            price_path=card["price_path"], price_sha256=price["sha256"], observed_queue=row["queue"]))
    require(bulk and len(covered) == len(set(covered)) == len(frame["records"]) and
        set(covered) == {r["uid"] for r in frame["records"]}, "original frame coverage incomplete or overlapping")
    successor = str(handoff["successor_job_id"])
    live = {r["job_id"]: r for r in h["controllers"] if r.get("queue")}
    require(successor in live and live[successor]["operational_version"] == "autodriver-r5", "successor not live")
    guards = sorted([j for j, r in live.items() if r["mode"] == "rescue" and
        successor in r["dependency"].split(":")[1:]], key=int)
    require(len(guards) == 2, "both continuation failure guards required")
    protection = {}
    for job in [successor, *guards]:
        r = live[job]
        require(r["queue"]["reason"] != "(JobHeldUser)" and
            load(r["prearm_path"])["sha256"] == r["prearm_sha256"] and
            load(r["release_path"])["sha256"] == r["release_sha256"], "unreleased/unbound continuation")
        protection[job] = {k: r[k] for k in ["dependency", "queue", "prearm_sha256", "release_sha256"]}
    require({r["job_id"] for r in bulk}.issubset(set(handoff["wait_job_ids"])), "handoff omits bulk allocations")
    readiness = load(base / "AUTOMATION_READINESS-r5.json")
    check_files(c["code_files"])
    check_files(readiness["code_files"])
    value = dict(schema="verification-reader-advancement-v3", campaign_sha256=c["sha256"], model=key,
        reporting_source_files={str(Path(__file__).resolve()): file_sha(__file__),
            str(Path(__file__).with_name("closeout_verification_discovery_r1.py")): file_sha(Path(__file__).with_name("closeout_verification_discovery_r1.py"))},
        discovery_advancement_sha256=previous["sha256"], audit_frame_sha256=frame["sha256"],
        audit_prefixes=len(frame["records"]), original_measurement_index=index, measurement_uid=uid,
        measurement_job_id=jid, measurement_elapsed_seconds=actual[jid]["elapsed_seconds"],
        measurement_actual_allocated_billing_core_hours=actual[jid]["actual_allocated_billing_core_hours"],
        measurement_draws=draws, measurement_annotation_output_tokens=sum(r["output_tokens"] for r in draws),
        measurement_runtime_sha256=runtime["sha256"], measurement_invocation_result_sha256=invocation["sha256"],
        cold_load_seconds=runtime["load_seconds"], measured_role_context_input_tokens=draws[0]["input_tokens"],
        bulk_reader_jobs=bulk, bulk_billing_core_hours_max=sum(r["resources"]["billing_core_hours_max"] for r in bulk),
        all_original_prefixes_covered_once=True, accounting=actual, controller_job_id=controller_id,
        controller_elapsed_seconds=actual[controller_id]["elapsed_seconds"], controller_result_sha256=result["sha256"],
        handoff_sha256=handoff["sha256"], successor_job_id=successor, released_rescue_job_ids=guards,
        continuation_protection=protection, observed_health_sha256=h["sha256"], observed_health_epoch=h["observed_epoch"],
        scientific_files_verified=len(c["code_files"]), operational_readiness_files_verified=len(readiness["code_files"]),
        semantic_qualification="NOT_MEASURED; original full audit remains required",
        new_causal_and_utility_effects="NOT_MEASURED", additional_manual_submissions=0)
    proof = save(output, value)
    require(load(output) == proof, "price advancement readback differs")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    p.add_argument("--model", required=True)
    p.add_argument("--controller-job-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = closeout(a.campaign, a.model, a.controller_job_id, a.output)
    print(json.dumps({k: r[k] for k in ["sha256", "model", "measurement_elapsed_seconds",
        "measured_role_context_input_tokens", "bulk_billing_core_hours_max", "successor_job_id"]}))


if __name__ == "__main__":
    main()
