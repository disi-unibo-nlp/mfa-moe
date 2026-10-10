"""Close out a completed, controller-derived v3 reader audit without dispatch.

Reporting helper outside the frozen executor/operational closure. Requires the
full original audit, successful terminal accounting and a saved independent
review. No inference, scientific writes, threshold changes or job submissions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/experimental_resume")]
from moe_exp.verification_study.contracts import load, save, require, check_files, file_sha
from moe_exp.verification_study import semantics
import verification_study_v3_reader_r1 as reader
import verification_study_v3_semantic_format_r1 as fmt
import verification_study_v3_health_status_r4 as health


def accounting(ids):
    require(ids and all(re.fullmatch(r"[0-9]+", str(i)) for i in ids), "unsafe job IDs")
    result = subprocess.run(["sacct", "-X", "-nP", "-j", ",".join(sorted(ids)),
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,Partition,Account"],
        check=True, capture_output=True, text=True, timeout=30)
    rows = {}
    for line in result.stdout.splitlines():
        v = line.split("|")
        if v[0] not in ids:
            continue
        require(v[0] not in rows and len(v) >= 7 and v[6] == "iscrc_miosr", "foreign accounting")
        rows[v[0]] = {"state": v[1], "exit_code": v[2], "elapsed_seconds": int(v[3]),
            "allocated_cpus": int(v[4]), "partition": v[5], "account": v[6]}
    require(set(rows) == set(ids), "missing accounting")
    return rows


def run(campaign_path, key, controller_id, review_path, output):
    campaign_path = campaign_path.resolve()
    c = load(campaign_path)
    check_files(c["code_files"])
    rd = load(campaign_path.parent / "AUTOMATION_READINESS-r5.json")
    check_files(rd["code_files"])
    root = Path(c["result_root"])
    frame_path = root / key / "AUDIT_FRAME.json"
    directory = root / key / "eligibility-ratings"
    _, contract, frame, executor = reader.deployment(campaign_path, frame_path)
    require(len(frame["records"]) == 128, "full original frame required")
    qpath = root / key / "RECOGNIZER.json"
    q = load(qpath)
    aggregate = load(q["semantic_decision_ledger_path"])
    require(aggregate["sha256"] == q["semantic_decision_ledger_sha256"] and
        aggregate["campaign_sha256"] == c["sha256"] and aggregate["model"] == key,
        "recognizer semantic action binding differs")
    binding = fmt.load_binding(campaign_path)
    require(q["semantic_format_binding_sha256"] == aggregate["semantic_format_binding_sha256"] ==
            binding["sha256"], "format annex differs")
    raw_reference = aggregate["decision_ledgers"][key]["eligibility"]
    ledger = load(raw_reference["path"])
    require(ledger["sha256"] == raw_reference["sha256"] and
        fmt.decision_ledger(campaign_path, frame_path, directory, "eligibility") == ledger,
        "original raw draw ledger does not reproduce")
    ratings = [reader.validate_rating(directory / (r["uid"] + f".reader-{d}.json"),
        r, d, "eligibility", c, contract, frame, executor) for r in frame["records"] for d in (0, 1)]
    require(len(ratings) == 256, "full original draws required")
    with fmt.installed_parser():
        original = semantics.qualify_recognizer(frame, ratings)
    require(all(q[k] == v for k, v in original.items() if k != "sha256"),
        "original recognition qualification does not reproduce")
    invocations = {}
    for draw in ratings:
        ip = Path(draw["invocation_runtime_path"])
        if str(ip) not in invocations:
            runtime = load(ip)
            result = load(ip.with_name("RESULT.json"))
            require(result["status"] == "COMPLETE" and result.get("error") is None and
                result["job_id"] == draw["job_id"] and result["frame_sha256"] == frame["sha256"] and
                result["campaign_sha256"] == c["sha256"] and result["role"] == "eligibility" and
                result["reader_executor_sha256"] == executor["sha256"] and
                result["slice"] == runtime["slice"] and result["selected_uids"] == runtime["selected_uids"],
                "reader invocation is incomplete or foreign")
            invocations[str(ip)] = {"runtime_sha256": runtime["sha256"],
                "result_sha256": result["sha256"], "job_id": draw["job_id"],
                "slice": runtime["slice"], "selected_uids": runtime["selected_uids"]}
    selected = Counter(uid for v in invocations.values() for uid in v["selected_uids"])
    require(selected == Counter(r["uid"] for r in frame["records"]),
        "invocation slices must cover the original full frame once without overlap")
    job_counts = Counter(str(r["job_id"]) for r in ratings)
    ar = accounting(set(job_counts) | {controller_id})
    for jid in set(job_counts) | {controller_id}:
        require(ar[jid]["state"] == "COMPLETED" and ar[jid]["exit_code"] == "0:0",
                "closeout requires successful terminal accounting")
    for jid, count in job_counts.items():
        ar[jid]["draws"] = count
        ar[jid]["actual_allocated_billing_core_hours"] = (
            ar[jid]["elapsed_seconds"] * ar[jid]["allocated_cpus"] / 3600)
    controller = load(root / "automation/runs" / controller_id / "RESULT.json")
    require(controller["campaign_sha256"] == c["sha256"] and
        controller["controller_job_id"] == controller_id and not controller["frontier"]["errors"],
        "controller closeout is foreign or has errors")
    h = health.observe(campaign_path, write=True)
    require(not h["errors"] and not h["unresolved_submission_intents"], "unresolved operational health")
    science_rows = {row["job_id"]: row for row in h["scientific_jobs"]}
    card_proofs = {}
    for jid in job_counts:
        row = science_rows[jid]
        card = load(row["card_path"])
        submission = load(row["submission_path"])
        shape = card["reader_input_shape"]
        expected = [r["uid"] for r in frame["records"][shape["begin"]:shape["begin"]+shape["count"]]]
        inv = [v for v in invocations.values() if v["job_id"] == jid]
        require(card["sha256"] == row["job_card_sha256"] and card["campaign_sha256"] == c["sha256"] and
            card["model"] == key and card["full_frame_sha256"] == frame["sha256"] and
            card["selected_uids"] == shape["uids"] == expected and shape["role"] == "eligibility" and
            len(inv) == 1 and inv[0]["selected_uids"] == expected and
            inv[0]["slice"] == [shape["begin"],shape["count"]] and
            str(submission["job_id"]) == jid and job_counts[jid] == 2*shape["count"],
            "reader job/card/submission/slice coverage differs")
        card_proofs[jid] = {"card_path":row["card_path"],"card_sha256":card["sha256"],
            "submission_path":row["submission_path"],"submission_sha256":submission["sha256"],
            "begin":shape["begin"],"count":shape["count"]}
    handoff = load(root / "automation/runs" / controller_id / "HANDOFF.json")
    successor = str(handoff["successor_job_id"])
    live = {x["job_id"]: x for x in h["controllers"] if x.get("queue")}
    require(successor in live and live[successor]["operational_version"] == "autodriver-r5",
        "successor protection requires fresh live readback")
    guards = sorted([jid for jid, row in live.items() if row["mode"] == "rescue" and
        successor in row["dependency"].split(":")[1:]], key=int)
    require(len(guards) == 2, "successor requires both failure guards")
    protection = {}
    for jid in [successor, *guards]:
        row = live[jid]
        require(row["queue"]["reason"] != "(JobHeldUser)", "successor/guard not released")
        require(load(row["prearm_path"])["sha256"] == row["prearm_sha256"] and
            load(row["release_path"])["sha256"] == row["release_sha256"], "protection seals differ")
        protection[jid] = {k: row[k] for k in ["dependency", "queue", "prearm_sha256", "release_sha256"]}
    pairs = defaultdict(dict)
    for row in ledger["records"]:
        pairs[row["uid"]][row["reader"]] = row
    usable_pairs = sum(all(p[d]["effective_parsed"] is not None for d in (0, 1)) for p in pairs.values())
    breakdown = {}
    for decision in ["eligible", "already_checked"]:
        for split in ["apparent", "held_out"]:
            proposed = [r for r in frame["records"] if r["split"] == split and r["proposed"][decision]]
            counts = Counter()
            for record in proposed:
                pair = [pairs[record["uid"]][d] for d in (0, 1)]
                if any(p["finish_reason"] != "stop" for p in pair):
                    status = "NONSTOP_UNKNOWN"
                elif any(p["effective_parsed"] is None for p in pair):
                    status = "UNPARSEABLE_UNKNOWN"
                else:
                    values = [p["effective_parsed"][decision] for p in pair]
                    status = ("AMBIGUOUS_UNKNOWN" if any(v is None for v in values) else
                        "DISAGREE_UNKNOWN" if values[0] != values[1] else
                        "CONFIRMED" if values[0] is True else "REJECTED")
                counts[status] += 1
            breakdown[decision + ":" + split] = {"proposal_rows": len(proposed),
                "proposal_families": len({r["family"] for r in proposed}), "pair_decisions": dict(counts)}
    review_path = review_path.resolve()
    require(review_path.is_file(), "saved independent review required")
    prior_path = campaign_path.parent / (key.upper() + "_READER_ADVANCEMENT-r1.json")
    prior = load(prior_path)
    result = {"schema": "verification-eligibility-full-closeout-v3", "campaign_sha256": c["sha256"],
        "model": key, "original_discovery_assignments": 48, "original_prefixes": 128,
        "original_draws": 256, "all_original_draws_validated": True,
        "frame_path": str(frame_path), "frame_sha256": frame["sha256"],
        "qualification_path": str(qpath), "qualification_sha256": q["sha256"],
        "qualification_status": q["status"], "decisions": q["decisions"],
        "failure_reasons": q["failure_reasons"], "qualification_recomputed_exactly": True,
        "raw_decision_ledger_path": raw_reference["path"], "raw_decision_ledger_sha256": ledger["sha256"],
        "ledger_recomputed_exactly": True, "semantic_action_inputs_path": q["semantic_decision_ledger_path"],
        "semantic_action_inputs_sha256": aggregate["sha256"], "semantic_format_binding_sha256": binding["sha256"],
        "draw_receipts": ledger["source_receipts"], "reader_invocations": invocations, "reader_job_cards": card_proofs,
        "draw_finish_reasons": dict(Counter(r["finish_reason"] for r in ratings)),
        "draw_status": dict(Counter(r["status"] for r in ledger["records"])),
        "format_status": dict(Counter(r["format_status"] for r in ledger["records"])),
        "recovered_stopped_fenced_draws": sum(r["recovered_stop"] for r in ledger["records"]),
        "both_draws_parseable_prefixes": usable_pairs, "proposal_pair_decisions": breakdown,
        "annotation_output_tokens": sum(r["output_tokens"] for r in ratings),
        "successful_reader_allocations": {jid: ar[jid] for jid in job_counts},
        "actual_allocated_billing_core_hours": sum(ar[jid]["actual_allocated_billing_core_hours"] for jid in job_counts),
        "controller_job_id": controller_id, "controller_elapsed_seconds": ar[controller_id]["elapsed_seconds"],
        "controller_result_sha256": controller["sha256"], "controller_model_status": controller["frontier"]["models"][key],
        "handoff_sha256": handoff["sha256"], "successor_job_id": successor,
        "released_rescue_job_ids": guards, "continuation_protection": protection,
        "observed_health_sha256": h["sha256"], "scientific_files_verified": len(c["code_files"]),
        "operational_readiness_files_verified": len(rd["code_files"]),
        "supersedes_pending_full_audit_in": prior_path.name, "preserves_reader_advancement_sha256": prior["sha256"],
        "additional_inference_draws": 0, "additional_manual_submissions": 0,
        "sol_review_path": str(review_path), "sol_review_file_sha256": file_sha(review_path),
        "closeout_helper_file_sha256": file_sha(Path(__file__).resolve()),
        "scope": "Completed qualification under the unchanged seven-model gates; causal/utility outcomes are unmeasured when recognition fails."}
    require(not output.exists(), "immutable closeout already exists")
    sealed = save(output, result)
    require(load(output) == sealed, "closeout readback differs")
    print(json.dumps({k: sealed[k] for k in ["model", "sha256", "qualification_status", "draw_status",
        "both_draws_parseable_prefixes", "annotation_output_tokens", "actual_allocated_billing_core_hours",
        "proposal_pair_decisions", "successor_job_id", "released_rescue_job_ids"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=REPO / "report/verification-study-v3/CAMPAIGN.json")
    p.add_argument("--model", required=True)
    p.add_argument("--controller-job-id", required=True)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.campaign, a.model, a.controller_job_id, a.review, a.output)

if __name__ == "__main__":
    main()
