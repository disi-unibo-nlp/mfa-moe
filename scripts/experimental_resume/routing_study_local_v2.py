"""Additive v2 preparation and finite CPU analyses. No submission side effects.

`prepare` is bounded metadata orchestration on LEONARDO. Context fitting and
187,524-row label reconciliation require an allocated CPU Slurm step. Existing
queued work and every historical bound source remain read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from moe_exp.routing_control.context_v2 import nested_context_model
from moe_exp.routing_control.anticipation_v2 import analyze_anticipation
from moe_exp.routing_control.families_v2 import family_boundary
from moe_exp.routing_control.study_v1 import file_sha, require, save as historical_save, sealed
from moe_exp.routing_control.study_v2 import protocol

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / "report/routing-study-v1"
OUT = REPO / "report/routing-study-v2"


def save(path, value):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root) for root in (Path("/leonardo/home/userexternal/lmolfett"),
            Path("/leonardo_work/IscrC_MIOSR/lmolfett"))), "output outside authorized owned directories")
    if "sha256" in value:
        from moe_exp.routing_control.study_v2 import verify
        verify(value)
    return historical_save(path, {k: v for k, v in value.items() if k != "sha256"})


def identity(compute=False):
    host = socket.gethostname()
    import getpass
    require(getpass.getuser() == "lmolfett" and ("leonardo" in host or
            compute and os.environ.get("SLURM_CLUSTER_NAME") == "leonardo"), "wrong cluster/user")
    if compute:
        require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and not host.startswith("login"),
                "analysis requires allocated CPU Slurm step")
    return host


def reference(path, *, check_seal=True):
    return {"path": str(path), "sha256": sealed(path)["sha256"] if check_seal else None,
            "file_sha256": file_sha(path)}


def prepare(verification=None):
    host = identity()
    spec = save(OUT / "EXECUTION_REQUIREMENTS.json", protocol())
    families = sealed(OLD / "INDEPENDENT_FAMILY_MANIFEST.json")
    require(families["status"] == "FROZEN" and len(families["validation"]) == 32 and len(families["final"]) == 96,
            "independent reservation changed")
    boundary = save(OUT / "FAMILY_BOUNDARY.json", family_boundary(
        sealed(REPO / "report/experimental-resume-v1/family-freeze.json"), families))
    chain = sealed(OLD / "submissions-v5/RECOVERY_CHAIN.json")
    inventory = sealed(OLD / "STUDY_INVENTORY_v3.json")
    jobs = chain["jobs"] + [chain["analysis_job"], "59495690"] + inventory["operator_pilot_jobs"] + inventory["utility_pilot_jobs"]
    command = ["sacct", "-X", "-nP", "-j", ",".join(sorted(set(jobs))),
               "--format=JobID,State%40,ExitCode,ElapsedRaw,AllocTRES%100"]
    observed = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    rows = []
    for line in observed.splitlines():
        if line.strip():
            job, state, code, elapsed, tres, *_ = line.split("|")
            rows.append({"job_id": job, "state": state, "exit_code": code, "elapsed_seconds": int(elapsed), "TRES": tres})
    sources = {name: reference(OLD / name) for name in ("STUDY_INVENTORY_v3.json", "INDEPENDENT_FAMILY_MANIFEST.json",
              "HISTORICAL_REPRODUCTION.json", "NATIVE_BOUNDARY_PROFILES.json", "REPORT_v4_BINDING.json")}
    # Stream hashes of larger historical tables; never load or refit them on a login node.
    sources.update({name: reference(OLD / name, check_seal=False) for name in
                    ("CANDIDATE_TABLE.json", "DENSE_INVENTORY.json", "HISTORICAL_INTEGRATION.json")})
    bound = {}
    for path in (OLD / "dense-recovery-v3/MANIFEST.json", OLD / "submissions-v5/RECOVERY_CHAIN.json"):
        artifact = sealed(path)
        bound.update(artifact.get("code_files", {}))
        bound.update(artifact.get("binding", {}).get("code_files", {}))
    require(all(file_sha(path) == sha for path, sha in bound.items()), "queued bound source changed")
    audit_rows = []
    for name in ("TRANSITION_DETECTOR_V2_DISCOVERY_AUDIT.json", "MECHANISM_DETECTOR_AUDIT_v1.json",
                 "MECHANISM_EXTENSION_DETECTOR_AUDIT_v1.json"):
        path = REPO / "report/experimental-resume-v1" / name
        audit = sealed(path)
        audit_rows.append({"reference": reference(path), "schema": audit["schema"],
            "scope": audit.get("scope", audit.get("interpretation")), "families": audit["families"],
            "rated_sample_counts": audit.get("rated_sample_counts"),
            "v2_qualification_status": "INCOMPLETE_SIX_DECISION_PREFIX_SEMANTICS_HELD_OUT_AND_SYMBOLIC_AUDITS"})
    prefix_audits = save(OUT / "PREFIX_AUDIT_INVENTORY.json", {"schema": "routing-prefix-audit-inventory-v2",
        "existing_audits": audit_rows, "status": "UNAVAILABLE_V2_RECOGNIZER_QUALIFICATION",
        "reuse": "Detector replay alone does not prove semantic readers saw no future text"})
    candidates = save(OUT / "CANDIDATE_MANIFEST.json", {"schema": "routing-study-candidates-v2",
        "protocol_sha256": spec["sha256"], "historical_candidates": sources["CANDIDATE_TABLE.json"],
        "selected_new_singletons": [], "status": "UNAVAILABLE_PREFIX_RECOGNITION_QUALIFICATION",
        "maximum_new_singletons": 2, "routing_context_model": "UNAVAILABLE_FOLD_LOCAL_FIT_AND_QUALIFIED_PREDICTION_FRAME",
        "prefix_audit_inventory_sha256": prefix_audits["sha256"],
        "confidence": "UNAVAILABLE_NATIVE_SEMANTICS", "failed_check_revision": "UNAVAILABLE_NEW_QUALIFYING_EVIDENCE"})
    experiments = save(OUT / "EXPERIMENT_MANIFEST.json", {"schema": "routing-study-v2-stage-inventory",
        "protocol_sha256": spec["sha256"], "independent_family_manifest_sha256": families["sha256"],
        "source_revision": families["source_revision"], "assignments": [],
        "stages": [{"stage": stage, "status": "UNAVAILABLE_QUALIFICATION_AND_MEASURED_PRICE",
                    "continuation_cap": 16384, "early_endpoint_window": 1024} for stage in
                   ("target", "operator", "timing", "context", "composition", "native_validation", "validation", "final")],
        "validation_families": 32, "final_families": 96, "maximum_final_assignments": 768,
        "incumbent": {"actions": [{"layer": 28, "experts": [9, 189], "operator": "force", "strength": 0},
                                  {"layer": 24, "experts": [159], "operator": "bias", "strength": 1}],
                      "controller_status": "REQUIRES_QUALIFICATION_AND_FREEZE_BEFORE_VALIDATION", "utility": "UNPROVEN"},
        "new_submissions": [], "historical_jobs_resubmitted": False})
    state = save(OUT / "STUDY_INVENTORY.json", {"schema": "routing-study-inventory-v2",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "hostname": host, "user": "lmolfett",
        "protocol_sha256": spec["sha256"], "candidate_manifest_sha256": candidates["sha256"],
        "family_boundary_sha256": boundary["sha256"],
        "experiment_manifest_sha256": experiments["sha256"], "sources": sources,
        "historical_dense_assigned": 187524, "dense_accounting": "QUEUED_RECOVERY_AND_EXACT_CLOSEOUT_NOT_YET_COMPLETE",
        "dense_chain_sha256": chain["sha256"], "accounting": {"command": command, "rows": rows},
        "protected_code_files": bound, "native_finalization": inventory["native_finalization"]["status"],
        "counterfactual_NLL": "FAILED_QUALIFICATION", "X3": "DEFERRED", "new_outcomes_observed": 0})
    report = OUT / "REPORT.md"
    require(not report.exists(), "v2 report already exists; use a new immutable output version")
    content = "# Routing study v2: 16k local-context execution\n\n"
    content += ("The implementation is additive: every new scientific continuation has a 16,384-token cap and a separate "
                "first-1,024-token mechanism endpoint. Historical 256/1,024-token results retain their original inference. "
                "No new target or independent outcome is qualified yet.\n\n")
    content += f'Observed on {state["observed_utc"]} as lmolfett on {host}. '
    content += "Existing recovery, operator and high-cap pricing jobs are preserved. Their current states are:\n\n"
    content += "| Job | State | Exit |\n|---|---|---|\n"
    content += "".join(f'| {r["job_id"]} | {r["state"]} | {r["exit_code"]} |\n' for r in rows)
    content += ("\nAll 187,524 dense identities remain bound to the historical frame. Complete identity reconciliation is pending "
                "the already queued recovery/closeout. `dense-ledger` creates a new identity ledger from committed receipts "
                "and explicit unresolved assignments; it never resubmits work. Pending calls are not outcomes.\n\n"
                "Existing discovery and mechanism detector audits have been indexed in PREFIX_AUDIT_INVENTORY.json. "
                "The mechanism audits contain lexical proposals, and the older discovery audit does not qualify the six "
                "required semantic decisions, held-out coverage and symbolic candidates. No audit is silently promoted "
                "to a v2 prefix recognizer.\n\n"
                "The v2 code implements fold-local eight-component PCA and K=1/2/4 clustering, nested family prediction, "
                "ten-family association support and the three-outer-fold gain rule. Multiple contexts fall back to K=1 "
                "when support fails. Native ID composition and saved execution weights remain separate signals. Actual "
                "local-context fits and 64/256-token anticipation remain unavailable until their prediction frame qualifies.\n\n"
                "The first qualified opportunity is final for each transition: context declines and random skips cannot "
                "retry later. Skip probabilities are frozen from discovery strata of source state, prefix status and "
                "position. Delayed early acts, closure, ambiguous decisions and overlapping composition pulses retain no-fire "
                "receipts. Controller generation is charged to policy cost. Snapshots bind request, prefix, controller "
                "ledger, capture and exact sampler/engine state.\n\n"
                "New manifests bind recognizer, context, sampler/resume, all-40-layer ID/selected-weight capture, 16k cap, "
                "early endpoint and immutable code. Target/operator/timing/context effects and achieved dose remain unobserved. "
                "Exploratory advancement requires verified matched dose, positive semantic point effects versus both controls, "
                "three positive family folds and reported simultaneous intervals; intervals may remain inconclusive.\n\n"
                "The existing 32/96 family reservation, two seeds and source revision are unchanged. Native validation runs "
                "first. State-only and context-aware counterparts use the same actions and settings. The incumbent must "
                "qualify before validation; its forced L28/E9+E189 and +1 L24/E159 actions remain fixed and unproven. Final "
                "enrollment is at most 768 assignments. Accuracy–cost frontiers remain unavailable.\n\n"
                "Measured stage prices require generation, prefill, model loads, controller, replay, capture/storage, annotation, "
                "grading, runtime and bounded recovery. Short or mismatched pilots cannot price 16k cells. No new jobs "
                "are submitted. Stage resource authorization follows a complete measured price.\n\n"
                "Native-finalization qualification failure, failed counterfactual-NLL qualification and deferred X3 remain "
                "separate limitations; qualified free generation can proceed independently. CPU tests validate contracts "
                "and information boundaries; GPU fidelity, capture, RNG continuity and semantic recognition still require "
                "observed passing receipts.\n\n"
                "The executable request loop requires a synchronous qualified backend that preserves its live sampler "
                "and supplies all-layer selected weights. Concrete vLLM integration and qualification for that interface "
                "remain outstanding.\n\n"
                "The following historical report is preserved evidence from its original observation time. Its progress "
                "counts are historical snapshots; the job table above is the current scheduler observation.\n\n")
    if verification is not None:
        checks = sealed(verification)
        require(checks.get("status") == "PASS_CPU_IMPLEMENTATION" and
                all(file_sha(p) == sha for p, sha in checks["code_files"].items()), "implementation verification changed")
        content += f'CPU implementation verification: {checks["tests_passed"]} tests passed. '
        content += "Historical and queued bound source hashes were verified unchanged. GPU and semantic qualification remain pending.\n\n"
    historical_relative = os.path.relpath(OLD, OUT)
    content += (OLD / "REPORT_v4.md").read_text().replace("](", f"]({historical_relative}/")
    report.write_text(content)
    save(OUT / "REPORT_BINDING.json", {"schema": "routing-study-report-binding-v2", "inventory_sha256": state["sha256"],
        "report_file_sha256": file_sha(report), "code_files": {str(p): file_sha(p) for p in
            [Path(__file__).resolve(), *sorted((REPO / "src/moe_exp/routing_control").glob("*v2.py"))]},
        "implementation_verification": reference(verification) if verification else None,
        "status": "IMPLEMENTED_INTERFACES_PENDING_SCIENTIFIC_QUALIFICATION_AND_AUTHORIZATION"})
    print(json.dumps({"status": "PREPARED", "report": str(report), "submitted": 0}), flush=True)


def dense_ledger(directory, output):
    identity(True)
    import dense_recovery_v2 as recovery
    _, _, _, frame, price = recovery.validate(directory)
    records = []
    for index, shard in enumerate(price["shards"]):
        path = directory / f"labels-shard-{index:03d}"
        binding = sealed(path / "BINDING.json")
        for start in range(shard["start"], shard["stop"], recovery.native.BATCH):
            stop = min(start + recovery.native.BATCH, shard["stop"])
            batch = recovery.native.committed_labels(path, start, stop, binding, price["prompt_records"])
            observed = batch["records"] if batch else [None] * (stop - start)
            attempt = path / "attempts" / f"{start:07d}.json"
            attempt_sha = sealed(attempt)["sha256"] if attempt.exists() else None
            for assignment, label in zip(frame["records"][start:stop], observed, strict=True):
                require(label is None or label["blind_id"] == assignment["blind_id"], "dense label identity changed")
                records.append({"blind_id": assignment["blind_id"], "status": "COMMITTED_VALID" if label and
                    label["finish_reason"] == "stop" and label["label"] in recovery.native.qualified.CLASSES else
                    "COMMITTED_UNUSABLE" if label else "UNRESOLVED_ATTEMPT" if attempt_sha else "UNATTEMPTED",
                    "batch_sha256": batch["sha256"] if batch else None, "attempt_sha256": attempt_sha,
                    "receipt_path": str(path / "batches" / f"{start:07d}.json") if batch else None,
                    "label": label["label"] if label else None, "finish_reason": label["finish_reason"] if label else None})
    require(len(records) == len({r["blind_id"] for r in records}) == frame["sentences"] == 187524,
            "dense accounting must retain every assigned identity exactly once")
    save(output, {"schema": "routing-study-dense-identity-ledger-v2", "frame_sha256": frame["sha256"],
        "assigned": 187524, "records": records, "source_directory": str(directory),
        "observed_utc": datetime.now(timezone.utc).isoformat(), "missing_labels_break_adjacency": True})


def main():
    global OUT
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output-dir", type=Path, default=OUT)
    preparation.add_argument("--verification", type=Path)
    context = commands.add_parser("context")
    context.add_argument("--frame", type=Path, required=True)
    context.add_argument("--output", type=Path, required=True)
    anticipation = commands.add_parser("anticipation")
    anticipation.add_argument("--frame", type=Path, required=True)
    anticipation.add_argument("--output", type=Path, required=True)
    dense = commands.add_parser("dense-ledger")
    dense.add_argument("--source", type=Path, default=OLD / "dense-recovery-v3")
    dense.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        OUT = args.output_dir.resolve()
        prepare(args.verification)
    elif args.mode == "dense-ledger": dense_ledger(args.source, args.output)
    else:
        identity(True)
        frame = sealed(args.frame)
        require(frame.get("schema") == "routing-context-prediction-frame-v2", "wrong prediction frame")
        result = (analyze_anticipation(frame) if args.mode == "anticipation" else
                  nested_context_model(frame["rows"], frame["baseline"], frame["selection"], family_boundary=frame["family_boundary"]))
        save(args.output, {**{k: v for k, v in result.items() if k != "sha256"}, "frame_sha256": frame["sha256"]})


if __name__ == "__main__":
    main()
