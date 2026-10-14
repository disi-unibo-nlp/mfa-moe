#!/usr/bin/env python3
"""One CPU-only recovery of r5-r2 validation from its committed import.

The bound campaign sources are never changed. Regression success is recovered
from the original sealed job/log, not fabricated by mocking a test runner.
All scientific identity joins and remaining CPU graph checks run again.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))

from moe_exp.research_campaign_v3_r5_r2.contracts import (
    MODELS, Units, allocated, check_files, digest, execution_id, file_sha, load,
    locked, private_path, publish, require, seal,
)
from moe_exp.research_campaign_v3_r5_r2.scheduler import (
    Slurm, card, serial_capacity, submit_card,
)


class RecoverySlurm(Slurm):
    def submit(self, value, *, test_only=False):
        require(value["stage"] == "validate" and value["attempt"] == 1 and
                value["gpus"] == 0 and value["cpus"] == 4 and
                value["memory_gib"] == 24 and value["minutes"] == 30,
                "recovery cannot change its authorized CPU envelope")
        check_files(value["validation_recovery_files"])
        args = ["sbatch", "--test-only" if test_only else "--parsable",
                "--account=iscrc_miosr", "--qos=normal", "--partition=lrd_all_serial",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=24G",
                "--time=30", "--no-requeue", "--job-name=rc3-validate-recovery",
                "--comment=" + value["comment"], "--chdir=" + str(REPO),
                "--output=" + value["root"] + "/logs/%x-%j.out",
                "--error=" + value["root"] + "/logs/%x-%j.err",
                str(REPO / "scripts/research_campaign_v3_r5_validation_recovery.sbatch"),
                str(Path(value["root"]) / "cards" / (value["sha256"] + ".json"))]
        response = self.command(args)
        if test_only:
            return {"test_only": True, "response": response}
        job = response.split(";", 1)[0]
        require(job.isdigit(), "ambiguous recovery submission")
        return {"job_id": job, "response": response}


def submit(campaign):
    from moe_exp.research_campaign_v3_r5_r2.controller import _check, jobs
    root = Path(campaign["root"])
    path = root / "VALIDATION_RECOVERY_LAUNCH.json"
    if path.exists():
        return load(path)  # Never allocate another recovery.
    check_files(campaign["code_files"])
    scheduler = RecoverySlurm()
    original = load(root / "VALIDATION_LAUNCH.json")
    accounting = scheduler.observe(original["allocation"]["job_id"])
    require(accounting["terminal"] and accounting["state"] == "TIMEOUT",
            "recovery requires the original interrupted deterministic operation")
    require(not (root / "READINESS.json").exists(), "validation already complete")
    log = root / "logs" / ("rc3-validate-" + accounting["job_id"] + ".err")
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK(?:\s|$)", log.read_text())
    require(match and int(match[1]) == 102, "original regression success is unproven")
    imported = load(root / "EVIDENCE_IMPORT.json")
    require(imported["status"] == "complete" and
            len(imported["validated_downstream_interfaces"]) == 7,
            "actual import did not commit")
    files = {str(p): file_sha(p) for p in (Path(__file__),
        REPO / "scripts/research_campaign_v3_r5_validation_recovery.sbatch")}
    binding = publish(root / "VALIDATION_RECOVERY_BINDING.json", {
        "campaign_sha256": campaign["sha256"], "original": original,
        "original_accounting": accounting,
        "regression_log": {"path": str(log), "sha256": file_sha(log)},
        "tests_run": int(match[1]), "regression_seconds": float(match[2]),
        "import_sha256": imported["sha256"], "files": files,
        "allowed_attempts": 1, "scientific_generations_or_draws": 0,
        "reservation_local_hours": 2., "reservation_with_margin": 2.5,
        "failure_disposition": "explicit terminal validation failure; no successor"})
    raw = {k: v for k, v in card(campaign, "validate", attempt=1, minutes=30).items()
           if k not in ("sha256", "comment")}
    raw.update(validation_recovery_files=files, validation_recovery_sha256=binding["sha256"])
    raw["comment"] = "rc3:" + digest(raw)
    value = seal(raw)
    snapshot = scheduler.snapshot()
    require(all(serial_capacity(snapshot, [value], handoff_slots=0).values()),
            "recovery exceeds live serial capacity")
    admission = _check(campaign, scheduler, jobs(campaign, scheduler),
                       hours=2., bytes_=512 * 1024**2)
    require(admission["admitted"], "bounded validation recovery cannot be admitted")
    accepted = submit_card(campaign, value, scheduler)
    require(accepted["readback"]["fields"]["Command"] ==
            str(REPO / "scripts/research_campaign_v3_r5_validation_recovery.sbatch"),
            "recovery command readback differs")
    return publish(path, {"allocation": accepted, "binding_sha256": binding["sha256"],
        "bounded_admission": admission, "scientific_work_released": False,
        "only_recovery_attempt": True})


def recover(campaign, value):
    allocated()
    started = time.monotonic()
    root = Path(campaign["root"])
    binding = load(root / "VALIDATION_RECOVERY_BINDING.json")
    require(value["validation_recovery_sha256"] == binding["sha256"] and
            value["attempt"] == 1 and binding["allowed_attempts"] == 1,
            "validation recovery binding differs")
    check_files(binding["files"])
    check_files(campaign["code_files"])
    check_files(campaign["source_files"])
    check_files({binding["regression_log"]["path"]: binding["regression_log"]["sha256"]})
    imported = load(root / "EVIDENCE_IMPORT.json")
    require(imported["sha256"] == binding["import_sha256"], "committed import differs")
    # Validate every original imported artifact; do not reconstruct/rename a draw.
    check_files({r["path"]: r["sha256"] for r in imported["references"]})
    for r in imported["references"]:
        require(file_sha(r["referenced_as"]) == r["sha256"], "import alias differs")
    publish(root / "VALIDATION_RECOVERY_START.json", {"job_id": os.environ["SLURM_JOB_ID"],
        "card_sha256": value["sha256"], "binding_sha256": binding["sha256"]})
    from moe_exp.research_campaign_v3_r5_r2.scoring import score
    from moe_exp.research_campaign_v3_r5_r2.annotation import prepare_contexts
    from moe_exp.research_campaign_v3_r5_r2.validation import traverse
    generation_count = shape_count = 0
    models = []
    for model in MODELS:
        memory = load(root / model / "NON_REAP_TASKS.json")
        require(len({r["uid"] for r in memory["records"]}) == len(memory["records"]),
                "execution aliases duplicated")
        count = sum(r["mask"] is not None for r in memory["records"])
        shapes = load(root / model / "NON_REAP_QUALIFICATION_TASKS.json")["records"]
        compact_count = sum(r["key"] != "native" for r in shapes)
        generation_count += count
        shape_count += compact_count
        sources = {r["attempt_id"]: r for r in load(root / model / "SOURCES.json")["records"]}
        for row in memory["records"]:
            original = sources[row["source"]["attempt_id"]]
            require(row["source"]["trace_sha256"] == original["trace_sha256"] and
                    row["uid"] == execution_id(row), "question/source/mask join differs")
        native = load(root / model / "NATIVE_TASKS.json")["records"]
        selected = max((r for r in native if r["source"]["prefix_tokens"]),
            key=lambda r: (r.get("prompt_tokens", 0) + r["source"]["prefix_tokens"], r["uid"]))
        score(campaign, model, [selected])
        suffix = "-native-validation-" + digest([selected["uid"]])[:16]
        context_path = root / model / ("CONTEXTS" + suffix + ".json")
        annotation_path = root / model / ("ANNOTATION_TASKS" + suffix + ".json")
        if not context_path.exists() or not annotation_path.exists():
            prepare_contexts(campaign, model, phase="native", selected_uids={selected["uid"]})
        contexts = load(context_path)
        annotations = load(annotation_path)
        require(contexts["phase"] == "native" and all(r["uid"] == selected["uid"]
            for r in contexts["records"]), "native validation sample identity differs")
        require(Units(root / model / "scores").state(selected["uid"])["status"] == "complete",
                "strict scoring has not committed")
        measure = load(root / model / "MEASURE_TASKS.json")
        require(len(measure["records"]) == 1547, "measurement manifest incomplete")
        models.append({"model": model, "non_reap_generations": count,
            "compact_shapes": compact_count, "native_contexts": len(contexts["records"]),
            "context_sha256": contexts["sha256"], "annotation_tasks_sha256": annotations["sha256"],
            "measure_manifest_sha256": measure["sha256"]})
        print("validated original joins, score and context:", model, flush=True)
    require(generation_count == 69459 and shape_count == 346, "authorized memory boundary differs")
    graph = traverse(campaign)  # Real CPU forecast and scheduler read-only traversal.
    proof = publish(root / "IMPORT_GRAPH_VALIDATION.json", {"status": "complete", "models": models,
        "non_reap_generations": generation_count, "compact_shapes": shape_count,
        "real_cpu_implementations_exercised": ["import", "masks", "identities", "strict_scoring",
            "native_contexts", "forecast", "graph"],
        "external_boundaries": ["scheduler submission prohibited", "GPU absent"],
        "graph": graph, "seconds": time.monotonic() - started,
        "job_id": os.environ["SLURM_JOB_ID"], "deterministic_recovery_sha256": binding["sha256"]})
    check_files(campaign["code_files"])
    check_files(binding["files"])
    readiness = publish(root / "READINESS.json", {"passed": True, "tests_run": binding["tests_run"],
        "files": campaign["code_files"], "source_unchanged_during_tests": True,
        "actual_import_graph_validated": True, "actual_validation_sha256": proof["sha256"],
        "seconds": time.monotonic() - started, "allocated_cpus": 4, "allocated_memory_gib": 24,
        "allocation_minutes": 30, "production_released": False,
        "regression_job_id": binding["original_accounting"]["job_id"],
        "recovery_files": binding["files"], "deterministic_recovery_sha256": binding["sha256"]})
    return publish(root / "VALIDATION_RECOVERY_RESULT.json", {"status": "complete",
        "job_id": os.environ["SLURM_JOB_ID"], "readiness_sha256": readiness["sha256"],
        "scientific_draws_repeated": False})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("submit", "worker"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--card", type=Path)
    args = parser.parse_args()
    value = load(args.card) if args.card else None
    root = private_path(args.campaign or value["root"])
    campaign = load(root / "CAMPAIGN.json")
    if args.command == "submit":
        result = submit(campaign)
        print(json.dumps({"job_id": result["allocation"]["job_id"], "sha256": result["sha256"]}))
    else:
        with locked(root, "validation-recovery"):
            require(not (root / "VALIDATION_RECOVERY_RESULT.json").exists(), "recovery exhausted")
            try:
                result = recover(campaign, value)
            except Exception as error:
                publish(root / "VALIDATION_RECOVERY_RESULT.json", {"status": "execution_failed",
                    "job_id": os.environ["SLURM_JOB_ID"], "error_type": type(error).__name__,
                    "error": str(error)[:4000], "terminal": True, "additional_recoveries": 0})
                raise
        print(json.dumps(result))


if __name__ == "__main__":
    main()
