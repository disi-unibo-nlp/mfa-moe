"""Finite after-workload inventory and report; no dispatch or background loop."""
from __future__ import annotations

import json
import os
from pathlib import Path

import dense_recovery_v2 as recovery
from moe_exp.routing_control.study_v1 import file_sha, require, save, sealed
from routing_study_v1 import compute, OUT, accounting


def final_inventory(directory):
    _manifest, _inventory, _packed_price, frame, price = recovery.validate(directory)
    complete, valid, unresolved = 0, 0, []
    for index, shard in enumerate(price["shards"]):
        out = directory / f"labels-shard-{index:03d}"
        binding = sealed(out / "BINDING.json")
        for start in range(shard["start"], shard["stop"], recovery.native.BATCH):
            stop = min(start + recovery.native.BATCH, shard["stop"])
            committed = recovery.native.committed_labels(out, start, stop, binding, price["prompt_records"])
            if committed is None:
                attempt = out / "attempts" / f"{start:07d}.json"
                unresolved.append({"shard": index, "start": start, "stop": stop,
                    "blind_ids": [r["blind_id"] for r in frame["records"][start:stop]],
                    "status": "ATTEMPT_WITHOUT_COMMITTED_OUTPUT" if attempt.exists() else "UNATTEMPTED",
                    "attempt_sha256": sealed(attempt)["sha256"] if attempt.exists() else None})
            else:
                complete += len(committed["records"])
                valid += sum(r["label"] in recovery.native.qualified.CLASSES and r["finish_reason"] == "stop"
                             for r in committed["records"])
    missing = sum(r["stop"]-r["start"] for r in unresolved)
    require(complete + missing == frame["sentences"] == 187524, "final label accounting lost assignments")
    return {"schema": "routing-study-exact-label-closeout-v1", "assigned": frame["sentences"],
        "committed_label_calls": complete, "valid_stopped_class_labels": valid,
        "committed_unusable_labels": complete - valid, "unresolved_assigned_labels": missing,
        "unresolved_inventory": unresolved, "frame_sha256": frame["sha256"],
        "status": "ALL_ASSIGNED_CALLS_COMMITTED" if not missing else "EXPLICIT_UNRESOLVED_ASSIGNMENTS",
        "class_measurement": "Unparsed or capped labels remain measurement gaps and break adjacency"}


def main():
    compute()
    directory = OUT / "dense-recovery-v3"
    chain = sealed(OUT / "submissions-v5/RECOVERY_CHAIN.json")
    inventory = final_inventory(directory)
    result = OUT / ("closeout-" + os.environ["SLURM_JOB_ID"])
    result.mkdir(exist_ok=True)
    labels = save(result / "LABEL_ACCOUNTING.json", inventory)
    jobs = chain["jobs"] + [chain["analysis_job"], "59405556", "59405561", "59405635", "59400762", "59400764", "59400766"]
    live = accounting(jobs)
    grades = sealed(OUT / "INDEPENDENT_FAMILY_MANIFEST.json")
    trajectory = directory / "TRAJECTORY_RESULT.json"
    summary = {"schema": "routing-study-finite-closeout-v1", "labels_sha256": labels["sha256"], "accounting": live,
        "source_chain_sha256": chain["sha256"], "independent_families_sha256": grades["sha256"],
        "dense_trajectory": {"path": str(trajectory), "sha256": sealed(trajectory)["sha256"]} if trajectory.exists() else None,
        "code_sha256": file_sha(__file__), "status": "STUDY_INCOMPLETE_PREFIX_AND_UTILITY_QUALIFICATION",
        "controller_and_independent_generations": "New stages require measured complete prices and concrete authorization"}
    seal = save(result / "STATUS.json", summary)
    base = OUT / "REPORT_v4.md"
    require(base.exists(), "current consolidated report unavailable")
    text = base.read_text() + f'\nAfter-workload exact label accounting: {inventory["committed_label_calls"]:,} committed calls; '
    text += f'{inventory["valid_stopped_class_labels"]:,} valid stopped labels; {inventory["committed_unusable_labels"]:,} '
    text += f'committed but unusable labels; {inventory["unresolved_assigned_labels"]:,} unresolved assigned labels. '
    text += "Every unresolved blind identity is listed in LABEL_ACCOUNTING.json. Saved trajectory analysis is "
    text += "available.\n" if trajectory.exists() else "unavailable or still unresolved.\n"
    report = result / "REPORT.md"
    report.write_text(text)
    save(result / "REPORT_BINDING.json", {"status_sha256": seal["sha256"], "report_file_sha256": file_sha(report)})
    print(json.dumps({"status": inventory["status"], "report": str(report), "unresolved_labels": inventory["unresolved_assigned_labels"]}), flush=True)


if __name__ == "__main__":
    main()
