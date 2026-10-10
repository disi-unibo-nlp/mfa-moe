"""Reconcile the existing scheduled question rosters before reserving fresh MATH.

No model execution, exposure-dependent ranking, dispatch or verification writes.
Prepared reservations are conservatively treated as exposed. The active v3
campaign's future draws use the same public catalog and reserved families.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from moe_exp.expert_memory_study.contracts import REPO, file_sha, load, require, save


def normalize(text):
    return re.sub(r"\s+", "", text).casefold()


def reconcile(output):
    root = REPO / "report"
    inventory = load(root / "routing-study-v1/STUDY_INVENTORY_v3.json")
    paths = [root / "routing-study-v1/STUDY_INVENTORY_v3.json",
             root / "routing-study-v2/verified/FAMILY_BOUNDARY.json",
             root / "routing-study-v2/implementation-20261006-v1/NATIVE_INPUTS.json",
             root / "verification-study-v3/PUBLIC_QUESTIONS.json"]
    for model in ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b"):
        for name in ("DISCOVERY_ASSIGNMENTS", "VALIDATION_ASSIGNMENTS", "FINAL_NATIVE_ASSIGNMENTS"):
            paths.append(root / "verification-study-v3" / model / (name + ".json"))
    for study in inventory["studies"].values():
        reference = study["manifest"]
        path = Path(reference["path"])
        require(file_sha(path) == reference["file_sha256"], "inventory roster source changed")
        paths.append(path)
    paths.append(Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/"
                      "candidate-verification-v3-20261007-r4/J1/AUDIT_FRAME.json"))
    question_ids, families, problems, per_source = set(), set(), set(), {}

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("question_id", "question") and isinstance(item, str):
                    question_ids.add(item)
                elif key in ("family", "family_id") and isinstance(item, str):
                    families.add(item)
                elif key == "problem" and isinstance(item, str):
                    problems.add(normalize(item))
                if key == "questions" and isinstance(item, dict):
                    question_ids.update(item)
                if key in ("discovery", "validation", "final") and isinstance(item, list):
                    families.update(x for x in item if isinstance(x, str))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in paths:
        require(path.stat().st_size < 4 * 1024**2, "exposure roster exceeds bounded metadata review")
        value = json.loads(path.read_text())
        # Some historical rosters predate the current receipt seal convention;
        # inventory file hashes bind those sources exactly.
        before = (len(question_ids), len(families), len(problems))
        walk(value)
        per_source[str(path)] = {"added_question_ids": len(question_ids) - before[0],
                                "added_family_ids": len(families) - before[1],
                                "added_problem_texts": len(problems) - before[2]}
    table_path = root / "routing-study-v1/FULL_MATH_QUESTION_TABLE.json"
    table = load(table_path)
    text_matches = []
    for row in table["records"]:
        text = normalize(row["problem"])
        if text in problems or (len(text) >= 30 and any(text in p for p in problems)):
            question_ids.add(row["id"])
            text_matches.append(row["id"])
    paths.append(table_path)
    queue = subprocess.run(["squeue", "-u", "lmolfett", "-h", "-o", "%i|%j|%T|%k"],
                           capture_output=True, text=True, check=True, timeout=30).stdout
    unknown = [line for line in queue.splitlines()
               if not line.split("|")[1].startswith(("cv3", "em1-", "ems1-"))]
    require(not unknown, "unreviewed active workload requires additional exposure reconciliation")
    return save(output, {"schema": "expert-memory-later-exposure-ledger-v1", "complete": True,
        "reviewed_through_utc": datetime.now(timezone.utc).isoformat(),
        "question_ids": sorted(question_ids), "family_ids": sorted(families),
        "source_files": {str(p): file_sha(p) for p in paths}, "per_source": per_source,
        "full_math_problem_text_matches": text_matches, "queue_observation": queue,
        "scope": "Reconciled A/B/C/FRESH rosters, v2 native-input/boundary, all21 v3 rosters, public catalog and original100 J1 items",
        "prepared_reservations_conservatively_excluded": True,
        "future_existing_campaign_exposure_covered_by_fixed_rosters": True,
        "outcomes_used": False, "dispatch": False,
        "review_code_files": {str(Path(__file__).resolve()): file_sha(__file__)}})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = reconcile(args.output)
    print(json.dumps({"sha256": result["sha256"], "question_ids": len(result["question_ids"]),
                      "families": len(result["family_ids"]), "complete": result["complete"]}))
