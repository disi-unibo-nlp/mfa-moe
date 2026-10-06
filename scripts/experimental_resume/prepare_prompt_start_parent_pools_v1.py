"""Seal exact original prompts for the frozen discovery/mechanism/utility families.

There is no start detector, local-screen outcome or model continuation in
selection.  Existing versioned parent-pool family order is retained exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from audit_transition_detector_v2 import REPO, ROOT, digest, sealed, trace_at

FREEZE = REPO / "report/experimental-resume-v1/family-freeze.json"
ATTEMPTS = ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet"
OUTDIR = REPO / "report/experimental-resume-v1/prompt-start-parent-pools-v1"
EXPECTED = {"discovery": 48, "mechanism": 128, "utility": 96}


def file_sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(part)
    return hasher.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("exact parent-pool reconstruction requires CPU Slurm")
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    freeze = sealed(FREEZE)
    pools = freeze["new_parent_pools"]
    if (freeze["schema"] != "routing-control-family-freeze-v1"
            or set(pools["parent_pools"]) != set(EXPECTED)
            or any(len(pools["parent_pools"][name]) != n for name, n in EXPECTED.items())
            or pools["order_rule"] != "ascending SHA256(routing-control-v1|family_id)"):
        raise ValueError("frozen parent pools changed")
    all_families = [family for name in EXPECTED for family in pools["parent_pools"][name]]
    if len(set(all_families)) != sum(EXPECTED.values()):
        raise ValueError("parent pools overlap")
    for name in EXPECTED:
        families = pools["parent_pools"][name]
        if families != sorted(families, key=lambda family: hashlib.sha256(
                f"routing-control-v1|{family}".encode()).hexdigest()):
            raise ValueError("frozen parent pool order changed")
    table = pd.read_parquet(ATTEMPTS, columns=["attempt_id", "question", "dataset",
                                              "source_problem_id", "source_location",
                                              "trace_sha256"])
    attempts_sha = file_sha(ATTEMPTS)
    driver_sha = file_sha(Path(__file__))
    by_question = {}
    for row in table.to_dict("records"):
        if row["question"] in by_question:
            raise ValueError("duplicate native source question")
        by_question[row["question"]] = row
    out = {}
    for name, expected_n in EXPECTED.items():
        records = []
        for family in pools["parent_pools"][name]:
            question_id = pools["representative_questions"][family]
            row = by_question.get(question_id)
            if row is None or row["dataset"] + "|" + row["source_problem_id"] != question_id:
                raise ValueError("parent family has no exact native question")
            trace = trace_at(row["source_location"])
            if trace_digest(TraceRecord(**trace)) != row["trace_sha256"]:
                raise ValueError("native trace digest changed")
            replay = trace["metadata"]["token_replay"]
            ids = replay["prompt_token_ids"]
            messages = trace["generation_messages"]
            if (not ids or any(type(token) is not int or token < 0 for token in ids)
                    or not isinstance(messages, list) or not messages
                    or any(set(message) - {"role", "content"} for message in messages)
                    or any(message.get("role") not in {"system", "user"} for message in messages)
                    or not all(isinstance(message.get("content"), str) for message in messages)
                    or not isinstance(trace["prompt"], str) or not trace["prompt"]
                    or not any(trace["prompt"] in message["content"] for message in messages)):
                raise ValueError("original prompt or native token replay unsupported")
            records.append({
                "family": family, "question_id": question_id,
                "attempt_id": row["attempt_id"], "trace_sha256": row["trace_sha256"],
                "tokenizer_sha256": replay["tokenizer_sha256"],
                "problem_statement": trace["prompt"],
                "generation_messages": messages,
                "system_prompt": trace["system_prompt"],
                "prompt_ids": ids, "prompt_ids_sha256": digest(ids),
            })
        if len(records) != expected_n or len({r["family"] for r in records}) != expected_n:
            raise ValueError("native prompt pool count or family disjointness changed")
        body = {
            "schema": "prompt-start-parent-pool-v1",
            "stage": name, "job_id": os.environ["SLURM_JOB_ID"],
            "family_freeze_sha256": freeze["sha256"],
            "native_attempts_file_sha256": attempts_sha,
            "driver_sha256": driver_sha,
            "selection": "exact fixed parent-pool family order and representative native question from family-freeze; no start labels, local-screen result, future completion or correctness used",
            "rows": len(records), "families": len(records),
            "records": records,
            "interpretation": "Exact original prompt IDs for prompt-start intention-to-treat enrollment; discovery is action-development, mechanism and utility are disjoint validation parent pools, pending eligibility and complete-stage price."}
        out[name] = {**body, "sha256": digest(body)}
    if OUTDIR.exists():
        raise FileExistsError(OUTDIR)
    OUTDIR.mkdir(parents=True)
    for name, value in out.items():
        (OUTDIR / f"{name}.json").write_text(json.dumps(value, separators=(",", ":"),
                                                     ensure_ascii=False) + "\n")
    print(json.dumps({"outdir": str(OUTDIR), "stages": {
        name: {"rows": value["rows"], "sha256": value["sha256"]}
        for name, value in out.items()}}), flush=True)


if __name__ == "__main__":
    main()
