"""Complete the approved family freeze using its valid sealed duplicate graph.

The first CPU job wrote a correctly sealed JSON graph, then failed a Python
tuple/list readback comparison. This entry reuses that immutable work and
performs no new tokenization or graph construction. No generation is run.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import socket

from moe_exp.routing_control.study_v1 import digest, file_sha, freeze_independent_families, require, save, sealed
from prepare_independent_math_v1 import OUT, OLD, QUESTIONS, TOKENIZER, normalize


def reservation(records, graph, source):
    require(graph["source_sha256"] == source["sha256"] and graph["outcomes_used"] is False and
            graph["matched_prior_math500"] == 500, "preserved source graph is unqualified")
    require(all(isinstance(edge, list) and len(edge) == 2 for edge in graph["edges"]),
            "duplicate graph is not canonical JSON")
    result = freeze_independent_families(records, graph["edges"], graph["previously_used"], revision=source["revision"])
    return {**result, "schema": "routing-study-independent-family-manifest-v1", "full_source_sha256": source["sha256"],
        "duplicate_graph_sha256": graph["sha256"], "code_sha256": file_sha(__file__),
        "source_dataset": source["dataset"], "source_count": len(records)}


def main():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            not socket.gethostname().startswith("login"), "reservation completion requires CPU Slurm")
    import pandas as pd
    source, graph = sealed(OUT / "full-math-source/SOURCE.json"), sealed(OUT / "FULL_MATH_DUPLICATE_GRAPH.json")
    prior, previous = sealed(OLD), sealed(QUESTIONS)
    require(graph["prior_family_sha256"] == prior["sha256"] and
            graph["prior_question_table_sha256"] == previous["sha256"] and
            graph["tokenizer_json_sha256"] == file_sha(TOKENIZER / "tokenizer.json") and
            graph["previously_used"] == sorted(previous["questions"]), "preserved graph source bindings changed")
    require(graph["method"] == {"token_ngram": 8, "max_bucket": 20, "min_shingles": 6, "containment": .85,
                               "exact_text": "NFKC and collapsed whitespace", "preserve_all_prior_family_edges": True},
            "registered duplicate method changed")
    records = []
    for part in source["files"]:
        require(file_sha(part["path"]) == part["file_sha256"], "pinned source changed")
        for i, row in enumerate(pd.read_parquet(part["path"]).to_dict("records")):
            level = re.fullmatch(r"Level ([1-5])", str(row["level"]))
            require(level is not None and row["problem"] and row["solution"], "source metadata incomplete")
            records.append({"id": "math-full|" + source["revision"] + "|" + part["subject"] + "|" + str(i),
                "problem": row["problem"], "subject": part["subject"], "level": int(level[1]),
                "source_row": i, "solution": row["solution"]})
    require(len(records) == 5000, "pinned full-source count changed")
    texts = {normalize(r["problem"]) for r in records}
    matched = sum(normalize(r["problem"]) in texts for r in previous["questions"].values() if r["dataset"] == "math500")
    require(matched == graph["matched_prior_math500"] == 500, "prior MATH-500 aliases do not all match")
    enrollment = save(OUT / "INDEPENDENT_FAMILY_MANIFEST.json", reservation(records, graph, source))
    save(OUT / "FULL_MATH_QUESTION_TABLE.json", {"schema": "routing-study-full-math-question-table-v1",
        "source_sha256": source["sha256"], "records": records})
    save(OUT / "INDEPENDENT_ENROLLMENT_COMPLETION.json", {"schema": "routing-study-independent-enrollment-completion-v2",
        "job_id": os.environ["SLURM_JOB_ID"], "manifest_sha256": enrollment["sha256"], "graph_sha256": graph["sha256"],
        "failed_job": "59493569", "recovery": "Reused valid canonical serialized graph; no historical overwrite or graph rebuild",
        "code_files": {str(Path(__file__)): file_sha(__file__)}, "outcomes_generated": 0})
    print(json.dumps({"status": enrollment["status"], "source_count": len(records), "eligible_families": enrollment["eligible"],
        "validation": len(enrollment["validation"]), "final": len(enrollment["final"]), "matched_prior_math500": matched,
        "manifest_sha256": enrollment["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
