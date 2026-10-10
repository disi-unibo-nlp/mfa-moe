"""Freeze 128 new full-MATH families with complete prior-family exclusions."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import socket
import unicodedata

from moe_exp.routing_control.study_v1 import (digest, file_sha, freeze_independent_families,
                                             require, save, sealed)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "report/routing-study-v1"
OLD = REPO / "report/experimental-resume-v1/family-freeze.json"
QUESTIONS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/manifests/question-table-v1.json")
TOKENIZER = REPO.parent / "cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def duplicate_edges(token_ids, texts, old_edges=()):
    """Exact text and registered 8-token containment graph, including old edges."""
    keys = sorted(token_ids)
    shingles, buckets, exact = {}, defaultdict(list), defaultdict(list)
    for i, key in enumerate(keys):
        tokens = token_ids[key]
        shingles[i] = {tuple(tokens[j:j + 8]) for j in range(max(0, len(tokens) - 7))}
        for shingle in shingles[i]: buckets[shingle].append(i)
        exact[normalize(texts[key])].append(key)
    uncommon = {shingle for shingle, members in buckets.items() if len(members) <= 20}
    effective = {i: len(values & uncommon) for i, values in shingles.items()}
    shared = Counter()
    for shingle in uncommon:
        members = buckets[shingle]
        for pos, a in enumerate(members):
            for b in members[pos + 1:]: shared[a, b] += 1
    edges = {tuple(sorted((a, b))) for a, b in old_edges}
    for (a, b), count in shared.items():
        denominator = min(effective[a], effective[b])
        if denominator >= 6 and count / denominator >= .85:
            edges.add((keys[a], keys[b]))
    exact_pairs = []
    for members in exact.values():
        exact_pairs.extend((members[0], member) for member in members[1:])
    edges.update(exact_pairs)
    return sorted(edges), exact_pairs


def main():
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID") and
            not socket.gethostname().startswith("login"), "full-source graph requires allocated CPU Slurm")
    import pandas as pd
    from transformers import AutoTokenizer
    source = sealed(OUT / "full-math-source/SOURCE.json")
    prior, previous = sealed(OLD), sealed(QUESTIONS)
    require(set(previous["questions"]) == set(q for family in prior["new_parent_pools"]["families"].values()
                                             for q in family), "prior text and family universes differ")
    records, text_by_id = [], {}
    for part in source["files"]:
        require(file_sha(part["path"]) == part["file_sha256"], "pinned full-source file changed")
        rows = pd.read_parquet(part["path"]).to_dict("records")
        for i, row in enumerate(rows):
            require(row.get("problem") and row.get("solution") and row.get("level") and row.get("type"),
                    "MATH source metadata incomplete")
            level = re.fullmatch(r"Level ([1-5])", str(row["level"]))
            require(level is not None, "unknown MATH difficulty stratum")
            key = "math-full|" + source["revision"] + "|" + part["subject"] + "|" + str(i)
            records.append({"id": key, "problem": row["problem"], "subject": part["subject"],
                            "level": int(level[1]), "source_row": i, "solution": row["solution"]})
            text_by_id[key] = row["problem"]
    require(len(records) == 5000, "full MATH test source must contain exactly 5000 questions")
    text_by_id.update({key: row["problem"] for key, row in previous["questions"].items()})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    keys = sorted(text_by_id)
    encoded = tokenizer([text_by_id[k] for k in keys], add_special_tokens=False)["input_ids"]
    tokens = dict(zip(keys, encoded, strict=True))
    old_edges = [(edge["a"], edge["b"]) for edge in prior["full_edges"]]
    for family in prior["new_parent_pools"]["families"].values():
        old_edges.extend((family[0], member) for member in family[1:])
    edges, exact = duplicate_edges(tokens, text_by_id, old_edges)
    by_text = defaultdict(set)
    for record in records: by_text[normalize(record["problem"])].add(record["id"])
    matched_math500 = sum(normalize(row["problem"]) in by_text
                          for row in previous["questions"].values() if row["dataset"] == "math500")
    require(matched_math500 == 500, "all previously used MATH-500 source texts must match full source")
    graph = save(OUT / "FULL_MATH_DUPLICATE_GRAPH.json", {"schema": "routing-study-full-source-duplicate-graph-v1",
        "source_sha256": source["sha256"], "prior_family_sha256": prior["sha256"],
        "prior_question_table_sha256": previous["sha256"], "edges": edges,
        "method": {"token_ngram": 8, "max_bucket": 20, "min_shingles": 6, "containment": .85,
                   "exact_text": "NFKC and collapsed whitespace", "preserve_all_prior_family_edges": True},
        "previously_used": sorted(previous["questions"]), "matched_prior_math500": matched_math500,
        "tokenizer_json_sha256": file_sha(TOKENIZER / "tokenizer.json"), "outcomes_used": False})
    result = freeze_independent_families(records, edges, list(previous["questions"]), revision=source["revision"])
    enrollment = save(OUT / "INDEPENDENT_FAMILY_MANIFEST.json", {**result,
        "schema": "routing-study-independent-family-manifest-v1", "full_source_sha256": source["sha256"],
        "duplicate_graph_sha256": graph["sha256"], "code_sha256": file_sha(__file__),
        "source_dataset": source["dataset"], "source_count": len(records)})
    # Frozen full source supports downstream original prompts and independent
    # grading; gold/solutions never enter policy triggers or target selection.
    save(OUT / "FULL_MATH_QUESTION_TABLE.json", {"schema": "routing-study-full-math-question-table-v1",
        "source_sha256": source["sha256"], "records": records})
    print(json.dumps({"status": enrollment["status"], "source_count": len(records),
        "eligible_families": enrollment["eligible"], "validation": len(enrollment["validation"]),
        "final": len(enrollment["final"]), "matched_prior_math500": matched_math500,
        "manifest_sha256": enrollment["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
