"""Seal exact native token IDs for every Qwen/native agreed discovery start.

This pool is separate from the earlier Qwen-only, 23-row selected pool.  The
13-family globally disjoint subset is frozen here before any new intervention.
GPT-OSS results and future target sentences are never used for selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from audit_transition_detector_v2 import BASE, REPO, ROOT, digest, sealed, trace_at

FRAME = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
AGREEMENT = REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json"
OUT = BASE / "JOINT_QWEN_NATIVE_EXACT_POOL_v1.json"
PRIORITY = {"candidate_to_verify": 0, "approach_to_commit": 1,
            "failed_check_to_revise": 2}


def main():
    # The serial allocation can run on a node named login08.  Slurm job and
    # step identity, not the node hostname, distinguish allocated CPU work.
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("exact trace reconstruction requires CPU Slurm")
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    frame, agreement = sealed(FRAME), sealed(AGREEMENT)
    if (frame["schema"] != "transition-v22-full-prefix-start-frame-v1"
            or frame["rows"] != 372
            or agreement["schema"] != "full-prefix-native-veto-agreement-v1"
            or agreement["frame_sha256"] != frame["sha256"]):
        raise ValueError("frozen source frame or agreement changed")
    agreed = [row for row in agreement["records"] if row["v22_fired"]
              and row["qwen_status"] == "accepted"
              and row["native_status"] == "accepted"]
    if (len(agreed) != 14 or len({row["uid"] for row in agreed}) != 14
            or Counter(row["transition"] for row in agreed) !=
            {"candidate_to_verify": 8, "approach_to_commit": 6}):
        raise ValueError("reader-agreed start population changed")
    source_by_uid = {row["uid"]: row for row in frame["records"]}
    if len(source_by_uid) != frame["rows"]:
        raise ValueError("duplicate source UID")
    attempts = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                               columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(row["attempt_id"]): row for row in attempts.to_dict("records")}
    traces = {}
    records = []
    for row in sorted(agreed, key=lambda x: x["uid"]):
        source = source_by_uid[row["uid"]]
        if (source["family"] != row["family"] or source["transition"] != row["transition"]
                or source["analysis_meta"]["v22_fired"] is not True):
            raise ValueError("agreement UID rebound to changed source")
        attempt_id = source["attempt_id"]
        if attempt_id not in traces:
            trace_source = by_attempt[attempt_id]
            trace = trace_at(trace_source["source_location"])
            if trace_digest(TraceRecord(**trace)) != trace_source["trace_sha256"]:
                raise ValueError("native trace digest differs")
            traces[attempt_id] = trace
        trace = traces[attempt_id]
        replay = trace["metadata"]["token_replay"]
        prompt_ids = replay["prompt_token_ids"]
        ids = replay["completion_token_ids"]
        offsets = replay["completion_offsets"]
        n = source["prefix_tokens"]
        if (not prompt_ids or n < 1 or n > 8192 or len(ids) != len(offsets)
                or n > len(ids) or digest(ids[:n]) != source["analysis_meta"]["prefix_ids_sha256"]
                or replay["tokenizer_sha256"] != source["analysis_meta"]["tokenizer_sha256"]):
            raise ValueError("native prompt/prefix token IDs changed")
        text = trace["cot_text"]
        end_char = int(offsets[n - 1][1])
        prefix_text = text[:end_char]
        if (hashlib.sha256(prefix_text.encode()).hexdigest() !=
                source["analysis_meta"]["prefix_text_sha256"]
                or prefix_text != source["reader_input"]["emitted_prefix"]
                or not prefix_text.rstrip().endswith(
                    source["reader_input"]["triggering_sentence"].rstrip())):
            raise ValueError("reader-agreed emitted text changed")
        reasoning = trace["metadata"].get("reasoning_content")
        reasoning_start = text.find(reasoning) if reasoning else -1
        if reasoning_start < 0 or end_char > reasoning_start + len(reasoning):
            raise ValueError("selected prefix has passed reasoning closure")
        records.append({
            "uid": row["uid"], "family": row["family"],
            "transition": row["transition"], "attempt_id": attempt_id,
            "sentence_index": source["sentence_index"],
            "question": source["reader_input"]["problem"],
            "trace_sha256": by_attempt[attempt_id]["trace_sha256"],
            "tokenizer_sha256": replay["tokenizer_sha256"],
            "prompt_ids": prompt_ids, "prompt_ids_sha256": digest(prompt_ids),
            "prefix_ids": ids[:n], "prefix_ids_sha256": digest(ids[:n]),
            "prefix_tokens": n,
            "prefix_text_sha256": source["analysis_meta"]["prefix_text_sha256"],
            "qwen_status": "two_natural_stop_start_true",
            "native_status": "two_natural_stop_start_true",
        })
    # Discovery enrollment is globally family-disjoint.  The only duplicate
    # family has one candidate and one approach start; candidate takes priority.
    used = set()
    selected = []
    for row in sorted(records, key=lambda x: (PRIORITY[x["transition"]],
                                               digest(["joint-reader-global-v1",
                                                       x["family"], x["uid"]]))):
        if row["family"] not in used:
            selected.append(row["uid"])
            used.add(row["family"])
    selected_rows = [r for r in records if r["uid"] in set(selected)]
    if (len(selected) != 13 or len(used) != 13 or
            Counter(r["transition"] for r in selected_rows) !=
            {"candidate_to_verify": 8, "approach_to_commit": 5}):
        raise ValueError("globally disjoint enrollment changed")
    body = {
        "schema": "joint-qwen-native-exact-pool-v1",
        "job_id": os.environ["SLURM_JOB_ID"],
        "frame_sha256": frame["sha256"],
        "agreement_sha256": agreement["sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "all v2.2 lexical fires with Qwen3.8 and native Qwen3.6 both-reader accepted starts; global family-disjoint subset uses candidate then approach priority and fixed UID hash; GPT-OSS and future outcomes excluded",
        "rows": len(records), "families": len({r["family"] for r in records}),
        "selected_global_uids": selected,
        "selected_global_count": len(selected),
        "selected_global_transition_counts": dict(Counter(r["transition"] for r in selected_rows)),
        "records": records,
        "interpretation": "Discovery-only LLM-consensus start pool; no human-truth or causal claim",
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)},
                              separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(OUT), "sha256": digest(body),
                      "rows": len(records), "families": len(used),
                      "selected": body["selected_global_transition_counts"]}), flush=True)


if __name__ == "__main__":
    main()
