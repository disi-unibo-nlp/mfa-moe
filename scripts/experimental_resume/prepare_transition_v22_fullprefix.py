"""Bind the fresh discovery audit rows to exact already-emitted native prefixes.

Future native sentences remain in the separate v2.1 target frame and never
enter this v3 start reader frame.  The v2.2 fire indicator is analysis-only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from audit_transition_detector_v2 import BASE, REPO, ROOT, digest, sealed, trace_at

SOURCE = BASE / "TRANSITION_V21_INDEPENDENT_AUDIT_FRAME.json"
SENSITIVITY = REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json"
OUT = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("full-prefix frame preparation requires CPU Slurm")
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    source = sealed(SOURCE)
    sensitivity = sealed(SENSITIVITY)
    units = sealed(BASE / "UNITS.json")
    if (source["schema"] != "transition-v21-independent-audit-frame-v1" or
        source["units_sha256"] != units["sha256"] or
        sensitivity["units_sha256"] != units["sha256"]):
        raise ValueError("discovery source/frame/replay seal mismatch")
    v22_fires = {transition: {(e["attempt_id"], e["sentence_index"])
                 for e in sensitivity["eligible_discovery_events"].get(
                     "v2.2|delimiter_aware|" + transition, [])}
                 for transition in ("candidate_to_verify", "approach_to_commit", "failed_check_to_revise")}
    by_unit = {(u["attempt_id"], u["sentence_index"]): u for u in units["records"]}
    table = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                            columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(r["attempt_id"]): r for r in table.to_dict("records")}
    trace_cache = {}
    records = []
    for row in source["records"]:
        attempt, index = row["attempt_id"], row["sentence_index"]
        left = by_unit[attempt, index]
        later_index = row["reader_input"]["later_sentences"][0]["sentence_index"]
        right = by_unit[attempt, later_index]
        if later_index != index + 1 or left["segment"] != right["segment"]:
            raise ValueError("full-prefix source lacks contiguous next sentence")
        if attempt not in trace_cache:
            record = by_attempt[attempt]
            trace = trace_at(record["source_location"])
            if trace_digest(TraceRecord(**trace)) != record["trace_sha256"]:
                raise ValueError("native trace changed")
            trace_cache[attempt] = trace
        trace = trace_cache[attempt]
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = replay["completion_token_ids"], replay["completion_offsets"], trace["cot_text"]
        n = left["token_end"]
        if n != row["prefix_tokens"] or n > 8192 or len(ids) != len(offsets) or n < 1:
            raise ValueError("full-prefix token count mismatch")
        prefix_end = int(offsets[n-1][1])
        if not (left["char_start"] < left["char_end"] <= prefix_end <= right["char_start"]):
            raise ValueError("full prefix reaches later sentence or omits trigger")
        if text[left["char_start"]:left["char_end"]] != row["reader_input"]["triggering_sentence"]:
            raise ValueError("saved trigger text differs from exact native stream")
        reasoning = trace["metadata"].get("reasoning_content")
        start = text.find(reasoning) if reasoning else -1
        if start < 0 or prefix_end > start + len(reasoning):
            raise ValueError("prefix leaves native reasoning segment")
        prefix = text[:prefix_end]
        key = (attempt, index)
        fired_v22 = key in v22_fires[row["transition"]]
        if fired_v22 and not row["fired"]:
            raise ValueError("v2.2 event is not a subset of v2.1 sampling fire")
        records.append({
            "uid": row["uid"], "transition": row["transition"],
            "family": row["family"], "attempt_id": attempt, "sentence_index": index,
            "prefix_tokens": n,
            "reader_input": {"problem": row["reader_input"]["problem"],
                             "emitted_prefix": prefix,
                             "triggering_sentence": row["reader_input"]["triggering_sentence"]},
            "analysis_meta": {
                "v21_fired": row["fired"], "v22_fired": fired_v22,
                "prefix_text_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "prefix_ids_sha256": digest(ids[:n]),
                "tokenizer_sha256": replay["tokenizer_sha256"],
                "native_trace_sha256": by_attempt[attempt]["trace_sha256"]}})
    if [r["uid"] for r in records] != [r["uid"] for r in source["records"]]:
        raise ValueError("new frame UID/order differs")
    body = {"schema": "transition-v22-full-prefix-start-frame-v1",
            "job_id": os.environ["SLURM_JOB_ID"],
            "source_frame_sha256": source["sha256"],
            "sensitivity_v22_sha256": sensitivity["sha256"],
            "units_sha256": units["sha256"],
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "families": source["families"], "rows": len(records),
            "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_sentence"],
            "prefix_token_cap_native": 8192,
            "scope": "discovery-only exact emitted prefixes; no later sentence, gold, correctness, source class, or detector fire in reader input",
            "records": records}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "rows": len(records),
                      "v22_fired": {t: sum(r["analysis_meta"]["v22_fired"] for r in records if r["transition"] == t)
                                    for t in v22_fires},
                      "prefix_token_max": max(r["prefix_tokens"] for r in records)}))


if __name__ == "__main__":
    main()
