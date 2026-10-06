"""Discovery-only exact token IDs for two-reader full-prefix accepted starts."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from audit_transition_detector_v2 import BASE, REPO, ROOT, digest, sealed, trace_at

FRAME = BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json"
RATINGS = BASE / "ratings-v22-fullprefix-v2-6c10499b-1ef8863f"
NATIVE = BASE / "ratings-native-veto-full-v0-6c10499b-191f97fe"
OUT = BASE / "FULLPREFIX_READER_AGREED_ELIGIBLE_POOL_v0.json"
TRANSITIONS = ("candidate_to_verify", "approach_to_commit", "failed_check_to_revise")
MAX_PER_TRANSITION = 12


def rated_starts(path, frame, binding, summary):
    if (binding["frame_sha256"] != frame["sha256"]
            or summary["binding_sha256"] != binding["sha256"]
            or summary.get("counts", {}).get("ratings") != 744):
        raise ValueError("full-prefix Qwen audit binding or rating coverage differs")
    batches = [sealed(batch) for batch in sorted((path / "batches").glob("[0-9]" * 6 + ".json"))]
    rated = [row for batch in batches for row in batch["records"]]
    if (len(rated) != len(frame["records"])
            or any(batch["binding_sha256"] != binding["sha256"] for batch in batches)
            or [r["uid"] for r in rated] != [r["uid"] for r in frame["records"]]):
        raise ValueError("Qwen full-prefix batches incomplete or reordered")
    return rated, [batch["sha256"] for batch in batches]


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("eligible-pool trace reconstruction requires CPU Slurm")
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    frame = sealed(FRAME)
    binding = sealed(RATINGS / "BINDING.json")
    summary = sealed(RATINGS / "SUMMARY.json")
    native_binding = sealed(NATIVE / "BINDING.json")
    native_summary = sealed(NATIVE / "SUMMARY.json")
    if frame["schema"] != "transition-v22-full-prefix-start-frame-v1" or frame["rows"] != 372:
        raise ValueError("frozen full-prefix discovery frame changed")
    rated, batch_shas = rated_starts(RATINGS, frame, binding, summary)
    native_rated, native_batch_shas = rated_starts(NATIVE, frame, native_binding, native_summary)
    native_by_uid = {row["uid"]: row for row in native_rated}
    pools = defaultdict(list)
    counts = Counter()
    for row, result in zip(frame["records"], rated, strict=True):
        if row["transition"] not in TRANSITIONS or row["uid"] != result["uid"]:
            raise ValueError("transition or rating UID differs")
        transition = row["transition"]
        counts[transition + "|sampled"] += 1
        if not row["analysis_meta"]["v22_fired"]:
            continue
        counts[transition + "|fires"] += 1
        accepted = all(r["finish_reason"] == "stop" and r["rating"] == {"start": True}
                       for r in result["readers"])
        counts[transition + "|reader_agreed_starts"] += accepted
        if accepted:
            pools[transition].append(row)
    chosen = []
    selection_counts = {}
    for transition in TRANSITIONS:
        rows = sorted(pools[transition], key=lambda r: digest(
            ["fullprefix-eligible-pool-v0", transition, r["family"], r["uid"]]))
        used_families = set()
        for row in rows:
            if row["family"] in used_families:
                continue
            chosen.append(row)
            used_families.add(row["family"])
            if len(used_families) == MAX_PER_TRANSITION:
                break
        selection_counts[transition] = {
            "agreed_rows": len(rows),
            "agreed_families": len({r["family"] for r in rows}),
            "selected_distinct_families": len(used_families)}
    table = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                            columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(r["attempt_id"]): r for r in table.to_dict("records")}
    trace_cache = {}
    out_rows = []
    for row in chosen:
        attempt = row["attempt_id"]
        if attempt not in trace_cache:
            source = by_attempt[attempt]
            trace = trace_at(source["source_location"])
            if trace_digest(TraceRecord(**trace)) != source["trace_sha256"]:
                raise ValueError("eligible native trace digest changed")
            trace_cache[attempt] = trace
        trace = trace_cache[attempt]
        replay = trace["metadata"]["token_replay"]
        ids, offsets = replay["completion_token_ids"], replay["completion_offsets"]
        prompt_ids = replay["prompt_token_ids"]
        n = row["prefix_tokens"]
        if (not prompt_ids or n < 1 or n > 8192 or len(ids) != len(offsets)
                or n > len(ids) or digest(ids[:n]) != row["analysis_meta"]["prefix_ids_sha256"]
                or replay["tokenizer_sha256"] != row["analysis_meta"]["tokenizer_sha256"]):
            raise ValueError("reader-agreed native prefix IDs changed")
        text = trace["cot_text"]
        end_char = int(offsets[n - 1][1])
        prefix_text = text[:end_char]
        if (hashlib.sha256(prefix_text.encode()).hexdigest() != row["analysis_meta"]["prefix_text_sha256"]
                or prefix_text != row["reader_input"]["emitted_prefix"]
                or not prefix_text.rstrip().endswith(row["reader_input"]["triggering_sentence"].rstrip())):
            raise ValueError("reader-agreed prefix text changed")
        reasoning = trace["metadata"].get("reasoning_content")
        reasoning_start = text.find(reasoning) if reasoning else -1
        if reasoning_start < 0 or end_char > reasoning_start + len(reasoning):
            raise ValueError("selected prefix has passed reasoning closure")
        out_rows.append({
            "uid": row["uid"], "transition": row["transition"],
            "family": row["family"], "attempt_id": attempt,
            "sentence_index": row["sentence_index"],
            "question": row["reader_input"]["problem"],
            "trace_sha256": by_attempt[attempt]["trace_sha256"],
            "tokenizer_sha256": replay["tokenizer_sha256"],
            "prompt_ids": prompt_ids, "prompt_ids_sha256": digest(prompt_ids),
            "prefix_ids": ids[:n], "prefix_ids_sha256": digest(ids[:n]),
            "prefix_tokens": n,
            "prefix_text_sha256": row["analysis_meta"]["prefix_text_sha256"],
            "qwen_fullprefix_reader_status": "both_natural_stop_start_true",
            "native_veto_reader_votes": [
                r["rating"]["start"] if r["rating"] is not None and r["finish_reason"] == "stop" else None
                for r in native_by_uid[row["uid"]]["readers"]]})
    if len({r["uid"] for r in out_rows}) != len(out_rows):
        raise ValueError("duplicate selected eligible UID")
    body = {
        "schema": "fullprefix-reader-agreed-eligible-pool-v0",
        "job_id": os.environ["SLURM_JOB_ID"],
        "frame_sha256": frame["sha256"],
        "qwen_binding_sha256": binding["sha256"],
        "qwen_summary_sha256": summary["sha256"],
        "qwen_batch_sha256s": batch_shas,
        "native_binding_sha256": native_binding["sha256"],
        "native_summary_sha256": native_summary["sha256"],
        "native_batch_sha256s": native_batch_shas,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "v2.2 fired and both full-prefix Qwen readers natural-stop start=true; fixed hash, first distinct discovery family per transition, at most 12; native-veto status recorded but never used for selection; no future target or intervention outcome",
        "max_native_prefix_tokens": 8192,
        "counts": dict(counts), "selection_counts": selection_counts,
        "records": out_rows,
        "interpretation": "LLM-audited start enrollment, not validated human truth or transition control; transition-specific families may overlap"}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts),
                      "selection_counts": selection_counts}))


if __name__ == "__main__":
    main()
