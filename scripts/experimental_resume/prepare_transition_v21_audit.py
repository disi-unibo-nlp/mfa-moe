"""Freeze new, disjoint discovery windows and exact short-prefix scout receipts.

The online detector never sees later sentences.  These records serve offline
arm-blind semantic audit and a same-prefix replay feasibility scout only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from audit_transition_detector_v2 import BASE, REPO, ROOT, digest, ratings, sealed, trace_at

FRAME_OUT = BASE / "TRANSITION_V21_INDEPENDENT_AUDIT_FRAME.json"
SCOUT_OUT = BASE / "CANDIDATE_PREFIX_SCOUT_v1.json"
SENSITIVITY = REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.1.json"
MAX_PREFIX_TOKENS = 8192


def selected(rows, per_family):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    chosen = []
    for family in sorted(grouped):
        ordered = sorted(grouped[family], key=lambda r: digest(["v21-audit-sample",
                                 r["transition"], family, r["attempt_id"], r["sentence_index"]]))
        chosen.extend(ordered[:per_family])
    return chosen


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("independent audit frame requires CPU Slurm")
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    units = sealed(BASE / "UNITS.json")
    sensitivity = sealed(SENSITIVITY)
    old_frame, by_uid, _ = ratings()
    if sensitivity["units_sha256"] != units["sha256"] or sensitivity["rating_frame_sha256"] != old_frame["sha256"]:
        raise ValueError("v2.1 replay input binding differs")
    excluded = {(row["transition"], row["analysis_meta"]["attempt_id"],
                 row["analysis_meta"]["source_sentence_index"]) for row in old_frame["records"]}
    fires = {transition: {(e["attempt_id"], e["sentence_index"])
             for e in sensitivity["eligible_discovery_events"].get(
                 "v2.1|delimiter_aware|" + transition, [])}
             for transition in ("candidate_to_verify", "approach_to_commit", "failed_check_to_revise")}
    grouped = defaultdict(list)
    for unit in units["records"]:
        grouped[unit["attempt_id"]].append(unit)
    candidates = defaultdict(list)
    for attempt, all_units in grouped.items():
        ordered = sorted(all_units, key=lambda u: u["sentence_index"])
        for index, left in enumerate(ordered[:-1]):
            later = []
            previous = left
            for right in ordered[index+1:index+5]:
                if right["sentence_index"] != previous["sentence_index"] + 1 or right["segment"] != previous["segment"]:
                    break
                if right["token_start"] >= left["token_end"] + 256:
                    break
                later.append({"sentence_index": right["sentence_index"],
                              "token_start_after_trigger": right["token_start"] - left["token_end"],
                              "text": right["inputs"]["sentence"]})
                previous = right
            if not later:
                continue
            for transition in fires:
                key = (transition, attempt, left["sentence_index"])
                if key in excluded:
                    continue
                fired = (attempt, left["sentence_index"]) in fires[transition]
                row = {"transition": transition, "family": left["family"],
                       "question": left["question"], "attempt_id": attempt,
                       "sentence_index": left["sentence_index"],
                       "segment": left["segment"],
                       "prefix_tokens": left["token_end"],
                       "prefix_text_end": left["char_end"],
                       "fired": fired,
                       "reader_input": {
                           "problem": left["inputs"]["problem_statement"],
                           "previous_sentence": left["inputs"]["previous_sentence"],
                           "triggering_sentence": left["inputs"]["sentence"],
                           "later_sentences": later}}
                candidates[transition, fired].append(row)
    sample = []
    for transition in fires:
        for fired in (True, False):
            # All rare approach/failure fires; fixed family-spread candidate
            # quotas.  Every arm remains discovery-only and independent of the
            # previously rated 619 local windows.
            quota = (3 if fired else 2) if transition == "candidate_to_verify" else (
                1000 if fired else 1)
            source = [r for r in candidates[transition, fired]
                      if r["prefix_tokens"] <= MAX_PREFIX_TOKENS]
            sample.extend(selected(source, quota))
    for row in sample:
        row["uid"] = digest(["transition-v21-independent-rating", row["transition"],
                             row["family"], row["attempt_id"], row["sentence_index"]])
    sample.sort(key=lambda row: row["uid"])
    if len({r["uid"] for r in sample}) != len(sample):
        raise ValueError("duplicate new audit UID")
    family_counts = defaultdict(set)
    counts = Counter()
    for row in sample:
        key = row["transition"] + "|" + ("fire" if row["fired"] else "nonfire")
        counts[key] += 1
        family_counts[key].add(row["family"])
    frame_body = {"schema": "transition-v21-independent-audit-frame-v1",
                  "job_id": os.environ["SLURM_JOB_ID"],
                  "units_sha256": units["sha256"],
                  "attribution_sensitivity_sha256": sensitivity["sha256"],
                  "excluded_previous_rating_frame_sha256": old_frame["sha256"],
                  "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "families": len(grouped),
                  "prefix_token_cap": MAX_PREFIX_TOKENS,
                  "target_window": "first 1-4 contiguous later native sentences beginning within 256 tokens after trigger",
                  "input_allowlist": ["problem", "previous_sentence", "triggering_sentence", "later_sentences"],
                  "sampling": "fixed hash within each discovery family, up to 3 candidate fires and 2 candidate nonfires; all approach/failure fires and 1 nonfire per family; exclude all v1 rated windows; no replacement",
                  "counts": dict(counts),
                  "family_counts": {k: len(v) for k, v in family_counts.items()},
                  "records": sample}
    frame = {**frame_body, "sha256": digest(frame_body)}

    # Exact native replay prefixes for a small context-matched feasibility
    # scout; select only candidate fires already accepted by both old readers.
    accepted = set()
    for old in old_frame["records"]:
        if old["transition"] != "candidate_to_verify":
            continue
        votes = by_uid[old["uid"]]["readers"]
        if all(x["rating"] is not None and x["finish_reason"] == "stop" and x["rating"]["start"]
               for x in votes):
            accepted.add((old["analysis_meta"]["attempt_id"],
                          old["analysis_meta"]["source_sentence_index"]))
    eligible = [u for u in units["records"]
                if (u["attempt_id"], u["sentence_index"]) in accepted
                and (u["attempt_id"], u["sentence_index"]) in fires["candidate_to_verify"]
                and u["token_end"] <= MAX_PREFIX_TOKENS]
    chosen = sorted(selected([{"family": u["family"], "transition": "candidate_to_verify",
                               "attempt_id": u["attempt_id"], "sentence_index": u["sentence_index"],
                               "unit": u} for u in eligible], 1),
                    key=lambda r: digest(["candidate-prefix-scout-family-v1", r["family"]]))[:12]
    table = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                            columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(row["attempt_id"]): row for row in table.to_dict("records")}
    scouts = []
    for chosen_row in chosen:
        unit = chosen_row["unit"]
        source = by_attempt[unit["attempt_id"]]
        trace = trace_at(source["source_location"])
        if trace_digest(TraceRecord(**trace)) != source["trace_sha256"]:
            raise ValueError("scout trace digest changed")
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = replay["completion_token_ids"], replay["completion_offsets"], trace["cot_text"]
        prompt_ids = replay["prompt_token_ids"]
        n = unit["token_end"]
        if (len(ids) != len(offsets) or n > MAX_PREFIX_TOKENS or
            not prompt_ids or int(offsets[n-1][1]) > len(text)):
            raise ValueError("scout prefix token/text mismatch")
        prefix_text = text[:int(offsets[n-1][1])]
        reasoning = trace["metadata"].get("reasoning_content")
        if not reasoning:
            raise ValueError("scout trace lacks saved native reasoning content")
        reasoning_start = text.find(reasoning)
        if reasoning_start < 0 or not (reasoning_start <= unit["char_start"] < unit["char_end"]
                                      <= len(prefix_text) <= reasoning_start + len(reasoning)):
            raise ValueError("scout prefix extends beyond native reasoning")
        messages = trace["generation_messages"]
        if not messages:
            raise ValueError("scout lacks original generation messages")
        scouts.append({"uid": digest(["candidate-prefix-scout-v1", unit["family"], unit["attempt_id"],
                                      unit["sentence_index"]]),
                       "family": unit["family"], "question": unit["question"],
                       "attempt_id": unit["attempt_id"], "sentence_index": unit["sentence_index"],
                       "trace_sha256": source["trace_sha256"],
                       "problem": unit["inputs"]["problem_statement"],
                       "generation_messages": messages,
                       "generation_messages_sha256": digest(messages),
                       "tokenizer_sha256": replay["tokenizer_sha256"],
                       "prompt_ids": prompt_ids,
                       "prompt_ids_sha256": digest(prompt_ids),
                       "prefix_ids": ids[:n],
                       "prefix_ids_sha256": digest(ids[:n]),
                       "prefix_text_sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
                       "prefix_tokens": n, "prefix_chars": len(prefix_text)})
    scout_body = {"schema": "candidate-prefix-scout-v1",
                  "job_id": os.environ["SLURM_JOB_ID"],
                  "units_sha256": units["sha256"],
                  "attribution_sensitivity_sha256": sensitivity["sha256"],
                  "old_rating_frame_sha256": old_frame["sha256"],
                  "new_rating_frame_sha256": frame["sha256"],
                  "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "selection": "up to 12 deterministic first-hash distinct-family v2.1 delimiter-aware candidate fires, <=8192 tokens, with both old LLM readers accepting the start; feasibility scout only",
                  "records": scouts}
    scout = {**scout_body, "sha256": digest(scout_body)}
    if FRAME_OUT.exists() or SCOUT_OUT.exists():
        raise FileExistsError("v2.1 audit frame/scout output already exists")
    FRAME_OUT.write_text(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    SCOUT_OUT.write_text(json.dumps(scout, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"frame": str(FRAME_OUT), "counts": frame["counts"],
                      "family_counts": frame["family_counts"],
                      "scout": str(SCOUT_OUT), "scout_rows": len(scouts)}))


if __name__ == "__main__":
    main()
