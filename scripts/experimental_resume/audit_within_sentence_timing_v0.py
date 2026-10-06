"""Discovery-only exact-token timing inventory for closed math candidate spans.

Sentence-end and within-sentence cutoffs are separate definitions. The new
cutoff never reads the later clause online; this CPU audit only measures where
a fully closed candidate expression first became visible in saved native text.
The original transition primary still excludes its triggering sentence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import pandas as pd

REPO = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo")
ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
BASE = ROOT / "steering-v1/runs/routing-control-v1/dense-discovery"
REPORT = REPO / "report/experimental-resume-v1/WITHIN_SENTENCE_TIMING_INVENTORY_v0.json"
FRAME_OUT = BASE / "WITHIN_SENTENCE_CLOSED_MATH_PREFIX_FRAME_v0.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/experimental_resume"))
from moe_exp.routing_control.transitions_v22 import _MATH, _RELATION, _NUMBER
from moe_exp.routing_control.prefix import candidates
from moe_exp.schemas import TraceRecord
from moe_exp.correlation_pipeline.spans import trace_digest
from audit_transition_detector_v2 import trace_at


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"changed sealed source: {path}")
    return value


def first_closed_evidence(sentence):
    """Recognition offset inside sentence; no unclosed/partial plain numbers."""
    options = []
    for match in _MATH.finditer(sentence):
        expr = next(group for group in match.groups() if group is not None)
        if _RELATION.search(expr) and _NUMBER.search(expr) and "..." not in expr and r"\ldots" not in expr:
            options.append((match.end(), "closed_math_relation"))
    for item in candidates(sentence):
        if item.kind in {"boxed", "answer_math"}:
            options.append((item.end, item.kind))
    return min(options) if options else None


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("native offset audit requires allocated CPU Slurm")
    source = sealed(BASE / "TRANSITION_V22_FULL_PREFIX_START_FRAME.json")
    units = sealed(BASE / "UNITS.json")
    agreement = sealed(REPO / "report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json")
    if (source["schema"] != "transition-v22-full-prefix-start-frame-v1"
            or source["units_sha256"] != units["sha256"]
            or agreement["frame_sha256"] != source["sha256"]):
        raise ValueError("discovery frame/units/rating binding changed")
    by_unit = {(u["attempt_id"], u["sentence_index"]): u for u in units["records"]}
    by_rating = {r["uid"]: r for r in agreement["records"]}
    table = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                            columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(r["attempt_id"]): r for r in table.to_dict("records")}
    trace_cache = {}
    inventory, frame_rows, counts = [], [], Counter()
    for row in source["records"]:
        if not row["analysis_meta"]["v22_fired"]:
            continue
        transition = row["transition"]
        counts[transition + "|audited_fires"] += 1
        unit = by_unit[row["attempt_id"], row["sentence_index"]]
        if unit["family"] != row["family"] or unit["token_end"] != row["prefix_tokens"]:
            raise ValueError("audited unit offset differs")
        sentence = row["reader_input"]["triggering_sentence"]
        if sentence != unit["inputs"]["sentence"]:
            raise ValueError("audited triggering sentence differs")
        if transition != "candidate_to_verify":
            inventory.append({"uid": row["uid"], "transition": transition,
                              "family": row["family"], "status": "early_phrase_not_syntactically_qualified",
                              "sentence_end_token": row["prefix_tokens"]})
            counts[transition + "|early_abstain"] += 1
            continue
        evidence = first_closed_evidence(sentence)
        if evidence is None:
            inventory.append({"uid": row["uid"], "transition": transition,
                              "family": row["family"], "status": "no_closed_math_evidence",
                              "sentence_end_token": row["prefix_tokens"],
                              "qwen_fullsentence_status": by_rating[row["uid"]]["qwen_status"]})
            counts[transition + "|no_closed_math_evidence"] += 1
            continue
        local_end, basis = evidence
        attempt = row["attempt_id"]
        if attempt not in trace_cache:
            src = by_attempt[attempt]
            trace = trace_at(src["source_location"])
            if trace_digest(TraceRecord(**trace)) != src["trace_sha256"]:
                raise ValueError("saved native trace digest changed")
            trace_cache[attempt] = trace
        trace = trace_cache[attempt]
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = replay["completion_token_ids"], replay["completion_offsets"], trace["cot_text"]
        full_n = row["prefix_tokens"]
        if (len(ids) != len(offsets) or full_n > len(ids) or not replay["prompt_token_ids"]
                or digest(ids[:full_n]) != row["analysis_meta"]["prefix_ids_sha256"]
                or replay["tokenizer_sha256"] != row["analysis_meta"]["tokenizer_sha256"]
                or text[:int(offsets[full_n - 1][1])] != row["reader_input"]["emitted_prefix"]):
            raise ValueError("native replay or full-prefix reader input differs")
        span = text[unit["char_start"]:unit["char_end"]]
        if span.count(sentence) != 1:
            raise ValueError("trigger sentence is not unique inside saved unit span")
        abs_start = unit["char_start"] + span.find(sentence)
        evidence_char_end = abs_start + local_end
        first_n = next((i + 1 for i in range(unit["token_start"], full_n)
                        if int(offsets[i][1]) >= evidence_char_end), None)
        if first_n is None:
            raise ValueError("closed math evidence exceeds sentence token boundary")
        prefix_end = int(offsets[first_n - 1][1])
        prefix_text = text[:prefix_end]
        reasoning = trace["metadata"].get("reasoning_content")
        reasoning_start = text.find(reasoning) if reasoning else -1
        if reasoning_start < 0 or prefix_end > reasoning_start + len(reasoning) or "</think>" in prefix_text:
            raise ValueError("within-sentence candidate crosses reasoning closure")
        early = first_n < full_n
        qwen_status = by_rating[row["uid"]]["qwen_status"]
        inventory.append({"uid": row["uid"], "transition": transition,
                          "family": row["family"], "status": "early_cut" if early else "same_token_as_sentence_end",
                          "basis": basis, "sentence_end_token": full_n,
                          "earliest_complete_token": first_n,
                          "token_headroom": full_n - first_n,
                          "trailing_chars_in_sentence": len(sentence) - local_end,
                          "token_overshoot_chars": prefix_end - evidence_char_end,
                          "qwen_fullsentence_status": qwen_status})
        counts[transition + "|closed_math_evidence"] += 1
        counts[transition + "|early_token_cut"] += early
        counts[transition + "|early_cut|qwen_" + qwen_status] += early
        if early:
            fragment = text[abs_start:prefix_end].strip()
            if not fragment or not prefix_text.rstrip().endswith(fragment.rstrip()):
                raise ValueError("early fragment does not end visible prefix")
            frame_rows.append({
                "source_uid": row["uid"], "transition": transition,
                "family": row["family"], "attempt_id": attempt,
                "sentence_index": row["sentence_index"],
                "trace_sha256": by_attempt[attempt]["trace_sha256"],
                "tokenizer_sha256": replay["tokenizer_sha256"],
                "prompt_ids": replay["prompt_token_ids"],
                "prompt_ids_sha256": digest(replay["prompt_token_ids"]),
                "prefix_ids": ids[:first_n], "prefix_ids_sha256": digest(ids[:first_n]),
                "prefix_tokens": first_n,
                "prefix_text_sha256": hashlib.sha256(prefix_text.encode()).hexdigest(),
                "reader_input": {"problem": row["reader_input"]["problem"],
                                 "emitted_prefix": prefix_text,
                                 "triggering_fragment": fragment},
            })
    report_body = {"schema": "within-sentence-timing-inventory-v0",
                   "job_id": os.environ["SLURM_JOB_ID"],
                   "source_frame_sha256": source["sha256"],
                   "units_sha256": units["sha256"],
                   "agreement_sha256": agreement["sha256"],
                   "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "counts": dict(counts), "records": inventory,
                   "primary_endpoint": "original triggered sentence is excluded; unchanged",
                   "interpretation": "Discovery-only candidate timing diagnostic; closed math span can precede sentence end, but semantic start at the earlier cut is unvalidated. Approach and failed-check phrases abstain."}
    frame_body = {"schema": "within-sentence-closed-math-prefix-frame-v0",
                  "source_frame_sha256": source["sha256"],
                  "timing_inventory_sha256": digest(report_body),
                  "visible_input_allowlist": ["problem", "emitted_prefix", "triggering_fragment"],
                  "rows": len(frame_rows), "records": frame_rows,
                  "interpretation": "Prospective exact native token cut for separate start-measurement diagnostic; no future clause in reader input and original primary unchanged"}
    if REPORT.exists() or FRAME_OUT.exists():
        raise FileExistsError("timing inventory/frame already exists")
    REPORT.write_text(json.dumps({**report_body, "sha256": digest(report_body)}, separators=(",", ":"), ensure_ascii=False) + "\n")
    FRAME_OUT.write_text(json.dumps({**frame_body, "sha256": digest(frame_body)}, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(REPORT), "frame": str(FRAME_OUT),
                      "counts": dict(counts), "frame_rows": len(frame_rows)}), flush=True)


if __name__ == "__main__":
    main()
