"""Exact discovery all-fire native side-judge token and context burden.

Cost-only CPU job: no inference, no labels, no intervention outcomes. The
reasoning prefix is sliced at its saved native token boundary; future text in
trace files is never inserted into a side-judge prompt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

from audit_transition_detector_v2 import BASE, REPO, ROOT, digest, sealed, trace_at
import native_prefix_semantic_veto_v0 as veto
from price_transition_ratings_v3 import count_prompt_tokens

SENSITIVITY = REPO / "report/experimental-resume-v1/TRANSITION_PREFIX_ATTRIBUTION_SENSITIVITY_v2.2.json"
UNITS = BASE / "UNITS.json"
OUT = REPO / "report/experimental-resume-v1/NATIVE_VETO_ALL_FIRE_BURDEN_v0.json"
TRANSITIONS = ("candidate_to_verify", "approach_to_commit", "failed_check_to_revise")
MAX_PREFIX = 16384  # utility scout's output horizon, not a model hard limit
CONTEXT = 32768  # proposed separate late-prefix side-judge profile


def main():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("all-fire tokenizer replay requires CPU Slurm")
    import pandas as pd
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "src"))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.models.token_replay import tokenizer_fingerprint

    sensitivity, units = sealed(SENSITIVITY), sealed(UNITS)
    if sensitivity["units_sha256"] != units["sha256"]:
        raise ValueError("all-fire sensitivity/units binding differs")
    by_unit = {(u["attempt_id"], u["sentence_index"]): u for u in units["records"]}
    events = [(transition, e) for transition in TRANSITIONS for e in
              sensitivity["eligible_discovery_events"]["v2.2|delimiter_aware|" + transition]]
    if len(events) != 1865:
        raise ValueError("all-fire discovery count changed")
    table = pd.read_parquet(ROOT / "v3_analysis/results-r2/qwen36/A/attempts.parquet",
                            columns=["attempt_id", "source_location", "trace_sha256"])
    by_attempt = {str(r["attempt_id"]): r for r in table.to_dict("records")}
    tokenizer = AutoTokenizer.from_pretrained(veto.MODEL, local_files_only=True)
    native_vocab_fingerprint = tokenizer_fingerprint(tokenizer)
    model_cache = veto.cache_receipt()
    trace_cache = {}
    counts = Counter()
    token_counts = Counter()
    lengths = defaultdict(list)
    family = defaultdict(Counter)
    for transition, event in events:
        key = (event["attempt_id"], event["sentence_index"])
        unit = by_unit[key]
        fam = event["family"]
        if (unit["family"] != fam or unit["token_end"] != event["prefix_tokens"]
                or transition not in TRANSITIONS):
            raise ValueError("native fire/units identity differs")
        counts[transition + "|all_fires"] += 1
        family[fam][transition + "|all_fires"] += 1
        n = unit["token_end"]
        tier = "at_most_8192" if n <= 8192 else (
            "utility_late_8193_to_16384" if n <= MAX_PREFIX else "beyond_utility_16384")
        counts[transition + "|" + tier] += 1
        family[fam][transition + "|" + tier] += 1
        if n > MAX_PREFIX:
            continue
        attempt = event["attempt_id"]
        if attempt not in trace_cache:
            source = by_attempt[attempt]
            trace = trace_at(source["source_location"])
            if trace_digest(TraceRecord(**trace)) != source["trace_sha256"]:
                raise ValueError("native cost trace digest changed")
            trace_cache[attempt] = trace
        trace = trace_cache[attempt]
        replay = trace["metadata"]["token_replay"]
        ids, offsets, text = replay["completion_token_ids"], replay["completion_offsets"], trace["cot_text"]
        if (len(ids) != len(offsets) or n < 1 or n > len(ids)
                or replay["tokenizer_sha256"] != native_vocab_fingerprint):
            raise ValueError("native token replay or tokenizer identity changed")
        end = int(offsets[n - 1][1])
        prefix = text[:end]
        reasoning = trace["metadata"].get("reasoning_content")
        start = text.find(reasoning) if reasoning else -1
        if start < 0 or end > start + len(reasoning):
            counts[transition + "|after_reasoning_closure"] += 1
            continue
        row = {"transition": transition,
               "reader_input": {"problem": unit["inputs"]["problem_statement"],
                                "emitted_prefix": prefix,
                                "triggering_sentence": unit["inputs"]["sentence"]}}
        try:
            messages = veto.messages(row)
        except ValueError:
            counts[transition + "|prefix_alignment_failure"] += 1
            continue
        prompt = count_prompt_tokens(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
        lengths[transition].append(prompt)
        counts[transition + "|context_checked"] += 1
        if prompt + veto.MAX_TOKENS > CONTEXT:
            counts[transition + "|context_overflow"] += 1
            continue
        counts[transition + "|online_context_eligible"] += 1
        family[fam][transition + "|online_context_eligible"] += 1
        token_counts[transition + "|prompt_tokens_once"] += prompt
        token_counts[transition + "|response_cap_tokens_once"] += veto.MAX_TOKENS
        token_counts[transition + "|" + tier + "|prompt_tokens_once"] += prompt
        token_counts[transition + "|" + tier + "|response_cap_tokens_once"] += veto.MAX_TOKENS
    summary = {}
    for transition in TRANSITIONS:
        values = sorted(lengths[transition])
        summary[transition] = {
            "tested_prompt_count": len(values),
            "prompt_tokens_min": values[0] if values else None,
            "prompt_tokens_median": values[len(values) // 2] if values else None,
            "prompt_tokens_p90": values[int(.9 * (len(values) - 1))] if values else None,
            "prompt_tokens_max": values[-1] if values else None}
    body = {
        "schema": "native-veto-all-fire-burden-v0", "job_id": os.environ["SLURM_JOB_ID"],
        "sensitivity_sha256": sensitivity["sha256"], "units_sha256": units["sha256"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_driver_sha256": hashlib.sha256(Path(veto.__file__).read_bytes()).hexdigest(),
        "model_cache": model_cache,
        "native_vocab_fingerprint": native_vocab_fingerprint,
        "native_prefix_cap": MAX_PREFIX, "proposed_side_model_context": CONTEXT,
        "response_cap": veto.MAX_TOKENS,
        "all_fire_count": len(events), "counts": dict(counts),
        "tokens": dict(token_counts), "prompt_length_summary": summary,
        "by_family": {k: dict(v) for k, v in sorted(family.items())},
        "interpretation": "Exact single-call side-judge token counts for v2.2 discovery fires through the 16,384-token utility-scout output horizon. The 32,768-token side-judge engine is proposed, not GPU-qualified; late starts beyond the frozen 8,192-prefix rating frame still need a separately priced semantic audit. Events beyond utility horizon are reported but excluded from that scout. No inference or semantic precision estimate."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps({**body, "sha256": digest(body)}, indent=1) + "\n")
    print(json.dumps({"out": str(OUT), "counts": dict(counts),
                      "tokens": dict(token_counts)}))


if __name__ == "__main__":
    main()
