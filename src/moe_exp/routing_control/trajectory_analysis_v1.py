"""Strict CPU reference for the frozen 1,024-token, seven-class local endpoint.

This consumes *post-generation* token ownership and arm-blind offline ratings.
It does not infer sentence boundaries, invent missing ratings, or run a detector.
All assignments become ITT receipts, including failures and nonfires.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

from .analysis import CLASSES, class_summary, paired_itt, trajectory_completion
from .design import TRANSITIONS


SCHEMA = "dense-trajectory-analysis-input-v1"
THINK_END_ID = 248069  # Frozen Qwen3.6 reasoning-closure token.
TARGET_CLASS = {"candidate_to_verify": {"Verify"},
                "approach_to_commit": {"Plan", "Implement"},
                "failed_check_to_revise": {"Explore", "Plan"}}
STATUSES = {"complete", "nonfire", "early_finish", "cap", "failure", "unscored"}
BINDINGS = {"generation_manifest_sha256", "generation_summary_sha256",
            "tokenizer_config_sha256", "sentence_label_binding_sha256",
            "semantic_rating_binding_sha256", "blind_map_sha256", "rubric_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_seal(value):
    if type(value) is not dict or value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("trajectory analysis input seal changed")
    return value


def _integer(value, lower=0):
    return type(value) is int and value >= lower


def _identity(row):
    fields = ("uid", "question", "family", "arm", "seed")
    if not all(k in row for k in fields):
        raise ValueError("assignment or observation identity is incomplete")
    if any(type(row[k]) is not str or not row[k] for k in fields[:-1]) or not _integer(row["seed"]):
        raise ValueError("invalid assignment or observation identity")
    return tuple(row[k] for k in fields)


def _trace(observation, trigger_index):
    """Return gradeable rows and a measurement status, preserving gaps as failures.

    `token_owner` binds each emitted reasoning token to its sentence index or null.
    The supplied routed positions must cover exactly the same reasoning tokens.
    This is a structural check; the producer must separately bind the tokenizer,
    decoder and raw routing arrays to their hashes.
    """
    tokens = observation["emitted_token_ids"]
    if type(tokens) is not list or any(not _integer(t) for t in tokens) or len(tokens) > 1024:
        raise ValueError("invalid or over-horizon continuation token IDs")
    closure = tokens.index(THINK_END_ID) if THINK_END_ID in tokens else len(tokens)
    if observation["reasoning_closed"] is not (closure < len(tokens)):
        raise ValueError("reasoning closure disagrees with emitted token IDs")
    if observation["reasoning_token_end"] != closure:
        raise ValueError("reasoning end disagrees with closure token position")
    owners = observation["token_owner"]
    routed = observation["routed_positions"]
    if type(owners) is not list or len(owners) != closure or any(
            x is not None and not _integer(x) for x in owners):
        raise ValueError("token ownership must cover exactly the emitted reasoning prefix")
    if type(routed) is not list or routed != list(range(closure)):
        raise ValueError("routed arrays do not align to every reasoning token")
    sentences = observation["sentences"]
    if type(sentences) is not list:
        raise ValueError("sentence rows must be a list")
    if not sentences:
        return [], "no_post_trigger_sentence", closure
    indices = [r.get("sentence_index") for r in sentences]
    if any(not _integer(i) for i in indices) or indices != list(
            range(trigger_index + 1, trigger_index + len(sentences) + 1)):
        raise ValueError("post-trigger sentences must be dense, ordered and exclude the trigger")
    segments = {r.get("segment") for r in sentences}
    if len(segments) != 1 or any(not _integer(s) for s in segments):
        raise ValueError("reasoning segment changed inside the local window")
    seen = set()
    previous_end = 0
    gradeable = []
    measure = "rated"
    for j, row in enumerate(sentences):
        index = indices[j]
        start, end = row.get("token_start"), row.get("token_end")
        if not _integer(start) or not _integer(end, 1) or start >= end or end > closure:
            raise ValueError("sentence token offsets exceed emitted reasoning tokens")
        if start < previous_end:
            raise ValueError("sentence token offsets overlap")
        if owners[start:end] != [index] * (end - start):
            raise ValueError("sentence offsets disagree with token ownership")
        if [p for p, owner in enumerate(owners) if owner == index] != list(range(start, end)):
            raise ValueError("sentence ownership is fragmented or extends outside its offsets")
        seen.update(range(start, end))
        previous_end = end
        if type(row.get("complete")) is not bool or (not row["complete"] and j != len(sentences)-1):
            raise ValueError("only the final sentence may be incomplete")
        if not row["complete"]:
            continue
        if row.get("label") not in (*CLASSES, None):
            raise ValueError("unknown seven-class label")
        if row.get("label") is None or row.get("label_finish_reason") != "stop":
            measure = "unscored_class"
            continue
        votes = row.get("behavior_votes")
        if type(votes) is not list or len(votes) != 2 or {
                v.get("reader") for v in votes if type(v) is dict} != {"reader0", "reader1"}:
            measure = "unscored_behavior"
            continue
        bad_vote = False
        for vote in votes:
            if vote.get("finish_reason") != "stop" or vote.get("transition") not in (*TRANSITIONS, None) or \
                    type(vote.get("substantive")) is not bool:
                bad_vote = True
        if bad_vote:
            measure = "unscored_behavior"
            continue
        if (votes[0]["transition"], votes[0]["substantive"]) != (
                votes[1]["transition"], votes[1]["substantive"]):
            measure = "reader_disagreement"
            continue
        event = votes[0]["transition"]
        if event is not None and row["label"] not in TARGET_CLASS[event]:
            measure = "class_behavior_disagreement"
            continue
        gradeable.append({"sentence_index": index, "segment": row["segment"],
                          "token_start": start, "token_end": end,
                          "label": row["label"], "behavioral_transition": event,
                          "substantive": votes[0]["substantive"]})
    if {owner for owner in owners if owner is not None} != set(indices):
        raise ValueError("token owner refers to an absent sentence")
    return (gradeable if measure == "rated" else []), measure, closure


def analyze(value, *, n_boot=5000):
    """Produce an all-assignment receipt table and frozen paired contrasts.

    Input is a sealed, versioned join of the generation ledger and independent
    arm-blind sentence/behavior ratings. A missing observation is an incomplete
    experiment, never an implicit failure receipt.
    """
    verify_seal(value)
    if value.get("schema") != SCHEMA or value.get("horizon") != 1024:
        raise ValueError("wrong trajectory schema or horizon")
    bindings = value.get("bindings")
    if type(bindings) is not dict or set(bindings) != BINDINGS or any(
            type(v) is not str or len(v) != 64 or set(v) - set("0123456789abcdef")
            for v in bindings.values()):
        raise ValueError("generation, tokenizer and blinded measurement bindings required")
    ordered = value.get("ordered_transitions")
    if type(ordered) is not list or not 1 <= len(ordered) <= 2 or any(t not in TRANSITIONS for t in ordered):
        raise ValueError("one or two frozen registered transitions required")
    pairs = value.get("comparison_pairs")
    if type(pairs) is not list or any(type(pair) is not list or len(pair) != 2 for pair in pairs):
        raise ValueError("comparison pairs must be explicitly frozen")
    assignments, observations = value.get("assignments"), value.get("observations")
    if type(assignments) is not list or type(observations) is not list or not assignments:
        raise ValueError("assignments and observations are required")
    assigned = {}
    for request in assignments:
        ident = _identity(request)
        if ident[0] in assigned:
            raise ValueError("duplicate assigned UID")
        if not _integer(request.get("trigger_sentence_index")):
            raise ValueError("missing trigger sentence index")
        assigned[ident[0]] = request
    observed = {}
    for row in observations:
        ident = _identity(row)
        if ident[0] in observed or ident[0] not in assigned or ident != _identity(assigned[ident[0]]):
            raise ValueError("duplicate, unassigned or mismatched observation")
        observed[ident[0]] = row
    if set(observed) != set(assigned):
        raise ValueError("missing ITT observation receipts")
    receipts, class_groups, measurement = [], [], Counter()
    for uid in sorted(assigned):
        assignment, row = assigned[uid], observed[uid]
        status = row.get("status")
        if status not in STATUSES or not _integer(row.get("injected_token_count")):
            raise ValueError("unknown status or injection count")
        if row.get("correct") not in (0, 1) or type(row["correct"]) is not int:
            raise ValueError("binary correctness receipt required")
        if status in {"failure", "unscored"} and row["correct"] != 0:
            raise ValueError("failed/unscored accuracy must be zero")
        if status == "failure":
            tokens = row.get("emitted_token_ids")
            if type(tokens) is not list or any(not _integer(t) for t in tokens) or len(tokens) > 1024:
                raise ValueError("failure/nonfire emitted token count is invalid")
            semantic, measure, rows = 0, status, []
        else:
            rows, measure, _ = _trace(row, assignment["trigger_sentence_index"])
            if measure == "rated":
                semantic = trajectory_completion(rows, triggering_sentence=assignment["trigger_sentence_index"],
                                                 ordered_transitions=ordered, horizon=1024)["success"]
                class_groups.append(rows)
            else:
                semantic = 0
            tokens = row["emitted_token_ids"]
        measurement[measure] += 1
        receipts.append({"uid": uid, "question": assignment["question"], "family": assignment["family"],
                         "arm": assignment["arm"], "seed": assignment["seed"], "status": status,
                         "semantic_success": semantic, "correct": row["correct"],
                         "tokens": len(tokens) + row["injected_token_count"],
                         "emitted_tokens": len(tokens), "injected_tokens": row["injected_token_count"],
                         "semantic_measurement": measure, "rated_complete_sentences": len(rows)})
    contrasts = paired_itt(assignments, receipts, comparison_pairs=[tuple(p) for p in pairs],
                           n_boot=n_boot) if pairs else None
    summary = _combine_class_summaries(class_groups)
    result = {"schema": "dense-trajectory-analysis-result-v1", "input_sha256": value["sha256"],
              "population": value.get("population"), "horizon": 1024,
              "ordered_transitions": ordered, "comparison_pairs": pairs,
              "assignments": len(assignments), "measurement_statuses": dict(measurement),
              "receipts": receipts, "class_summary": summary, "paired_itt": contrasts,
              "interpretation": "Arm-blind LLM ratings of a frozen local 1024-token endpoint; no latent-reasoning or universal-optimum claim."}
    result["sha256"] = digest(result)
    return result


def _combine_class_summaries(groups):
    """Sum within-request class dynamics; never join two requests or sparse gaps."""
    counts = [[0 for _ in CLASSES] for _ in CLASSES]
    dwell = {name: [] for name in CLASSES}
    reentries, loops = Counter(), Counter()
    for rows in groups:
        if not rows:
            continue
        one = class_summary(rows)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                counts[i][j] += one["transition_counts"][i][j]
        for name in CLASSES:
            dwell[name].extend(one["dwell_sentences_observed_including_censoring"][name])
        reentries.update(one["reentries"])
        loops.update(one["loop_counts"])
    probabilities = [[counts[i][j] / sum(counts[i]) if sum(counts[i]) else 0.0
                      for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    return {"classes": list(CLASSES), "transition_counts": counts,
            "transition_probabilities": probabilities,
            "dwell_sentences_observed_including_censoring": dwell,
            "reentries": dict(reentries), "loop_counts": dict(loops),
            "note": "Within-request observed windows only; boundaries censor dwell."}
