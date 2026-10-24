"""Blinded views derived from token-bound, already-emitted native prefixes."""
from __future__ import annotations

from .contracts import digest, identity, require, seal, verify
from .semantics import CHECK, INTENT, candidate_proposals, eligible_proposal
from moe_exp.routing_control.transitions_v24 import _completed_sentences


def reasoning_limit(receipt):
    tokens = receipt["emitted_token_ids"]
    markers = receipt["closure_token_sequences"]
    for index in range(len(tokens)):
        if any(tokens[index:index+len(m)] == m for m in markers):
            return index
    return len(tokens)


def prefix_tokens(tokenizer, ids, text_end):
    # Find the first token boundary that has emitted the completed unit.
    low, high = 0, len(ids)
    while low < high:
        middle = (low+high)//2
        if len(tokenizer.decode(ids[:middle], skip_special_tokens=False)) >= text_end:
            high = middle
        else:
            low = middle+1
    actual = tokenizer.decode(ids[:low], skip_special_tokens=False)
    require(len(actual) >= text_end, "cannot locate completed prefix token boundary")
    return ids[:low], actual


def audit_records(receipt, problem, tokenizer):
    verify(receipt)
    assignment = receipt["assignment"]
    require(assignment["stage"] == "discovery" and assignment["arm"] == "native" and
            not receipt["engineering_only"], "audit source is not native discovery")
    ids = receipt["emitted_token_ids"][:reasoning_limit(receipt)]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    rows = []
    for start, end, sentence in _completed_sentences(text):
        emitted, prefix = prefix_tokens(tokenizer, ids, end)
        candidates = candidate_proposals(problem, prefix)
        current = [c for c in candidates if c["char_end"] == end]
        checked = [c for c in candidates if c["checked"]]
        candidate = (current or checked or candidates or [{}])[-1]
        eligible = eligible_proposal(problem, prefix)
        already_checked = bool(CHECK.search(sentence) and not INTENT.search(sentence) and checked)
        stratum = "proposal" if eligible else "checked" if already_checked else "intent" if INTENT.search(sentence) else "unfiltered"
        row = {"uid": identity("prefix-audit", assignment["uid"], len(emitted)), "source_uid": assignment["uid"],
            "family": assignment["family"], "population": "discovery", "problem": problem,
            "prefix_text": prefix, "candidate": eligible or candidate, "prefix_token_ids": emitted,
            "prefix_ids_sha256": digest(emitted), "position": len(emitted), "prefix_only": True,
            "stratum": stratum, "proposed": {"eligible": eligible is not None, "already_checked": already_checked}}
        rows.append(row)
    # A uniform hash-sampled coverage frame includes proposals/checks/intents too.
    # Keep coverage copies distinct so their probability of selection is intact.
    return rows + [{**r, "uid": identity("unfiltered-copy", r["uid"]), "stratum": "unfiltered"} for r in rows]


def outcome_view(receipt, problem, prefix_text):
    verify(receipt)
    assignment = receipt["assignment"]
    require(assignment["stage"] == "conditional" and assignment["cap"] == 1024,
            "primary semantic view requires conditional assay")
    prefix = assignment["prefix"]
    require(prefix["candidate"]["problem_sha256"] == digest(problem), "candidate has foreign problem")
    return {"uid": identity("blind-outcome", assignment["uid"]), "family": assignment["family"],
        "problem": problem, "prefix_text": prefix_text, "candidate": prefix["candidate"],
        "horizon": 1024, "arm_blinded": True, "execution_complete": receipt["execution_complete"],
        "closure_policy": "exclude first native closure marker and all subsequent text",
        "source_receipt_sha256": receipt["sha256"], "window_token_ids": receipt["emitted_token_ids"][:reasoning_limit(receipt)]}


def past_profile_rows(frame, ratings, captures, model, *, width=256):
    """Checked-vs-unchecked destination profiles, not an anticipation claim."""
    import numpy as np
    from .semantics import parse_rating
    verify(frame); verify(model)
    by_uid = {}
    for rating in ratings:
        verify(rating)
        require(rating["frame_sha256"] == frame["sha256"], "foreign nomination label")
        by_uid.setdefault(rating["uid"], {})[rating["reader"]] = rating
    rows = []
    for record in frame["records"]:
        pair = by_uid.get(record["uid"], {})
        parsed = [parse_rating(pair[i]["text"], "eligibility") if i in pair and pair[i]["finish_reason"] == "stop" and
                  pair[i]["input_sha256"] == digest(record) else None for i in (0,1)]
        if not all(p is not None for p in parsed):
            continue
        positive = all(p["already_checked"] is True for p in parsed)
        comparison = all(p["eligible"] is True for p in parsed)
        if not (positive or comparison):
            continue
        # Audit holdout is not used to nominate or tune a target.
        if record["split"] != "apparent":
            continue
        arrays = captures[record["source_uid"]]
        for name, ids in arrays.items():
            if not name.endswith("_native_ids"):
                continue
            layer = int(name.split("_")[0][1:])
            positions = arrays[f"L{layer}_positions"]
            mask = (positions < record["position"]) & (positions >= max(0, record["position"]-width))
            if not mask.any():
                continue
            for expert in range(model["experts"]):
                rows.append({"family": record["family"], "population": "discovery", "information_boundary": "past_only",
                    "matched": True, "condition": "positive" if positive else "comparison", "layer": layer,
                    "expert": expert, "inclusion_rate": float((ids[mask] == expert).any(-1).mean()),
                    "source_uid": record["source_uid"], "prefix_uid": record["uid"], "width": width,
                    "claim_scope": "destination-state association; not next-state anticipation",
                    "matching_scope": "family/source-native trace; not position or candidate matched"})
    # Every retained comparison is matched within family by nominate().
    return rows
