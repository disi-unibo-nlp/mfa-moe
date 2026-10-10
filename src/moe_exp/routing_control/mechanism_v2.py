"""Early substantive endpoint separate from the 16k answer-utility endpoint."""
from __future__ import annotations

from .policy_stages_v2 import THINK_END_ID
from .state_analysis_v1 import transition_summary
from .study_v1 import STATES, require
from .study_v2 import CAP, EARLY_WINDOW, TRANSITIONS


def early_endpoint(tokens, sentences, transition, *, finish_reason):
    require(transition in TRANSITIONS and finish_reason in ("stop", "length", "error") and
            len(tokens) <= CAP and all(type(t) is int and t >= 0 for t in tokens), "invalid v2 mechanism trace")
    require(finish_reason != "length" or len(tokens) == CAP, "16k length finish required")
    closure = tokens.index(THINK_END_ID) if THINK_END_ID in tokens else None
    end = min(EARLY_WINDOW, closure if closure is not None else len(tokens))
    require(len({s["sentence_index"] for s in sentences}) == len(sentences), "duplicate sentence identity")
    ordered = sorted(sentences, key=lambda s: s["token_start"])
    require(all(a["token_end"] <= b["token_start"] for a, b in zip(ordered, ordered[1:])),
            "overlapping sentence ownership")
    rows, successes, missing = [], [], []
    for sentence in sentences:
        start, stop = sentence["token_start"], sentence["token_end"]
        require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(tokens), "invalid sentence ownership")
        if start >= end:
            continue
        if stop > end or sentence.get("complete") is not True:
            rows.append({**sentence, "valid": False})
            missing.append({"sentence_index": sentence["sentence_index"], "reason": "INCOMPLETE_EARLY_SENTENCE"})
            continue
        votes = sentence.get("behavior_votes", [])
        qualified = (len(votes) == 2 and {v.get("reader") for v in votes} == {0, 1} and
            all(v.get("finish_reason") == "stop" and v.get("prefix_only") is True and
                v.get("visible_token_end") == stop and type(v.get("substantive")) is bool and
                v.get("transition") in (*TRANSITIONS, None) for v in votes) and
            len({(v["transition"], v["substantive"]) for v in votes}) == 1)
        valid = sentence.get("label") in STATES and sentence.get("label_finish_reason") == "stop" and qualified
        rows.append({**sentence, "valid": valid})
        if not valid:
            missing.append({"sentence_index": sentence["sentence_index"], "reason": "MISSING_AMBIGUOUS_OR_FUTURE_VISIBLE_RATING"})
        elif votes[0]["substantive"] and votes[0]["transition"] == transition:
            successes.append(stop)
    closed = closure is not None and closure <= EARLY_WINDOW
    covered = sorted((r["token_start"], r["token_end"]) for r in rows if r.get("valid"))
    cursor = 0
    for start, stop in covered:
        if start != cursor:
            missing.append({"token_start": cursor, "token_end": start, "reason": "UNRATED_TOKEN_GAP"})
        cursor = stop
    if cursor < end:
        missing.append({"token_start": cursor, "token_end": end, "reason": "UNRATED_TOKEN_GAP"})
    # A witnessed act remains success even if other labels are missing; a
    # negative endpoint requires complete usable measurement of the window.
    success = True if successes else None if missing or finish_reason == "error" else False
    if rows and any(r["token_end"] > end for r in rows):
        # The trailing partial sentence is a gap, never a future state.
        rows = [{**r, "token_end": min(r["token_end"], end)} for r in rows]
    return {"schema": "routing-study-early-mechanism-endpoint-v2", "transition": transition,
        "early_endpoint_window": EARLY_WINDOW, "continuation_cap": CAP, "observed_tokens": end,
        "substantive_success": success, "first_substantive_token_end": min(successes) if successes else None,
        "missing_measurement": missing, "closure_competes": closed,
        "window_censored": not closed and (end == EARLY_WINDOW or finish_reason == "length"),
        "state_trajectory": transition_summary(rows, closed=closed,
            hit_cap=not closed and (end == EARLY_WINDOW or finish_reason == "length")),
        "answer_utility": "SEPARATE_INDEPENDENT_GRADE_REQUIRED"}
