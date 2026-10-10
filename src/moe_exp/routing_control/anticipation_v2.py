"""Qualified token-row anticipation frame with fixed, past-only text features."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import re

import numpy as np

from .prefix import PrefixInput
from .state_analysis_v1 import predictive_ablation, trailing_features, validate_capture
from .study_v1 import STATES, digest, require
from .study_v2 import seal, verify


def prefix_baseline(prefix, recognition):
    """Hashing has no fitted vocabulary/IDF; all later fitting is fold local."""
    require(recognition.get("prefix_sha256") == digest(asdict(prefix)) and recognition["source_state"] in STATES and
            all(state in STATES for state in recognition["state_history"]) and
            type(recognition["run_age_tokens"]) is int and 0 <= recognition["run_age_tokens"] <= len(prefix.emitted_token_ids),
            "prefix baseline receipt differs")
    text = np.zeros(128)
    for word in re.findall(r"\w+|[^\w\s]", prefix.emitted_text.lower()):
        code = hashlib.sha256(word.encode()).digest()
        text[int.from_bytes(code[:4], "little") % 128] += 1 if code[4] % 2 else -1
    norm = np.linalg.norm(text)
    if norm: text /= norm
    state = np.array([recognition["source_state"] == s for s in STATES], float)
    history = np.array([recognition["state_history"].count(s) for s in STATES], float)
    if history.sum(): history /= history.sum()
    return np.r_[text, np.log1p(len(prefix.emitted_token_ids)), state, history,
                 np.log1p(recognition["run_age_tokens"])]


def build_prediction_frame(traces, recognizer_audit, alignment_audit, decode, *, family_boundary):
    """Future outcomes are training labels, never prefix features.

    All traces must be native discovery captures. Recognition receipts bind the
    exact prefix, with audited source state/history/run age and no future text.
    Unknown recognition rows stay excluded with explicit coverage identities.
    """
    verify(recognizer_audit)
    verify(alignment_audit)
    from .families_v2 import require_discovery
    require_discovery([r["family"] for r in traces], family_boundary)
    require(recognizer_audit.get("status") == alignment_audit.get("status") == "PASS" and
            recognizer_audit.get("prefix_only") is True and alignment_audit.get("prediction_row_alignment") is True and
            alignment_audit.get("closure_row_included") is True, "anticipation qualification unavailable")
    rows, baseline, static, temporal, shuffled, selection, excluded = [], [], [], [], [], [], []
    for trace in traces:
        require(trace.get("population") == "discovery" and trace.get("arm") == "native" and
                trace.get("alignment_audit_sha256") == alignment_audit["sha256"] and trace.get("seed") in (0, 1) and
                trace.get("prediction_positions") == list(range(len(trace["emitted_token_ids"]))),
                "nonnative trace, foreign family pool or shifted prediction rows")
        ids, weights = validate_capture(trace["emitted_token_ids"], trace["executed_ids"], trace.get("selected_weights"))
        for opportunity in trace["prediction_rows"]:
            position = opportunity["position"]
            require(type(position) is int and 0 <= position <= len(ids) and type(opportunity["outcome"]) is bool,
                    "invalid source boundary or prediction label")
            prefix_ids = trace["emitted_token_ids"][:position]
            if 248069 in prefix_ids:
                excluded.append({"uid": opportunity["uid"], "reason": "REASONING_CLOSED"})
                continue
            recognition = opportunity.get("recognition")
            if not recognition or recognition.get("audit_sha256") != recognizer_audit["sha256"] or recognition.get("status") != "DECIDED":
                excluded.append({"uid": opportunity["uid"], "reason": "MISSING_OR_AMBIGUOUS_PREFIX_RECOGNITION"})
                continue
            prefix = PrefixInput.from_record({"problem": trace["problem"], "emitted_token_ids": prefix_ids,
                                              "emitted_text": decode(prefix_ids)})
            features = [trailing_features(ids[:position], position, weights=None if weights is None else weights[:position], window=w)
                        for w in (64, 256)]
            if any(f["status"] != "AVAILABLE" or f["acceleration"] is None for f in features):
                excluded.append({"uid": opportunity["uid"], "reason": "INSUFFICIENT_TRAILING_HISTORY"})
                continue
            rng = np.random.default_rng(int(digest(["order-shuffle-v2", trace["uid"], position])[:16], 16))
            order = np.arange(position)
            # Align from the current boundary and shuffle within 64-token
            # blocks. Every trailing 64/256 composition is thereby preserved.
            for right in range(position, 0, -64):
                left = max(0, right - 64)
                order[left:right] = rng.permutation(order[left:right])
            permuted = ids[:position][order]
            shuffled_features = [trailing_features(permuted, position, window=w) for w in (64, 256)]
            def dynamics(frame):
                return np.r_[frame["turnover"], frame["drift"], frame["velocity"].ravel(), frame["acceleration"].ravel()]
            # Static controls use each exact window's composition; shuffle only
            # within the trailing history for temporal features.
            row = {"uid": opportunity["uid"], "family": trace["family"], "population": "discovery", "seed": trace["seed"],
                "position": position, "source_state": recognition["source_state"], "prefix_status": recognition["candidate_status"],
                "transition": opportunity["transition"], "outcome": opportunity["outcome"],
                "baseline_contract": "prefix_text_position_state_history_run_age", "prediction_alignment_qualified": True,
                "prefix_ids_sha256": digest(prefix_ids)}
            rows.append(row)
            baseline.append(prefix_baseline(prefix, recognition))
            static.append(np.concatenate([f["selection"].ravel() for f in features]))
            temporal.append(np.concatenate([dynamics(f) for f in features]))
            shuffled.append(np.concatenate([dynamics(f) for f in shuffled_features]))
            selection.append(features[0]["selection"])
    require(rows and len({r["uid"] for r in rows}) == len(rows), "no qualified or duplicate prediction rows")
    return seal({"schema": "routing-context-prediction-frame-v2", "rows": rows,
        "baseline": np.asarray(baseline).tolist(), "static": np.asarray(static).tolist(),
        "temporal": np.asarray(temporal).tolist(), "order_shuffled": np.asarray(shuffled).tolist(),
        "selection": np.asarray(selection).tolist(), "excluded": excluded,
        "family_boundary": family_boundary,
        "recognizer_audit_sha256": recognizer_audit["sha256"], "alignment_audit_sha256": alignment_audit["sha256"],
        "weights_analysis": "SEPARATE_FROM_ID_CONTEXT_MODEL"})


def analyze_anticipation(frame):
    verify(frame)
    return seal({"schema": "routing-anticipation-results-v2", "frame_sha256": frame["sha256"],
        "variants": predictive_ablation(frame["rows"], frame["baseline"], frame["static"], frame["temporal"], frame["order_shuffled"]),
        "excluded": frame["excluded"], "inference": "Nested family prediction; no causal anticipation claim"})
