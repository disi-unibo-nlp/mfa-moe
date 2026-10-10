"""Additive descriptions of frozen pulse studies; no replacement estimators."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .state_analysis_v1 import transition_summary, validate_capture
from .study_v1 import STATES, require


REFERENCE_IDENTITIES = {"candidate_to_verify": ((28, 189), (28, 9)),
                        "approach_to_commit": ((24, 159),)}


def describe_windows(record, mapped, ids, *, horizon, width=256):
    """Retain closure, missing labels, and exposure denominators independently.

    Saved IDs include every emitted token's prediction row. Reasoning windows
    exclude answer production according to the frozen reasoning_end. Class
    occupancy uses exact sentence token ownership; it cannot establish a
    substantive verification or a correctness outcome.
    """
    require(record["uid"] == mapped["uid"] and horizon in (256, 1024), "historical contract mismatch")
    require(record["emitted_tokens"] == len(mapped["emitted_token_ids"]), "emitted token count changed")
    validate_capture(mapped["emitted_token_ids"], ids)
    end = mapped["reasoning_end"]
    require(type(end) is int and 0 <= end <= len(ids) and end == record["reasoning_tokens"],
            "reasoning boundary changed")
    require(len(mapped["token_owner"]) == end, "reasoning ownership alignment changed")
    owners = mapped["token_owner"]
    labels, measured = {}, set()
    for sentence in record["sentences"]:
        if sentence["measurement_status"] == "parsed_stop":
            require(sentence["label"] in STATES, "invalid measured reasoning class")
            labels[sentence["sentence_index"]] = sentence["label"]
            indices = sentence["token_indices"]
            require(all(0 <= i < end and owners[i] == sentence["sentence_index"] for i in indices) and
                    not measured.intersection(indices), "sentence/token ownership differs")
            measured.update(indices)
    identities = REFERENCE_IDENTITIES[record["transition"]]
    windows = []
    for left in range(0, horizon, width):
        right = min(left + width, horizon)
        observed_end = max(left, min(end, right))
        n = observed_end - left
        counts = {state: 0 for state in STATES}
        for i in range(left, observed_end):
            label = labels.get(owners[i])
            if label is not None:
                counts[label] += 1
        windows.append({"start": left, "end": right, "intended_tokens": right - left,
            "observed_reasoning_tokens": n, "measured_reasoning_tokens": sum(counts.values()),
            "unknown_reasoning_tokens": n - sum(counts.values()), "class_token_counts": counts,
            "class_occupancy_per_intended_token": {state: count / (right - left) for state, count in counts.items()},
            "class_occupancy_among_measured_tokens": {state: count / sum(counts.values())
                if sum(counts.values()) else None for state, count in counts.items()},
            "reference_expert_selection": [{"layer": layer, "expert": expert,
                "selected_rows": int(np.any(ids[left:observed_end, layer] == expert, axis=1).sum()),
                "per_intended_token": float(np.any(ids[left:observed_end, layer] == expert, axis=1).sum()) / (right-left),
                "conditional_on_observed_reasoning": float(np.any(ids[left:observed_end, layer] == expert, axis=1).mean())
                    if n else None} for layer, expert in identities],
            "closure_observed_by_end": bool(record["reasoning_closed"] and end <= right),
            "closure_in_window": bool(record["reasoning_closed"] and left < end <= right),
            "cap_censored": bool(record["hit_cap"] and not record["reasoning_closed"] and end < right),
            "generation_error": bool(record["generation_error"])})
    sentences = [{**s, "valid": s["measurement_status"] == "parsed_stop"} for s in record["sentences"]]
    trajectory = transition_summary(sentences, closed=record["reasoning_closed"],
                                    hit_cap=record["hit_cap"] and not record["reasoning_closed"])
    return {"uid": record["uid"], "family": record["family"], "prefix_uid": record["prefix_uid"],
        "transition": record["transition"], "arm": record["arm"], "seed": record["seed"],
        "windows": windows, "trajectory": trajectory,
        "weights": "UNAVAILABLE_PER_TOKEN_EXECUTED_WEIGHTS_NOT_SAVED",
        "confidence": "UNAVAILABLE_GATE_PROBABILITIES_AND_NATIVE_BOUNDARY_NOT_SAVED"}


def window_contrasts(records, contrasts):
    """Descriptive paired effects: seeds, then starts, then equal families.

    No new intervals are attached to post hoc windows. Unknown labels are
    reported as a separate metric; a missing route archive remains missing.
    """
    indexed = {}
    for record in records:
        key = record["prefix_uid"], record["seed"], record["arm"]
        require(key not in indexed, "duplicate historical continuation")
        indexed[key] = record
    result = []
    for left_arm, right_arm in contrasts:
        cells = defaultdict(list)
        for (prefix, seed, arm), left in indexed.items():
            if arm != left_arm:
                continue
            right = indexed.get((prefix, seed, right_arm))
            if right is None or "windows" not in left or "windows" not in right:
                continue
            require(left["family"] == right["family"] and left["transition"] == right["transition"],
                    "historical paired family differs")
            for a, b in zip(left["windows"], right["windows"], strict=True):
                require((a["start"], a["end"]) == (b["start"], b["end"]), "paired windows differ")
                metrics = {"observed_reasoning_tokens": a["observed_reasoning_tokens"] - b["observed_reasoning_tokens"],
                    "unknown_reasoning_tokens": a["unknown_reasoning_tokens"] - b["unknown_reasoning_tokens"],
                    "closure_by_end": int(a["closure_observed_by_end"]) - int(b["closure_observed_by_end"])}
                for state in STATES:
                    metrics["class_occupancy:" + state] = (a["class_occupancy_per_intended_token"][state] -
                                                           b["class_occupancy_per_intended_token"][state])
                for x, y in zip(a["reference_expert_selection"], b["reference_expert_selection"], strict=True):
                    require((x["layer"], x["expert"]) == (y["layer"], y["expert"]), "expert identity differs")
                    metrics[f'selection:L{x["layer"]}E{x["expert"]}'] = x["per_intended_token"] - y["per_intended_token"]
                for metric, difference in metrics.items():
                    cells[left["transition"], a["start"], a["end"], metric, left["family"], prefix].append(difference)
        families = defaultdict(lambda: defaultdict(list))
        for (transition, start, end, metric, family, _prefix), seeds in cells.items():
            families[transition, start, end, metric][family].append(float(np.mean(seeds)))
        for (transition, start, end, metric), values in sorted(families.items()):
            result.append({"left": left_arm, "right": right_arm, "transition": transition,
                "start": start, "end": end, "metric": metric,
                "estimate": float(np.mean([np.mean(starts) for starts in values.values()])),
                "families": len(values), "assigned_paired_starts": sum(len(starts) for starts in values.values()),
                "paired_seed_cells": sum(len(seeds) for key, seeds in cells.items() if key[:4] == (transition, start, end, metric)),
                "status": "DESCRIPTIVE_POST_HOC_WINDOW_NO_NEW_SIGNIFICANCE_CLAIM"})
    return result


def dose_descriptions(telemetry, actions):
    """Keep TP rank observations separate; sparse L1/2 is weight displacement."""
    action_map = {a["name"]: a for a in actions}
    result = []
    for row in telemetry["records"]:
        action = action_map[row["action"]]
        experts = dict(action["experts"]).get(row["layer"])
        if not experts:
            continue  # Worker-padded layers do not represent an intervention.
        totals = row["totals"]
        exposure = totals["active_rows"]
        result.append({**row, "target_experts": experts, "operator": action.get("kind", "bias"),
            "executed_target_mass_per_active_row": totals["target_mass"] / exposure if exposure else None,
            "selected_weight_L1_per_active_row": totals["weight_l1"] / exposure if exposure else None,
            "selected_weight_TV_per_active_row": totals["weight_l1"] / (2 * exposure) if exposure else None,
            "net_added_target_hits_per_active_row": (totals["actual_target_hits"] - totals["native_target_hits"]) / exposure
                if exposure else None,
            "rank_handling": "Separate execution-local telemetry; TP ranks are never summed as observations.",
            "weight_semantics": "Normalized executed sparse top-k weights at the same hidden state; not full router probabilities."})
    return result
