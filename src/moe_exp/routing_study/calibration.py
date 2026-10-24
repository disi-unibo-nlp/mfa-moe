"""Compact exact-FP8 dispatch summaries, never NF4 or pooled expert identities."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import numpy as np
from .common import CHECKPOINT, MODEL, seal, verify, checkpoint_ok, digest
from .prepare import validate_split
from .temporal import fit_timing

MARKERS = ("<think>", "</think>", "Alternatively")


def npmi(joint, marker, expert, total):
    if not 0 <= joint <= min(marker, expert) <= total or marker <= 0 or expert <= 0:
        raise ValueError("Unavailable marker/expert or inconsistent contingency counts")
    if joint == 0:
        return -1.
    if joint == total:
        return 1.
    return math.log(joint * total / (marker * expert)) / -math.log(joint / total)


def fit_identity(samples, min_support=4):
    """Same problem-balanced frequency-weighted accuracy lift as colleague fit()."""
    repeats = Counter(s["problem_key"] for s in samples)
    baseline = sum(s["is_correct"] / repeats[s["problem_key"]] for s in samples) / len(repeats)
    layers = {}
    for layer in range(40):
        mass, correct = np.zeros(256), np.zeros(256)
        support = [set() for _ in range(256)]
        for s in samples:
            info = s["layers"][str(layer)]
            rate = np.asarray(info["reasoning_counts"]) / info["reasoning_tokens"]
            mass += rate / repeats[s["problem_key"]]
            correct += rate * s["is_correct"] / repeats[s["problem_key"]]
            for expert in np.flatnonzero(rate):
                support[expert].add(s["problem_key"])
        lifts = np.divide(correct, mass, out=np.zeros(256), where=mass > 0) - baseline
        eligible = [e for e in range(256) if len(support[e]) >= min_support and mass[e] > 0 and lifts[e] > 0]
        eligible.sort(key=lambda e: (-lifts[e], -len(support[e]), e))
        scores = np.zeros(256)
        if eligible:
            for expert in eligible[:8]:
                scores[expert] = lifts[expert] / lifts[eligible[0]]
        layers[str(layer)] = {"scores": scores.tolist()}
    if not any(any(info["scores"]) for info in layers.values()):
        return {"unavailable": "No supported positive accuracy-lift targets"}
    return {"policy": {"layers": layers, "estimator": "problem_balanced_trace_frequency_weighted_accuracy_lift",
                       "min_support": min_support, "max_experts": 8, "baseline_accuracy": baseline}}


def fit_steer(samples, min_support=10):
    grouped = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        rates = np.array([sample["layers"][str(l)]["reasoning_counts"] for l in range(40)], float)
        lengths = np.array([sample["layers"][str(l)]["reasoning_tokens"] for l in range(40)])
        grouped[sample["problem_key"]][sample["is_correct"]].append(rates / lengths[:, None])
    # Every ranked difference uses both outcome arms of the SAME question.
    # Pure-outcome questions provide no within-question evidence and cannot
    # increase support, even if plentiful in one benchmark or difficulty group.
    paired = [g for g in grouped.values() if True in g and False in g]
    arms = {outcome: [np.mean(g[outcome], axis=0) for g in paired]
            for outcome in (False, True)}
    within = [yes - no for yes, no in zip(arms[True], arms[False])]
    delta = np.mean(within, axis=0) if within else None
    within_report = {"problems": len(within), "differences": delta.tolist() if within else None,
                     "status": "paired_question_ranking" if len(within) >= min_support else
                               "descriptive_only; insufficient_mixed_outcome_questions"}
    unavailable = {"unavailable": f"Fewer than {min_support} mixed-outcome supporting questions"}
    if len(within) < min_support:
        return {"promote": unavailable, "suppress": unavailable, "contrasts": [], "within_question": within_report}
    stats = []
    for layer in range(40):
        for expert in range(256):
            support = {str(outcome): int(sum(r[layer, expert] > 0 for r in arms[outcome]))
                       for outcome in (False, True)}
            joint_support = sum(yes[layer, expert] > 0 and no[layer, expert] > 0
                                for yes, no in zip(arms[True], arms[False]))
            if joint_support < min_support:
                continue
            stats.append(dict(layer=layer, expert=expert, difference=float(delta[layer, expert]),
                support=support, paired_expert_support=int(joint_support),
                group_problems={str(o): len(arms[o]) for o in (False, True)},
                within_question_problems=len(within),
                within_question_difference=float(delta[layer, expert])))
    def choose(sign):
        ranked = sorted((s for s in stats if sign*s["difference"] > 0),
                        key=lambda s: (-sign*s["difference"], s["layer"], s["expert"]))
        if len(ranked) < 2:
            return {"unavailable": "Fewer than two layer-qualified targets with paired outcome support"}
        return {"targets": [[s["layer"], s["expert"]] for s in ranked[:2]]}
    return {"promote": choose(1), "suppress": choose(-1), "contrasts": stats, "within_question": within_report,
            "method_status": "equal_question_paired_outcome_rate_difference"}


def fit_rice(samples):
    stats, unavailable = [], []
    for layer in range(40):
        entries = [s["layers"][str(layer)] for s in samples]
        total = sum(e["all_tokens"] for e in entries)
        expert_counts = np.sum([e["all_counts"] for e in entries], axis=0)
        marker_counts = {m: sum(e["marker_tokens"][m] for e in entries) for m in MARKERS}
        if not all(marker_counts.values()):
            unavailable.append(layer)
            continue
        hits = {m: np.sum([e["marker_experts"][m] for e in entries], axis=0) for m in MARKERS}
        for expert, count in enumerate(expert_counts):
            if count == 0:
                continue
            scores = {m: npmi(int(hits[m][expert]), marker_counts[m], int(count), total) for m in MARKERS}
            stats.append(dict(layer=layer, expert=expert,
                score=scores[MARKERS[0]]-scores[MARKERS[1]]-scores[MARKERS[2]], marker_npmi=scores,
                expert_tokens=int(count), marker_tokens=marker_counts, total_tokens=total))
    if unavailable:
        return {"unavailable": "Missing observed marker opportunities in layers " + str(unavailable)}
    stats.sort(key=lambda s: (-s["score"], s["layer"], s["expert"]))
    if len(stats) < 2:
        return {"unavailable": "Fewer than two supported marker experts"}
    return {"targets": [[s["layer"], s["expert"]] for s in stats[:2]], "ranking": stats,
            "alignment": "binary_event_on_final_token_of_actual_marker_occurrence",
            "actual_occurrences": {m: sum(s["marker_occurrences"][m] for s in samples) for m in MARKERS}}


def validate_samples(samples, study):
    verify(study)
    validate_split(study["split"])
    seen = set()
    for s in samples:
        verify(s)
        checkpoint_ok(s["checkpoint"])
        if s.get("backend") != study.get("backend") or s.get("code") != study.get("code"):
            raise ValueError("Calibration backend/code provenance differs")
        if s["id"] in seen or s["id"] not in study["calibration_sources"]:
            raise ValueError("Duplicate or non-calibration attempt")
        seen.add(s["id"])
        original = study["calibration_sources"][s["id"]]
        if (s["problem_key"] != original["problem_key"]
                or study["split"]["membership"][s["problem_key"]] != "calibration"):
            raise ValueError("Calibration split mismatch")
        if s["study_sha256"] != study["sha256"] or s["capture"] != "verified_modular_dispatch":
            raise ValueError("Routing checkpoint/path/study mismatch")
        if type(s["is_correct"]) is not bool or set(s["layers"]) != set(map(str, range(40))):
            raise ValueError("Need scored native routing from all layers")
        if s["infrastructure_failure"]:
            raise ValueError("Calibration capture failed")
        for info in s["layers"].values():
            if info["reasoning_tokens"] <= 0 or info["all_tokens"] <= 0:
                raise ValueError("No native reasoning routing")
            for key, length in (("reasoning_counts", info["reasoning_tokens"]),
                                ("all_counts", info["all_tokens"])):
                counts = np.asarray(info[key])
                if counts.shape != (256,) or (counts < 0).any() or (counts > length).any() or counts.sum() != length * 8:
                    raise ValueError("Native selected-ID counts invalid")
            for marker in MARKERS:
                n = info["marker_tokens"][marker]
                hits = np.asarray(info["marker_experts"][marker])
                if (type(n) is not int or not 0 <= n <= info["all_tokens"] or hits.shape != (256,)
                        or (hits < 0).any() or (hits > n).any() or hits.sum() != n * 8
                        or (hits > np.asarray(info["all_counts"])).any()):
                    raise ValueError("Invalid actual marker/expert alignment counts")
            if (sum(n for _,n in info["gap_histogram"]) != info["reasoning_tokens"]
                    or any(not np.isfinite(v) or v < 0 or type(n) is not int or n <= 0 for v,n in info["gap_histogram"])):
                raise ValueError("Unique native gaps missing")
    if seen != set(study["calibration_sources"]):
        raise ValueError("Require complete planned calibration capture")
    if {s["is_correct"] for s in samples} != {False, True}:
        raise ValueError("Calibration needs both outcomes")


def calibrate(samples, study):
    samples = list(samples)
    validate_samples(samples, study)
    from moe_exp.moe_margin_guiding.calibration import fit as fit_margin
    cached = []
    for s in samples:
        cached.append(dict(dataset=study["problems"][s["problem_key"]]["dataset"],
            problem_id=s["problem_key"], source_problem_id=s["problem_key"], model_id=MODEL,
            is_correct=s["is_correct"], metadata={"correlation_features": {
                "num_experts": 256, "top_k": 8, "num_layers": 40, "layer_indices": list(range(40)),
                "values": {f"router_boundary_margin_l{l:02d}": s["layers"][str(l)]["mean_probability_gap"]
                           for l in range(40)}}}))
    try:
        margin = {"policy": fit_margin(cached, model=MODEL, num_experts=256, top_k=8)}
    except ValueError as error:
        margin = {"unavailable": str(error)}
    binding = {"split_sha256": study["split"]["sha256"], "checkpoint": CHECKPOINT,
               "attempts": sorted(s["id"] for s in samples)}
    result = dict(schema=1, study_sha256=study["sha256"], **binding, sample_hashes=sorted(s["sha256"] for s in samples),
                  identity=fit_identity(samples), steer=fit_steer(samples), rice=fit_rice(samples), margin=margin)
    try:
        result["timing"] = fit_timing(samples, binding)
    except ValueError as error:
        result["timing"] = {"unavailable": str(error)}
    return seal(result)
