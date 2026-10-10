"""Family-equal policy resource accounting and descriptive exposure diagnostics."""
from collections import defaultdict

import numpy as np

from .study_v1 import require

RESOURCE_FIELDS = ("generated_tokens", "controller_tokens", "reasoning_tokens", "processed_input_tokens",
                   "controller_input_tokens", "intervened_positions", "controller_seconds", "wall_seconds")


def itemize(rows, families, arms):
    expected = {(f, s, a) for f in families for s in (0, 1) for a in arms}
    require(len(expected) == len(families)*2*len(arms), "duplicate family or arm")
    indexed = {}
    for row in rows:
        key = row["family"], row["seed"], row["arm"]
        require(key in expected and key not in indexed, "duplicate or foreign resource assignment")
        for field in RESOURCE_FIELDS:
            value = row.get(field)
            require(value is None or isinstance(value, (int, float)) and not isinstance(value, bool) and
                    np.isfinite(value) and value >= 0, "invalid measured resource count")
            require(value is None or field.endswith("seconds") or type(value) is int, "token/position count must be integer")
        require(row.get("reasoning_tokens") is None or row.get("generated_tokens") is None or
                row["reasoning_tokens"] <= row["generated_tokens"], "reasoning tokens exceed generation")
        indexed[key] = row
    result = []
    for arm in arms:
        metrics = {}
        for field in (*RESOURCE_FIELDS, "total_policy_tokens"):
            values = []
            for family in families:
                seeds = []
                for seed in (0, 1):
                    row = indexed.get((family, seed, arm), {})
                    value = (row["generated_tokens"] + row["controller_tokens"] if field == "total_policy_tokens" and
                             row.get("generated_tokens") is not None and row.get("controller_tokens") is not None else
                             None if field == "total_policy_tokens" else row.get(field))
                    seeds.append(value)
                if all(v is not None for v in seeds):
                    values.append(float(np.mean(seeds)))
            metrics[field] = {"mean": float(np.mean(values)) if values else None,
                "measured_families": len(values), "assigned_families": len(families),
                "status": "COMPLETE" if len(values) == len(families) else "UNAVAILABLE_OR_PARTIAL"}
        result.append({"arm": arm, "metrics": metrics})
    return {"schema": "routing-study-policy-resources-v1", "arms": result,
            "offline_annotation_and_grading": "Separate research ledger, excluded from policy-token cost",
            "weighting": "Two seeds averaged within family; families equally weighted; field-specific missingness explicit"}


def exposure_diagnostics(rows, families, policy):
    indexed = {}
    for row in rows:
        key = row["family"], row["seed"], row["arm"]
        require(key not in indexed, "duplicate diagnostic assignment")
        indexed[key] = row
    by_family, missing = [], []
    for family in families:
        seeds = []
        for seed in (0, 1):
            a, n = indexed.get((family, seed, policy), {}), indexed.get((family, seed, "native"), {})
            if type(a.get("correct")) is not bool or type(n.get("correct")) is not bool or a.get("intervened_positions") is None:
                missing.append({"family": family, "seed": seed})
                continue
            exposure = a["intervened_positions"]
            require(type(exposure) is int and exposure >= 0, "invalid intervention exposure")
            damage = int(n["correct"]) - int(a["correct"])
            c_to_i = int(n["correct"] and not a["correct"])
            i_to_c = int(a["correct"] and not n["correct"])
            seeds.append([damage, c_to_i, i_to_c, exposure])
        if len(seeds) == 2:
            by_family.append(np.mean(seeds, axis=0))
    if not by_family:
        return {"status": "UNAVAILABLE_PAIRED_GRADES_AND_EXPOSURE", "policy": policy, "missing": missing}
    damage, c_to_i, i_to_c, exposure = np.mean(by_family, axis=0)
    return {"schema": "routing-study-exposure-diagnostics-v1", "policy": policy,
        "measured_families": len(by_family), "assigned_families": len(families), "missing": missing,
        "mean_intervened_positions": float(exposure), "accuracy_damage_probability": float(damage),
        "correct_to_incorrect_probability": float(c_to_i), "incorrect_to_correct_probability": float(i_to_c),
        "accuracy_damage_per_intervened_position": float(damage / exposure) if exposure else None,
        "net_correct_to_incorrect_flips_per_intervened_position": float((c_to_i - i_to_c) / exposure) if exposure else None,
        "status": "DESCRIPTIVE_RATIO" if exposure else "UNAVAILABLE_ZERO_EXPOSURE",
        "claim_limit": "Descriptive ratio of family-equal means; unstable near zero dose; no causal specialist or accuracy-retention inference."}
