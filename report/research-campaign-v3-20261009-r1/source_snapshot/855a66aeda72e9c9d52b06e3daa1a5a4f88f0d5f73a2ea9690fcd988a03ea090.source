"""Prospective v3 amendment: fixed conditional assay and policy ITT."""
from __future__ import annotations

from .contracts import identity, require, seal, verify
from .registry import MODEL_KEYS

ARMS = ("native", "target", "matched_random")
FULL_CAP = 16384
SEMANTIC_CAP = 1024
PULSE = 256


def validate_assignment(row, *, calibration=False):
    stage=row["stage"]
    require(stage in ("discovery","validation","final_native","utility","conditional") and
            row["model"] in MODEL_KEYS and row["seed"] in ((0,) if stage=="discovery" else (0,1)),
            "assignment stage/model/seed differs from protocol")
    arms=ARMS if stage=="conditional" else ("target","matched_random") if stage=="utility" else ("native",)
    cap=64 if calibration and stage=="discovery" else SEMANTIC_CAP if stage=="conditional" else FULL_CAP
    require(row["arm"] in arms and row["cap"]==cap and
            bool(row.get("prefix")) == (stage=="conditional" or calibration), "assignment arm/cap/prefix differs")
    return row


def protocol():
    return seal({"schema": "verification-protocol-v3", "models": list(MODEL_KEYS),
        "target": "specific unchecked numeric or symbolic candidate against an original problem constraint",
        "candidate_kinds": ["intermediate", "complete"], "event": "substantive_verification",
        "arms": list(ARMS), "seeds": [0, 1], "discovery_families": 48,
        "validation_families": 32, "final_families": 96, "native_source_seed": 0,
        "source_seed_fallback": False, "conditional_cap": SEMANTIC_CAP, "utility_cap": FULL_CAP,
        "pulse_tokens": PULSE, "operator": "native_logit_selection_bias", "targets_per_model": 1,
        "target_inclusion_increment": .25, "inclusion_tolerance": .025,
        "native_exposure_tolerance": .025, "relative_weight_displacement_mismatch": .20,
        "calibration_bracket_expansions": 8, "calibration_bisections": 12,
        "recognizer_support_families": 10, "apparent_precision": .70,
        "audit_prefix_cap": 128, "audit_family_split": [36, 12],
        "veto_call_cap": 16, "veto_output_cap": 1024,
        "validation_role": "pipeline qualification; no favorable-effect selection",
        "conditional_population": "first eligible native seed-0 prefix per final family",
        "utility_population": "all original-prompt assignments including no-fire and failures",
        "closure": "competing event; pulse stops at native reasoning closure",
        "cost": "generator_output_tokens + controller_output_tokens",
        "semantic_contrasts": 14, "utility_contrasts": 28, "headline_simultaneous_contrasts": 42,
        "bootstrap_draws": 50000, "bootstrap_unit": "shared family; seeds averaged first",
        "accuracy_loss_margin": None, "final_effect_based_expansion": False,
        "negative_or_null_advancement_gate": False, "automatic_assignment_retries": 1,
        "supersedes": "unrun v2 screens only; historical artifacts remain under original contracts",
        "scope_changes": ["candidate-specific eligibility", "verification-only recognizer",
            "short conditional assay", "one nominated action", "three-arm policy ITT",
            "validation for qualification", "intervention-layer scientific capture"],
        "dispatch": "source-bound qualification and exact role prices required"})


def family_boundary(discovery, independent):
    verify(discovery); verify(independent)
    pools = {"discovery": list(discovery["new_parent_pools"]["parent_pools"]["discovery"]),
             **{s: [r["family"] for r in independent[s]] for s in ("validation", "final")}}
    require([len(pools[s]) for s in pools] == [48, 32, 96], "family roster differs")
    require(len(set(sum(pools.values(), []))) == 176, "family overlap or duplicate")
    return seal({"schema": "verification-families-v3", **pools,
        "discovery_sha256": discovery["sha256"], "independent_sha256": independent["sha256"],
        "source_revision": independent["source_revision"]})


def assignments(model, stage, records, *, policy=None, prefixes=None):
    verify(model)
    require(stage in ("discovery", "validation", "final_native", "utility", "conditional"), "unknown stage")
    require(len({r["family"] for r in records}) == len(records), "duplicate family")
    if stage in ("utility", "conditional"):
        verify(policy)
        require(policy["model_sha256"] == model["sha256"], "policy/model binding differs")
    if stage == "conditional":
        verify(prefixes, "verification-prefix-enrollment-v3")
        require(prefixes["model_sha256"] == model["sha256"] and prefixes["source_seed"] == 0 and
                prefixes["frozen_before_treatment"] is True, "prefix source or freeze differs")
        by_family = {r["family"]: r for r in prefixes["prefixes"]}
        require(set(by_family) <= {r["family"] for r in records}, "foreign enrolled prefix")
    arms = ARMS if stage == "conditional" else ("target", "matched_random") if stage == "utility" else ("native",)
    seeds = [0] if stage == "discovery" else [0, 1]
    rows = []
    for record in records:
        if stage == "conditional" and record["family"] not in by_family:
            continue
        for seed in seeds:
            for arm in arms:
                body = {"stage": stage, "model": model["key"], "model_sha256": model["sha256"],
                    "family": record["family"], "question_id": record["question_id"], "seed": seed, "arm": arm,
                    "cap": SEMANTIC_CAP if stage == "conditional" else FULL_CAP,
                    "policy_sha256": policy["sha256"] if policy else None,
                    "prefix": by_family[record["family"]] if stage == "conditional" else None,
                    "all_assignment_accounting": True}
                rows.append({**body, "uid": identity(body)})
    return seal({"schema": "verification-assignments-v3", "model_sha256": model["sha256"],
        "protocol_sha256": protocol()["sha256"], "stage": stage, "assignments": rows,
        "assigned": len(rows), "prefix_enrollment_sha256": prefixes["sha256"] if prefixes else None})
