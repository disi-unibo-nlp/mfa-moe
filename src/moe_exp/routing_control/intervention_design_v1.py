"""Frozen target/operator/timing/composition enrollment before GPU dispatch."""
from dataclasses import asdict

from .policy_stages_v1 import RoutingAction
from .study_v1 import REFERENCE_TARGETS, SCHEMA, TRANSITIONS, digest, require

EXECUTION_FIELDS = frozenset({"checkpoint_revision", "engine_profile_sha256", "sampler_sha256",
    "worker_qualification_sha256", "prefix_recognition_audit_sha256", "capture_executed_ids",
    "capture_selected_weights", "capture_confidence_qualified", "top_k", "shared_expert"})


def require_execution(contract):
    require(set(contract) == EXECUTION_FIELDS and all(contract[k] for k in
        ("checkpoint_revision", "engine_profile_sha256", "sampler_sha256", "worker_qualification_sha256",
         "prefix_recognition_audit_sha256")), "incomplete frozen execution contract")
    require(contract["top_k"] == 8 and contract["shared_expert"] == "unchanged" and
            contract["capture_executed_ids"] is True and contract["capture_selected_weights"] is True and
            type(contract["capture_confidence_qualified"]) is bool, "new generation capture/top-k contract differs")


def eligible_prefixes(prefixes, transition):
    eligible = [p for p in prefixes if p["transition"] == transition and p.get("eligible") is True]
    require(all(p.get("prefix_recognition_qualified") is True and p.get("population") == "discovery" and
                p.get("prefix_ids_sha256") and p.get("family") and p.get("uid") for p in eligible),
            "prefix is unqualified or outside discovery")
    require(len({p["uid"] for p in eligible}) == len(eligible), "duplicate eligible prefix")
    return sorted(eligible, key=lambda p: (digest([SCHEMA, p["family"]]), p["uid"]))[:24]


def freeze_stage(stage, targets, prefixes, controls, execution, *, evidence=()):
    """All assigned non-fires/errors are retained by the downstream controller.

    Controls are keyed by (layer, expert, family, seed, operator, sign). Their
    expert pools follow the existing native-exposure rule; their calibrated
    strength/displacement and unmatched status remain explicit in every cell.
    """
    require_execution(execution)
    require(stage in ("target", "operator", "timing"), "unknown singleton stage")
    require(1 <= len(targets) <= 2 and len({(t["layer"], t["expert"]) for t in targets}) == len(targets),
            "at most two layer-specific singleton targets")
    qualified = {(r["layer"], r["expert"]): r for r in evidence}
    rows = []
    for target in targets:
        identity = target["layer"], target["expert"]
        require(target.get("prefix_recognition_qualified") is True and target["transition"] in TRANSITIONS,
                "target lacks qualified prefix support")
        if stage == "target":
            require(identity not in REFERENCE_TARGETS and target.get("status") == "SUPPORTED_ASSOCIATION" and
                    min(target["positive_families"], target["comparison_families"]) >= 10 and
                    target["consistent_folds"] >= 3, "new singleton fails discovery support")
            options = [("positive", "bias", 1., 0), ("negative", "bias", -1., 0)]
        else:
            support = qualified.get(identity, {})
            require(support.get("individually_supported") is True and support.get("population") == "discovery" and
                    support.get("sha256"), "later stage requires frozen individual discovery support")
            if stage == "operator":
                options = [("bias", "bias", 1., 0), ("force", "force", 0., 0), ("reweight", "reweight", 1., 0)]
            else:
                require(support.get("selected_operator") in ("bias", "force", "reweight") and
                        (support["selected_operator"] == "force" or support.get("selected_strength", 0) > 0),
                        "negative screening action is not automatically a useful policy")
                options = [("first", support["selected_operator"], support["selected_strength"], 0),
                           ("delayed64", support["selected_operator"], support["selected_strength"], 64)]
        for prefix in eligible_prefixes(prefixes, target["transition"]):
            for seed in (0, 1):
                base = {"stage": stage, "target": list(identity), "prefix_uid": prefix["uid"],
                    "prefix_ids_sha256": prefix["prefix_ids_sha256"], "family": prefix["family"],
                    "transition": target["transition"], "seed": seed, "continuation_cap": 1024,
                    "pulse_tokens": 256, "execution_contract_sha256": digest(execution), "nonfires_retained": True}
                native = {**base, "arm": "native", "action": None, "control": None}
                rows.append({**native, "uid": digest(native)})
                for arm, operator, strength, delay in options:
                    action = RoutingAction(f"L{identity[0]}E{identity[1]}_{arm}", target["transition"],
                                           *identity, operator, strength, delay)
                    cell = {**base, "arm": arm, "action": asdict(action), "control": None}
                    rows.append({**cell, "uid": digest(cell)})
                    sign = -1 if strength < 0 else 1
                    control = controls.get((*identity, prefix["family"], seed, operator, sign))
                    require(control and control.get("population") == "discovery" and
                            control.get("pool_rule") == "existing_native_exposure_matching" and control.get("sha256"),
                            "control calibration is unavailable")
                    # The initial ±1 screen is operator-matched at the exact
                    # signed strength. Later stages use discovery-calibrated
                    # strengths and retain unmatched displacement explicitly.
                    random_strength = strength if stage == "target" else control["strength"]
                    require(type(control["expert"]) is int and control["expert"] != identity[1], "random target coincides")
                    random = RoutingAction(action.action_id + "_random", action.transition, action.layer,
                                           control["expert"], operator, random_strength, delay)
                    random_cell = {**base, "arm": "random_" + arm, "action": asdict(random), "control": control}
                    rows.append({**random_cell, "uid": digest(random_cell)})
    require(len({r["uid"] for r in rows}) == len(rows), "stage assignment identity collision")
    return {"schema": "routing-study-focused-stage-v1", "stage": stage, "targets": targets,
        "execution": execution, "assignments": rows, "assigned": len(rows), "seeds": [0, 1],
        "horizon": 1024, "maximum_prefixes_per_transition_target": 24,
        "ranking": ["accuracy_change", "token_cost_change", "substantive_transition_change",
                    "accuracy_damage_per_exposure", "net_correct_to_incorrect_flips_per_exposure"],
        "accuracy_scope": "Unavailable in short capped screen unless an independently qualified final-answer endpoint exists",
        "counterfactual_NLL": "UNAVAILABLE_FAILED_MEASUREMENT_QUALIFICATION",
        "dispatch_status": "FROZEN_REQUIRES_COMPLETE_PRICE_AND_RESOURCE_AUTHORIZATION"}


def composition(actions, evidence, opportunities, *, reverse=False):
    require(len(actions) == 2 and len({a.action_id for a in actions}) == 2, "composition requires two supported actions")
    require(all(evidence.get(a.action_id, {}).get("individually_supported") is True and
                evidence[a.action_id].get("population") == "discovery" for a in actions),
            "composition advanced an unsupported action")
    require(all(a.operator == "force" or a.strength > 0 for a in actions), "negative action is not automatically useful")
    if reverse:
        require(actions[0].transition != actions[1].transition, "reversal requires distinct substantive actions")
    forward_order = tuple(a.transition for a in actions)
    reverse_order = tuple(reversed(forward_order))
    eligible = [p for p in opportunities if p.get("prefix_recognition_qualified") is True and
                p.get("population") == "discovery" and forward_order in map(tuple, p.get("eligible_orders", [])) and
                (not reverse or reverse_order in map(tuple, p.get("eligible_orders", [])))]
    if not eligible:
        return {"status": "UNAVAILABLE_GENUINE_COMPOSITION_OPPORTUNITIES", "assignments": []}
    return {"schema": "routing-study-composition-design-v1", "status": "FROZEN_REQUIRES_EXECUTION_QUALIFICATION_AND_PRICE",
        "actions": [asdict(a) for a in actions], "opportunities": eligible,
        "arms": ["native", actions[0].action_id, actions[1].action_id, "combined"] + (["reversed"] if reverse else []),
        "comparison": "Combined policy versus each constituent, with all no-fires retained"}


def reusable_cell(saved, planned):
    """Never equate cells across population, horizon, sampler or measurement."""
    return (saved.get("status") == "COMPLETE" and saved.get("execution_contract") == planned.get("execution_contract") and
            saved.get("measurement_contract") == planned.get("measurement_contract") and
            saved.get("assignment") == planned.get("assignment") and all(key in saved and key in planned for key in
                ("execution_contract", "measurement_contract", "assignment")))
