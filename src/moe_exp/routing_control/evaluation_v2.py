"""Independent v2 enrollment and supported sequential composition."""
from __future__ import annotations

from dataclasses import asdict

from .policy_stages_v1 import freeze_independent_assignments as historical_enrollment
from .study_v1 import digest, require
from .study_v2 import CAP, EARLY_WINDOW, SCHEMA, _execution, earliest_prefixes, incumbent_actions, seal, verify


def action_settings(actions, *, experts=True):
    return [{k: v for k, v in a.items() if k != "action_id" and (experts or k != "experts")} for a in actions]


def freeze_incumbent(recognizer, controller_qualification):
    verify(recognizer)
    verify(controller_qualification)
    require(recognizer.get("status") == controller_qualification.get("status") == "PASS" and
            set(a.transition for a in incumbent_actions()) <= set(recognizer["supported_transitions"]) and
            controller_qualification.get("recognizer_sha256") == recognizer["sha256"] and
            controller_qualification.get("actions_sha256") == digest([asdict(a) for a in incumbent_actions()]),
            "incumbent qualified controller unavailable")
    return seal({"schema": "routing-incumbent-controller-v2", "frozen_before_validation": True,
        "actions": [asdict(a) for a in incumbent_actions()], "recognizer_sha256": recognizer["sha256"],
        "controller_qualification_sha256": controller_qualification["sha256"], "utility_status": "UNPROVEN_COMPARATOR"})


def freeze_independent_assignments(families, stage, execution, policies=(), *, selection=None,
                                   selected_policy=None, random_policy=None, incumbent=None):
    verify(families)
    _execution(execution)
    require(execution["family_boundary"]["independent_freeze_sha256"] == families["sha256"],
            "independent reservation differs from execution family boundary")
    verify(incumbent)
    require(incumbent.get("frozen_before_validation") is True and
            digest(incumbent.get("actions")) == digest([asdict(a) for a in incumbent_actions()]),
            "incumbent actions changed or controller not frozen before validation")
    if stage == "validation":
        require(policies and len(policies) <= 2, "at most two qualified policies")
        for p in policies:
            verify(p)
            require(p.get("mode") in ("state_only", "routing_context") and p.get("qualification_status") == "PASS" and
                    p.get("selection_population") == "discovery" and
                    p.get("execution_contract_sha256") == execution["sha256"], "unqualified independent policy")
        require(len({p["mode"] for p in policies}) == len(policies), "duplicate validation policy type")
        if len(policies) == 2:
            require(action_settings(policies[0]["actions"]) == action_settings(policies[1]["actions"]),
                    "state and context counterparts must use identical actions/operators/timing")
    if stage == "final":
        for value in (selection, selected_policy, random_policy): verify(value)
        require(selected_policy["sha256"] == selection["selected_policy_sha256"] and
                selected_policy["policy_id"] == selection["selected_policy"] and
                random_policy.get("context_gate_sha256") == selected_policy.get("context_gate_sha256") and
                random_policy.get("mode") == selected_policy.get("mode") and
                action_settings(random_policy["actions"], experts=False) == action_settings(selected_policy["actions"], experts=False),
                "random comparator context/operator/timing differs")
        require(all(len(r["experts"]) == len(a["experts"]) and not set(r["experts"]) & set(a["experts"])
                    for r, a in zip(random_policy["actions"], selected_policy["actions"], strict=True)), "random target geometry differs")
    # Reuse only enrollment logic; v2 rebinds all IDs to its new contract.
    prior = historical_enrollment(families, stage, policies, selection=selection,
                                 random_policy=random_policy, incumbent=incumbent)
    rows = []
    for row in prior["rows"]:
        body = {k: v for k, v in row.items() if k != "uid"}
        body.update(continuation_cap=CAP, early_endpoint_window=EARLY_WINDOW,
                    execution_contract_sha256=execution["sha256"], nonfires_retained=True)
        rows.append({**body, "uid": digest([SCHEMA, families["sha256"], body])})
    pairs = ([["selected", "native"], ["selected", "matched_random"], ["selected", "incumbent"]] if stage == "final" else
             [[p["policy_id"], "native"] for p in policies] if stage == "validation" else [])
    if stage == "validation" and len(policies) == 2:
        pairs.append([policies[0]["policy_id"], policies[1]["policy_id"]])
    return seal({"schema": "routing-study-experiment-manifest-v2", "stage": stage, "execution": execution,
        "family_freeze_sha256": families["sha256"], "source_revision": families["source_revision"],
        "assignments": rows, "assigned": len(rows), "continuation_cap": CAP, "early_endpoint_window": EARLY_WINDOW,
        "incumbent_sha256": incumbent["sha256"], "prespecified_accuracy_cost_pairs": pairs,
        "final_outcomes_used_for_selection": False,
        "dispatch_status": "REQUIRES_MEASURED_PRICE_AND_STAGE_RESOURCE_AUTHORIZATION"})


def freeze_composition(actions, evidence, opportunities, execution, *, reverse=False):
    _execution(execution)
    from .families_v2 import require_discovery
    require_discovery([p["family"] for p in opportunities], execution["family_boundary"])
    require(len(actions) == 2 and actions[0].transition != actions[1].transition and
            len({a.action_id for a in actions}) == 2, "two distinct individually supported actions required")
    for a in actions:
        support = verify(evidence[a.action_id])
        require(support.get("individually_supported") is True and support.get("population") == "discovery" and
                (a.operator == "force" or a.strength > 0), "composition action lacks exploratory support")
    order = [a.transition for a in actions]
    eligible = [p for p in opportunities if order in p.get("eligible_orders", []) and
                (not reverse or list(reversed(order)) in p.get("eligible_orders", []))]
    prefixes = earliest_prefixes(eligible, actions[0].transition)
    if not prefixes:
        return seal({"schema": "routing-study-composition-manifest-v2", "status": "UNAVAILABLE_GENUINE_COMPOSITION_OPPORTUNITIES",
                     "assignments": [], "assigned": 0})
    rows = []
    arms = [("native", ()), (actions[0].action_id, (actions[0],)), (actions[1].action_id, (actions[1],)), ("combined", tuple(actions))]
    if reverse: arms.append(("reversed", tuple(reversed(actions))))
    for p in prefixes:
        for seed in (0, 1):
            for arm, sequence in arms:
                body = {"stage": "composition", "prefix_uid": p["uid"], "family": p["family"], "seed": seed,
                    "prefix_ids_sha256": p["prefix_ids_sha256"], "arm": arm, "actions": [asdict(a) for a in sequence],
                    "ordered": True, "nonfires_retained": True, "continuation_cap": CAP,
                    "early_endpoint_window": EARLY_WINDOW, "execution_contract_sha256": execution["sha256"]}
                rows.append({**body, "uid": digest([SCHEMA, body])})
    return seal({"schema": "routing-study-experiment-manifest-v2", "stage": "composition", "execution": execution,
        "assignments": rows, "assigned": len(rows), "continuation_cap": CAP, "early_endpoint_window": EARLY_WINDOW,
        "pulse_overlap": "NO_FIRE_RETAINED", "dispatch_status": "REQUIRES_MEASURED_PRICE_AND_STAGE_RESOURCE_AUTHORIZATION"})
