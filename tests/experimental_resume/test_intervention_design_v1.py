import pytest

from moe_exp.routing_control.intervention_design_v1 import composition, freeze_stage, reusable_cell
from moe_exp.routing_control.policy_stages_v1 import RoutingAction


def inputs():
    target = {"layer": 20, "expert": 12, "transition": "candidate_to_verify", "status": "SUPPORTED_ASSOCIATION",
              "prefix_recognition_qualified": True, "positive_families": 10, "comparison_families": 10, "consistent_folds": 3}
    prefixes = [{"uid": str(i), "family": str(i), "transition": "candidate_to_verify", "population": "discovery",
                 "eligible": True, "prefix_recognition_qualified": True, "prefix_ids_sha256": str(i)} for i in range(30)]
    controls = {(20, 12, str(i), seed, operator, sign): {"expert": 13, "strength": 0 if operator == "force" else 1,
                "population": "discovery", "pool_rule": "existing_native_exposure_matching", "sha256": "calibrated", "matched": False}
                for i in range(30) for seed in (0, 1) for operator in ("bias", "force", "reweight") for sign in (-1, 1)}
    execution = {"checkpoint_revision": "revision", "engine_profile_sha256": "engine", "sampler_sha256": "sampler",
        "worker_qualification_sha256": "worker", "prefix_recognition_audit_sha256": "audit", "capture_executed_ids": True,
        "capture_selected_weights": True, "capture_confidence_qualified": False, "top_k": 8, "shared_expert": "unchanged"}
    return target, prefixes, controls, execution


def test_focused_stages_keep_caps_signs_controls_and_no_fires():
    target, prefixes, controls, execution = inputs()
    result = freeze_stage("target", [target], prefixes, controls, execution)
    assert result["assigned"] == 24 * 2 * 5
    assert {r["action"]["strength"] for r in result["assignments"] if r["arm"] == "random_negative"} == {-1}
    evidence = [{"layer": 20, "expert": 12, "individually_supported": True, "population": "discovery", "sha256": "effect",
                 "selected_operator": "reweight", "selected_strength": 1}]
    assert freeze_stage("operator", [target], prefixes, controls, execution, evidence=evidence)["assigned"] == 24 * 2 * 7
    timing = freeze_stage("timing", [target], prefixes, controls, execution, evidence=evidence)
    assert timing["assigned"] == 24 * 2 * 5
    assert all(r["nonfires_retained"] for r in timing["assignments"])
    assert {r["action"]["delay"] for r in timing["assignments"] if r["arm"] == "delayed64"} == {64}
    with pytest.raises(ValueError, match="later stage"):
        freeze_stage("operator", [target], prefixes, controls, execution)
    with pytest.raises(ValueError, match="capture/top-k"):
        freeze_stage("target", [target], prefixes, controls, {**execution, "capture_selected_weights": False})


def test_composition_reversal_requires_support_and_actual_orders():
    actions = [RoutingAction("verify", "candidate_to_verify", 20, 12, "bias", 1),
               RoutingAction("commit", "approach_to_commit", 24, 159, "force", 0)]
    evidence = {a.action_id: {"individually_supported": True, "population": "discovery"} for a in actions}
    opportunities = [{"prefix_recognition_qualified": True, "population": "discovery",
                      "eligible_orders": [["candidate_to_verify", "approach_to_commit"]]}]
    assert composition(actions, evidence, opportunities, reverse=True)["status"] == "UNAVAILABLE_GENUINE_COMPOSITION_OPPORTUNITIES"
    opportunities[0]["eligible_orders"].append(["approach_to_commit", "candidate_to_verify"])
    assert "reversed" in composition(actions, evidence, opportunities, reverse=True)["arms"]
    assert composition(actions, evidence, opportunities)["arms"] == ["native", "verify", "commit", "combined"]


def test_reuse_requires_complete_matching_execution_and_measurement():
    planned = {"execution_contract": {"horizon": 1024}, "measurement_contract": "semantic", "assignment": "id"}
    assert reusable_cell({**planned, "status": "COMPLETE"}, planned)
    assert not reusable_cell({**planned, "status": "COMPLETE", "execution_contract": {"horizon": 256}}, planned)
    assert not reusable_cell({**planned, "status": "COMPLETE", "measurement_contract": "accuracy"}, planned)
