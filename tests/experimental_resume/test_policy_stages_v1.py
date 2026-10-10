from dataclasses import asdict

import numpy as np
import pytest

from moe_exp.routing_control.policy_stages_v1 import (
    PrefixPolicyController, RoutingAction, freeze_independent_assignments, matched_control, price_workers)
from moe_exp.routing_control.study_v1 import digest


def audit():
    return {"status": "PASS", "prefix_only": True, "recognizes_substantive_acts": True,
            "supported_transitions": ["candidate_to_verify", "approach_to_commit"], "sha256": "audit"}


def response(prefix, acts=()):
    return {"prefix_sha256": digest(asdict(prefix)), "boundary": True,
            "candidate_status": "complete_unchecked", "approach_identified": True,
            "substantive_acts": list(acts), "controller_tokens": 4,
            "controller_input_tokens": len(prefix.emitted_token_ids) + 2, "controller_seconds": .2}


class HostileRecord(dict):
    def __getitem__(self, key):
        assert key in ("problem", "emitted_token_ids"), "read forbidden answer/future/offline field"
        return super().__getitem__(key)


def test_online_controller_reads_only_native_prefix_and_keeps_batch_isolation():
    action = RoutingAction("verification", "candidate_to_verify", 28, 189, "bias", 1)
    controller = PrefixPolicyController([action], response, audit())
    record = HostileRecord(problem="Question", emitted_token_ids=[1, 2],
                           gold="secret", future_ids=[9], emitted_text="future", offline_label="Verify")
    fired = controller.observe(record, lambda ids: "Answer is 2." if len(ids) == 2 else "")
    assert fired[0]["position"] == 2
    names = controller.routing_rows(np.array([1, 2, 257, 258]), [True, True, False, True])
    assert names.tolist() == ["native", "verification", "native", "native"]
    other = PrefixPolicyController([action], response, audit())
    assert other.routing_rows(np.arange(4), [True] * 4).tolist() == ["native"] * 4
    assert controller.ledger()["controller_tokens"] == 4
    assert controller.ledger()["controller_input_tokens"] == 4


def test_delayed_policy_keeps_no_fire_if_substantive_act_precedes_delay():
    action = RoutingAction("verification", "candidate_to_verify", 28, 189, "bias", 1, delay=64)
    def recognize(prefix):
        return response(prefix, ["candidate_to_verify"] if len(prefix.emitted_token_ids) > 2 else [])
    controller = PrefixPolicyController([action], recognize, audit())
    assert controller.observe({"problem": "q", "emitted_token_ids": [1, 2]}, lambda ids: "x" * len(ids)) == []
    assert controller.observe({"problem": "q", "emitted_token_ids": [1, 2, 3]}, lambda ids: "x" * len(ids)) == []
    assert controller.ledger()["decisions"]["verification"]["status"] == "NO_FIRE_DESIRED_ACT_OCCURRED"
    assert controller.observe({"problem": "q", "emitted_token_ids": [1, 2, 3] + [4] * 63}, lambda ids: "x" * len(ids)) == []


def test_delayed_policy_fires_at_exact_64_token_edge_and_abstains_on_closure():
    action = RoutingAction("verification", "candidate_to_verify", 28, 189, "bias", 1, delay=64)
    controller = PrefixPolicyController([action], response, audit())
    controller.observe({"problem": "q", "emitted_token_ids": [1]}, lambda ids: "x" * len(ids))
    fired = controller.observe({"problem": "q", "emitted_token_ids": [1] * 65}, lambda ids: "x" * len(ids))
    assert fired[0]["position"] == 65
    closed = PrefixPolicyController([action], response, audit())
    closed.observe({"problem": "q", "emitted_token_ids": [1]}, lambda ids: "x" * len(ids))
    closed.observe({"problem": "q", "emitted_token_ids": [1, 2]}, lambda ids: "x</think>")
    assert closed.ledger()["decisions"]["verification"]["status"] == "NO_FIRE_CLOSED"
    assert len(closed.ledger()["controller_calls"]) == 1
    capped = PrefixPolicyController([action], response, audit())
    capped.observe({"problem": "q", "emitted_token_ids": [1]}, lambda ids: "x" * len(ids))
    assert capped.finish(hit_cap=True)["decisions"]["verification"]["status"] == "NO_FIRE_CAP"


def test_unsupported_recognizer_and_reversed_repeat_are_rejected():
    action = RoutingAction("verification", "candidate_to_verify", 28, 189, "bias", 1)
    with pytest.raises(ValueError, match="passing semantic audit"):
        PrefixPolicyController([action], response, {**audit(), "prefix_only": False})
    second = RoutingAction("verification2", "candidate_to_verify", 28, 9, "force", 0)
    with pytest.raises(ValueError, match="distinct substantive"):
        PrefixPolicyController([action, second], response, audit(), ordered=True)


def family_freeze():
    return {"status": "FROZEN", "sha256": "family-freeze", "source_revision": "revision",
            "validation": [{"family": f"v{i}", "question_id": f"qv{i}"} for i in range(32)],
            "final": [{"family": f"t{i}", "question_id": f"qt{i}"} for i in range(96)]}


def test_stage_enrollment_and_no_final_outcome_leakage():
    families = family_freeze()
    policies = [{"policy_id": "policy", "sha256": "policy-sha", "qualification_status": "PASS", "selection_population": "discovery"}]
    native = freeze_independent_assignments(families, "native_validation")
    assert native["maximum_assignments"] == 64
    validation = freeze_independent_assignments(families, "validation", policies)
    assert validation["maximum_assignments"] == 128
    selection = {"status": "SELECTED_ON_VALIDATION", "sha256": "selected",
        "selected_policy": "policy", "selected_policy_sha256": "policy-sha",
        "validation_families_sha256": digest(families["validation"])}
    random = {"operator_timing_matched_to": "policy", "calibration_population": "discovery", "sha256": "random"}
    incumbent = {"frozen_before_validation": True, "sha256": "incumbent"}
    final = freeze_independent_assignments(families, "final", selection=selection, random_policy=random, incumbent=incumbent)
    assert final["maximum_assignments"] == 768
    assert {r["arm"] for r in final["rows"]} == {"native", "selected", "matched_random", "incumbent"}
    assert not {r["family"] for r in final["rows"]} & {r["family"] for r in validation["rows"]}
    assert len({r["uid"] for r in final["rows"]}) == 768
    with pytest.raises(ValueError, match="outcome leakage"):
        freeze_independent_assignments(families, "validation", [{**policies[0], "selection_population": "final"}])
    with pytest.raises(ValueError, match="winner"):
        freeze_independent_assignments(families, "final", selection={"status": "NO_COST_IMPROVING_WINNER"})


def test_unmatched_random_control_is_retained_with_operator_and_timing():
    action = RoutingAction("verification", "candidate_to_verify", 28, 189, "reweight", 1, delay=64)
    control = matched_control(action, 120, .8, {"pool_rule": "existing_native_exposure_matching",
        "population": "discovery", "sha256": "calibration", "matched": False,
        "target_displacement": .1, "random_displacement": .07})
    assert control["unmatched_retained"] and not control["matched"]
    assert control["action"]["delay"] == 64 and control["action"]["operator"] == "reweight"


def test_complete_prices_use_role_specific_measured_throughput_and_recovery():
    workloads = {role: {"input_tokens": 100, "output_tokens": 20, "loads": 1, "recovery_loads": 1}
                 for role in ("generation", "controller", "labels", "grading")}
    telemetry = {role: {"measurement_sha256": role, "prefill_tps": 10, "decode_tps": rate,
        "load_seconds": 5, "shutdown_seconds": 1, "overhead_seconds": 2, "call_recovery_factor": 1.25, "gpus": 2}
        for role, rate in zip(workloads, (1, 2, 4, 5))}
    result = price_workers(workloads, telemetry)
    assert result["status"] == "PRICED_REQUIRES_CONCRETE_AUTHORIZATION"
    gen = next(r for r in result["workers"] if r["role"] == "generation")
    assert gen["seconds"] == 1.25 * (10 + 20) + 12 + 2
    assert {r["cost_ledger"] for r in result["workers"]} == {"policy", "research"}
    missing = price_workers(workloads, {"generation": telemetry["generation"]})
    assert missing["GPU_hours"] is None and len(missing["unavailable_roles"]) == 3
