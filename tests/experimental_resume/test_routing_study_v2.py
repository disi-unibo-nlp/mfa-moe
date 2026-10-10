from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from moe_exp.routing_control import context_v2 as context
from moe_exp.routing_control.families_v2 import family_boundary
from moe_exp.routing_control.evaluation_v2 import freeze_composition, freeze_incumbent, freeze_independent_assignments
from moe_exp.routing_control.policy_stages_v2 import PrefixPolicyController, THINK_END_ID, freeze_context_gate
from moe_exp.routing_control.resource_analysis_v2 import ROLES, price_stage
from moe_exp.routing_control.study_v1 import digest, file_sha, fold
from moe_exp.routing_control.study_v2 import (
    Action, CAP, CHECKS, EARLY_WINDOW, RECOGNIZES, advance_action, dispatch_gate, earliest_prefixes,
    execution_contract, freeze_stage, incumbent_actions, protocol, qualify_recognizer, seal, select_operator, verify)


def recognition():
    return seal({"status": "PASS", "prefix_only": True, "recognizes_substantive_acts": True,
                 "family_boundary_sha256": boundary()["sha256"],
                 "supported_transitions": ["candidate_to_verify", "approach_to_commit", "checked_candidate_to_close"]})


def independent_freeze():
    return seal({"status": "FROZEN", "source_revision": "preserved",
        "validation": [{"family": f"v{i}", "question_id": f"qv{i}"} for i in range(32)],
        "final": [{"family": f"t{i}", "question_id": f"qt{i}"} for i in range(96)]})


def boundary():
    discovery = [f"f{i}" for i in range(60)] + [f"family-{i}" for i in range(60)] + [str(i) for i in range(12)] + ["family", "held"]
    return family_boundary(seal({"new_parent_pools": {"parent_pools": {"discovery": discovery}}}), independent_freeze())


def execution():
    return execution_contract(checkpoint_revision="frozen-model-revision", engine_profile=seal({"profile": "test"}),
        family_boundary=boundary(),
        sampler=seal({"max_tokens": CAP, "seeds": [0, 1]}),
        resume=seal({"status": "PASS", "engine_rng_state": True, "seed_only_restart": False}), recognizer=recognition(),
        worker=seal({"status": "PASS", "checks": {k: True for k in CHECKS}, "code_files": {__file__: file_sha(__file__)},
                     "cross_run_variability_audited": True}))


def action(delay=0):
    return Action("verify", "candidate_to_verify", 20, (12,), "bias", 1, delay)


def response(prefix, **extra):
    return {"prefix_sha256": digest(asdict(prefix)), "boundary": True, "source_state": "Analyze",
            "candidate_status": "complete_unchecked", "approach_identified": False, "substantive_acts": [],
            "controller_tokens": 3, "controller_input_tokens": len(prefix.emitted_token_ids), "controller_seconds": .1, **extra}


def controller(*, delay=0, recognizer=response, **extra):
    return PrefixPolicyController([action(delay)], recognizer, recognition(), uid="request", seed=0, **extra)


def observe(c, ids, **extra):
    return c.observe({"problem": "q", "emitted_token_ids": ids}, lambda t: "x" * len(t), **extra)


def test_contracts_are_additive_and_require_actual_qualification():
    spec = protocol()
    verify(spec)
    assert spec["continuation_cap"] == CAP and spec["early_endpoint_window"] == EARLY_WINDOW
    assert execution()["capture_layers"] == list(range(40))
    from moe_exp.routing_control.intervention_design_v1 import require_execution
    with pytest.raises(ValueError): require_execution(execution())
    with pytest.raises(ValueError, match="cap"):
        execution_contract(checkpoint_revision="x", engine_profile=seal({}), sampler=seal({"max_tokens": 1024, "seeds": [0, 1]}),
            resume=seal({}), recognizer=recognition(), worker=seal({}), family_boundary=boundary())


def test_recognition_precision_symbolic_and_future_visibility_gates():
    audits = [seal({"decision": d, "population": "discovery", "prefix_only": True,
        "eligible_families": [f"f{i}" for i in range(10)], "apparent_positive_count": 10, "confirmed_positive_count": 7,
        "held_out_families": ["held"], "held_out_precision_ci95": [.4, .9], "held_out_coverage": .3,
        "symbolic_candidates_audited": True}) for d in RECOGNIZES]
    assert qualify_recognizer(audits, family_boundary=boundary(), supported_transitions=["candidate_to_verify"])["status"] == "PASS"
    bad = deepcopy(audits)
    bad[0] = seal({**{k: v for k, v in bad[0].items() if k != "sha256"}, "symbolic_candidates_audited": False})
    assert qualify_recognizer(bad, family_boundary=boundary(), supported_transitions=["candidate_to_verify"])["status"] == "FAIL"
    bad[0] = seal({**{k: v for k, v in bad[0].items() if k != "sha256"}, "prefix_only": False})
    with pytest.raises(ValueError, match="future"):
        qualify_recognizer(bad, family_boundary=boundary(), supported_transitions=["candidate_to_verify"])
    with pytest.raises(ValueError, match="priced"):
        qualify_recognizer(audits, family_boundary=boundary(), supported_transitions=["candidate_to_verify"], method="native_semantic_veto")


def inputs():
    target = {"layer": 20, "expert": 12, "transition": "candidate_to_verify", "prefix_recognition_qualified": True,
              "positive_families": 10, "comparison_families": 10, "consistent_folds": 3}
    prefixes = [{"uid": f"p{i}-{p}", "family": f"f{i}", "transition": target["transition"], "population": "discovery",
        "position": p, "eligible": True, "prefix_recognition_qualified": True, "prefix_ids_sha256": digest([i, p])}
        for i in range(30) for p in (100, 50)]
    controls = {(20, 12, f"f{i}", s, op, sign): seal({"expert": 13, "strength": 0 if op == "force" else sign,
        "population": "discovery", "pool_rule": "existing_native_exposure_matching", "matched": True})
        for i in range(30) for s in (0, 1) for op in ("bias", "force", "reweight") for sign in (-1, 1)}
    support = seal({"layer": 20, "expert": 12, "individually_supported": True, "population": "discovery",
                    "selected_operator": "force", "selected_strength": 0, "selected_delay": 64})
    return target, prefixes, controls, support


def test_earliest_prefix_before_hash_limit_and_frozen_16k_screen():
    target, prefixes, controls, support = inputs()
    selected = earliest_prefixes(list(reversed(prefixes)), target["transition"])
    assert len(selected) == len({p["family"] for p in selected}) == 24
    assert all(p["position"] == 50 for p in selected)
    for stage, n in (("target", 5), ("operator", 7), ("timing", 5)):
        manifest = freeze_stage(stage, [target], prefixes, controls, execution(), evidence=[support])
        assert manifest["assigned"] == 24 * 2 * n
        assert len({r["uid"] for r in manifest["assignments"]}) == manifest["assigned"]
        assert all(r["continuation_cap"] == CAP and r["early_endpoint_window"] == EARLY_WINDOW for r in manifest["assignments"])
    with pytest.raises(ValueError, match="artifact seal"):
        freeze_stage("operator", [target], prefixes, controls, {**execution(), "continuation_cap": 1024}, evidence=[support])


class HostileRecord(dict):
    def __getitem__(self, key):
        assert key in ("problem", "emitted_token_ids")
        return super().__getitem__(key)


def test_future_answer_grade_and_offline_labels_cannot_change_a_decision():
    c = controller()
    record = HostileRecord(problem="q", emitted_token_ids=[], emitted_text="</think>", future_ids=[THINK_END_ID],
                           gold="future", grade=True, offline_label="Verify")
    assert c.observe(record, lambda t: "")[0]["position"] == 0
    snapshot = c.ledger()
    record.update(gold="changed", offline_label="Explore", grade=False)
    assert c.observe(record, lambda t: "") == [] and c.ledger() == snapshot
    other = controller()
    assert other.routing_rows(np.array([0, 1])).tolist() == ["native", "native"]


def test_prediction_alignment_closure_row_and_all_subsequent_native():
    c = controller()
    observe(c, [])
    observe(c, [1])
    observe(c, [1, THINK_END_ID])
    assert c.routing_rows(np.arange(4)).tolist() == ["verify", "verify", "native", "native"]
    assert c.ledger()["closure_position"] == 1
    from moe_exp.routing_control.state_analysis_v1 import validate_capture
    ids = np.tile(np.arange(8), (3, 40, 1))
    validate_capture([1, THINK_END_ID, 2], ids, np.full(ids.shape, .125))
    with pytest.raises(ValueError, match="align"):
        validate_capture([1, THINK_END_ID, 2], ids[:2])


def test_delay_early_act_missing_decision_and_cap_no_fire():
    c = controller(delay=64, recognizer=lambda p: response(p, substantive_acts=["candidate_to_verify"] if p.emitted_token_ids else []))
    assert observe(c, []) == [] and observe(c, [1]) == []
    assert c.finish()["decisions"]["verify"]["status"] == "NO_FIRE_DESIRED_ACT_OCCURRED"
    delayed = controller(delay=64)
    for i in range(65): fired = observe(delayed, [1] * i)
    assert fired[0]["position"] == 64 and delayed.routing_rows(np.array([63, 64, 319, 320])).tolist() == ["native", "verify", "verify", "native"]
    bad = controller(delay=64, recognizer=lambda p: response(p, substantive_acts=None if len(p.emitted_token_ids) == 64 else []))
    for i in range(65): observe(bad, [1] * i)
    assert bad.finish()["decisions"]["verify"]["status"] == "NO_FIRE_AMBIGUOUS_DELAY"
    skipped = controller(origin_position=1)
    observe(skipped, [1])
    with pytest.raises(ValueError, match="skipped"): observe(skipped, [1, 2, 3])
    with pytest.raises(ValueError, match="cap finish"): controller().finish(hit_cap=True)


def model_and_gate():
    # Serialized transform fixture with two distinguishable local contexts.
    transform = seal({"signal": "native_expert_ID_inclusion", "components": 8, "K": 2,
        "mean": [0.] * 10240, "scale": [1.] * 10240, "pca_mean": [0.] * 10240,
        "pca_components": np.eye(8, 10240).tolist(), "centroids": [[0.] * 8, [1.] * 8]})
    model = seal({"population": "discovery", "status": "MULTIPLE_CONTEXTS_SUPPORTED", "selected_K": 2, "model": transform,
                  "family_boundary_sha256": boundary()["sha256"], "discovery_families": boundary()["discovery"]})
    telemetry = [{"family": str(i), "seed": s, "transition": "candidate_to_verify", "population": "discovery",
        "first_qualified_opportunity": True, "model_sha256": model["sha256"], "source_state": "Analyze",
        "prefix_status": "complete_unchecked", "position": 64, "context": i % 2} for i in range(12) for s in (0, 1)]
    gate = freeze_context_gate(model, telemetry, accepted_contexts={"candidate_to_verify": [0]})
    return model, gate


def test_context_decline_and_random_skip_never_retry_later_opportunities():
    model, gate = model_and_gate()
    c = controller(mode="routing_context", context_model=model, gate=gate, origin_position=64)
    ids = np.tile(np.arange(8), (64, 40, 1))
    assert observe(c, [1] * 64, routing_ids=ids) == []  # context 1 is declined
    observe(c, [1] * 65, routing_ids=np.tile(np.arange(8, 16), (65, 40, 1)))
    assert c.ledger()["decisions"]["verify"]["status"] == "NO_FIRE_CONTEXT_DECLINED"
    assert c.ledger()["opportunities"]["verify"]["position"] == 64
    assert next(iter(gate["strata"].values()))["accept_probability"] == .5
    skip = controller(mode="random_skip", context_model=model, gate=gate, origin_position=64)
    first = observe(skip, [1] * 64)
    assert observe(skip, [1] * 65) == []
    repeated = controller(mode="random_skip", context_model=model, gate=gate, origin_position=64)
    assert observe(repeated, [1] * 64) == first
    bad = controller(mode="routing_context", context_model=model, gate=gate, origin_position=64)
    with pytest.raises(ValueError, match="align"):
        observe(bad, [1] * 64, routing_ids=np.tile(np.arange(8), (65, 40, 1)))


def test_resume_preserves_cost_pulses_and_sampler_cursor():
    c = controller(delay=64)
    for i in range(20): observe(c, [1] * i)
    sampler = seal({"assignment_uid": "request", "next_prediction_index": 19,
                    "rng_state_sha256": "rng", "engine_state_sha256": "engine"})
    snapshot = c.snapshot(sampler, manifest_sha256="manifest", capture_sha256="capture")
    resumed = controller(delay=64)
    resumed.restore(snapshot, sampler_state=sampler, manifest_sha256="manifest", capture_sha256="capture")
    assert observe(resumed, [1] * 19) == []  # replaying an observed prefix adds no charge
    for i in range(20, 65):
        assert observe(resumed, [1] * i) == observe(c, [1] * i)
    assert resumed.ledger() == c.ledger()
    with pytest.raises(ValueError, match="resume binding"):
        controller(delay=64).restore(snapshot, sampler_state=seal({**{k: v for k, v in sampler.items() if k != "sha256"},
            "rng_state_sha256": "other"}), manifest_sha256="manifest", capture_sha256="capture")


def test_composition_overlap_and_no_simultaneous_multilayer_steering():
    a, b = action(), Action("commit", "approach_to_commit", 24, (159,), "force", 0)
    c = PrefixPolicyController([a, b], lambda p: response(p, approach_identified=True), recognition(), uid="r", seed=1, ordered=True)
    assert len(observe(c, [])) == 1
    assert c.ledger()["decisions"]["commit"]["status"] == "NO_FIRE_PULSE_OVERLAP"
    evidence = {x.action_id: seal({"population": "discovery", "individually_supported": True}) for x in (a, b)}
    _, prefixes, _, _ = inputs()
    p = {**prefixes[0], "eligible_orders": [[a.transition, b.transition]]}
    assert freeze_composition([a, b], evidence, [p], execution(), reverse=True)["assigned"] == 0
    assert freeze_composition([a, b], evidence, [p], execution())["assigned"] == 8


def test_context_stage_and_independent_enrollment_keep_families_and_incumbent():
    target, prefixes, controls, support = inputs()
    model, gate = model_and_gate()
    ex = seal({**{k: v for k, v in execution().items() if k != "sha256"},
               "context_model_sha256": model["sha256"], "opportunity_gate_sha256": gate["sha256"]})
    manifest = freeze_stage("context", [target], prefixes, controls, ex, evidence=[support], gate=gate)
    assert manifest["assigned"] == 24 * 2 * 4
    assert {r["arm"] for r in manifest["assignments"]} == {"native", "state_only", "routing_context", "random_skip"}
    families = independent_freeze()
    incumbent = freeze_incumbent(recognition(), seal({"status": "PASS", "recognizer_sha256": recognition()["sha256"],
                               "actions_sha256": digest([asdict(a) for a in incumbent_actions()])}))
    native = freeze_independent_assignments(families, "native_validation", execution(), incumbent=incumbent)
    assert native["assigned"] == 64 and native["source_revision"] == "preserved"
    policy = seal({"policy_id": "p", "mode": "state_only", "actions": [asdict(action())],
        "qualification_status": "PASS", "selection_population": "discovery", "execution_contract_sha256": execution()["sha256"]})
    selection = seal({"status": "SELECTED_ON_VALIDATION", "selected_policy": "p", "selected_policy_sha256": policy["sha256"],
                      "validation_families_sha256": digest(families["validation"])})
    random = seal({"actions": [asdict(Action("random", "candidate_to_verify", 20, (13,), "bias", 1))],
        "operator_timing_matched_to": "p", "mode": "state_only", "calibration_population": "discovery"})
    final = freeze_independent_assignments(families, "final", execution(), selected_policy=policy,
             selection=selection, random_policy=random, incumbent=incumbent)
    assert final["assigned"] == 768 and all(r["continuation_cap"] == CAP for r in final["assignments"])
    assert not {r["family"] for r in native["assignments"]} & {r["family"] for r in final["assignments"]}


def test_exploratory_advancement_allows_uncertain_intervals_but_requires_dose():
    result = seal({"population": "discovery", "continuation_cap": CAP, "early_endpoint_window": EARLY_WINDOW,
        "simultaneous_ci95": [-.2, .3], "dose_verified": True, "random_dose_matched": True,
        "semantic_vs_native": .1, "semantic_vs_random": .05, "family_fold_effects": [.1, .2, .1, -.1],
        "operator": "bias", "strength": 1, "token_cost": 100, "intervened_positions": 256})
    assert advance_action(result) and select_operator([result]) == result
    unmatched = seal({**{k: v for k, v in result.items() if k != "sha256"}, "random_dose_matched": False})
    assert not advance_action(unmatched) and select_operator([unmatched]) is None


def test_complete_prices_reject_short_pilots_and_unapproved_dispatch():
    manifest = seal({"stage": "target", "assigned": 2, "execution": execution()})
    workloads = {r: {"input_tokens": 0, "output_tokens": 0, "loads": 0, "calls": 0, "no_work_reason": "test no work"} for r in ROLES}
    workloads["generation"].update(input_tokens=100, output_tokens=2 * CAP, loads=1, calls=2)
    workloads["capture"]["metadata_storage_bytes"] = 100
    workloads["recovery"]["maximum_assignment_retries"] = 0
    measured = seal({"status": "MEASURED", "execution_contract_sha256": execution()["sha256"],
        "measurement_artifact_sha256": "measured", "prefill_tps": 10, "decode_tps": 100, "load_seconds": 420,
        "shutdown_seconds": 10, "seconds_per_call": 1, "gpus": 2, "cpus": 2, "billing_cores": 16, "continuation_cap": CAP})
    price = price_stage(manifest, workloads, {"generation": measured})
    assert price["capture_storage_bytes"] == 2 * CAP * (4 * 40 * 8 * 6 + 40 * 2 * 4 + 4) + 100
    assert price["complete_seconds"] == 10 + 2 * CAP / 100 + 430 + 2
    with pytest.raises(ValueError, match="pilot"):
        price_stage(manifest, workloads, {"generation": seal({**{k: v for k, v in measured.items() if k != "sha256"}, "continuation_cap": 1024})})
    unavailable = price_stage(manifest, workloads, {})
    assert unavailable["status"] == "UNAVAILABLE_MEASURED_PRICE" and unavailable["maximum_GPU_hours"] is None
    auth = seal({"approved": False})
    with pytest.raises(ValueError, match="authorization"):
        dispatch_gate(manifest, price, auth, seal({}))


def test_actual_eight_component_context_transform_and_profile_validation():
    rng = np.random.default_rng(22)
    X = np.zeros((12, 40, 256))
    for row in X:
        for layer in row: layer[rng.choice(256, 8, replace=False)] = 1
    model = context.fit_context(X, [str(i) for i in range(12)], 2)
    labels, features = context.transform_context(model, X)
    assert len(model["pca_components"]) == 8 and labels.shape == (12,) and features.shape == (12, 26)
    with pytest.raises(ValueError, match="composition"): context.profiles(X / 8)


@pytest.mark.filterwarnings("ignore:Features .* are constant:UserWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_nested_cv_transforms_never_fit_on_held_families_and_falls_back(monkeypatch):
    families = [f"family-{i}" for i in range(60)]
    assert {fold(f) for f in families} == {0, 1, 2, 3}
    rows = [{"family": f, "population": "discovery", "outcome": bool(s), "source_state": "Analyze",
        "prefix_status": "complete_unchecked", "transition": "candidate_to_verify",
        "baseline_contract": "prefix_text_position_state_history_run_age", "prediction_alignment_qualified": True}
        for f in families for s in (0, 1)]
    X = np.tile(np.r_[np.ones(8), np.zeros(248)], (len(rows), 40, 1))
    # Instrument transform fits while keeping real family CV/classifier tuning.
    calls = []
    def fitted(value, training_families, k):
        calls.append(set(training_families))
        return {"training_families": sorted(set(training_families)), "K": k, "sha256": digest([training_families, k])}
    def transformed(model, value):
        return np.zeros(len(value), int), np.zeros((len(value), 8))
    monkeypatch.setattr(context, "fit_context", fitted)
    monkeypatch.setattr(context, "transform_context", transformed)
    monkeypatch.setattr(context, "context_associations", lambda *args, **kwargs: [])
    baseline = np.array([[i % 2, (i // 2) % 5] for i in range(len(rows))], float)
    result = context.nested_context_model(rows, baseline, X, family_boundary=boundary(), regularization=(1.,))
    assert result["status"] == "FALLBACK_K1" and result["selected_K"] == 1
    for outer in result["outer_folds"]:
        assert not set(outer["training_families"]) & set(outer["held_families"])
        assert set(outer["training_families"]) in calls
    bad = [{**r, "population": "final"} for r in rows]
    with pytest.raises(ValueError, match="population"):
        context.nested_context_model(bad, baseline, X, family_boundary=boundary())
    disguised = [{**r, "family": "t0"} for r in rows]
    with pytest.raises(ValueError, match="reserved/foreign"):
        context.nested_context_model(disguised, baseline, X, family_boundary=boundary())


@pytest.mark.filterwarnings("ignore:Features .* are constant:UserWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_nested_context_retention_requires_positive_outer_gains_and_local_associations(monkeypatch):
    # XOR fixture: the association reverses between routing contexts.
    families = [f"family-{i}" for i in range(48)]
    rows, profiles = [], []
    for family in families:
        for cluster in (0, 1):
            for x in (0, 1):
                profile = np.tile(np.r_[np.ones(8), np.zeros(248)], (40, 1))
                profile[0] = np.roll(profile[0], 8 * cluster)
                profile[1] = np.roll(profile[1], 8 * x)
                profiles.append(profile)
                rows.append({"family": family, "population": "discovery", "outcome": bool(cluster ^ x),
                    "source_state": "Analyze", "prefix_status": "complete_unchecked", "transition": "candidate_to_verify",
                    "baseline_contract": "prefix_text_position_state_history_run_age", "prediction_alignment_qualified": True})
    def fitted(value, training_families, k):
        return {"K": k, "sha256": digest([training_families, k]), "training_families": sorted(set(training_families))}
    def transformed(model, value):
        cluster, x = 1 - value[:, 0, 0], 1 - value[:, 1, 0]
        features = np.column_stack([cluster, x])
        if model["K"] == 1:
            return np.zeros(len(value), int), features
        if model["K"] == 2:
            return cluster.astype(int), np.column_stack([features, cluster * x])
        return (cluster * 2 + x).astype(int), np.column_stack([features, cluster * x])
    monkeypatch.setattr(context, "fit_context", fitted)
    monkeypatch.setattr(context, "transform_context", transformed)
    result = context.nested_context_model(rows, np.zeros((len(rows), 1)), np.asarray(profiles), family_boundary=boundary(), regularization=(1.,))
    assert result["status"] == "MULTIPLE_CONTEXTS_SUPPORTED" and result["selected_K"] == 2
    assert result["positive_outer_folds"] == 4
    assert all(r["predictive_gain"] > 0 for r in result["outer_folds"])
    assert {r["context"] for r in result["association_support"] if r["supported"]} == {0, 1}
