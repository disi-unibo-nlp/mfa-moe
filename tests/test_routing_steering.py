"""Scientific contract tests; tiny synthetic inputs, no checkpoint or GPU."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from moe_exp.correlation_pipeline.dynamics.common import digest
from moe_exp.correlation_pipeline.dynamics.contracts import (
    deduplicate, freeze_folds, require_development, sampling_weight, save, seal, validate,
    validate_feature,
)
from moe_exp.correlation_pipeline.dynamics.contrasts import (
    estimate_contrast, joint_question_draws, sibling_sequence_controls,
)
from moe_exp.correlation_pipeline.dynamics.prospective import design, evaluate, weights, estimated_histories
from moe_exp.correlation_pipeline.dynamics.experiment import freeze_branches, add_timing_arms, analyze_pilot
from moe_exp.correlation_pipeline.dynamics.evaluation import routing_features
from moe_exp.correlation_pipeline.continuation import exact_prefix, run_branches
from moe_exp.moe_guiding.native import NativePolicy, TokenContext, TimedNativeRouter, manipulate, calibrate_dose
from moe_exp.moe_guiding.native_integration import RequestClock
from moe_exp.correlation_pipeline.dynamics.annotation_plan import prepare, contiguous_blocks
from moe_exp.correlation_pipeline.dynamics.campaign import _scope_summaries
from moe_exp.moe_guiding.mechanisms import matched_sham, likelihood_panel


def member(cohort="A", probability=1.):
    return dict(cohort=cohort, inclusion_probability=probability if cohort == "A" else None,
                arm_inclusion_probability=probability if cohort == "B" else None)


def feature(q, *, correct=True, attempt=0):
    return dict(question_id=str(q), attempt_id=f"{q}-{attempt}", trace_id=f"{q}-{attempt}",
        model="m", dataset="d", decision_token=64, feature_end_token=64,
        budget_population="original", base={"position": 64.}, text={"word": .1}, state={},
        is_correct=correct, remaining_tokens=100, completion_tokens=164, memberships=[member()],
        routing_windows=[dict(layer=5, window=64, start_token=0, end_token=64,
            expert_rates={"0": .9 if correct else .1, "1": .1 if correct else .9},
            expert_weight_occupancy={"0": .9 if correct else .1, "1": .1 if correct else .9},
            boundary_gaps=[.1] * 64, set_turnover=.3, full_jsd=.01, full_shuffled_jsd=.005,
            full_local_entropy=.1, full_marginal_entropy=.3, full_entropy_difference=.2,
            overlap_lag1=.7, overlap_shuffled_lag1=.4)])


def test_contract_tamper_and_resume(tmp_path):
    artifact = seal("features", {"rows": []}, inputs={"source": "abc"}, config={}, population="test")
    save(tmp_path / "a.json", artifact)
    save(tmp_path / "a.json", artifact)
    modified = copy.deepcopy(artifact)
    modified["payload"]["rows"] = [1]
    with pytest.raises(ValueError, match="binding"):
        validate(modified)
    other = seal("features", {"rows": [1]}, inputs={"source": "abc"}, config={}, population="test")
    with pytest.raises(ValueError, match="overwrite"):
        save(tmp_path / "a.json", other)


def test_merge_extractions_requires_complete_unique_frozen_union(tmp_path):
    from moe_exp.correlation_pipeline.dynamics.campaign import merge_extractions
    from moe_exp.correlation_pipeline.dynamics.contracts import file_binding
    population = "original_budget_A_and_B_union"
    code = {"source_sha256": "test"}
    attempts = [{**feature(q), "capped": False} for q in range(2)]
    inv = seal("inventory", {"attempts": attempts}, inputs={"source": "test"}, config={},
               population=population, code=code)
    inventory = tmp_path / "inventory.json"
    save(inventory, inv)
    paths = []
    for i, attempt in enumerate(attempts):
        config = dict(include_weights=True, shard_index=i, shard_count=2)
        inputs = {"inventory": inv["binding"]}
        artifact = seal("features", {"attempt": attempt, "rows": []}, inputs=inputs,
                        config=config, population=population, code=code)
        path = tmp_path / f"feature-{i}.json"
        save(path, artifact)
        manifest = seal("analysis", {"features": [file_binding(path)], "measurements": [
            {"model": attempt["model"], "attempt_id": attempt["attempt_id"]}]}, inputs=inputs,
            config=config, population=population, code=code)
        path = tmp_path / f"manifest-{i}.json"
        save(path, manifest)
        paths.append(path)
    result = merge_extractions(inventory, paths)
    assert len(result["payload"]["features"]) == 2
    assert result["config"] == {"include_weights": True}
    with pytest.raises(ValueError, match="Incomplete"):
        merge_extractions(inventory, paths[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        merge_extractions(inventory, [paths[0], paths[0]])


def test_resource_projection_separates_only_first_startup_observation():
    from moe_exp.correlation_pipeline.dynamics.campaign import measured_resource_estimate
    inv = {"binding": "i", "population": "original", "payload": {"attempts": [
        {"model": "m", "completion_tokens": 100}] * 10}}
    points = [dict(model="m", attempt_id=str(i), completion_tokens=t, seconds=s,
                   output_bytes=50, peak_rss_kib=100) for i, t, s in ((0, 50, 80.), (1, 10, 1.), (2, 100, 10.))]
    ex = {"binding": "e", "inputs": {"inventory": "i"}, "payload": {"measurements": points}}
    result = measured_resource_estimate(inv, ex, startup_attempt_id="0")
    assert result["payload"]["startup_allowance_seconds_per_worker"] == 80.
    assert result["payload"]["models"][0]["projected_cpu_seconds"] == pytest.approx(100.)
    with pytest.raises(ValueError, match="first"):
        measured_resource_estimate(inv, ex, startup_attempt_id="1")


def test_extraction_storage_abort_preserves_completed_work(tmp_path, monkeypatch):
    from moe_exp.correlation_pipeline.dynamics import campaign
    inv = seal("inventory", {"attempts": [{**feature(1), "capped": False}]}, inputs={"source": "t"},
               config={}, population="original_budget_A_and_B_union")
    path = tmp_path / "inventory.json"
    save(path, inv)
    monkeypatch.setattr(campaign, "extract_attempt", lambda row, **kwargs: (row, []))
    with pytest.raises(RuntimeError, match="storage limit"):
        campaign.extract(path, tmp_path / "out", max_output_bytes=1)
    assert (tmp_path / "out/run.json").is_file()
    assert not (tmp_path / "out/m/1-0.json").exists()


def test_dedup_retains_sampling_and_capped_correct():
    base = dict(model="m", attempt_id="a", question_id="q", budget_population="original",
                is_correct=True, termination="length", memberships=[member()], captures=[{"cohort": "A"}])
    b = {**base, "memberships": [member("B", .25)], "captures": [{"cohort": "B"}]}
    joined = deduplicate([base, b])[0]
    assert len(joined["captures"]) == 2
    assert joined["is_correct"] is True and joined["termination"] == "length"
    assert sampling_weight(joined, "B") == 4
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate([base, {**b, "is_correct": False}])
    assert len(deduplicate([base, {**base, "budget_population": "extended"}])) == 2


def test_global_folds_reserve_excludes_b_and_all_siblings():
    rows = [dict(question_id=str(q), memberships=[member()]) for q in range(100)]
    rows += [dict(question_id=str(q), memberships=[member("B", .2)]) for q in range(10)]
    folds = freeze_folds(rows, "inventory")
    assert len(folds["payload"]["reserved_questions"]) == 18
    assert not set(folds["payload"]["reserved_questions"]) & set(map(str, range(10)))
    assert freeze_folds(list(reversed(rows)), "inventory")["payload"] == folds["payload"]
    for f in range(4):
        assert all(folds["payload"]["outer"][q] != f for q in folds["payload"]["inner"][str(f)])
    with pytest.raises(ValueError, match="Reserved"):
        require_development([{"question_id": folds["payload"]["reserved_questions"][0]}], folds)


def test_features_reject_future_labels_and_destination_leakage():
    row = feature(1)
    validate_feature(row)
    for change in ({"feature_end_token": 65}, {"destination_start_token": 63},
                   {"state": {"oracle": 1}, "state_provenance": "lookahead"}):
        with pytest.raises(ValueError):
            validate_feature({**row, **change})
    future = copy.deepcopy(row)
    future["routing_windows"][0]["end_token"] = 65
    with pytest.raises(ValueError):
        validate_feature(future)


def test_contrast_recovers_question_balanced_rd_and_occupancy():
    rows = [feature(q, correct=y, attempt=int(y)) for q in range(12) for y in (True, False)]
    # Adding 30 clones to one question cannot increase its weight or support.
    rows.extend(copy.deepcopy(rows[0]) for _ in range(30))
    specs = {"m": dict(layers=[5], num_experts=2, top_k=1)}
    estimates = estimate_contrast(rows, specs, "accuracy", bootstrap=100)
    selected = next(r for r in estimates if r["expert"] == 0 and r["metric"] == "selection_probability")
    assert selected["effect"] == pytest.approx(.8)
    assert selected["questions"] == 12 and selected["p"] is not None
    assert selected["ci"] == pytest.approx([.8, .8])
    assert len(estimates) == 4  # both experts and both primary metrics, including zeros
    fewer = estimate_contrast(rows[:18], specs, "accuracy", bootstrap=20)
    assert all(r["p"] is None and r["status"] == "descriptive" for r in fewer)


def test_entropy_variability_is_in_static_baseline():
    routing = routing_features(feature(0)["routing_windows"])
    assert "L5_W64_full_entropy_difference" in routing["static"]
    assert "L5_W64_full_entropy_difference" not in routing["temporal"]
    assert routing["temporal"]["L5_W64_full_jsd"] == .01
    assert routing["shuffled"]["L5_W64_full_jsd"] == .005


def test_efficiency_matching_keeps_models_separate():
    from moe_exp.correlation_pipeline.dynamics.contrasts import _matched_rows
    rows = [feature("shared", attempt=i) for i in range(4)]
    for row, model, length in zip(rows, ("m", "m", "other", "other"), (100, 200, 1000, 2000)):
        row.update(model=model, completion_tokens=length)
    matched = list(_matched_rows(rows, "efficient_success"))
    assert [arm for _, _, arm in matched] == [True, False, True, False]


def test_vector_question_bootstrap_matches_scalar_with_missing_questions():
    from moe_exp.correlation_pipeline.dynamics.contrasts import _vector_estimates, cluster_estimate
    draws = joint_question_draws(range(20), bootstrap=31)
    values = {q: np.array([q / 7 - .7, q / 3 + 1]) for q in range(0, 20, 2)}
    counts = np.stack([np.bincount(i, minlength=20) for i in draws[1]])
    result = _vector_estimates(values, draws, counts, 2, 10)
    for expert in range(2):
        scalar = cluster_estimate({q: v[expert] for q, v in values.items()}, draws)
        assert result[expert]["questions"] == scalar["questions"]
        for key in ("effect", "ci", "p", "standardized_effect"):
            assert result[expert][key] == pytest.approx(scalar[key])


def test_shared_question_bootstrap_and_sibling_permutation():
    names, a = joint_question_draws(["b", "a", "a"], bootstrap=10)
    other_names, b = joint_question_draws(["a", "b"], bootstrap=10)
    assert names == other_names
    np.testing.assert_equal(a, b)
    rows = [feature(q, correct=y, attempt=int(y)) for q in range(2) for y in (True, False)]
    controls = sibling_sequence_controls(rows)
    assert len(controls) == 4
    for r in controls:
        assert r["control_donor"] != r["attempt_id"]
        assert r["control_donor"].split("-")[0] == r["question_id"]


def test_prospective_design_train_only_and_prefix_invariance():
    train = [feature(q, correct=bool(q % 2)) for q in range(12)]
    test = [feature(100)]
    a, b, audit = design(train, test, 4)
    future = copy.deepcopy(test)
    future[0].update(is_correct=False, remaining_tokens=99999, source_class="Explore", next_class="Verify")
    future[0]["routing_windows"][0]["boundary_gaps"] = [999.] * 64
    a2, _, audit2 = design(train, future, 4)
    np.testing.assert_equal(a, a2)
    assert audit == audit2
    assert audit["thresholds"]["L5"] == pytest.approx(.1)
    future[0]["routing_windows"] = copy.deepcopy(test[0]["routing_windows"])
    _, b2, _ = design(train, future, 4)
    np.testing.assert_equal(b, b2)


def test_b_weights_are_eligible_question_balanced():
    rows = [feature("a"), feature("a"), feature("b")]
    rows[0]["memberships"] = [member("B", .25)]
    rows[1]["memberships"] = [member("B", .5)]
    rows[2]["memberships"] = [member("B", .1)]
    np.testing.assert_allclose(weights(rows, "B"), [1/3, 1/6, .5])
    with pytest.raises(ValueError):
        weights(rows, "A")


def test_nested_prediction_known_signal_and_missing_state_gate():
    original = [feature(q, correct=bool(q % 2)) for q in range(32)]
    folds = freeze_folds(original, "test")
    rows = [r for r in original if r["question_id"] in folds["payload"]["outer"]]
    result = evaluate(rows, folds, bootstrap=20, min_questions=3)
    assert result["status"] == "exploratory_predictive"
    assert result["models"]["static_routing"]["log_loss"] < result["models"]["prefix_text"]["log_loss"]
    assert result["models"]["prefix_state"]["status"] == "unavailable"
    assert not result["dynamic_trigger_qualified"]


def native_selector(k, scale=1.):
    def select(logits):
        ids = logits.topk(k, -1).indices
        w = logits.gather(-1, ids).softmax(-1) * scale
        return w, ids.to(torch.int32)
    return select


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_native_k_noop_parity_and_reweight_scale(k):
    scores = torch.arange(12, dtype=torch.float32)[None].repeat(3, 1)
    select = native_selector(k, scale=2.5)
    p = NativePolicy(3, (10,), "noop")
    w, ids, log = manipulate(scores, select, p, top_k=k)
    reference_w, reference_ids = select(scores)
    assert torch.equal(w, reference_w) and torch.equal(ids, reference_ids)
    assert not log["membership_changed"].any()
    w2, ids2, _ = manipulate(scores, select, replace(p, action="selected_reweight", dose=1.), top_k=k)
    assert torch.equal(ids, ids2)
    torch.testing.assert_close(w2.sum(-1), w.sum(-1))


def test_selection_bias_realized_membership_and_calibration():
    scores = torch.tensor([[3., 2., 1.], [3., 2., .9]])
    p = NativePolicy(0, (2,), "selection_bias", dose=1.2)
    _, ids, log = manipulate(scores, native_selector(2), p, top_k=2)
    assert (ids == 2).any(-1).all() and log["membership_changed"].all()
    calibrated = calibrate_dose(scores, native_selector(2), p, top_k=2,
                               doses=[0., .1, 1.2], target=1.)
    assert calibrated["selected"]["dose"] == 1.2
    assert calibrated["selected"]["weight_l1"] > 0


def test_timing_request_isolation_and_phase_end():
    p = NativePolicy(0, (2,), "selection_bias", dose=5., start_token=2, pulse_tokens=2)
    events = []
    route = TimedNativeRouter(p, top_k=2, sink=events.append)
    contexts = [TokenContext("a", 1, True), TokenContext("b", 2, True),
                TokenContext("a", 2, True), TokenContext("a", 3, False), TokenContext("b", 4, True)]
    scores = torch.tensor([[3., 2., 1.]]).repeat(5, 1)
    route(scores, native_selector(2), contexts)
    assert [e["reason"] for e in events] == ["before_pulse", "applied", "applied", "not_reasoning", "after_pulse"]
    with pytest.raises(ValueError, match="Duplicate"):
        route(scores[:1], native_selector(2), [TokenContext("a", 3, True)])


def test_request_clock_exact_prefix_and_next_output_index():
    clock = RequestClock("r", [10, 11], [12, 13], closing_sequences=[[20, 21]])
    c = clock.observe(torch.tensor([10, 11, 12, 13]), torch.arange(4))
    assert [r.output_token for r in c] == [-1, 0, 1, 2]
    assert c[-1].reasoning
    c = clock.observe(torch.tensor([20, 21]), torch.tensor([4, 5]))
    assert c[0].reasoning and not c[1].reasoning
    assert clock.transitions[0]["output_token"] == 4
    with pytest.raises(ValueError, match="contiguous"):
        clock.observe(torch.tensor([9]), torch.tensor([8]))


def test_arbitrary_exact_prefix_and_original_budget():
    record = {"metadata": {"token_replay": {"prompt_token_ids": [9], "completion_token_ids": [1, 2, 3, 4]},
                           "generation_config": {"max_tokens": 10, "temperature": .7, "top_p": .9, "top_k": 20}}}
    value = exact_prefix(record, 2)
    record["metadata"]["token_replay"]["completion_token_ids"][2:] = [99, 98, 97]
    assert exact_prefix(record, 2) == value
    assert value["max_new_tokens"] == 8
    with pytest.raises(ValueError):
        exact_prefix(record, 6)
    record["metadata"]["generation_config"] = dict(max_tokens=None, max_model_len=16,
        sampler=dict(temperature=1., top_p=1., top_k=0, min_p=None), seed=7)
    record["metadata"]["effective_max_tokens"] = 15
    actual = exact_prefix(record, 2)
    assert actual["max_new_tokens"] == 13
    assert actual["generation_config"]["temperature"] == 1.


def pilot():
    folds = freeze_folds([dict(question_id=str(q), memberships=[member()]) for q in range(100)], "test")
    prefixes = []
    for q in folds["payload"]["reserved_questions"]:
        prefixes.append(dict(question_id=q, attempt_id=q, model="m", prefix_tokens=2048,
            prefix_token_ids=[1] * 2048, prompt_token_ids=[9], prefix_sha256=digest([[9], [1] * 2048]),
            generation_config=dict(max_tokens=4096, temperature=.7, top_p=.9, top_k=20),
            reasoning_at_branch=True, budget_population="original", closing_sequences=[[2]], source_binding="source"))
    candidate = dict(model="m", layer=5, experts=[0], eligible_for_mechanism_pilot=True, stable_folds=3)
    calibration = {a: dict(dose=-.1 if a == "negative" else .1, development_only=True,
        prefix_binding="development", membership_change=.05, weight_l1=.02)
        for a in ("positive", "negative", "sham", "reweight")}
    calibration["sham"].update(experts=[1], prevalence_matched=True)
    qualification = dict(status="passed", historical_backend_revalidated=True,
                         selected_ids_equal=True, dispatched_weights_equal=True, output_equal=True, attribution_equal=True)
    return freeze_branches(prefixes, folds, candidate, calibration, qualification)


def test_pilot_fixed_size_budget_and_timing():
    manifest = pilot()
    assert len(manifest["payload"]["branches"]) == 160
    for b in manifest["payload"]["branches"]:
        assert b["max_new_tokens"] == 2048
        assert b["policy"]["start_token"] == 2048
    timing = add_timing_arms(manifest, dict(manifest_binding=manifest["binding"], measurable_routing_change=True))
    assert len(timing["payload"]["branches"]) == 64
    assert all(b["policy"]["start_token"] >= 2048 for b in timing["payload"]["branches"])


def test_branch_resume_and_paired_question_effects(tmp_path):
    manifest = pilot()
    calls = []
    def generate(b):
        calls.append(b["branch_id"])
        return dict(suffix_token_ids=[1, 2], prompt_echo_verified=True, telemetry_verified=True, finish_reason="stop")
    results = run_branches(manifest, tmp_path, generate)
    again = run_branches(manifest, tmp_path, generate)
    assert len(calls) == 160 and results == again
    for r in results:
        r["is_correct"] = True
    analysis = analyze_pilot(manifest, results, bootstrap=20)
    assert analysis["status"] == "complete"
    effect = next(r for r in analysis["contrasts"] if r["metric"] == "is_correct")
    assert effect["questions"] == 16 and effect["effect"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_pilot(manifest, results + results[:1], bootstrap=20)


def test_contiguous_annotation_budget_and_exact_reuse():
    rows, units = [], {}
    for q in range(20):
        for outcome in (True, False):
            r = feature(q, correct=outcome, attempt=int(outcome))
            r.update(memberships=[member("B", .2)], trace_sha256=digest(r["attempt_id"]))
            rows.append(r)
            units[r["attempt_id"]] = [dict(index=i, start=i*2, end=i*2+1, text="s") for i in range(200)]
    plan = prepare(rows, units, model="m", input_binding="source")
    assert len(plan["payload"]["questions"]) == 16
    assert plan["payload"]["new_lookahead_labels"] == 3840
    assert plan["payload"]["new_prefix_only_labels"] == 200
    assert len(contiguous_blocks(list(range(50)))) == 50
    repeated = prepare(rows, units, model="m", input_binding="source", reusable=plan["payload"]["units"][:1])
    assert repeated["payload"]["new_lookahead_labels"] == 3839


def test_matched_scope_entropy_and_occupancy():
    tensors = dict(ids=torch.tensor([[[0], [0], [1], [1]]]),
                   weights=torch.ones((1, 4, 1)), reasoning_tokens=torch.tensor([0, 1]))
    row = dict(layers=[9], num_experts=2, completion_tokens=4)
    scopes = _scope_summaries(tensors, row)
    full, reasoning = scopes
    assert full["layer"] == reasoning["layer"] == 9
    assert full["mixture_marginal_entropy"] == pytest.approx(np.log(2))
    assert reasoning["mixture_marginal_entropy"] == 0
    assert full["expert_rates"] == [.5, .5]
    assert reasoning["expert_rates"] == [1., 0.]


def test_sham_matching_and_equal_compute_likelihood_is_separate():
    sham = matched_sham([0], {0: .2, 1: .21, 2: .8}, layer=5)
    assert sham["experts"] == [1]
    def score(prefix, targets, dose):
        return dict(target_token_ids=targets, log_probabilities=[-1. + dose] * len(targets), expert_executions=8)
    panel = likelihood_panel([1], [2, 3], {"baseline": 0., "bias": .1}, score, verification_binding="verified")
    assert panel["routes"]["bias"]["mean_log_probability_change"] == pytest.approx(.1)
    assert panel["claim"] == "teacher_forced_likelihood_only"
    def unequal(prefix, targets, dose):
        return {**score(prefix, targets, dose), "expert_executions": 8 if dose == 0 else 9}
    with pytest.raises(ValueError, match="equal compute"):
        likelihood_panel([1], [2], {"baseline": 0., "bias": .1}, unequal, verification_binding="verified")


def test_native_instance_hook_and_restoration():
    from types import SimpleNamespace
    from moe_exp.moe_guiding.native_integration import instrument
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            def compute(hidden_states, router_logits, indices_type=None, *, input_ids=None):
                return native_selector(2)(router_logits)
            self.router = SimpleNamespace(_compute_routing=compute, top_k=2,
                                          global_num_experts=3, eplb_state=None)
        def forward(self, scores):
            return self.router._compute_routing(scores, scores)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].mlp = torch.nn.Module()
            self.layers[0].mlp.experts = Experts()
        def forward(self, input_ids, positions):
            scores = torch.tensor([[3., 2., 1.]]).repeat(len(input_ids), 1)
            return self.layers[0].mlp.experts(scores)
    model, log = Model(), []
    router = model.layers[0].mlp.experts.router
    original = router._compute_routing
    clock = RequestClock("r", [9], [8], closing_sequences=[[7]])
    policy = NativePolicy(0, (2,), "selection_bias", dose=5., start_token=1)
    with instrument(model, policy, clock, log.append):
        _, ids = model(torch.tensor([9, 8]), torch.tensor([0, 1]))
    assert router._compute_routing is original
    assert 2 not in ids[0].tolist() and 2 in ids[1].tolist()
    assert [e["output_token"] for e in log] == [0, 1]


def test_timing_analysis_reuses_original_baseline():
    first = pilot()
    timing = add_timing_arms(first, dict(manifest_binding=first["binding"], measurable_routing_change=True))
    results = []
    for manifest, branches in ((first, timing["payload"]["reference_branches"]),
                               (timing, timing["payload"]["branches"])):
        for b in branches:
            results.append(dict(branch_id=b["branch_id"], manifest_binding=manifest["binding"],
                                status="complete", total_tokens=2050, is_correct=True))
    result = analyze_pilot(timing, results, bootstrap=20)
    assert result["status"] == "complete"
    assert {r["arm"] for r in result["contrasts"]} == {"delayed", "continuous"}


def test_estimated_history_is_prefix_invariant():
    rows = [feature(1) for _ in range(3)]
    for i, r in enumerate(rows):
        r["decision_token"] = 64 * (i + 1)
    p = np.zeros((3, 7))
    p[:, 0] = 1.
    original = estimated_histories(rows, p)
    p[-1] = [0., 1., 0., 0., 0., 0., 0.]
    changed = estimated_histories(rows, p)
    assert original[:2] == changed[:2]
    assert original[-1]["estimated_switches"] == 0
    assert changed[-1]["estimated_switches"] == 1
