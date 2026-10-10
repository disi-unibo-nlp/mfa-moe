"""Scientific contract regressions for the prospective routing study."""
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/experimental_resume")]
from moe_exp.routing_control import study_v1 as study
from moe_exp.routing_control import state_analysis_v1 as state
import dense_recovery_v2 as recovery


def test_gap_breaks_adjacency_reentry_and_terminal_event():
    rows = [{"sentence_index": i, "label": label, "token_start": i * 10,
             "token_end": (i + 1) * 10} for i, label in enumerate(("Analyze", None, "Verify"))]
    result = state.transition_summary(rows, closed=True)
    assert sum(map(sum, result["transition_counts"])) == 0
    assert result["loops"] == {}
    assert result["destination_specific_exit_counts"] == [{"source": "Verify", "destination": "closure", "count": 1}]
    assert result["dwell"][0]["censored"] is True
    assert result["dwell"][-1]["censored"] is False
    result = state.transition_summary(rows, hit_cap=True)
    assert result["termination"] == "cap_censored"
    assert all(row["censored"] for row in result["dwell"])


def test_capture_includes_closure_producing_row():
    ids = np.tile(np.arange(8), (3, 40, 1))
    state.validate_capture([1, 248069, 2], ids, np.full(ids.shape, 1 / 8))
    with pytest.raises(ValueError, match="align exactly"):
        state.validate_capture([1, 248069, 2], ids[:2])
    with pytest.raises(ValueError, match="top-k"):
        state.validate_capture([1, 248069, 2], np.zeros_like(ids))


def test_trailing_features_are_invariant_to_future_and_never_invent_gate_confidence():
    ids = np.tile(np.arange(8), (256, 40, 1))
    ids[64:128] += 8
    before = state.trailing_features(ids, 128)
    ids[128:] += 40
    after = state.trailing_features(ids, 128)
    np.testing.assert_array_equal(before["selection"], after["selection"])
    np.testing.assert_array_equal(before["velocity"], after["velocity"])
    assert before["executed_weight"] is None
    assert before["native_selection_boundary_gap"] is None
    assert before["router_probability_mass"] is None
    assert state.trailing_features(ids, 20)["status"] == "UNAVAILABLE_SHORT_PREFIX"


def test_delayed_assignment_no_fire_after_act_or_closure():
    assert not study.delayed_fire(opportunity_position=10, position=73, desired_act_occurred=False, closed=False)
    assert study.delayed_fire(opportunity_position=10, position=74, desired_act_occurred=False, closed=False)
    assert not study.delayed_fire(opportunity_position=10, position=90, desired_act_occurred=True, closed=False)
    assert not study.delayed_fire(opportunity_position=10, position=90, desired_act_occurred=False, closed=True)


def test_duplicate_connected_exclusion_and_independent_family_reservation():
    questions = [{"id": str(i), "subject": "algebra" if i % 2 else "geometry", "level": i % 5 + 1}
                 for i in range(140)]
    result = study.freeze_independent_families(questions, [("old", "0"), ("0", "1")], ["old"], revision="revision")
    assert result["eligible"] == 138
    assert len(result["validation"]) == 32 and len(result["final"]) == 96
    chosen = result["validation"] + result["final"]
    assert not {r["question_id"] for r in chosen} & {"0", "1"}
    assert len({r["family"] for r in chosen}) == 128
    assert result == study.freeze_independent_families(questions, [("old", "0"), ("0", "1")], ["old"], revision="revision")
    shortage = study.freeze_independent_families(questions[:100], [], [], revision="revision")
    assert shortage["status"] == "UNAVAILABLE_FAMILY_SHORTAGE" and not shortage["validation"]


def utility_rows(families):
    return [{"family": f, "seed": s, "arm": arm, "correct": True,
             "generated_tokens": 100 if arm == "native" else 80,
             "controller_tokens": 0 if arm == "native" else 5, "intervened_positions": 1}
            for f in families for s in (0, 1) for arm in ("native", "policy")]


def test_frontier_counts_online_controller_cost_and_requires_all_validation_families():
    families = [f"f{i}" for i in range(32)]
    result = study.utility_frontier(utility_rows(families), families, ["policy"], draws=1000)
    assert result["contrasts"][1]["estimate"] == -15
    assert result["accuracy_loss_margin"] is None
    assert result["frontier"] == ["policy"]
    assert study.choose_policy(result)["selected_policy"] == "policy"
    rows = utility_rows(families)
    missing = study.utility_frontier(rows[:1] + rows[2:], families, ["policy"], draws=1000)
    assert missing["missing_execution_assignments"] == 1
    assert missing["contrasts"][0]["missing_accuracy_bounds"] == [-100 / 64, 0]
    assert missing["contrasts"][1]["missing_cost_upper_bound"] == "unbounded"
    with pytest.raises(ValueError, match="all 32"):
        study.choose_policy(missing)


def test_no_claimed_cost_improving_policy_if_all_cost_more():
    families = [f"f{i}" for i in range(32)]
    rows = utility_rows(families)
    for row in rows:
        if row["arm"] == "policy": row["controller_tokens"] = 50
    result = study.utility_frontier(rows, families, ["policy"], draws=100)
    assert study.choose_policy(result)["status"] == "NO_COST_IMPROVING_WINNER"


def test_screen_limits_and_matched_random_sign():
    targets = [{"layer": 2, "expert": 3, "transition": "candidate_to_verify", "prefix_recognition_qualified": True}]
    prefixes = [{"uid": str(i), "family": str(i), "transition": "candidate_to_verify", "eligible": True,
                 "prefix_ids_sha256": str(i)} for i in range(30)]
    controls = {("L2E3", str(i), s): 4 for i in range(30) for s in (0, 1)}
    assignments = study.screen_assignments(targets, prefixes, controls)
    assert len(assignments) == 24 * 2 * 5
    assert len({r["uid"] for r in assignments}) == len(assignments)
    assert {r["bias"] for r in assignments if r["arm"] == "random_negative"} == {-1.}
    assert all(r["top_k"] == 8 and r["pulse_tokens"] == 256 for r in assignments)


def test_pack_reuses_original_shards_and_prices_all_remaining_labels():
    inv = {"sha256": "inventory", "missing": 96, "shards": [
        {"shard": i, "missing_batches": [{"start": i * 32, "stop": (i + 1) * 32}]} for i in range(3)]}
    price = {"sha256": "old", "cold_load_seconds": 400, "shutdown_seconds": 100,
             "per_shard_overhead_seconds": 100, "prompt_records": [{"prompt_tokens": 200}] * 96,
             "repeat_factor": 1.25, "bounded_prefill_tokens_per_second": 1000,
             "max_tokens": 1024, "bounded_decode_tokens_per_second": 100}
    result = recovery.pack(inv, price, max_wall=86400)
    assert len(result["groups"]) == 1 and result["groups"][0]["shards"] == [0, 1, 2]
    assert result["loads_including_bounded_recovery"] == 2
    expected = 2 * (1.25 * (96 * 200 / 1000 + 96 * 1024 / 100) + 1100) / 3600
    assert abs(result["maximum_GPU_hours"] - expected) < 2 / 3600


def test_singleton_selection_requires_semantic_recognition_and_excludes_references():
    rows = [{"status": "SUPPORTED_ASSOCIATION", "prefix_recognition_qualified": False,
             "layer": 1, "expert": 3, "transition": "candidate_to_verify", "risk_difference": .5,
             "consistent_folds": 4}]
    assert study.select_singletons(rows) == []
    rows[0]["prefix_recognition_qualified"] = True
    assert len(study.select_singletons(rows)) == 1
    rows[0].update(layer=28, expert=189)
    assert study.select_singletons(rows) == []


def test_multi_shard_runner_uses_one_model_load_and_keeps_exact_batch_bindings(workdir, monkeypatch):
    frame = {"sha256": "frame", "code_digest": "code", "records": [{"blind_id": str(i)} for i in range(64)]}
    price = {"sha256": "price", "shards": [{"start": 0, "stop": 32}, {"start": 32, "stop": 64}],
             "prompt_records": [{"blind_id": str(i), "prompt_tokens": 1, "prompt_ids_sha256": "exact"}
                                for i in range(64)]}
    inv = {"shards": [{"shard": i, "missing_batches": [{"shard": i, "start": i * 32,
             "stop": (i + 1) * 32, "blind_ids": [str(j) for j in range(i * 32, (i + 1) * 32)]}]} for i in range(2)]}
    manifest = {"sha256": "manifest"}
    recovery.save(workdir / "PACKAGING_QUALIFICATION.json", {"status": "PASS", "manifest_sha256": "manifest"})
    for i in range(2):
        directory = workdir / f"labels-shard-{i:03d}"
        for name in ("batches", "attempts", "loads"): (directory / name).mkdir(parents=True)
        recovery.save(directory / "BINDING.json", recovery.native.label_binding(frame, price, i))
    loads, calls = [], []
    monkeypatch.setattr(recovery, "compute", lambda *a: None)
    monkeypatch.setattr(recovery, "validate", lambda *a: (manifest, inv, {"groups": [{"shards": [0, 1]}]}, frame, price))
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setattr(recovery, "model_and_params", lambda *a: (loads.append(1), None))
    def call(model, params, block, prompts):
        calls.append(len(block))
        return [{**p, "label": "Read", "raw_completion": "Read", "finish_reason": "stop", "generated_tokens": 1}
                for p in prompts]
    monkeypatch.setattr(recovery, "call", call)
    recovery.run(workdir, 0)
    assert loads == [1] and calls == [32, 32]
    for i in range(2):
        summary = recovery.sealed(workdir / f"labels-shard-{i:03d}/SUMMARY.json")
        assert summary["sentences"] == 32
    recovery.run(workdir, 0)
    assert loads == [1] and calls == [32, 32]
