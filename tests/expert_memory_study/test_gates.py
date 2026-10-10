from types import SimpleNamespace

import numpy as np
import pytest

from moe_exp.expert_memory_study.contracts import digest, load, save, seal
from moe_exp.expert_memory_study.journal import Journal
from moe_exp.expert_memory_study.population import assignments
from moe_exp.expert_memory_study.pricing import approve_final_plan, final_plan
from moe_exp.expert_memory_study.protocol import ARMS, MODELS
from moe_exp.expert_memory_study.selection import coverage
from moe_exp.expert_memory_study.stages import materialize_aliases, resolve_result


def masks():
    return {m: seal({"campaign_sha256": "campaign", "final_masks": {
        "state_balanced": [[0, 1]], "labelled_frequency": [[0, 1]], "random": [[1, 2]]}}) for m in MODELS}


def population():
    return seal({"pilot": [{"family": f"p{i}", "question_id": f"p{i}"} for i in range(32)],
                 "final_order": [{"family": f"f{i}", "question_id": f"f{i}"} for i in range(640)]})


def test_minimum_class_coverage_is_distinct_from_minimum_layer_cell():
    counts = np.array([[[[1, 9], [9, 1]], [[4, 6], [4, 6]]]])
    result = coverage(counts, ["f"], [[0], [0]])
    assert result["class_mean_layer_coverage"] == [.5, .4]
    assert result["minimum_class_coverage"] == .4
    assert result["minimum_class_layer_coverage"] == .1


def test_population_reservation_rejects_pilot_final_overlap_and_small_final():
    source = population()
    source["final_order"][0]["family"] = "p0"
    with pytest.raises(ValueError, match="isolation"):
        assignments(source, masks(), stage="final", final_size=640)
    with pytest.raises(ValueError, match="size"):
        assignments(population(), masks(), stage="final", final_size=32)


def test_different_cards_cannot_submit_the_same_unresolved_assignment(tmp_path):
    journal = Journal(tmp_path)
    journal.submission_intent(seal({"stage": "pilot", "request": {"assignment_uids": ["x"]}}))
    with pytest.raises(ValueError, match="overlapping"):
        journal.submission_intent(seal({"stage": "final", "request": {"assignment_uids": ["x"]}}))


def test_accounting_no_match_never_authorizes_a_retry(tmp_path, monkeypatch):
    from moe_exp.expert_memory_study.cards import reconcile_submission
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="lmolfett\n" if command[0] == "id" else
                               "login01.leonardo.local\n" if command[0] == "hostname" else "")

    monkeypatch.setattr("moe_exp.expert_memory_study.cards.subprocess.run", run)
    card = save(tmp_path / "card.json", {"stage": "measure-cpu", "request": {"output": "p"}})
    Journal(tmp_path / "submissions").submission_intent(card)
    result = reconcile_submission(tmp_path / "card.json", tmp_path)
    assert result["status"] == "UNKNOWN_NO_MATCH" and not result["performs_dispatch"]
    assert all(c[0] != "sbatch" for c in calls)
    assert not list((tmp_path / "submissions").glob("*.terminal.json"))


def test_aliases_commit_without_duplicate_inference_and_keep_source_binding(tmp_path):
    roster = assignments(population(), masks(), stage="pilot")
    source = next(r for r in roster["records"] if r["model"] == "gpt" and r["arm"] == "state_balanced")
    journal = Journal(tmp_path / "results")
    journal.begin(source)
    original = journal.commit(source, {"execution_complete": True, "text": "answer"})
    materialize_aliases(journal, tmp_path, source, roster)
    alias = next(r for r in roster["records"] if r["model"] == "gpt" and r["family"] == source["family"]
                 and r["arm"] == "labelled_frequency")
    assert resolve_result(tmp_path, alias, roster) == original
    assert load(tmp_path / "results" / (alias["uid"] + ".result.json"))["no_duplicate_inference"]


def test_complete_allocation_prices_only_canonical_draws_and_never_shrinks_precision():
    campaign = {"sha256": "campaign"}
    freeze = seal({"models": masks()})
    roster = assignments(population(), masks(), stage="final", final_size=640, frozen_masks_sha256=freeze["sha256"])
    prices = {(m, a): seal({"campaign_sha256": "campaign", "model": m, "arm": a, "context": 40960,
                           "load_seconds": 1., "worst_cap_assignment_seconds": .01}) for m in MODELS for a in ARMS}
    availability = seal({"complete": True, "source_files": {"observed": "hash"}, "reviewed_epoch": 1.,
                         "remaining_core_hours": 100., "available_storage_bytes": 1024**4,
                         "remaining_seconds": 1000., "parallel_gpu_jobs": 2})
    plan = final_plan(campaign, roster, freeze, prices, availability)
    assert plan["N"] == 640 and plan["remaining_canonical_draws"] == 640 * 21
    assert all(s["arm"] != "labelled_frequency" for s in plan["shards"])
    review = seal({"approved_plan_sha256": plan["sha256"], "user_quote": "Approve this exact allocation",
                   "complete_allocation_authorized": True, "scheduler_capacity_confirmed": True,
                   "deadline_confirmed": True, "reserved_core_hours": 100., "reserved_storage_bytes": 1024**4})
    assert approve_final_plan(plan, review)["dispatch_authorized"]
    with pytest.raises(ValueError, match="confirmation"):
        approve_final_plan(plan, {**review, "scheduler_capacity_confirmed": False})
    failed = final_plan(campaign, roster, freeze, prices, {**availability, "available_storage_bytes": 0})
    assert failed["status"] == "INFEASIBLE" and failed["N"] == 640
    with pytest.raises(ValueError, match="authorization"):
        approve_final_plan(failed, review)
