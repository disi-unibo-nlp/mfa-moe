import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from moe_exp.expert_memory_study.contracts import digest, load, save, seal
from moe_exp.expert_memory_study.journal import Journal
from moe_exp.expert_memory_study.population import assignments, components, proportional_order
from moe_exp.expert_memory_study.protocol import KEEP, MODELS, protocol
from moe_exp.expert_memory_study.statistics import blinded_discordance, paired_analysis, sample_size


def test_seal_tamper_and_immutable_publication(tmp_path):
    path = tmp_path / "receipt.json"
    receipt = save(path, {"value": 1})
    assert load(path) == receipt
    with pytest.raises(FileExistsError):
        save(path, {"value": 2})
    path.write_text(json.dumps({**receipt, "value": 2}))
    with pytest.raises(ValueError, match="seal"):
        load(path)


def test_concurrent_publication_has_one_winner(tmp_path):
    path = tmp_path / "concurrent.json"

    def write(value):
        try:
            save(path, {"value": value})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(write, (1, 2))) == 1
    assert load(path)["value"] in (1, 2)


def test_interrupted_assignment_needs_terminal_authorized_reconciliation(tmp_path):
    journal = Journal(tmp_path)
    body = {"model": "gpt", "family": "a"}
    assignment = {**body, "uid": digest(body)}
    journal.begin(assignment)
    with pytest.raises(ValueError, match="unresolved"):
        journal.begin(assignment)
    reconciliation = {"uid": assignment["uid"], "terminal": True,
                      "retry_authorized": True, "draw_committed": False}
    journal.begin(assignment, retry_receipt=reconciliation, maximum_retries=1)
    journal.commit(assignment, {"execution_complete": False})
    assert journal.begin(assignment)["status"] == "ALREADY_COMMITTED"


def test_ambiguous_submission_cannot_duplicate(tmp_path):
    journal = Journal(tmp_path)
    card = seal({"stage": "qualification"})
    journal.submission_intent(card)
    with pytest.raises(ValueError, match="reconciled"):
        journal.submission_intent(card)


def test_components_preserve_transitive_historical_aliases():
    result = components(["new1", "new2", "isolated"], [("new1", "alias"), ("alias", "new2")])
    assert sorted(map(len, result.values())) == [1, 3]


def test_proportional_order_unique_reproducible():
    rows = [{"family": str(i), "subject": "a" if i < 60 else "b", "level": 1} for i in range(100)]
    order = proportional_order(rows)
    assert order == proportional_order(list(reversed(rows)))
    assert len({r["family"] for r in order}) == 100
    assert abs(sum(r["subject"] == "a" for r in order[:32]) - 32 * .6) <= 1


def test_blinded_sizing_extremes_and_unknowns():
    assert sample_size([0] * 21)["N"] == 640
    assert sample_size([32] * 21)["N"] == 3712
    grid = np.zeros((32, 7, 4))
    assert blinded_discordance(grid) == [0] * 21
    grid[0, 0, 1] = np.nan
    assert blinded_discordance(grid)[:3] == [1] * 3
    flipped = 1 - grid
    assert blinded_discordance(flipped) == blinded_discordance(grid)


def test_no_zero_width_uncertainty_for_degenerate_data():
    result = paired_analysis(np.zeros((12, 7, 4)), draws=1000)
    assert all(r["simultaneous_interval"] is None for r in result["contrasts"])
    assert all(r["uncertainty_status"] == "UNAVAILABLE_DEGENERATE" for r in result["contrasts"])


def test_unknown_worst_case_and_shared_family_replicates():
    grid = np.tile(np.array([0., 1., 0., 1.]), (20, 7, 1))
    grid[:10, :, :] = np.array([1., 0., 1., 0.])
    grid[0, :, 1] = np.nan
    result = paired_analysis(grid, draws=1000)
    rows = result["contrasts"]
    assert len(rows) == 21 and result["shared_family_indices"]
    assert rows[0]["identified_difference"] == [0., .05]
    assert rows[0]["simultaneous_interval"] == rows[3]["simultaneous_interval"]


def test_assignment_seeds_and_identical_policy_aliases():
    population = seal({"pilot": [{"family": str(i), "question_id": f"q{i}"} for i in range(32)], "final_order": []})
    masks = {m: seal({"campaign_sha256": "campaign", "final_masks": {"state_balanced": [[0, 1]], "labelled_frequency": [[0, 1]],
                                     "random": [[1, 2]]}}) for m in MODELS}
    roster = assignments(population, masks, stage="pilot")["records"]
    assert len(roster) == 896 and len({r["seed"] for r in roster}) == 32
    assert len({r["seed"] for r in roster if r["family"] == "0"}) == 1
    assert all(r["alias_of_arm"] == "state_balanced" for r in roster if r["arm"] == "labelled_frequency")
    assert set(protocol()["models"]) == set(KEEP)
