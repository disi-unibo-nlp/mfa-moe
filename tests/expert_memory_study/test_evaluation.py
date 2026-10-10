import pytest
from moe_exp.expert_memory_study.cards import card
from moe_exp.expert_memory_study.contracts import allocated, seal
from moe_exp.expert_memory_study.grading import score
from moe_exp.expert_memory_study.pricing import shard_price
from moe_exp.expert_memory_study.stages import score_grid


@pytest.mark.parametrize("complete,accepted,capped,answer,expected", [
    (True, True, False, "2", (1, 1, 1)),
    (True, False, True, "1", (0, 0, 0)),
    (True, False, False, "", (0, 0, 0)),
    (True, False, False, "1", (0, 0, 1)),
    (True, None, False, "1", (0, 0, 1)),
    (False, False, False, "", (0, 0, 1)),
])
def test_all_assignment_grading(complete, accepted, capped, answer, expected):
    scorer = lambda *args, **kwargs: {"model_answer": answer, "is_correct": accepted}
    result = score({"execution_complete": complete, "text": "response", "finish_reason": "length" if capped else "stop"},
                   r"The answer is \boxed{2}.", scorer=scorer)
    assert (result["strict_score"], result["semantic_lower"], result["semantic_upper"]) == expected


def test_bounded_first_measurement_and_no_unpriced_bulk():
    campaign = seal({"saved_source_files": {"saved": "hash"}})
    measurement = card(campaign, "measure-cpu", request={"output": "price.json"})
    assert measurement["resources"]["seconds"] == 1200
    assert measurement["authorization"] is None and not measurement["submission_allowed"]
    with pytest.raises(ValueError, match="measured"):
        card(campaign, "final")
    with pytest.raises(ValueError, match="limit"):
        card(campaign, "qualification", seconds=7201)


def test_measured_shards_include_load_and_margin():
    price = seal({"load_seconds": 600, "worst_cap_assignment_seconds": 100})
    assert shard_price(price, 2)["reservation_seconds"] == 1000
    with pytest.raises(ValueError, match="walltime"):
        shard_price(price, 1000)


def test_real_data_and_inference_reject_unallocated_login_execution(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        allocated(gpu=True)


def test_incomplete_grades_cannot_become_final_analysis():
    roster = seal({"records": [{"uid": "a", "family": "f", "model": "gpt", "arm": "native"}]})
    grades = seal({"assignments_sha256": roster["sha256"], "records": []})
    with pytest.raises(ValueError, match="incomplete"):
        score_grid(grades, roster)
