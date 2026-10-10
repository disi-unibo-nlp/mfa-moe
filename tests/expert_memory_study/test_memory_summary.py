from moe_exp.expert_memory_study.contracts import seal
from moe_exp.expert_memory_study.protocol import ARMS, MODELS
from moe_exp.expert_memory_study.stages import arm_summaries


def test_accuracy_uses_all_assignments_while_memory_keeps_its_observed_denominator():
    assigned, grades = [], []
    for family in ("a", "b"):
        for model in MODELS:
            for arm in ARMS:
                assignment = {"uid": f"{family}/{model}/{arm}", "family": family, "model": model, "arm": arm}
                assigned.append(assignment)
                observed = family == "a"
                memory = [{"total_expert_bytes": 10, "total_model_tensor_bytes": 110,
                           "memory": {"allocated": 200, "reserved": 300, "peak_allocated": 250,
                                      "nvml_process_residence": [{"bytes": 350}]}}] * 2 if observed else None
                grades.append({"assignment": assignment, "strict_score": int(observed),
                               "semantic_lower": int(observed), "semantic_upper": 1,
                               "status": "STRICT_ACCEPTED" if observed else "ASSIGNED_MISSING",
                               "memory": memory, "prefill_memory": [{"peak_allocated": 240}] * 2 if observed else None})
    roster = seal({"records": assigned})
    result = arm_summaries(seal({"records": grades, "assignments_sha256": roster["sha256"]}), roster)
    assert len(result) == 28
    for row in result:
        assert row["assigned_families"] == 2 and row["memory_observed_families"] == 1
        assert row["strict_accuracy"] == .5 and row["semantic_bounds"] == [.5, 1.]
        assert row["assigned_failures"] == 1
        assert row["memory"]["expert_bytes"]["mean"] == 20
        assert row["memory"]["nvml_process_bytes"]["mean"] == 700
