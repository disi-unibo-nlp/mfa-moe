from moe_exp.routing_control.resource_analysis_v1 import exposure_diagnostics, itemize


def rows():
    return [{"family": f, "seed": s, "arm": arm, "correct": not (f == "a" and arm == "policy"),
             "generated_tokens": 80 if arm == "policy" else 100, "controller_tokens": 5 if arm == "policy" else 0,
             "reasoning_tokens": 50, "intervened_positions": 10 if arm == "policy" else 0}
            for f in ("a", "b") for s in (0, 1) for arm in ("native", "policy")]


def test_missing_runtime_is_not_zero_and_controller_tokens_count():
    result = itemize(rows(), ["a", "b"], ["native", "policy"])
    metrics = result["arms"][1]["metrics"]
    assert metrics["total_policy_tokens"]["mean"] == 85
    assert metrics["wall_seconds"]["mean"] is None
    assert metrics["wall_seconds"]["status"] == "UNAVAILABLE_OR_PARTIAL"


def test_damage_ratios_retain_zero_exposure_and_missing_grades():
    result = exposure_diagnostics(rows(), ["a", "b"], "policy")
    assert result["accuracy_damage_probability"] == .5
    assert result["accuracy_damage_per_intervened_position"] == .05
    assert result["net_correct_to_incorrect_flips_per_intervened_position"] == .05
    zero = rows()
    for row in zero:
        row["intervened_positions"] = 0
    result = exposure_diagnostics(zero, ["a", "b"], "policy")
    assert result["status"] == "UNAVAILABLE_ZERO_EXPOSURE"
    assert result["accuracy_damage_per_intervened_position"] is None
    zero[1]["correct"] = None
    result = exposure_diagnostics(zero, ["a", "b"], "policy")
    assert result["measured_families"] == 1 and len(result["missing"]) == 1
