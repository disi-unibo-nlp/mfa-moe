import math
import pytest

from moe_exp.routing_control.counterfactual import (
    designated_logprob, prediction_input, pricing, verify_route,
)


def test_only_past_native_tokens_enter_prediction_prefix():
    row = {'prompt_ids': [10, 11], 'prefix_ids': [20, 21], 'native_suffix_ids': [30, 31, 32, 33]}
    prompt, target = prediction_input(row, 1)
    assert prompt == [10, 11, 20, 21, 30] and target == 31
    changed = {**row, 'native_suffix_ids': [30, 31, 99, 100]}
    assert prediction_input(changed, 1) == (prompt, target)
    with pytest.raises(ValueError):
        prediction_input(row, True)


def test_exact_designated_token_is_required():
    assert designated_logprob([(8, -0.1, 1), (42, -3.2, 9)], 42) == -3.2
    for entries in ([(8, -0.1, 1)], [(42, -3.2, 9), (42, -3.2, 9)], [(42, math.inf, 1)]):
        with pytest.raises(ValueError):
            designated_logprob(entries, 42)


def test_force_keeps_native_expert_budget_and_enforces_membership():
    route = [list(range(8)) for _ in range(40)]
    assert verify_route(route, (28, [1, 2]), 'force_positive') == 2
    assert verify_route(route, (28, [9, 189]), 'force_negative') == 0
    with pytest.raises(ValueError):
        verify_route(route, (28, [1, 2]), 'force_negative')
    bad = [*route[:28], [0] * 8, *route[29:]]
    with pytest.raises(ValueError):
        verify_route(bad, (28, [9, 189]), 'bias')


def test_complete_price_counts_long_prefixes_and_fixed_cost():
    value = pricing(600_000, 148, 137_676, 12_288, 760, 20, 13)
    assert value['maximum_generated_tokens'] == 148
    assert value['proposed_wall_minutes'] >= 30
    assert value['two_A100_allocation_GPU_hours'] == value['proposed_wall_minutes'] / 30
