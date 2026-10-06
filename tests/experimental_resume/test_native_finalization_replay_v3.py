from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
from native_finalization_replay_v3 import snapshot, compare


def test_owned_snapshot_does_not_follow_buffer_or_output_mutation():
    original = SimpleNamespace(token_ids=[1, 2], routed_experts=np.array([[[3, 4]]]))
    frozen = snapshot(original)
    original.token_ids.append(5); original.routed_experts[:] = 9
    assert frozen['tokens'] == [1, 2] and frozen['routes'].tolist() == [[[3, 4]]]


def test_unknown_routes_never_count_as_equal():
    a = {'tokens': [1], 'routes': None}
    result = compare(a, a)
    assert result['tokens_equal'] and result['routes_equal'] is None
    assert result['routes_status'] == 'UNKNOWN'


def test_comparison_distinguishes_order_membership_and_token_divergence():
    a = {'tokens': [1, 2], 'routes': np.array([[[3, 4]]])}
    b = {'tokens': [1, 8], 'routes': np.array([[[4, 3]]])}
    result = compare(a, b)
    assert not result['tokens_equal'] and result['first_token_difference'] == 1
    assert not result['routes_equal'] and result['route_sets_equal']
    assert result['first_route_difference']['position'] == [0, 0, 0]
