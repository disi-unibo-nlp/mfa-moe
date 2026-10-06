import numpy as np

from moe_exp.routing_control.uncertainty import paired_binary_bounds


def test_zero_discordance_and_seed_cancellation_do_not_imply_zero_uncertainty():
    n = 39
    baseline = np.zeros((n, 2), int)
    result = paired_binary_bounds(baseline, baseline, [str(i) for i in range(n)], contrasts=564)
    assert result['estimate'] == 0
    assert result['simultaneous_interval'][0] < 0 < result['simultaneous_interval'][1]
    cancellation = paired_binary_bounds([[1, 0]], [[0, 1]], ['family'], contrasts=564)
    assert cancellation['discordant_families'] == 1
    assert cancellation['simultaneous_interval'] == [-1., 1.]


def test_family_weights_are_preserved_in_conservative_bound():
    zero = np.zeros((4, 2), int)
    result = paired_binary_bounds(zero, zero, ['a', 'a', 'a', 'b'], contrasts=2)
    assert result['n_families'] == 2
    assert result['simultaneous_interval'][1] == 1.
