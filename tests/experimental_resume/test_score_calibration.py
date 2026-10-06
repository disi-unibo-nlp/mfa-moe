import pytest

from moe_exp.routing_control.score_calibration import calibration


def test_exact_repeat_does_not_hide_cross_api_error():
    scores = {'designated_a': -.1, 'designated_b': -.1,
        'full_vocab_a': -.11, 'full_vocab_b': -.11}
    assert not calibration([scores])['pass']
    scores['full_vocab_a'] = scores['full_vocab_b'] = -.1
    assert calibration([scores])['pass']


def test_repeat_noise_is_measured_without_substituting_teacher_forcing():
    row = {'designated_a': -.1, 'designated_b': -.102,
        'full_vocab_a': -.101, 'full_vocab_b': -.103}
    result = calibration([row])
    assert result['pass'] and result['checks']['mean_abs']['repeat_baseline'] == pytest.approx(.002)
    with pytest.raises(ValueError):
        calibration([{**row, 'full_vocab_a': float('nan')}])
