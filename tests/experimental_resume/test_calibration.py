import numpy as np
import pytest

from moe_exp.routing_control.calibration import calibrate_training_generator, monte_carlo_precision


@pytest.mark.parametrize('gain', [0., .001, .003, .005, .010])
def test_calibrates_conditional_information_while_preserving_baseline(gain):
    q = np.array([.03, .15, .35, .6, .85, .97])
    generator = calibrate_training_generator(q, gain)
    a = np.array(generator.negative_signal_probability)
    b = np.array(generator.positive_signal_probability)
    assert (a+b)/2 == pytest.approx(q, abs=1e-12)
    def entropy(p):
        return -(p*np.log(p)+(1-p)*np.log1p(-p))
    assert float(np.mean(entropy(q)-(entropy(a)+entropy(b))/2)) == pytest.approx(gain, abs=1e-10)
    first, second = generator.draw(42, 20), generator.draw(42, 20)
    assert first['augmentation'].shape == (len(q), 20)
    assert np.array_equal(first['labels'], second['labels'])
    assert np.array_equal(first['augmentation'], second['augmentation'])


def test_zero_simulated_errors_do_not_have_zero_uncertainty():
    assert monte_carlo_precision(0)['Wilson_95'][1] > 0
    assert monte_carlo_precision(200)['Wilson_95'][0] < 1
    with pytest.raises(ValueError):
        calibrate_training_generator([.2, .8], .02)
    with pytest.raises(ValueError):
        calibrate_training_generator([0., 1.], .003)
