"""Training-only binary generator with a known conditional Bayes log-loss gain.

This defines a prospective power generator, not an observed correctness result.
Given baseline context C with probability q(C), an independent balanced signal Z
is revealed by an augmentation. Offsets preserve E[p(Y=1|C,Z)|C]=q(C).
The oracle improvement is I(Y;Z|C), in nats per attempt. Pipeline power still
requires fully refitting the registered controls, encoders and learner for each
training-only simulation; this helper alone cannot establish equivalence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _sigmoid(x):
    return 1/(1+np.exp(-np.asarray(x, float)))


def _probabilities(q, strength):
    logits = np.log(q)-np.log1p(-q)
    low, high = logits-strength, logits+strength
    for _ in range(64):
        middle = (low+high)/2
        marginal = (_sigmoid(middle-strength)+_sigmoid(middle+strength))/2
        low = np.where(marginal < q, middle, low)
        high = np.where(marginal >= q, middle, high)
    offset = (low+high)/2
    return _sigmoid(offset-strength), _sigmoid(offset+strength)


def _kl(p, q):
    # Masking preserves exact 0/1 endpoint limits without 0*log(0).
    result = np.zeros_like(p)
    positive, negative = p > 0, p < 1
    result[positive] += p[positive]*np.log(p[positive]/q[positive])
    result[negative] += (1-p[negative])*np.log((1-p[negative])/(1-q[negative]))
    return result


@dataclass(frozen=True)
class ConditionalGenerator:
    baseline_probability: tuple[float, ...]
    negative_signal_probability: tuple[float, ...]
    positive_signal_probability: tuple[float, ...]
    signal_strength: float
    target_gain_nats_per_attempt: float
    realized_oracle_gain_nats_per_attempt: float
    population: str = 'provided training-fold contexts only'

    def draw(self, seed: int, augmentation_dimension: int):
        """Known signal plus matched-dimensional noise; no held-out labels are input."""
        if type(augmentation_dimension) is not int or augmentation_dimension < 1:
            raise ValueError('augmentation dimension must be a positive integer')
        rng = np.random.default_rng(seed)
        n = len(self.baseline_probability)
        signal = rng.integers(0, 2, n)
        negative = np.asarray(self.negative_signal_probability)
        positive = np.asarray(self.positive_signal_probability)
        p = np.where(signal, positive, negative)
        features = rng.normal(size=(n, augmentation_dimension))
        features[:, 0] = 2*signal-1
        labels = rng.binomial(1, p)
        return {'augmentation': features, 'labels': labels, 'oracle_probability': p,
            'baseline_oracle_probability': np.asarray(self.baseline_probability),
            'oracle_gain': self.realized_oracle_gain_nats_per_attempt,
            'scope': self.population,
            'limitation': 'power for this explicitly balanced signal generator; actual pipeline power requires refitting and cannot be inferred here'}


def calibrate_training_generator(training_probabilities, target_gain):
    """Calibrate expected conditional information, preserving each context's marginal."""
    q = np.asarray(training_probabilities, float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q <= 0) or np.any(q >= 1):
        raise ValueError('finite strictly interior training-fold probabilities required')
    if target_gain not in (0., .001, .003, .005, .010):
        raise ValueError('use only the five preregistered conditional gains')
    def evaluate(strength):
        negative, positive = _probabilities(q, strength)
        info = float(np.mean((_kl(negative, q)+_kl(positive, q))/2))
        return info, negative, positive
    if target_gain == 0:
        strength, info, negative, positive = 0., 0., q.copy(), q.copy()
    else:
        low, high = 0., 20.
        if evaluate(high)[0] < target_gain:
            raise ValueError('training contexts cannot support the requested Bayes gain')
        for _ in range(60):
            middle = (low+high)/2
            if evaluate(middle)[0] < target_gain:
                low = middle
            else:
                high = middle
        strength = (low+high)/2
        info, negative, positive = evaluate(strength)
    if np.max(np.abs((negative+positive)/2-q)) > 1e-12 or abs(info-target_gain) > 1e-10:
        raise ValueError('conditional generator failed marginal/gain calibration')
    return ConditionalGenerator(tuple(q), tuple(negative), tuple(positive), strength, target_gain, info)


def monte_carlo_precision(successes, simulations=200):
    """Wilson interval for independent simulation rejection/coverage counts."""
    if type(successes) is not int or type(simulations) is not int or simulations < 1 or not 0 <= successes <= simulations:
        raise ValueError('valid integer simulation counts required')
    z = 1.959963984540054
    p = successes/simulations
    denominator = 1+z*z/simulations
    center = (p+z*z/(2*simulations))/denominator
    half = z*np.sqrt(p*(1-p)/simulations+z*z/(4*simulations*simulations))/denominator
    return {'estimate': p, 'Wilson_95': [max(0., center-half), min(1., center+half)],
            'simulations': simulations, 'independence_assumed': True}
