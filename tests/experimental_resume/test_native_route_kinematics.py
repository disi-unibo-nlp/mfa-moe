import numpy as np
import pytest

from scripts.experimental_resume.native_route_kinematics import motion, window_profiles


def test_fixed_windows_and_motion():
    ids = np.tile(np.arange(8, dtype=np.int32), (1, 192, 1))
    ids[:, 64:128, :] += 8
    ids[:, 128:, :] += 16
    weights = np.ones_like(ids, dtype=np.float32)
    freq, gate, starts, ends, tail = window_profiles(ids, weights, np.arange(192), experts=32)
    assert freq.shape == (3, 1, 32)
    assert starts.tolist() == [0, 64, 128] and ends.tolist() == [64, 128, 192]
    assert tail == 0
    velocity, acceleration, signed, turnover = motion(freq, gate)
    assert velocity[:, 0].tolist() == [1., 1.]
    assert acceleration.shape == (1, 1) and acceleration[0, 0] == 2
    assert signed[0, 0] == 0
    assert turnover[:, 0].tolist() == [1., 1.]


def test_invalid_routing_shape_rejected():
    ids = np.zeros((1, 64, 7), dtype=np.int32)
    with pytest.raises(ValueError):
        window_profiles(ids, np.ones_like(ids), np.arange(64))
