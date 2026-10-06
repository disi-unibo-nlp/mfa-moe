"""Exact native expert-frequency and gate aggregation for a sentence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/extract_discovery_routes.py'
SPEC = importlib.util.spec_from_file_location('discovery_routes_test', SOURCE)
ROUTES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTES)


def test_sentence_profiles_keep_native_topk_frequency_and_normalized_gate_mass():
    ids = np.array([[[0, 1], [1, 2], [2, 3]], [[3, 2], [2, 1], [1, 0]]])
    weights = np.array([[[2., 1.], [4., 1.], [1., 1.]], [[1., 1.], [1., 3.], [2., 1.]]])
    freq, gate = ROUTES.profiles(ids, weights, [0, 1], experts=4)
    np.testing.assert_allclose(freq[0], [.25, .5, .25, 0])
    np.testing.assert_allclose(gate[0], [1/3, 1/6 + 2/5, 1/10, 0])
    np.testing.assert_allclose(freq.sum(axis=1), 1.)
    np.testing.assert_allclose(gate.sum(axis=1), 1.)


def test_sentence_profiles_reject_duplicate_ownership_and_zero_gate_mass():
    ids = np.zeros((1, 2, 1), dtype=np.int64)
    weights = np.ones_like(ids, dtype=float)
    with pytest.raises(ValueError, match='owned'):
        ROUTES.profiles(ids, weights, [0, 0], experts=2)
    with pytest.raises(ValueError, match='zero selected'):
        ROUTES.profiles(ids, np.zeros_like(weights), [0], experts=2)
