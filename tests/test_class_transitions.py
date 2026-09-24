"""Protect directed adjacency and outcome-group averaging in report figures."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

report = Path(__file__).resolve().parents[1] / 'report'
sys.path.insert(0, str(report))
spec = importlib.util.spec_from_file_location('class_transitions', report / 'generate_class_transitions.py')
transitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transitions)
sys.path.pop(0)


def test_direction_self_transitions_and_sampling_gaps():
    units = [{'index': i, 'label': label} for i, label in
             [(5, 'B'), (0, 'A'), (1, 'B'), (2, 'B'), (4, 'A')]]
    matrix = transitions.transition_counts(units, ['A', 'B'])
    np.testing.assert_array_equal(matrix, [[0, 2], [0, 1]])


def test_mean_includes_labelled_attempts_without_pairs():
    matrix = np.array([[0, 2], [1, 0]])
    result = transitions.summarize([matrix, np.zeros((2, 2), dtype=int)], 2)
    assert result['attempts'] == 2
    assert result['attempts_with_pairs'] == 1
    assert result['pairs'] == 3
    assert result['mean_counts_per_attempt'] == [[0, 1], [.5, 0]]
    assert transitions.summarize([], 2)['mean_counts_per_attempt'] is None


def test_duplicate_indices_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        transitions.transition_counts([{'index': 0, 'label': 'A'}] * 2, ['A'])


def test_unknown_labels_are_rejected_instead_of_bridged():
    with pytest.raises(ValueError, match='Unknown'):
        transitions.transition_counts([{'index': 0, 'label': 'unknown'}], ['A'])
