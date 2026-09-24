"""Partial-run metrics must use matched IDs and distinguish missing coverage."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'partial_guiding', Path(__file__).resolve().parents[1] / 'report/analyze_partial_guiding.py')
partial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partial)


def row(correct, tokens, finish='stop'):
    return {'input': {'dataset': 'd', 'source_problem_id': 'p'},
            'is_correct': correct, 'generated_token_count': tokens, 'finish_reason': finish}


def test_completed_subset_uses_same_ids_not_full_baseline_denominator():
    baseline = {'a': row(True, 10), 'b': row(False, 20), 'missing': row(True, 5)}
    guided = {'a': row(False, 30, 'length'), 'b': row(False, 30, 'length')}
    result = partial.summarize(baseline, guided, ['a', 'b'])
    assert result['attempts'] == 2 and result['problems'] == 1
    assert result['baseline']['accuracy'] == .5
    assert result['guided']['accuracy'] == 0
    assert result['accuracy_delta_pp'] == -50
    assert result['mean_tokens_delta'] == 15
    assert result['guided']['limit_hits'] == 2
    assert result['wrong_to_right'] == 0 and result['right_to_wrong'] == 1
    assert partial.summarize(baseline, guided, []) is None
    with pytest.raises(KeyError):
        partial.summarize(baseline, guided, ['missing'])
