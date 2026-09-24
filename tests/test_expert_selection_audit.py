"""Show why equal-sign lift and risk difference need not select equal sets."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'selection_audit', Path(__file__).resolve().parents[1] / 'report/audit_expert_selection.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('polarity,accuracies', [('positive', [.9, .7]), ('negative', [.1, .3])])
def test_rank_reversal_and_preserved_sign(polarity, accuracies):
    experts = [{'expert': i, 'frequency_mass': mass, 'accuracy': accuracy,
                'lift': accuracy - .5, 'problem_support': 4}
               for i, (mass, accuracy) in enumerate(zip([1., 4.], accuracies))]
    policy = {'baseline_accuracy': .5, 'calibration_problems': list(range(10)),
              'expert_polarity': polarity, 'min_support': 4, 'max_experts': 1, 'top_k': 1,
              'layers': {'3': {'experts': experts, 'scores': [1., 0.]}}}
    result = audit.audit_policy(policy)['3']
    assert result['saved_selected'] == [0]
    assert result['rd_selected'] == [1]
    assert result['overlap'] == 0
    assert all(s['lift'] * s['problem_balanced_rd'] > 0 for s in result['experts'])
    policy['layers']['3']['scores'] = [0., 1.]
    with pytest.raises(AssertionError):
        audit.audit_policy(policy)
