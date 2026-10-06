from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
import utility_outcomes_v2 as outcomes


def factorial():
    return [{'family': family, 'seed': seed, 'arm': arm,
             'operational_correct': bool((int(family) + seed) % 2),
             'tokens': 1000 + 100 * int(family) - (100 if arm == 'frozen_policy' else 0)}
            for family in ('0', '1', '2', '3') for seed in (0, 1) for arm in ('native', 'frozen_policy')]


def test_paired_seeds_preserve_accuracy_and_known_token_change():
    result = outcomes.inference(factorial(), ['0','1','2','3'], replicates=1000)
    assert result['point_estimates'] == [0., -100.]
    assert result['simultaneous_95_intervals'] == [[0.,0.],[-100.,-100.]]
    assert 'noninferiority' in result['interpretation']


def test_unknown_endpoint_remains_in_itt_bounds():
    rows = factorial(); rows[1]['tokens'] = None
    result = outcomes.inference(rows, ['0','1','2','3'], replicates=1000)
    assert result['status'] == 'INCOMPLETE_ENDPOINTS'
    assert result['point_estimates'] is None and result['joint_region'] is None
    low, high = result['identification_bounds'][1]
    assert low < -100 < high
    assert high - low == 16384 / 8


def test_missing_assignment_and_duplicate_cell_fail():
    rows = factorial()
    with pytest.raises(ValueError, match='factorial'):
        outcomes.paired_values(rows[:-1], ['0','1','2','3'])
    with pytest.raises(ValueError, match='factorial'):
        outcomes.paired_values(rows[:-1] + [rows[0]], ['0','1','2','3'])


def test_missing_generation_is_not_silently_scored_as_wrong():
    assignment={'uid':'test', 'family':'f', 'seed':0, 'arm':'native'}
    row=outcomes.extract_rows({'records':[{'assignment':assignment, 'status':'MISSING',
        'receipt_path':None,'receipt_sha256':None}]})[0]
    assert row['operational_correct'] is None and row['tokens'] is None and not row['gradeable']


def test_committed_generation_error_is_wrong_but_partial_tokens_are_not_savings():
    assignment={'uid':'test','family':'f','seed':0,'arm':'native'}
    receipt={'sha256':'receipt','assignment':assignment,'status':'GENERATION_ERROR',
        'binding_sha256':'binding','error':'infrastructure failure','result':None}
    row=outcomes.extract_rows({'records':[{'assignment':assignment,'status':'GENERATION_ERROR',
        'receipt_path':'unused','receipt_sha256':'receipt','source_binding_sha256':'binding'}]},
        receipt_loader=lambda _: receipt)[0]
    assert row['operational_correct'] is False and row['tokens'] is None and not row['gradeable']


def test_rebound_receipt_fails():
    item={'assignment':{'uid':'new'},'status':'GENERATION_ERROR','receipt_path':'unused',
          'receipt_sha256':'receipt','source_binding_sha256':'binding'}
    with pytest.raises(ValueError, match='assignment'):
        outcomes.extract_rows({'records':[item]},receipt_loader=lambda _: {'sha256':'receipt','assignment':{'uid':'old'}})
