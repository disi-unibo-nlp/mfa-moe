"""Inserted expert slots differ from changed token rows when two targets enter."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/experimental_resume'))
import overnight_routing_runner_v1 as old
import overnight_routing_dose_audit_v3 as corrected


def fixture(kind='force', target_count=2):
    from moe_steer.engine import THINK_END_ID
    action = {'name': 'test', 'kind': kind, 'sign': 1, 'magnitude': 0 if kind == 'force' else 1,
              'experts': [[28, [9, 189][:target_count]]]}
    layer = {'active_rows': 24., 'actual_target_hits': 48., 'native_target_hits': 21.,
             'membership_changes': 27., 'weight_l1': 12.473389625549316,
             'target_mass': 9.692789077758789}
    entry = {'rows': {'test': 24}, 'segments': {'test': [[0, 24]]}, 'dose': {'test': {'28': layer}}}
    result = {'tokens': [7]*23+[THINK_END_ID]+[7]*100, 'error': False, 'routed_present': True,
              'finish': 'stop', 'role': 'target', 'policies': ['test'], 'slots': [0],
              'inactive_native_checks': {rank: {'28': {'expert_identity_mismatches': 0,
                'weight_mismatches': 0}} for rank in ('0', '1')},
              'action_dose': {rank: copy.deepcopy(entry) for rank in ('0', '1')}}
    return result, {'horizon': 1024, 'actions': [action]}


def test_observed_force_receipt_is_valid_27_insertions_in_24_rows_with_two_targets():
    result, manifest = fixture()
    with pytest.raises(ValueError, match='membership'):
        old.audit_output_dose(result, manifest)
    assert corrected.audit_output_dose(result, manifest)['active_rows'] == 24
    layer = result['action_dose']['0']['dose']['test']['28']
    assert layer['membership_changes'] == layer['actual_target_hits'] - layer['native_target_hits']


@pytest.mark.parametrize('metric,value', [('membership_changes', 193), ('actual_target_hits', 49),
                                       ('native_target_hits', 49), ('weight_l1', 49)])
def test_impossible_two_target_slot_or_mass_counts_still_fail(metric, value):
    result, manifest = fixture()
    for rank in ('0', '1'): result['action_dose'][rank]['dose']['test']['28'][metric] = value
    with pytest.raises(ValueError): corrected.audit_output_dose(result, manifest)


def test_reweight_preserves_membership_and_rank_checks_remain_active():
    result, manifest = fixture('reweight')
    with pytest.raises(ValueError, match='slot turnover'):
        corrected.audit_output_dose(result, manifest)
    for rank in ('0', '1'):
        layer = result['action_dose'][rank]['dose']['test']['28']
        layer['membership_changes'] = 0
        layer['actual_target_hits'] = layer['native_target_hits']
    corrected.audit_output_dose(result, manifest)
    result['action_dose']['1']['dose']['test']['28']['weight_l1'] += 1
    with pytest.raises(ValueError, match='TP ranks'):
        corrected.audit_output_dose(result, manifest)


def test_tighter_singleton_bound_is_diagnostic_without_assuming_stable_tie_selection():
    result, manifest = fixture(target_count=1)
    for rank in ('0', '1'):
        result['action_dose'][rank]['dose']['test']['28']['actual_target_hits'] = 24
    report = corrected.audit_output_dose(result, manifest)
    assert all(not d['within_tighter_target_bound'] for d in report['slot_turnover_diagnostics'])
