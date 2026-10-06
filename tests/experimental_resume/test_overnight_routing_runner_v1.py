"""Tests for frozen same-prefix assignment and operator/ordered-mode boundaries."""
from collections import Counter
import copy
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/experimental_resume'))
import overnight_routing_runner_v1 as run


def design(mode='ordered_positive'):
    transition = 'candidate_to_verify'
    actions = []
    for name, expert in [('target', 189), *[(f'random{i}', 20+i) for i in range(4)]]:
        item = {'name': name, 'transition': transition, 'experts': [[28, [expert]]]}
        item.update({'bias': 1.} if mode == 'ordered_positive' else
                    {'kind': 'force', 'sign': 1, 'magnitude': 0.})
        actions.append(item)
    slots = [0] if mode == 'ordered_positive' else []
    arms = [{'name': n, 'role': 'native', 'policies': {}, 'slots': []}
            for n in ('native', 'native_duplicate')]
    arms += [{'name': 'target', 'role': 'target', 'policies': {transition: ['target']}, 'slots': slots},
             {'name': 'random', 'role': 'random', 'policies': {transition: ['random{random_set}']}, 'slots': slots}]
    return {'schema': 'overnight-routing-design-v1', 'mode': mode, 'transitions': [transition],
            'actions': actions, 'arms': arms, 'planned_contrasts': [['target', 'native']],
            'analysis_scopes': ['all', transition], 'analysis_seed': 20261004}


def assignment(d):
    source = run.base.sealed(run.SOURCE)
    rows = [r for r in source['rows'] if r['transition'] in d['transitions']]
    random_sets, orders = run.schedules(rows, d['arms'])
    return {'sha256': 'fixture-sha', 'rows': rows, 'arms': d['arms'], 'seeds': [0,1],
            'random_set_by_family_seed': random_sets, 'arm_order_by_uid_seed': orders}


def test_pairing_and_sharding_preserve_all_assigned_uids():
    d = design()
    run.validate_design(d)
    m = assignment(d)
    full = list(run.request_metadata(m, m['rows'], m['arms']))
    assert len(full) == 8*2*4
    assert len({meta['uid'] for _, meta in full}) == len(full)
    for offset in range(0, len(full), 4):
        group = full[offset:offset+4]
        assert len({meta['prompt_sha256'] for _, meta in group}) == 1
        assert len({meta['seed'] for _, meta in group}) == 1
        assert {meta['arm'] for _, meta in group} == {'native', 'native_duplicate', 'target', 'random'}
    split = list(run.request_metadata(m,m['rows'][:3],m['arms'])) + list(
        run.request_metadata(m,m['rows'][3:],m['arms']))
    assert split == full
    counts = Counter((meta['arm'],meta['execution_position']) for _, meta in full)
    for name in ('native', 'native_duplicate', 'target', 'random'):
        values = [counts[name,i] for i in range(4)]
        assert max(values)-min(values) <= 1


def test_native_operators_never_enter_ordered_pulse_adapter():
    d = design('native_operator')
    run.validate_design(d)
    table = run.build_policy_table(d['actions'])
    policy = table.policies[table.index_of('target')]
    assert policy.operator.kind == 'force' and policy.operator.sign == 1
    assert policy.schedule.kind == 'always'
    d['arms'][2]['slots'] = [0]
    with pytest.raises(ValueError, match='pulse slots'):
        run.validate_design(d)


def test_ordered_mode_rejects_delayed_only_and_arbitrary_dose():
    d = design()
    d['arms'][2]['slots'] = [512]
    with pytest.raises(ValueError, match='pulse slots'):
        run.validate_design(d)
    d = design()
    d['actions'][0]['bias'] = 2.
    with pytest.raises(ValueError):
        run.validate_design(d)


def test_repeated_same_policy_counts_two_pulses_without_early_or_late_edit():
    from moe_exp.routing_control.worker_adapter import OrderedPulse
    from moe_exp.routing_control.design import digest
    import numpy as np
    d = design()
    table = run.build_policy_table(d['actions'])
    body = {'action_policy_names': ['target','target'], 'slots': [0,512], 'horizon':1024}
    pulse = OrderedPulse.load({**body,'sha256':digest(body)},table,'target')
    mask,_ = pulse.rows(np.arange(1024),np.ones(1024,dtype=bool))
    assert mask.sum() == 512
    assert mask[:256].all() and mask[512:768].all()
    assert not mask[256:512].any() and not mask[768:].any()
    active = np.arange(1024)<600
    clipped,_ = pulse.rows(np.arange(1024),active)
    assert clipped.sum() == 344 and not clipped[600:].any()


def test_prices_cover_full_horizon_and_reject_too_large_shard():
    d = design()
    m = assignment(d)
    price = run.projection(m['rows'],m['arms'],2,14400)
    assert len(price['shards']) == 4
    assert all(s['estimated_wall_seconds'] <=14400 for s in price['shards'])
    assert run.workload(m['rows'],m['arms'],1024)['maximum_decode_tokens'] == 65536
    assert run.workload(m['rows'],m['arms'],256)['maximum_decode_tokens'] == 16384
    with pytest.raises(ValueError,match='exceeds specified wall'):
        run.projection(m['rows'],m['arms'],8,3600)


def test_completion_rejects_wrong_pulse_and_clips_on_closure():
    from moe_steer import engine
    d = design()
    manifest = {'horizon': 1024, 'actions': d['actions']}
    values = {'active_rows': 10., 'native_target_hits': 0., 'actual_target_hits': 0.,
              'membership_changes': 0., 'target_mass': 0., 'weight_l1': 0.}
    record = {'rows': {'target': 10}, 'segments': {'target': [[0,10]]},
              'dose': {'target': {'28': values}}}
    result = {'tokens': [2]*9 + [engine.THINK_END_ID] + [3]*10,
              'error': False, 'routed_present': True, 'finish': 'stop',
              'role': 'target', 'policies': ['target'], 'slots': [0],
              'action_dose': {'0': copy.deepcopy(record), '1': copy.deepcopy(record)},
              'inactive_native_checks': {rank: {'28': {'expert_identity_mismatches':0,
                  'weight_mismatches':0}} for rank in ('0','1')}}
    assert run.audit_output_dose(result,manifest)['active_rows'] == 10
    result['action_dose']['0']['segments']['target'] = [[0,20]]
    with pytest.raises(ValueError,match='boundary/order/closure'):
        run.audit_output_dose(result,manifest)
