"""Source-bound operator masks, fixed horizons, and per-transition shared controls."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/experimental_resume'))
import overnight_routing_runner_v2 as runner
import overnight_routing_qualify_v2 as qualification
from overnight_routing_worker_v2 import OperatorPulse
from moe_exp.routing_control.design import digest


def fresh_design():
    return runner.base.sealed(runner.common.DOC/'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json')


def test_frozen_common_horizon_design_has14_and10_arms_and28_valid_scoped_contrasts():
    design=fresh_design()
    transitions,horizon,actions,arms=runner.normalized(design)
    assert horizon==1024
    assert [len(arms[t]) for t in transitions]==[14,10]
    contrasts=runner.scoped_contrasts(design,arms)
    assert len(contrasts)==28
    assert all(x['scope'] in transitions for x in contrasts)
    assert {a['kind'] for a in actions}=={'bias','force','reweight'}


def test_operator_masks_are_native_before_between_after_pulses_and_stop_at_closure():
    _,_,actions,_=runner.normalized(fresh_design());table=runner.common.build_policy_table(actions)
    for action in actions:
        body={'action_policy_names':[action['name']],'slots':[0],'horizon':1024}
        pulse=OperatorPulse.load({**body,'sha256':digest(body)},table,action['name'])
        mask,indices=pulse.rows(np.arange(-10,1030),np.arange(-10,1030)<150)
        assert mask.sum()==150
        assert not mask[:10].any() and not mask[160:].any()
        assert set(indices[mask])=={table.index_of(action['name'])}


def test_repeated_force_and_unqualified_magnitude_rejected():
    _,_,actions,_=runner.normalized(fresh_design());table=runner.common.build_policy_table(actions)
    action=next(a for a in actions if a['kind']=='force')
    body={'action_policy_names':[action['name']]*2,'slots':[0,512],'horizon':1024}
    with pytest.raises(ValueError,match='second pulse'):
        OperatorPulse.load({**body,'sha256':digest(body)},table,action['name'])
    design=copy.deepcopy(fresh_design());next(a for a in design['actions'] if a['kind']=='reweight')['magnitude']=2.
    with pytest.raises(ValueError,match='unqualified operator'):
        runner.normalized(design)


def test_C_frozen_diagnostic_is_normalized_to256_pulse_without_changing_targets():
    design=runner.base.sealed(runner.common.DOC/'OVERNIGHT_DISCOVERY_C_DESIGN_v1.json')
    transitions,horizon,actions,arms=runner.normalized(design)
    assert horizon==256 and [len(arms[t]) for t in transitions]==[8,8]
    assert all(arm['slots']==[0] for values in arms.values() for arm in values if arm['role']!='native')
    _,_,rows=runner.enrollment(design)
    price=runner.projection(rows,arms,horizon,7200,4)
    assert all(s['estimated_wall_seconds']<=7200 for s in price['shards'])
    assert sum((s['end_row']-s['start_row'])*16 for s in price['shards'])==208


def test_qualification_complete_price_and_operator_preemption_fixture_coverage():
    manifest=qualification.prepared()
    assert manifest['maximum_requests']==28
    assert manifest['maximum_decode_tokens']==24576
    assert manifest['estimated_wall_seconds']<=7200
    assert {p['role'] for p in manifest['plan'] if p['preempt']}=={'force','reweight'}
    assert {p['role'] for p in manifest['plan'] if p['closed']}=={'force','reweight'}
    assert {p['role'] for p in manifest['plan'] if p['horizon']==256}=={'bias','force','reweight'}


def test_fresh_unequal_arm_count_whole_row_shards_and_unique_assignments():
    design=fresh_design();_,horizon,actions,arms=runner.normalized(design)
    source=runner.base.sealed(runner.common.SOURCE)
    rows=sorted(source['rows'],key=lambda r:(list(arms).index(r['transition']),r['uid']))
    sets,orders=runner.schedules(rows,arms)
    m={'sha256':'fixture','rows':rows,'arms_by_transition':arms,'seeds':[0,1],
       'random_set_by_family_transition_seed':sets,'arm_order_by_uid_seed':orders}
    allmeta=[meta for _,meta in runner.request_metadata(m,rows)]
    assert len(allmeta)==8*28+5*20 and len({x['uid'] for x in allmeta})==len(allmeta)
    price=runner.projection(rows,arms,horizon,8100)
    recovered=[]
    for shard in price['shards']:
        selected=rows[shard['start_row']:shard['end_row']]
        assert len({r['transition'] for r in selected})==1
        recovered.extend(meta for _,meta in runner.request_metadata(m,selected))
    assert recovered==allmeta


def test_completion_rejects_force_dose_miss_and_reweight_expert_substitution():
    result={'error':False,'role':'target','action_dose':{}}
    for rank in ('0','1'):
        result['action_dose'][rank]={'dose':{'action':{'28':{
            'active_rows':256.,'actual_target_hits':256.,'native_target_hits':5.,'membership_changes':251.}}}}
    manifest={'actions':[{'name':'action','kind':'force','experts':[[28,[189]]]}]}
    assert runner.audit_operator_dose(result,manifest,lambda *_:{})
    result['action_dose']['0']['dose']['action']['28']['actual_target_hits']=255.
    with pytest.raises(ValueError,match='force-in'):
        runner.audit_operator_dose(result,manifest,lambda *_:{})
    manifest['actions'][0]['kind']='reweight'
    with pytest.raises(ValueError,match='same-hidden-state'):
        runner.audit_operator_dose(result,manifest,lambda *_:{})


def test_qualification_gate_rejects_missing_or_failed_result_without_generation():
    with pytest.raises((KeyError,ValueError)):
        qualification.validate_result({'schema':'overnight-routing-qualification-result-v2','pass':False})
