"""Combined qualification adds coverage without erasing original failed receipts."""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))
import combine_overnight_operator_qualification_v4 as combined
import adjudicate_overnight_operator_qualification_v3 as adjudication
import overnight_routing_dose_audit_v3 as corrected
import overnight_routing_runner_v2 as runner
from test_overnight_routing_dose_audit_v3 import fixture


def pulse_case():
    record, manifest = fixture('bias')
    record.update(uid='fixture', tokens=[7]*1024, finish='length', closed=False, horizon=1024,
                  operator='bias', policy='test', policies=['test','test'], slots=[0,512])
    for entry in record['action_dose'].values():
        entry['rows'] = {'test':512}; entry['segments'] = {'test':[[0,256],[512,768]]}
        entry['dose']['test']['28'].update(active_rows=512.,actual_target_hits=900.,native_target_hits=600.,
                                         membership_changes=300.,weight_l1=256.,target_mass=250.)
    route = np.broadcast_to(np.arange(8), (1024,40,8)).copy()
    old = {'uid':'fixture','pass':True,'reasons':[]}
    return record, manifest, route, old


def test_full_second_pulse_is_required_by_independent_raw_audit():
    record, manifest, route, old = pulse_case()
    assert adjudication.check_one(record,old,manifest,route,runner,corrected)['pass']
    for entry in record['action_dose'].values():
        entry['rows']['test'] = 256
        entry['segments']['test'] = [[0,256]]
    assert not adjudication.check_one(record,old,manifest,route,runner,corrected)['pass']


def test_rank_disagreement_and_top8_corruption_still_fail():
    record, manifest, route, old = pulse_case()
    record['action_dose']['1']['dose']['test']['28']['weight_l1'] += 1
    assert not adjudication.check_one(record,old,manifest,route,runner,corrected)['pass']
    record, manifest, route, old = pulse_case(); route[600,28,0] = route[600,28,1]
    assert not adjudication.check_one(record,old,manifest,route,runner,corrected)['pass']


def test_combined_receipt_explicitly_preserves_both_original_failures(monkeypatch):
    prior = {'original_result_path':'original','original_result_sha256':'old-fail', 'sha256':'cpu-fail',
             'checks':[{'pass':i!=20} for i in range(28)], 'artifacts':{'old-raw':'old'}}
    new = {'sha256':'gpu-pass','job_id':'42','code_files':{},'artifacts':{'new-raw':'new'}}
    monkeypatch.setattr(combined,'evidence',lambda:(runner,{'sha256':'original-manifest'},prior,
                                                    {'sha256':'new-manifest'},new,[{'pass':True}]*3))
    result = combined.prepared()
    assert result['original_pass'] is False and result['original_adjudication_pass'] is False
    assert result['requests'] == 31 and result['supplemental_requests'] == 3
    assert result['original_gpu_requests_rerun'] == 0 and result['worker_changed'] is False
    assert result['preserved_accepted_original_case_indices'] == [i for i in range(28) if i!=20]
    assert result['original_coverage_gap']['status'].startswith('PRESERVED_FAIL')


def test_adapter_binds_combined_sources_and_supplement_without_schema_changes(monkeypatch):
    import overnight_routing_adjudicated_v4 as adapter
    monkeypatch.setattr(adapter,'INSTALLED',False)
    receipt={'sha256':'combined','original_result_path':'original','original_result_sha256':'original-fail',
             'supplemental_requests':3,'supplemental_result_sha256':'supplement',
             'original_adjudication_sha256':'cpu-fail'}
    monkeypatch.setattr(combined,'qualified_result',lambda:receipt)
    monkeypatch.setattr(runner,'qualified_result',runner.qualified_result)
    monkeypatch.setattr(runner,'validate',lambda manifest,driver:('validated',))
    monkeypatch.setattr(runner.common,'audit_output_dose',runner.common.audit_output_dose)
    monkeypatch.setattr(runner.common,'write_once',lambda path,body:{**body,'sha256':runner.base.digest(body)})
    adapter.install()
    value=runner.common.write_once(Path('/unused'),{'schema':'overnight-routing-manifest-v2','code_files':{}})
    assert value['cpu_checker_adjudication']['result_sha256']=='combined'
    assert value['cpu_checker_adjudication']['supplemental_gpu_requests']==3
    assert runner.validate(value,'unused')==('validated',)
    value['cpu_checker_adjudication']['supplemental_result_sha256']='changed'
    with pytest.raises(ValueError,match='adjudicated checker'):runner.validate(value,'unused')
