import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from measurement_study.campaign_v3_capture import resume_records,validate_verification,_deadline_reached,approved_output,differential_receipt,claim_capture_output
from measurement_study.hf_chunked_capture import chunk_ranges
STAGE=Path(__file__).resolve().parents[2]/'results/measurement-study/campaign-v3/tests'
STAGE.mkdir(parents=True,exist_ok=True)

class CaptureRunnerTests(unittest.TestCase):
    def test_chunk_completion_coverage_across_prompt_boundary(self):
        ranges=list(chunk_ranges(13,5,4))
        self.assertEqual(ranges,[(0,4,4),(4,8,1),(8,12,0),(12,13,0)])
        positions=[x-5 for a,b,p in ranges for x in range(a+p,b)]
        self.assertEqual(positions,list(range(8)))
        with self.assertRaises(ValueError):list(chunk_ranges(5,5,4))

    def test_unchanged_resume_and_corrupt_rejection(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            out=Path(temp);row={'id':'trace','completion_tokens':9,'trace_sha256':'trace-sha'}
            manifest={'binding':'manifest-sha','attempts':[row]}
            files=[]
            for i in range(2):
                f=out/str(i);f.write_bytes(b'capture')
                files.append({'path':str(f),'bytes':7,'sha256':hashlib.sha256(b'capture').hexdigest()})
            marker={'id':'trace','tokens':9,'trace_sha256':'trace-sha','manifest_binding':'manifest-sha','files':files}
            (out/'trace.complete.json').write_text(json.dumps(marker))
            completed,remaining,used=resume_records(manifest,out)
            self.assertEqual((len(completed),remaining,used),(1,[],14))
            manifest['binding']='new-code-or-outcome'
            with self.assertRaisesRegex(ValueError,'Incompatible'):resume_records(manifest,out)
            manifest['binding']='manifest-sha';(out/'0').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt'):resume_records(manifest,out)

    def test_verification_requires_longest_and_differential(self):
        m={'source':'glm','validation_binding':'v','maximum_input_tokens':130000,'spec':{'layers':[0,2]}}
        r={'status':'passed','validation_binding':'v','maximum_input_tokens':130000,'all_routed_layers':[0,2],
           'exact_saved_token_coverage':True,'unchanged_resume_verified':True,'differential':{'status':'passed'}}
        validate_verification(m,r)
        with self.assertRaises(ValueError):validate_verification(m,{**r,'maximum_input_tokens':32000})
        with self.assertRaises(ValueError):validate_verification(m,{**r,'differential':{}})

    def test_deadline_drains_before_new_attempt(self):
        with patch.dict('os.environ',{'MEASUREMENT_DEADLINE_EPOCH':'1000'}),patch('time.time',return_value=950):
            self.assertTrue(_deadline_reached({'seconds_per_token':.1},{'completion_tokens':1000},.2))
        sizing={'status':'measured','seconds_per_token':.1,'per_attempt_seconds':0.,'drain_margin_seconds':900}
        measured={'status':'measured','seconds_per_token':.1,'per_attempt_seconds':60.,'drain_margin_seconds':900}
        with patch.dict('os.environ',{'MEASUREMENT_DEADLINE_EPOCH':'100000'}),patch('time.time',return_value=950):
            self.assertFalse(_deadline_reached({'sizing':sizing},{'completion_tokens':10},0.))
        with patch.dict('os.environ',{'MEASUREMENT_DEADLINE_EPOCH':'2500'}),patch('time.time',return_value=950):
            self.assertFalse(_deadline_reached({'sizing':sizing},{'completion_tokens':10},0.))
            # The drain margin grows with the measured cost of the attempt that
            # would start now, so long attempts are not started near the end.
            self.assertTrue(_deadline_reached({'sizing':measured},{'completion_tokens':10000},0.))
        unmeasured={'status':'unmeasured_verification','drain_margin_seconds':1800}
        with patch.dict('os.environ',{'MEASUREMENT_DEADLINE_EPOCH':'2000'}),patch('time.time',return_value=100):
            self.assertFalse(_deadline_reached({'sizing':unmeasured},{'completion_tokens':1},0.))
        with patch.dict('os.environ',{'MEASUREMENT_DEADLINE_EPOCH':'2000'}),patch('time.time',return_value=300):
            self.assertTrue(_deadline_reached({'sizing':unmeasured},{'completion_tokens':1},0.))

    def test_stale_differential_rejected_without_capture_markers(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            path=Path(temp)/'differential.json'
            binding={'code':'old','short':'trace'}
            differential_receipt(path,binding,{'status':'passed'})
            self.assertEqual(differential_receipt(path,binding)['status'],'passed')
            with self.assertRaisesRegex(ValueError,'Stale'):
                differential_receipt(path,{**binding,'code':'new'})
            with self.assertRaisesRegex(ValueError,'did not pass'):
                differential_receipt(path,binding,{'status':'failed'})

    def test_partial_capture_root_cannot_rebind_after_crash(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temp:
            manifest={'binding':'input','code_binding':'code'}
            claim_capture_output(manifest,temp)
            claim_capture_output(manifest,temp)
            with self.assertRaisesRegex(ValueError,'changed'):
                claim_capture_output({**manifest,'binding':'new'},temp)

    def test_old_storage_rejected(self):
        with self.assertRaises(ValueError):approved_output('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/measurement-study-v1/capture')
        self.assertIn('campaign-v3',str(approved_output(STAGE)))

if __name__=='__main__':unittest.main()


def test_annotation_inputs_reject_stale_or_conflicting_sources(tmp_path):
    from measurement_study.annotation_provenance import verify_annotation_inputs
    from measurement_study.bundle import file_sha
    import pytest
    path=tmp_path/'annotations.json';path.write_text('[]')
    item={'path':str(path),'sha256':file_sha(path)}
    binding={'inputs':[item,dict(item)],'annotations_sha256':'embedded-payloads-digest'}
    assert verify_annotation_inputs(binding)=={str(path.resolve()):item['sha256']}
    with pytest.raises(ValueError,match='Conflicting'):
        verify_annotation_inputs({**binding,'inputs':[item,{**item,'sha256':'different'}]})
    path.write_text('[{}]')
    with pytest.raises(ValueError,match='changed'):
        verify_annotation_inputs(binding)
    assert verify_annotation_inputs(None)=={}


def test_capture_measurements_bind_observed_runtime_and_storage():
    from measurement_study.campaign_v3_capture import capture_measurements
    import pytest
    records=[{'id':'short','tokens':100,'seconds':4.,'files':[{'bytes':1000},{'bytes':1000}]},
             {'id':'long','tokens':1000,'seconds':80.,'files':[{'bytes':10000},{'bytes':3000}]}]
    value=capture_measurements(records,120.)
    assert value['p90_seconds_per_token']==.08
    assert value['bytes_per_token']==20.
    assert value['startup_seconds']==120.
    # The fitted fixed-plus-per-token model must cover both measured traces.
    assert value['runtime_model']['startup_seconds']==120.
    assert value['seconds_per_token']==value['runtime_model']['seconds_per_token']
    assert value['per_attempt_seconds']==value['runtime_model']['per_attempt_seconds']>=0
    for row in value['attempts']:
        # attempt seconds exclude the 120 s startup, so the per-attempt model alone covers them
        assert value['per_attempt_seconds']+value['seconds_per_token']*row['tokens']>=row['seconds']*(1-1e-9)
    for seconds in (float('nan'),float('inf'),0.,-1.):
        with pytest.raises(ValueError):capture_measurements(records,seconds)
    with pytest.raises(ValueError):capture_measurements([{**records[0],'seconds':float('nan')}],120.)


def test_capture_manifest_rejects_outcome_binding_and_checks_cohort(tmp_path):
    import pytest
    from measurement_study.campaign_v3_plan import load_cohort_manifest
    from measurement_study.campaign_v3_capture import load_manifest
    from measurement_study.production_plan import digest, sha
    import json
    from pathlib import Path
    core=dict(schema_version=1,model='gpt',seed=1,label_total=10,per_b=2,floor=1,traces_sha256='e'*64)
    cohort=dict(core,manifest_sha256=digest(core),corpus='card-v3',rows=[])
    cohort_path=tmp_path/'cohorts.json';cohort_path.write_text(json.dumps(cohort))
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'SHA256.json').write_text('{}')
    manifest=dict(spec={'capture_version':2},code_binding=sha(bundle/'SHA256.json'),attempts=[],
                  cohort_binding=load_cohort_manifest(cohort_path)[1],cohort_manifest_path=str(cohort_path))
    manifest['binding']=digest(manifest)
    path=tmp_path/'0.json';path.write_text(json.dumps(manifest))
    assert load_manifest(path,bundle)['attempts']==[]
    cohort_path.write_text(cohort_path.read_text()+' ')
    with pytest.raises(ValueError,match='changed since planning'):
        load_manifest(path,bundle)
    cohort_path.write_text(json.dumps(cohort))
    manifest['outcome_binding']={'sha256':'legacy'}
    manifest['binding']=digest({k:v for k,v in manifest.items() if k!='binding'})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='must not bind outcomes'):
        load_manifest(path,bundle)


def test_hf_reference_is_handed_over_on_cpu_only():
    import torch
    from measurement_study.campaign_v3_capture import _to_cpu, _hf_reference_child
    value = {'router': torch.ones(2, 3), 'nested': [torch.zeros(1), {'ids': torch.arange(3)}],
             'audit': {'native_dispatch': True}, 'semantics': 'x'}
    moved = _to_cpu(value)
    assert moved['audit'] == {'native_dispatch': True} and moved['semantics'] == 'x'
    assert all(t.device.type == 'cpu' for t in (moved['router'], moved['nested'][0], moved['nested'][1]['ids']))
    assert torch.equal(moved['nested'][1]['ids'], torch.arange(3))
    # the child is a module-level function so a spawn context can pickle it by reference
    assert _hf_reference_child.__module__ == 'measurement_study.campaign_v3_capture'
