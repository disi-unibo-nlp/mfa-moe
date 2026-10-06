"""The one-shot dispatcher must submit complete chains and never an empty GPU array."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/experimental_resume'))
import dispatch_overnight_stage_v3 as dispatch


def fixture(monkeypatch,workdir,phase,reader_shards=2):
    shared=dispatch.shared
    monkeypatch.setattr(shared,'DOC',workdir)
    monkeypatch.setattr(shared,'RUNS',workdir/'runs')
    monkeypatch.setenv('SLURM_JOB_ID','cpu-test')
    monkeypatch.setenv('SLURM_JOB_PARTITION','lrd_all_viz')
    monkeypatch.setenv('OVERNIGHT_DISPATCH_PHASE',phase)
    manifest={'sha256':'test-manifest-sha','max_wall_seconds':7200,
              'shards':[{'start_row':i,'end_row':i+1} for i in range(4)]}
    generation_price={'sha256':'generation-price-sha','manifest_sha256':manifest['sha256'],
                      'status':'PASS_COMPLETE_STAGE_GENERATION_ONLY','shards':manifest['shards']}
    envelope={'sha256':'envelope-sha','manifest_sha256':manifest['sha256'],
              'generation_price_sha256':generation_price['sha256'],
              'status':'PASS_COMPLETE_GENERATION_AND_SEMANTIC_ENVELOPE',
              'execution_files':{str(ROOT/'scripts/experimental_resume/overnight_routing_run_v3.sbatch'):'test-file-sha'},
              'semantic_measurement':{'measurement_gpu_hour_ceiling':20.}}
    frame={'sha256':'frame-sha','generation_manifest_sha256':manifest['sha256']}
    price={'sha256':'reader-price-sha','frame_sha256':frame['sha256'],
           'estimated_complete_gpu_hours':float(reader_shards),
           'shards':[{'start':i*8,'end':(i+1)*8} for i in range(reader_shards)]}
    values={'OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json':manifest,'OVERNIGHT_DISCOVERY_C_PRICE_v2.json':generation_price,
            'OVERNIGHT_DISCOVERY_C_ENVELOPE_v2.json':envelope,'BLIND_FRAME.json':frame,'READER_PRICE.json':price}
    monkeypatch.setattr(shared.base,'sealed',lambda path:values[Path(path).name])
    monkeypatch.setattr(shared.base,'file_sha',lambda path:'test-file-sha')
    monkeypatch.setattr(dispatch.subprocess,'run',lambda *a,**kw:None)
    validated=[]
    monkeypatch.setattr(dispatch.reader,'validate',lambda f,p:validated.append((f,p)))
    calls=[];saved={}
    def submit(directory,name,arguments,env,binding):
        identifier=str(9001+len(calls))
        calls.append({'name':name,'arguments':arguments,'env':dict(env),'binding':binding,'job':identifier})
        return identifier
    monkeypatch.setattr(shared,'submit',submit)
    monkeypatch.setattr(shared,'save',lambda path,body:saved.update({Path(path).name:body}))
    monkeypatch.setattr(sys,'argv',['dispatch','--stage','C','--phase',phase])
    return calls,saved,values,validated


def test_generation_submits_all_shards_then_frame_price_then_reader_dispatch(monkeypatch,workdir):
    calls,saved,_,_=fixture(monkeypatch,workdir,'generation')
    dispatch.main()
    assert [x['name'] for x in calls]==['generation','frame-price','reader-dispatch']
    assert '--array=0-3' in calls[0]['arguments'] and '--time=02:00:00' in calls[0]['arguments']
    assert '--dependency=afterok:9001' in calls[1]['arguments']
    assert '--dependency=afterok:9002' in calls[2]['arguments']
    assert calls[2]['env']['OVERNIGHT_DISPATCH_PHASE']=='readers'
    assert calls[0]['env']['MANIFEST']==calls[0]['env']['OVERNIGHT_MANIFEST']
    assert saved['GENERATION_CHAIN.json']['generation_job']=='9001'


def test_empty_gradeable_population_queues_only_ITT_analysis(monkeypatch,workdir):
    calls,saved,_,validated=fixture(monkeypatch,workdir,'readers',reader_shards=0)
    dispatch.main()
    assert len(validated)==1
    assert [x['name'] for x in calls]==['analysis']
    assert not any(x.startswith('--array=') or x.startswith('--dependency=') for x in calls[0]['arguments'])
    assert saved['MEASUREMENT_CHAIN.json']['reader_job'] is None


def test_reader_array_dependency_and_complete_price_failure(monkeypatch,workdir):
    calls,saved,values,_=fixture(monkeypatch,workdir,'readers',reader_shards=2)
    dispatch.main()
    assert [x['name'] for x in calls]==['reader-array','analysis']
    assert '--array=0-1' in calls[0]['arguments']
    assert '--dependency=afterok:9001' in calls[1]['arguments']
    values['READER_PRICE.json']['estimated_complete_gpu_hours']=21.
    calls.clear()
    with pytest.raises(ValueError,match='price/envelope'):
        dispatch.main()
    assert not calls


def test_rebound_generation_manifest_price_fails_before_submission(monkeypatch,workdir):
    calls,_,values,_=fixture(monkeypatch,workdir,'generation')
    values['OVERNIGHT_DISCOVERY_C_PRICE_v2.json']['manifest_sha256']='other-manifest'
    with pytest.raises(ValueError,match='generation price/envelope'):
        dispatch.main()
    assert not calls


@pytest.mark.parametrize("drift",["shards", "empty_shards", "execution_hash", "missing_execution_files"])
def test_generation_fails_closed_on_unpriced_or_changed_execution(monkeypatch,workdir,drift):
    calls,_,values,_=fixture(monkeypatch,workdir,'generation')
    if drift == 'shards':
        values['OVERNIGHT_DISCOVERY_C_PRICE_v2.json']['shards']=[{'start_row':0,'end_row':2}]
    elif drift == 'empty_shards':
        values['OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json']['shards']=[]
        values['OVERNIGHT_DISCOVERY_C_PRICE_v2.json']['shards']=[]
    elif drift == 'execution_hash':
        values['OVERNIGHT_DISCOVERY_C_ENVELOPE_v2.json']['execution_files']['runner']='other-hash'
    else:
        values['OVERNIGHT_DISCOVERY_C_ENVELOPE_v2.json'].pop('execution_files')
    with pytest.raises(ValueError,match='generation price/envelope'):
        dispatch.main()
    assert not calls
