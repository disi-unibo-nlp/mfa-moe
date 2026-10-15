"""Bounded development contracts and a real-tokenizer CPU integration fixture."""
from pathlib import Path
import io
import os
import tempfile
import time
import unittest
from .common import REPO, WORK, Units, digest, file_sha, load, publish, require


def fixture(binding):
    from transformers import AutoTokenizer
    from .renderer import Renderer
    from .lfm import trace_record
    from .contexts import build,measured
    from .design import allocate
    from .export import export
    with tempfile.TemporaryDirectory(prefix='final-routing-fixture-',dir=WORK/'tools/tmp') as temporary:
        root=Path(temporary);bulk=root/'bulk'
        cfg={**binding['config'],'root':str(root),'bulk_root':str(bulk)}
        test={**binding,'root':str(root),'config':cfg}
        row={'model':'lfm25','benchmark':'math500','family':'engineering-fixture','fold':0,
            'native_example':{'dataset':'math500','problem_id':'cpu-fixture','source_problem_id':'cpu-fixture',
                'sample_id':0,'prompt':'What is 2 + 2?','gold_answer':'4'},
            'source':None,'seed':17,'arm':'native','class':None,'depth_third':None,'dose':'central',
            'action':None,'pulse_tokens':1,'horizon':64,'prefix_status':'enrolled','nomination_status':'nominated'}
        row['uid']=digest(row)
        tokenizer=AutoTokenizer.from_pretrained(binding['models']['lfm25']['local_path'],local_files_only=True)
        ids=tokenizer.encode('We compute the sum. Two plus two equals four. </think>\\boxed{4}',add_special_tokens=False)
        units=Units(bulk/'lfm25/executions');units.intent(row,'CPU-FIXTURE');units.admitted(row,'CPU-FIXTURE')
        raw=units.output(row,{'token_ids':ids,'finish_reason':'stop','status':'complete','engineering_only':True})
        trace=trace_record(test,row,raw,tokenizer)
        require(trace['metadata']['token_replay']['completion_token_ids']==ids,'fixture replay lost native token IDs')
        contexts=build(test,'lfm25',[row]);require(contexts['tasks'],'fixture produced no reader contexts')
        renderer=Renderer(load(Path(binding['source_campaign'])/'CAMPAIGN.json'))
        labels=Units(bulk/'reader/annotations')
        try:
            for task in contexts['tasks']:
                conversation=renderer.request('render',inputs=task['inputs'])['conversation']
                require(conversation and all('content' in m for m in conversation),'original renderer input contract failed')
                parsed=renderer.request('parse',content='[[ ## label ## ]]\nImplement\n[[ ## completed ## ]]')
                require(parsed.get('label')=='Implement','original parser fixture failed')
                labels.intent(task,'CPU-FIXTURE');labels.commit(task,{'status':'complete','label':parsed['label'],'engineering_only':True})
        finally:renderer.close()
        outcome=measured(test,row,contexts['records'][0])
        require(outcome['strict_correct'] is True,'fixture final-answer scoring failed')
        require(outcome['first']['Implement']==[1,1],'fixture first-complete-sentence endpoint failed')
        require(outcome['occupancy']['unknown']==0,'fixture closure left unknown occupancy')
        units.commit(row,{'status':'complete','raw_sha256':raw['sha256'],'engineering_only':True})
        prefix=load(Path(binding['source_campaign'])/'gpt/PREFIXES.json')['records']
        publish(root/'ALLOCATION.json',allocate(prefix,cfg['seed']))
        publish(root/'lfm25/ASSIGNMENTS-full.json',{'records':[row]})
        result=export(test,'full')
        require(not result['required_results_complete'],'one fixture cannot imply complete eight-model results')
        manifest=load(Path(result['path'])/'paper_results_manifest.json')
        require(all(Path(s['path']).is_file() for a in manifest['artifacts'] for s in a['sources']), 'fixture export has broken provenance')
        return {'status':'complete','scientific_draws':0,'real_local_tokenizer':True,'original_renderer_and_parser':True,
            'contexts':len(contexts['tasks']),'raw_to_export_passed':True,'partial_eight_model_completion_rejected':True}


def validate(binding,*,allocated=False):
    import importlib
    import subprocess
    from .config import sources
    from .common import check_files
    started=time.monotonic()
    source_files=sources(binding['config'])
    for path in (REPO/'src/moe_exp/final_routing').glob('*.py'):
        importlib.import_module('moe_exp.final_routing.'+path.stem)
    subprocess.run(['bash','-n',str(REPO/'scripts/slurm/final_routing.sbatch')],check=True)
    suite=unittest.TestLoader().discover(str(REPO/'tests/final_routing'),top_level_dir=str(REPO/'tests/final_routing'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    require(result.wasSuccessful(),stream.getvalue())
    evidence=fixture(binding)
    check_files(source_files)
    return {'status':'complete','allocated':allocated,'tests_run':result.testsRun,'test_output':stream.getvalue(),
            'fixture':evidence,'seconds':time.monotonic()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'source_files':source_files}
