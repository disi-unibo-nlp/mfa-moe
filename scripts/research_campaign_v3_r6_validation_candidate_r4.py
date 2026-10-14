#!/usr/bin/env python3
"""Single bound CPU recovery: migrate one obsolete local-as-remote test assertion."""
import argparse
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts/experimental_resume'))
__package__='moe_exp.research_campaign_v3_r6'
from moe_exp.research_campaign_v3_r6.validation import *
from moe_exp.research_campaign_v3_r6.contracts import check_files,load,publish

def validate(campaign):
    allocated();start=time.monotonic();root=Path(campaign['root'])
    check_files(campaign['code_files']);check_files(campaign['source_files'])
    import warnings
    warnings.filterwarnings('ignore',category=PendingDeprecationWarning)
    tests=unittest.defaultTestLoader.discover(str(REPO/'tests/research_campaign_v3_r6_validation_r2'))
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    require(result.wasSuccessful(),'r6 regression contracts failed')
    from .resume import import_evidence
    imported=import_evidence(campaign)
    from .controller import stage_key
    publish(root/'stages'/stage_key('resolve')/'RESULT.json',{'stage':'resolve','model':'all',
        'status':'complete','validated_actual_import':True,'evidence_import_sha256':imported['sha256']})
    generation_count=0;shape_count=0;rows=[]
    from .scoring import score
    from .annotation import prepare_contexts
    for model in MODELS:
        memory=load(root/model/'NON_REAP_TASKS.json')
        require(len({r['uid'] for r in memory['records']})==len(memory['records']),'execution aliases duplicated')
        generation_count+=sum(r['mask'] is not None for r in memory['records'])
        shapes=load(root/model/'NON_REAP_QUALIFICATION_TASKS.json')['records']
        shape_count+=sum(r['key']!='native' for r in shapes)
        sources={r['attempt_id']:r for r in load(root/model/'SOURCES.json')['records']}
        from .contracts import execution_id
        for row in memory['records']:
            original=sources[row['source']['attempt_id']]
            require(row['source']['trace_sha256']==original['trace_sha256'] and row['uid']==execution_id(row),
                'question/source/mask scientific identity join differs')
        # Exercise real strict scoring and context preparation on original native
        # continuations. No GPU, reader draw, repeat or external scheduler stub.
        native=load(root/model/'NATIVE_TASKS.json')['records']
        selected=max((r for r in native if r['source']['prefix_tokens']),
            key=lambda r:(r.get('prompt_tokens',0)+r['source']['prefix_tokens'],r['uid']))
        score(campaign,model,[selected])
        contexts=prepare_contexts(campaign,model,phase='native',selected_uids={selected['uid']})
        # The complete native-context bank remains explicit independent CPU work.
        # This replay exercises the real implementation on an original identity;
        # it never marks the unbuilt full bank or its downstream role complete.
        rows.append({'model':model,'non_reap_generations':sum(r['mask'] is not None for r in memory['records']),
            'compact_shapes':len(shapes)-1,'native_contexts':len(contexts['records']),
            'measure_manifest_sha256':load(root/model/'MEASURE_TASKS.json')['sha256']})
    require(generation_count==69459,'frozen non-REAP generation count differs from authorized plan')
    require(shape_count==346,'frozen non-REAP compact shape count differs from authorized plan')
    reader=validate_reader(campaign)
    full_export=validate_full_export(campaign)
    graph=traverse(campaign)
    proof=publish(root/'IMPORT_GRAPH_VALIDATION.json',{'status':'complete','models':rows,
        'non_reap_generations':generation_count,'compact_shapes':shape_count,
        'reader_validation_sha256':reader['sha256'],'full_export_validation_sha256':full_export['sha256'],
        'real_cpu_implementations_exercised':['import','masks','identities','strict_scoring','native_contexts',
            'installed_tokenizer','installed_input_validator','sampler','full_export','checkpoint_recovery','pdf_build','forecast','graph'],
        'external_boundaries':['scheduler submission prohibited','GPU absent'],'graph':graph,
        'seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID']})
    check_files(campaign['code_files'])
    return publish(root/'READINESS.json',{'passed':True,'tests_run':result.testsRun,
        'files':campaign['code_files'],'source_unchanged_during_tests':True,'actual_import_graph_validated':True,
        'actual_validation_sha256':proof['sha256'],'seconds':time.monotonic()-start,
        'allocated_cpus':4,'allocated_memory_gib':24,'allocation_minutes':180,'production_released':False})


def validate_reader(campaign):
    from collections.abc import Mapping
    from types import SimpleNamespace
    import json
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    from .annotation import Renderer
    from .reader_contract import token_ids
    from .contracts import digest,SEED
    root=Path(campaign['root']);path=root/'READER_INPUT_VALIDATION.json'
    if path.exists():return load(path)
    a=campaign['annotator'];tokenizer=AutoTokenizer.from_pretrained(a['path'],local_files_only=True)
    config=json.loads((Path(a['path'])/'config.json').read_text())
    config=config.get('text_config',config)
    validator=object.__new__(InputProcessor)
    validator.model_config=SimpleNamespace(max_model_len=a['native_context'],runner_type='generate',
                                          get_vocab_size=lambda:config['vocab_size'])
    validator.renderer=SimpleNamespace(tokenizer=SimpleNamespace(max_token_id=max(tokenizer.get_vocab().values())))
    validator.skip_prompt_length_check=False;validator.supports_mm_inputs=False;validator.mm_encoder_cache_size=0
    renderer=Renderer(campaign);records=[];reproduced=0
    try:
        for row in load(root/'ANNOTATOR_PRICE_TASKS.json')['records']:
            conversation=renderer.request('render',inputs=row['inputs'])['conversation']
            cfg=a['config']
            original=tokenizer.apply_chat_template(conversation,tokenize=True,add_generation_prompt=True,
                enable_thinking=cfg['enable_thinking'],preserve_thinking=cfg['preserve_thinking'],reasoning_effort=cfg['reasoning_effort'])
            require(isinstance(original,Mapping),'installed original reader mismatch was not reproduced')
            try:validator._validate_model_input({'type':'token','prompt_token_ids':original},'decoder')
            except TypeError:reproduced+=1
            else:raise ValueError('original dictionary input unexpectedly passed installed vLLM validation')
            prompt=token_ids(tokenizer,conversation,cfg)
            require(prompt==original['input_ids'],'corrected reader token sequence changed intended original content')
            eligible=len(prompt)<a['native_context']
            if eligible:
                validator._validate_model_input({'type':'token','prompt_token_ids':prompt},'decoder')
                SamplingParams(**a['decoding'],max_tokens=min(cfg['max_tokens'],a['native_context']-len(prompt)),
                    seed=int(digest([SEED,'reader',row['uid']])[:8],16),skip_special_tokens=False)
            records.append({'uid':row['uid'],'model':row['model'],'input_tokens':len(prompt),
                'prompt_sha256':digest(prompt),'context_eligible':eligible,'intended_content_equal':True})
    finally:renderer.close()
    require(reproduced==64,'real retained reader roster did not reproduce original failure')
    return publish(path,{'status':'complete','records':records,'original_failures_reproduced':reproduced,
        'installed_input_validator_exercised':True,'gpu_loaded':False,'scientific_draws':0,
        'runtime_files':campaign['reader_runtime_files']})

def main():
    import json
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=load(args.campaign/'CAMPAIGN.json')
    binding=load(args.campaign/'VALIDATION_CPU_FIXTURE_BINDING.json');check_files(binding['files'])
    check_files(campaign['code_files']);check_files(campaign['source_files'])
    require(binding['original_campaign_sha256']==campaign['sha256'],'validation recovery source binding differs')
    import time
    started=time.monotonic()
    from moe_exp.research_campaign_v3_r6.controller import stage_key
    directory=args.campaign/'stages'/stage_key('validate')
    publish(directory/'START-3.json',{'job_id':os.environ['SLURM_JOB_ID'],
        'recovery_binding_sha256':binding['sha256'],'original_failed_job':'60481682'})
    result=validate(campaign)
    publish(directory/'RESULT.json',{'stage':'validate','model':'all','shard_index':0,'threshold_index':0,
        'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'seconds':time.monotonic()-started,
        'detail':{'readiness_sha256':result['sha256'],'recovery_binding_sha256':binding['sha256']},
        'failure_scope':None})
    proof=publish(args.campaign/'VALIDATION_CPU_FIXTURE_COMPLETED.json',{
        'readiness_sha256':result['sha256'],'recovery_binding_sha256':binding['sha256'],
        'job_id':os.environ['SLURM_JOB_ID'],'original_failed_job_preserved':'60481682',
        'scientific_implementation_unchanged':True,'stochastic_draws':0})
    print(json.dumps(proof,sort_keys=True))

if __name__=='__main__':main()
