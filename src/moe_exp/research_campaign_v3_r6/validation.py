"""Replay the actual seven-model import and CPU graph without submitting work."""
from pathlib import Path
import os
import time
import unittest
from .contracts import MODELS, REPO, Units, allocated, check_files, file_sha, load, publish, require


class NoSubmit:
    def __init__(self,snapshot):self.value=snapshot
    def snapshot(self):return self.value
    def submit(self,*a,**k):raise AssertionError('validation attempted a scheduler mutation')
    def release(self,*a,**k):raise AssertionError('validation attempted release')
    def command(self,args):
        require(args[0] in ('saldo','sacct','squeue','sacctmgr','lfs') or args[:2]==['scontrol','show'],
            'graph validation attempted a scheduler mutation')
        from .scheduler import Slurm
        return Slurm().command(args)
    def observe(self,job):
        from .scheduler import Slurm
        return Slurm().observe(job)
    def observe_many(self,jobs):
        from .scheduler import Slurm
        return Slurm().observe_many(jobs)


def traverse(campaign):
    from .controller import _measurement_forecast, ready
    from .scheduler import Slurm
    root=Path(campaign['root'])
    nodes=load(root/'GRAPH.json')['nodes'];names={n['stage'] for n in nodes}
    require(all(set(n['dependencies'])<=names for n in nodes),'registered graph has a dangling dependency')
    require((root/'EVIDENCE_IMPORT.json').exists(),'import must validate before graph traversal')
    for model in MODELS:
        require(len(load(root/model/'MEASURE_TASKS.json')['records'])==1547,'measurement boundary is incomplete')
        require((root/model/'NON_REAP_TASKS.json').exists(),'independent memory interface is missing')
    hours,bytes_,missing=_measurement_forecast(campaign)
    snapshot=Slurm().snapshot()
    from .controller import jobs
    scheduler=NoSubmit(snapshot)
    work=ready(campaign,scheduler,jobs(campaign,scheduler))
    return {'status':'complete','scheduler_submissions':0,'registered_graph_sha256':load(root/'GRAPH.json')['sha256'],
        'saved_sequence_projection_local_hours':hours,'saved_sequence_projection_bytes':bytes_,
        'unpriced_saved_sequence_models':missing,
        'runnable_transitions':[{'stage':c['stage'],'model':c['model'],'shard_index':c['shard_index']} for c in work],
        'live_resources':snapshot}


def validate(campaign):
    allocated();start=time.monotonic();root=Path(campaign['root'])
    check_files(campaign['code_files']);check_files(campaign['source_files'])
    import warnings
    warnings.filterwarnings('ignore',category=PendingDeprecationWarning)
    tests=unittest.defaultTestLoader.discover(str(REPO/'tests/research_campaign_v3_r6'))
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
    validator.tokenizer=SimpleNamespace(max_token_id=max(tokenizer.get_vocab().values()))
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


def validate_full_export(campaign):
    """Use real imported receipts, isolated output, and a no-publication PDF build."""
    import json,subprocess
    from .contracts import STAGING
    root=Path(campaign['root']);path=root/'FULL_EXPORT_VALIDATION.json'
    if path.exists():return load(path)
    isolated=root/'validation/full';isolated.mkdir(parents=True,exist_ok=True)
    for name in (*MODELS,'COHORTS.json','FOLDS.json','MEASUREMENT_COMPONENTS.json','READER_RECOVERY_ADJUDICATION.json'):
        source=root/name;target=isolated/name
        if source.exists() and not target.exists():target.symlink_to(source.resolve(),target_is_directory=source.is_dir())
    from .export import closeout
    value={**campaign,'root':str(isolated),'validation_only':True}
    result=closeout(value)
    report=load(isolated/'output/REPORT.json')
    require(result['REPORT_COMPLETE'] and report['cells']==243747 and not report['SCIENCE_COMPLETE'],
            'actual full-size validation export changed coverage or claimed scientific completion')
    before=file_sha(isolated/'output/EXPORT_STATE.json')
    recovery=closeout(value)
    require(recovery==result and before==file_sha(isolated/'output/EXPORT_STATE.json'),
            'committed deterministic export recovery changed scientific evidence')
    # Compilation snapshots the real canonical manuscript; no live paper pointer
    # or scientific receipt is changed by this validation build.
    build=STAGING/('research-r6-validation-'+campaign['sha256'][:16])
    if not (build/'BUILD.json').exists():
        completed=subprocess.run([os.sys.executable,'-B',str(REPO/'scripts/paper/build_campaign_paper_r3.py'),
            '--only-cached','--output',str(build)],capture_output=True,text=True,timeout=60)
        require(completed.returncode==0,'validation manuscript build failed: '+completed.stderr[-2000:])
    built=json.loads((build/'BUILD.json').read_text())
    require(built['state']=='COMPILED_FULL_CLASS' and not built['changed_sources'] and
            file_sha(Path(built['pdf_path']))==built['pdf_sha256'],'validation PDF source binding differs')
    return publish(path,{'status':'complete','cells':report['cells'],'report_sha256':report['sha256'],
        'export_state_sha256':before,'full_size_recovery_verified':True,'build_receipt':str(build/'BUILD.json'),
        'pdf_sha256':built['pdf_sha256'],'isolated_scientific_outputs':str(isolated/'output'),
        'accepted_scientific_journals_not_simulated':True,'publication_attempts':0})
