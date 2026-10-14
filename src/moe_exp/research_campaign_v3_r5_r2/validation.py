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
    tests=unittest.defaultTestLoader.discover(str(REPO/'tests/research_campaign_v3_r5_r2'))
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    require(result.wasSuccessful(),'r5 regression contracts failed')
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
    graph=traverse(campaign)
    proof=publish(root/'IMPORT_GRAPH_VALIDATION.json',{'status':'complete','models':rows,
        'non_reap_generations':generation_count,'compact_shapes':shape_count,
        'real_cpu_implementations_exercised':['import','masks','identities','strict_scoring','native_contexts','forecast','graph'],
        'external_boundaries':['scheduler submission prohibited','GPU absent'],'graph':graph,
        'seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID']})
    check_files(campaign['code_files'])
    return publish(root/'READINESS.json',{'passed':True,'tests_run':result.testsRun,
        'files':campaign['code_files'],'source_unchanged_during_tests':True,'actual_import_graph_validated':True,
        'actual_validation_sha256':proof['sha256'],'seconds':time.monotonic()-start,
        'allocated_cpus':4,'allocated_memory_gib':24,'allocation_minutes':30,'production_released':False})
