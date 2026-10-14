"""Retained CPU role pricing, restartable pair shards and serial cache lifetime."""
from pathlib import Path
import math
import time
import resource
from .contracts import load,publish,require,MODELS


def storage_price(campaign, model, pair):
    """Price the complete retained pair, including inference and sufficient data."""
    root = Path(campaign['root'])
    directory = root/model/'coordination'/f'pair-{pair:03d}'
    receipt = load(directory/'RESULT.json')
    require(receipt['status'] == 'complete', 'cannot price an incomplete analysis pair')
    paths = [p for p in directory.iterdir() if p.is_file() and p.name != 'STORAGE_PRICE.json']
    return publish(directory/'STORAGE_PRICE.json', {
        'status': 'complete', 'pair_output_bytes': sum(p.stat().st_size for p in paths),
        'result_sha256': receipt['sha256'], 'pair_index': pair,
        'includes_family_sufficient_statistics': True, 'includes_inference_and_nominations': True})


def pair_storage_bound(campaign, model):
    """Measured upper forecast; an unpriced model gets one retained pilot only."""
    root = Path(campaign['root'])
    measured = [load(p)['pair_output_bytes'] for p in (root/model/'coordination').glob('pair-*/STORAGE_PRICE.json')]
    imported = root/model/'ANALYSIS_STORAGE_PRICE.json'
    if imported.exists():
        measured.append(load(imported)['maximum_pair_output_bytes'])
    if not measured:
        return None
    return math.ceil(max(measured)*1.25)


def pilot(campaign,model):
    from .derive import derive
    from .extraction import extract
    from .observations import supplementary,coordination
    from .controller import stage_key
    root=Path(campaign['root']);started=time.monotonic()
    if (root/model/'ANALYSIS_PRICE.json').exists():return load(root/model/'DERIVED.json')
    result=load(root/model/"DERIVED.json") if (root/model/"DERIVED.json").exists() else derive(campaign,model)
    extraction=extract(campaign,model)
    supplementary(campaign,model)
    metrics={};pair=coordination(campaign,model,1,metrics=metrics);seconds=metrics['coordination_seconds']
    storage=storage_price(campaign,model,1)
    publish(root/'stages'/stage_key('coordination',model,1)/'RESULT.json',{
        'stage':'coordination','model':model,'shard_index':1,'status':'complete','seconds':seconds,
        'detail':pair,'retained_in_analysis_pilot':True})
    price=publish(root/model/'ANALYSIS_PRICE.json',{'status':'complete','pair_seconds':seconds,
        'extraction_seconds':extraction['seconds'],'seconds':time.monotonic()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'output_bytes':extraction['output_bytes'],'pair_output_bytes':storage['pair_output_bytes'],
        'storage_price_sha256':storage['sha256'],'compressed_input_bytes':extraction['compressed_input_bytes'],
        'extraction_bytes_per_second':extraction['compressed_input_bytes']/max(.001,extraction['seconds']),
        'worker_cpus':4,**metrics,'first_original_pair_retained':True,'bootstrap_draws':10000})
    return {**result,'analysis_price_sha256':price['sha256']}


def plan(campaign,model):
    from .observations import pair_tasks
    root=Path(campaign['root']);path=root/model/'COORDINATION_SHARDS.json'
    if path.exists():return load(path)
    price=load(root/model/'ANALYSIS_PRICE.json')
    require(price['peak_rss_kib']<=24*1024**2,'analysis peak RSS exceeds initial resource envelope')
    size=max(1,int((170*60)/(max(1,price['pair_seconds'])*1.25)))
    registered = pair_tasks(campaign, model)
    tasks=[r['shard_index'] for r in registered if r['shard_index']!=1 and
           not (root/model/'coordination'/f"pair-{r['shard_index']:03d}"/'RESULT.json').exists()]
    return publish(path,{'shards':[[1]]+[tasks[i:i+size] for i in range(0,len(tasks),size)],
        'seconds_per_pair_upper':max(1,price['pair_seconds'])*1.25,'price_sha256':price['sha256'],
        'peak_storage_bytes':price['output_bytes'],'all_pairs':len(registered),
        'missing_pairs':tasks,'previous_pairs_retained':len(registered)-len(tasks)})


def finalize(campaign, model):
    from .contracts import check_files, file_sha
    from .observations import pair_tasks
    root = Path(campaign['root']); inherited = root/model/'INHERITED_CACHE.json'
    if not inherited.exists():
        from .extraction import retire
        return retire(campaign, model)
    proof = load(inherited); dependencies = []
    require((root/model/'compact').resolve() == Path(proof['path']), 'inherited cache moved')
    for task in pair_tasks(campaign, model):
        path = root/model/'coordination'/f"pair-{task['shard_index']:03d}"/'RESULT.json'
        result = load(path)
        require(result['status'] == 'complete', 'inherited cache still has unfinished dependent pairs')
        check_files({a['path']:a['sha256'] for a in result.get('artifacts', [])})
        dependencies.append({'path':str(path), 'sha256':file_sha(path)})
    require(load(root/model/'SUPPLEMENTARY.json')['status'] == 'complete', 'missing supplementary output')
    return publish(root/model/'CACHE_RETIRED.json', {'status':'reused',
        'child_temporary_cache_never_allocated':True, 'ancestor_cache_preserved':True,
        'dependencies':dependencies, 'inherited_cache_sha256':proof['sha256']})


def shard(campaign,model,index,deadline):
    from .observations import coordination
    root=Path(campaign['root']);indices=load(root/model/'COORDINATION_SHARDS.json')['shards'][index-1]
    for pair in indices:
        path=root/model/'coordination'/f'pair-{pair:03d}'/'RESULT.json'
        if not path.exists():
            require(time.time()<deadline-60,'analysis shard deadline before pair')
            coordination(campaign,model,pair)
        storage_price(campaign,model,pair)
    return {'status':'complete','pair_indices':indices,'restartable':True}


def ready(campaign,scheduler,observed,want):
    """Only one model owns temporary extraction arrays at any instant."""
    from .controller import _success,_failure,_stage_state,_check,_mark,terminal,stage_key
    root=Path(campaign['root'])
    for model in MODELS:
        if (root/model/'CACHE_RETIRED.json').exists():continue
        if _failure(root,'derive',model):
            from .extraction import retire_failed
            retire_failed(campaign,model,_stage_state(root,'derive',model));continue
        if not _success(root,'mapping',model):
            want('mapping',model,minutes=180);return False
        if not _success(root,'derive',model):
            want('derive',model,minutes=180);return False
        if not (root/model/'ANALYSIS_PRICE.json').exists():
            if _failure(root,'analysis_pilot',model):
                from .extraction import retire_failed
                retire_failed(campaign,model,_stage_state(root,'analysis_pilot',model))
                continue
            want('analysis_pilot',model,minutes=180);return False
        table=plan(campaign,model)
        if not (root/model/'ANALYSIS_ADMISSION.json').exists():
            residual=sum(len(s) for s in table['shards'][1:])*table['seconds_per_pair_upper']*4/3600
            check=_check(campaign,scheduler,observed,hours=0,bytes_=0)
            publish(root/model/'ANALYSIS_ADMISSION.json',{'scope':'complete remaining original layer pairs for this model',**check,
                'runtime_projection_worker_hours':residual/4,'queue_uncertainty':'not estimable from execution timings'})
            if not check['admitted']:terminal(campaign,'resource_limited',check);return False
        if not load(root/model/'ANALYSIS_ADMISSION.json')['admitted']:return False
        for index in range(2,len(table['shards'])+1):
            want('coordination',model,index,minutes=180)
        if not all(_stage_state(root,'coordination',model,i) for i in range(1,len(table['shards'])+1)):return False
        if any(_failure(root,'coordination',model,i) for i in range(1,len(table['shards'])+1)):
            failures=[_stage_state(root,'coordination',model,i) for i in range(1,len(table['shards'])+1)
                if _failure(root,'coordination',model,i)]
            if all(f.get('failure_scope')=='model' for f in failures):
                from .extraction import retire_failed
                retire_failed(campaign,model,failures[0]);continue
            from .extraction import retire_failed
            retire_failed(campaign,model,failures[0]);continue
        want('analysis_finalize',model,minutes=30)
        return False
    return True
