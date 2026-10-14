"""Outcome-blind memory preparation independent of REAP and steering calibration."""
from pathlib import Path
import numpy as np
from .contracts import Units, check_files, digest, file_sha, load, publish, require


def freeze(campaign, model):
    from .tasks import freeze_masks, task_table
    root=Path(campaign['root']); contract=campaign['models'][model]
    path=root/model/'NON_REAP_TASKS.json'
    if path.exists():return load(path)
    selection=publish(root/model/'NON_REAP_SELECTION.json', {
        'algorithm_source_sha256':file_sha(Path(__file__).with_name('design.py')),
        'scores_sha256':file_sha(root/model/'SCORES.npz'),
        'sources_sha256':load(root/model/'SOURCES.json')['sha256'],
        'cohorts_sha256':load(root/'COHORTS.json')['sha256'],
        'discovery_outcomes_read':False,'evaluation_outcomes_read':False})
    freeze_masks(campaign,model,np.full((5,len(contract['layers']),contract['experts']),np.nan),component='non_reap')
    table=task_table(campaign,model,[],component='non_reap')
    from .workers import qualification_rows, price_rows
    qualification_rows(campaign,model,'non_reap'); price_rows(campaign,model,'non_reap')
    return table


def price(campaign, model, deadline, shard):
    from .workers import price as retained_price, price_rows
    root=Path(campaign['root'])
    result=retained_price(campaign,model,deadline,shard,component='non_reap')
    if shard:
        row=price_rows(campaign,model,'non_reap')['records'][shard-1]['row']
        from .scoring import score
        score(campaign,model,[row])
        execution=Units(root/model/'executions').state(row['uid'])
        scored=Units(root/model/'scores').state(row['uid'])
        require(scored and scored.get('status')=='complete','retained memory assignment lacks strict score')
        from .export import _resident
        memory=_resident(execution,root,model,campaign=campaign)
        publish(root/model/'memory_price_exports'/(row['uid']+'.json'),{
            'status':'complete','execution_uid':row['uid'],'original_assignment':row,
            'price_sha256':result['sha256'],'strict_score':scored,'resident_memory':memory,
            'generation_repeated':False,'unknown_outcomes_retained':True})
    return result


def ready(campaign, observed, want):
    from .controller import _success, _stage_state, _mark
    from .workers import qualification_rows, price_rows
    root=Path(campaign['root'])
    for model in campaign['models']:
        if not _success(root,'derive',model) or not _success(root,'native',model):continue
        if not (root/model/'NON_REAP_TASKS.json').exists():
            want('memory_prepare',model,minutes=180);continue
        shapes=qualification_rows(campaign,model,'non_reap')['records']
        prices={p['role']:p for p in price_rows(campaign,model,'non_reap')['records']}
        for shape in shapes:
            index=shape['shard_index']; state=_stage_state(root,'memory_qualify',model,index)
            if state is None:
                want('memory_qualify',model,index,minutes=120);break
            if state['status'] not in ('complete','reused'):
                if shape['key']=='native':
                    if not _stage_state(root,'memory_qualify',model):
                        _mark(root,'memory_qualify',model,'upstream_unavailable','native physical qualification failed')
                    break
                continue
            task=prices.get('memory:'+shape['key'])
            if task and _stage_state(root,'memory_price',model,task['shard_index']) is None:
                want('memory_price',model,task['shard_index'],minutes=1440);break
        else:
            want('memory_qualify',model,0,minutes=30)
            if _success(root,'memory_qualify',model):want('memory_price',model,0,minutes=30)
