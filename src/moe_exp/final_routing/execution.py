"""Allocated scientific workers; no controller process performs inference."""
from pathlib import Path
import time
from .common import MODELS, Units, digest, file_sha, load, publish, require, verified


def task_rows(card):
    path=Path(card['task_file'])
    require(file_sha(path)==card['task_file_sha256'],'bound worker task file changed')
    return verified(path)['records']


def generate(binding,card,deadline):
    from .runtime import Backend,recover_outputs
    from .contexts import build
    from .reservations import check_storage
    root,bulk=Path(binding['root']),Path(binding['config']['bulk_root'])
    admission=verified(root/'ADMISSION.json')
    require(admission['status']=='admitted' and admission['lfm_full_obligation_reserved'],'complete LFM obligation is not reserved')
    model=card['model'];native=card['stage']=='lfm_native'
    if not native:
        require(verified(root/'DESIGN.json')['mode']==card['mode'],'primary design changed')
    price=verified(root/model/'QUALIFIED.json')
    require(price['profile_sha256']==binding['models'][model]['profile_sha256'],'uncovered generation profile')
    namespace='native' if native else 'executions';units=Units(bulk/model/namespace)
    rows=task_rows(card);recover_outputs(units,rows)
    pending=[r for r in rows if units.state(r['uid']) is None]
    if pending:
        from transformers import AutoTokenizer
        from .runtime import prompt_prefix,remaining
        tokenizer=AutoTokenizer.from_pretrained(binding['models'][model]['local_path'],local_files_only=True)
        for row in pending:
            prompt,prefix,_=prompt_prefix(binding,model,row,tokenizer)
            remaining(binding['models'][model],prompt,prefix)
        check_storage(binding)
        backend=Backend(binding,model,price['capacity'],deadline)
        try:
            for start in range(0,len(pending),price['capacity']):
                batch=pending[start:start+price['capacity']]
                reserve=price['seconds_per_token_upper']*binding['models'][model]['native_context_limit']*len(batch)
                reserve+=price['full_context_prefill_seconds_upper']
                require(time.time()+reserve+60<deadline,'full native allowance does not fit remaining allocation; untouched assignments retained')
                check_storage(binding)
                backend.run(batch,units)
        finally:
            backend.close()
    if not native:
        build(binding,model,rows)
    states=[units.state(r['uid']) for r in rows]
    require(all(s is not None and s['status']!='output' for s in states),'unprocessed worker assignments remain')
    return {'status':'complete','assigned':len(rows),'known_outputs':sum(s['status']=='complete' for s in states),
            'unknown':sum(s['status']=='unknown' for s in states),'stochastic_redraws':0}


def annotation_inventory(binding,phase):
    """Verify every input profile on CPUs before any annotation GPU is loaded."""
    from transformers import AutoTokenizer
    from .renderer import Renderer
    from moe_exp.research_campaign_v3_r6.reader_contract import token_ids
    root,bulk=Path(binding['root']),Path(binding['config']['bulk_root'])
    target=root/('ANNOTATION_INVENTORY-'+phase+'.json')
    if target.exists():return verified(target)
    parent=load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    price=verified(root/'READER_PRICE.json')['profiles']['covered']
    if phase=='lfm':
        rows=verified(root/'lfm25/ANNOTATION_TASKS.json')['records']
    else:
        mode=verified(root/'DESIGN.json')['mode'];rows={}
        for model in MODELS:
            assignments=verified(root/model/('ASSIGNMENTS-'+mode+'.json'))['records']
            for row in assignments:
                p=bulk/model/'contexts'/(row['uid']+'.json')
                if p.exists():
                    for task in verified(p)['tasks']:rows[task['uid']]=task
        rows=list(rows.values())
    units=Units(bulk/'reader/annotations')
    rows=sorted([r for r in rows if units.state(r['uid']) is None],key=lambda r:r['uid'])
    renderer=Renderer(parent);tokenizer=AutoTokenizer.from_pretrained(parent['annotator']['path'],local_files_only=True)
    covered=[];uncovered=[];started=time.monotonic()
    try:
        for row in rows:
            conversation=renderer.request('render',inputs=row['inputs'])['conversation']
            n=len(token_ids(tokenizer,conversation,parent['annotator']['config']))
            if not price['input_min']<=n<=price['input_max'] or n+16384>parent['annotator']['native_context']:
                uncovered.append({'uid':row['uid'],'input_tokens':n})
            else:covered.append({**row,'input_tokens':n})
    finally:renderer.close()
    path=bulk/('ANNOTATION_TASKS-'+phase+'.json')
    publish(path,{'records':covered,'uncovered':uncovered,'reader_price_sha256':verified(root/'READER_PRICE.json')['sha256']})
    return publish(target,{'status':'complete' if not uncovered else 'resource_limited','phase':phase,
        'task_file':str(path),'task_file_sha256':file_sha(path),'contexts':len(covered),'uncovered':uncovered,
        'seconds':time.monotonic()-started,'unpriced_executions_released':False})


def annotate(binding,card,deadline):
    from .reader import Reader
    from .reservations import check_storage
    root=Path(binding['root']);proof=verified(root/('ANNOTATION_INVENTORY-'+card['annotation_phase']+'.json'))
    require(proof['status']=='complete','annotation input coverage failed before model loading')
    price=verified(root/'READER_PRICE.json')['profiles']['covered']
    rows=task_rows(card)
    require(all(price['input_min']<=r['input_tokens']<=price['input_max'] for r in rows),'uncovered reader profile')
    check_storage(binding)
    reader=Reader(binding,deadline)
    try:
        reader.run(rows,input_bin='covered',admission=price)
        require(all(reader.units.state(r['uid']) is not None for r in rows),'untouched annotation assignments remain')
        return {'status':'complete','contexts':len(rows),'unknown':sum(reader.units.state(r['uid'])['status']=='unknown' for r in rows)}
    finally:reader.close()


def handler(binding,card,deadline):
    stage=card['stage'];root=Path(binding['root'])
    if stage=='validate':
        from .validation import validate
        return validate(binding,allocated=True)
    if stage=='prepare':
        from .checkpoint import verify
        from .importer import prepare
        from .lfm import population
        from .inherited import inventory
        from .common import check_files
        parent=load(Path(binding['source_campaign'])/'CAMPAIGN.json')
        check_files(parent['code_files']);check_files(parent['source_files'])
        if not (root/'LFM_CHECKPOINT_VERIFIED.json').exists():verify(binding['config'])
        detail=prepare(binding,deadline);population(binding)
        if not (root/'INHERITED_TASKS.json').exists():inventory(binding)
        return detail
    if stage=='cpu_price':
        from .cpu_price import measure
        return measure(binding,deadline)
    if stage=='legacy_price':
        from .inherited import price
        return price(binding,deadline)
    if stage=='qualify':
        from .qualification import qualify
        return qualify(binding,card['model'],deadline,card['attempt'])
    if stage=='reader_select':
        from .pilot import select
        return {'status':'complete','selection':select(binding)}
    if stage=='reader_pilot':
        from .pilot import run
        return run(binding,deadline)
    if stage=='admit':
        from .admission import admit
        value=admit(binding)
        return {'status':'complete' if value['status']=='admitted' else 'resource_limited','admission':value}
    if stage in ('generate','lfm_native'):return generate(binding,card,deadline)
    if stage=='lfm_materialize':
        from .lfm import materialize
        return materialize(binding)
    if stage=='lfm_capture':
        from .lfm import replay
        return replay(binding,task_rows(card),deadline)
    if stage=='lfm_derive':
        from .lfm import derive
        from .importer import derive_model
        if not (root/'lfm25/DISCOVERY_INPUTS.json').exists():derive(binding)
        derive_model(binding,'lfm25')
        return {'status':'complete'}
    if stage=='lfm_readout':
        from .readouts import lfm
        return lfm(binding)
    if stage=='annotation_inventory':return annotation_inventory(binding,card['annotation_phase'])
    if stage=='annotate':return annotate(binding,card,deadline)
    if stage=='legacy':
        from .inherited import execute
        return execute(binding,card['model'],card['pair_indices'],deadline)
    if stage=='freeze_primary':
        from .admission import freeze_primary
        return freeze_primary(binding)
    if stage=='analyze':
        from .delivery import analyze
        return analyze(binding)
    if stage=='export':
        from .export import export
        design=root/'DESIGN.json'
        return {'status':'complete',**export(binding,verified(design)['mode'] if design.exists() else 'full')}
    if stage in ('advance','guard','finalize'):
        from .workflow import advance,guard,finalize
        return {'advance':advance,'guard':guard,'finalize':finalize}[stage](binding,card,deadline)
    raise ValueError('unregistered allocated stage: '+stage)
