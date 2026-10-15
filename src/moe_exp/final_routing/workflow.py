"""Finite protected Slurm graph. Every edge is afterany plus receipt checks."""
from pathlib import Path
import os
from .common import MODELS, REPO, digest, file_sha, load, locked, publish, require, verified
from .scheduler import Slurm,card,command,job_fields
from .journal import result


def payload(value):return {k:v for k,v in value.items() if k!='sha256'}


def node(config,stage,model='all',*,depends=(),phase='production',**kw):
    return card(config,stage,model,required_cards=list(depends),depends_on=list(depends),budget_phase=phase,**kw)


def pair(config,stage,model='all',*,depends=(),phase='production',recovery_minutes=None,**kw):
    first=node(config,stage,model,depends=depends,phase=phase,**kw)
    extra={**kw}
    if recovery_minutes is not None:extra['minutes']=recovery_minutes
    second=node(config,stage,model,depends=depends,phase=phase,attempt=1,
                recovery_of=digest(first),**extra)
    second['depends_on'].append(digest(first))
    # The job name remains a deterministic prefix; the complete immutable
    # card, including the recovery edge, is the journal identity.
    return [first,second]


def initial_graph(config,coordination=()):
    rows=[]
    validate=node(config,'validate',phase='preparation_cpu',minutes=15)
    rows.append(validate)
    prep=pair(config,'prepare',depends=[digest(validate)],phase='preparation_cpu',minutes=60,recovery_minutes=30)
    rows+=prep
    cpu=pair(config,'cpu_price',depends=[digest(validate)],phase='preparation_cpu',minutes=120,recovery_minutes=60)
    rows+=cpu
    legacy=pair(config,'legacy_price',depends=[digest(validate)],phase='preparation_cpu',minutes=120,recovery_minutes=30,
                external_dependencies=list(coordination))
    rows+=legacy
    qualification=[]
    for model in MODELS:
        jobs=pair(config,'qualify',model,depends=[digest(prep[-1])],phase='preparation_qualification')
        rows+=jobs;qualification.append(digest(jobs[-1]))
    select=pair(config,'reader_select',depends=qualification,phase='preparation_cpu',minutes=15)
    rows+=select
    reader=pair(config,'reader_pilot',depends=[digest(select[-1])],phase='preparation_reader')
    rows+=reader
    admission=node(config,'admit',depends=[digest(reader[-1]),digest(cpu[-1]),digest(legacy[-1])],phase='preparation_cpu',minutes=15)
    rows.append(admission)
    cpu_minutes=sum(r['minutes'] for r in rows if r['budget_phase']=='preparation_cpu')
    require(cpu_minutes==480,'preparation CPU graph must fit eight aggregate wall-hours')
    return protect(config,rows,0,[digest(admission)],control_phase='finalization')


def protect(config,workers,wave,terminals,control_phase='production'):
    controller=node(config,'advance',depends=terminals,phase=control_phase,minutes=5,wave=wave)
    guards=[]
    for attempt in (0,1):
        g=node(config,'guard',phase='finalization',minutes=10,wave=wave,attempt=attempt,
               controller_card=digest(controller),depends_on_guard=digest(guards[0]) if guards else None)
        g['depends_on']=[digest(controller)]+([digest(guards[0])] if guards else [])
        guards.append(g)
    return {'wave':wave,'cards':workers+[controller]+guards,'worker_cards':[digest(c) for c in workers],
            'terminal_cards':terminals,'controller':digest(controller),'guards':[digest(c) for c in guards]}


def accepted(root,identity):
    p=Path(root)/'submissions'/identity/'ACCEPTED.json'
    return verified(p) if p.exists() else None


def submit_wave(binding,plan,*,release=True):
    from .reservations import reserve_many
    root=Path(binding['root']);scheduler=Slurm(binding['config'])
    path=root/'waves'/('%02d'%plan['wave'])/'PLAN.json'
    if path.exists():require(payload(verified(path))==plan,'frozen wave plan changed')
    else:publish(path,plan)
    reserve_many(binding,plan['cards'])
    receipts={}
    for c in plan['cards']:
        deps=[]
        for identity in c.get('depends_on',[]):
            old=receipts.get(identity) or accepted(root,identity)
            require(old is not None,'dependency has not been accepted')
            deps.append(old['job_id'])
        deps+=c.get('external_dependencies',[])
        receipts[digest(c)]=scheduler.submit(c,deps,hold=True)
    launch=publish(path.parent/'ACCEPTED.json',{'wave':plan['wave'],
        'jobs':{identity:r['job_id'] for identity,r in receipts.items()},'plan_sha256':verified(path)['sha256'],
        'all_acceptances_verified':True})
    if release:release_wave(binding,plan,launch)
    return launch


def release_wave(binding,plan,launch):
    root=Path(binding['root']);path=root/'waves'/('%02d'%plan['wave'])/'RELEASED.json'
    if path.exists():return verified(path)
    by_id={digest(c):c for c in plan['cards']}
    # Failure owners are released before any scientific worker.
    order=plan['guards']+[plan['controller']]+[i for i in by_id if i not in plan['guards']+[plan['controller']]]
    scheduler=Slurm(binding['config']);observed={}
    for identity in order:
        jid=launch['jobs'][identity]
        raw=command(['scontrol','show','job','-o',jid]);fields=job_fields(raw)
        if fields['JobState']=='PENDING' and fields.get('Reason')=='JobHeldUser':command(['scontrol','release',jid])
        proof=scheduler.readback(jid,by_id[identity]);observed[identity]=proof
        require(proof['fields'].get('Reason')!='JobHeldUser','accepted worker remains held')
    return publish(path,{'wave':plan['wave'],'observed':observed,'owners_released_before_workers':True})


def planned_tasks(binding,wave,stage,model,groups,**extra):
    root=Path(binding['config']['bulk_root'])/'tasks'/('%02d'%wave)
    cards=[]
    for index,group in enumerate(groups):
        path=root/(stage+'-'+model+'-'+str(index)+'.json')
        publish(path,{'records':group['records'],'planned_seconds':group['planned_seconds']})
        cards+=pair(binding['config'],stage,model,shard=index,minutes=360,
            task_file=str(path),task_file_sha256=file_sha(path),planned_seconds=group['planned_seconds'],**extra)
    return cards


def parallel_lanes(binding,cards):
    admission=verified(Path(binding['root'])/'ADMISSION.json')
    design=Path(binding['root'])/'DESIGN.json'
    if design.exists():admission={**admission,**verified(design)}
    tails={'gpu':[],'cpu':[]};output=[]
    for i in range(0,len(cards),2):
        first,second=cards[i:i+2]
        kind='gpu' if first['gpus'] else 'cpu';limit=admission[kind+'_concurrency']
        require(limit>=1,'no admitted worker concurrency')
        lanes=tails[kind]
        if len(lanes)>=limit:
            previous=lanes.pop(0)
            first['depends_on'].append(previous)
            second['depends_on'].append(previous)
        # Scheduling-only lane edges deliberately do not become scientific
        # prerequisites: one independent failed assignment cannot cancel peers.
        second['recovery_of']=digest(first)
        # Replace the pre-lane first-attempt edge with its final identity.
        second['depends_on']=[digest(first)]+[d for d in first['depends_on']]
        output += [first,second];lanes.append(digest(second))
    return output


def production_plan(binding,wave):
    from .admission import pack,generation_cost
    root=Path(binding['root']);cfg=binding['config'];cards=[]
    if wave==1:
        price=verified(root/'lfm25/QUALIFIED.json');rows=verified(root/'lfm25/NATIVE_TASKS.json')['records']
        cards+=planned_tasks(binding,wave,'lfm_native','lfm25',pack(rows,lambda r:generation_cost(price),price['load_seconds'],capacity=price['capacity']))
        forecast=verified(root/'ADMISSION.json')['forecasts'][verified(root/'ADMISSION.json')['mode']]
        for r in forecast['legacy']:
            size=max(1,int(18000/r['pair_seconds']))
            for i in range(0,len(r['missing_pairs']),size):
                cards+=pair(cfg,'legacy',r['model'],shard=i//size,minutes=360,pair_indices=r['missing_pairs'][i:i+size])
    elif wave==2:cards+=pair(cfg,'lfm_materialize',minutes=360)
    elif wave==3:
        cards+=pair(cfg,'annotation_inventory',minutes=360,annotation_phase='lfm')
        rows=verified(root/'lfm25/SOURCES.json')['records'];p=verified(root/'lfm25/QUALIFIED.json')['replay_price']
        seconds=p['seconds_per_input_token_upper']*binding['models']['lfm25']['native_context_limit']
        cards+=planned_tasks(binding,wave,'lfm_capture','lfm25',pack(rows,lambda r:seconds,p['load_seconds']))
    elif wave in (4,9):
        phase='lfm' if wave==4 else 'primary';inventory=verified(root/('ANNOTATION_INVENTORY-'+phase+'.json'))
        require(inventory['status']=='complete','actual annotation inventory has unpriced inputs')
        rows=verified(inventory['task_file'])['records'];price=verified(root/'READER_PRICE.json')['profiles']['covered']
        groups=pack(rows,lambda r:price['batch_seconds']/64,price['load_seconds'],capacity=64)
        # Pack charges every final short reader batch at the full measured cost.
        for group in groups:
            group['planned_seconds']=price['load_seconds']+((len(group['records'])+63)//64)*price['batch_seconds']
            require(group['planned_seconds']<=18000,'reader full tail exceeds packing allowance')
        cards+=planned_tasks(binding,wave,'annotate','all',groups,annotation_phase=phase)
    elif wave==5:cards+=pair(cfg,'lfm_derive',minutes=360)
    elif wave==6:cards+=pair(cfg,'freeze_primary',minutes=360)
    elif wave==7:
        mode=verified(root/'DESIGN.json')['mode']
        for model in MODELS:
            price=verified(root/model/'QUALIFIED.json')
            rows=[r for r in verified(root/model/('ASSIGNMENTS-'+mode+'.json'))['records']
                  if r['prefix_status']=='enrolled' and r['nomination_status']=='nominated']
            cards+=planned_tasks(binding,wave,'generate',model,pack(rows,lambda r:generation_cost(price),price['load_seconds'],capacity=price['capacity']),mode=mode)
        cards+=pair(cfg,'lfm_readout',minutes=360)
    elif wave==8:cards+=pair(cfg,'annotation_inventory',minutes=360,annotation_phase='primary')
    elif wave==10:cards+=pair(cfg,'analyze',minutes=360)
    else:raise ValueError('finite scientific graph has no wave '+str(wave))
    for candidate in cards:candidate['previous_wave']=wave-1
    cards=parallel_lanes(binding,cards)
    require(len(cards)+3<950,'phase exceeds admitted Slurm submission capacity; no partial release')
    return protect(cfg,cards,wave,[digest(c) for c in cards if c['attempt']==1])


def handoff(root,controller,value):
    return publish(Path(root)/'controllers'/controller/'HANDOFF.json',value)


def advance(binding,c,deadline):
    root=Path(binding['root']);identity=digest(c)
    old=root/'controllers'/identity/'HANDOFF.json'
    if old.exists():return {'status':'complete','handoff':verified(old)}
    plan=verified(root/'waves'/('%02d'%c['wave'])/'PLAN.json')
    failures=[]
    for required in plan['terminal_cards']:
        p=root/'stages'/required/'RESULT.json'
        if not p.exists() or verified(p)['status']!='complete':failures.append({'card':required,'result':verified(p) if p.exists() else None})
    if failures or c['wave']==10:
        final=queue_finalization(binding,{'wave':c['wave'],'failures':failures,'reason':'stage_failure' if failures else 'scientific_graph_terminal'})
        return {'status':'complete','handoff':handoff(root,identity,{'finalization':final})}
    next_wave=c['wave']+1;require(next_wave<=10,'finite graph exhausted')
    p=root/'waves'/('%02d'%next_wave)/'PLAN.json'
    candidate=payload(verified(p)) if p.exists() else production_plan(binding,next_wave)
    launch=submit_wave(binding,candidate)
    return {'status':'complete','handoff':handoff(root,identity,{'wave':next_wave,'launch_sha256':launch['sha256']})}


def guard(binding,c,deadline):
    root=Path(binding['root']);handoff_path=root/'controllers'/c['controller_card']/'HANDOFF.json'
    if handoff_path.exists() or (root/'FINALIZATION_LAUNCH.json').exists():return {'status':'complete','owner_preserved':True}
    if c['attempt']==0:
        controller=payload(verified(root/'cards'/(c['controller_card']+'.json')))
        return advance(binding,controller,deadline)
    return {'status':'complete','finalization':queue_finalization(binding,{'wave':c['wave'],'reason':'controller_recovery_exhausted'})}


def queue_finalization(binding,reason):
    from .reservations import reserve
    root=Path(binding['root']);path=root/'FINALIZATION_INTENT.json';scheduler=Slurm(binding['config'])
    with locked(root,'finalization-intent'):
        if not path.exists():
            deps=[]
            live=set(command(['squeue','-h','-u','lmolfett','-o','%i']).splitlines())
            for p in (root/'submissions').glob('*/ACCEPTED.json'):
                a=verified(p);candidate=payload(verified(root/'cards'/(a['card_sha256']+'.json')))
                if candidate['stage'] in ('advance','guard','finalize'):continue
                if a['job_id'] not in live:continue
                fields=job_fields(command(['scontrol','show','job','-o',a['job_id']]))
                if fields['JobState']=='PENDING' and fields.get('Reason')=='JobHeldUser':
                    require(fields['RunTime']=='00:00:00' and fields['StartTime']=='Unknown','unused allocation already started')
                    command(['scancel',a['job_id']])
                    fields=job_fields(command(['scontrol','show','job','-o',a['job_id']]))
                    require(fields['JobState']=='CANCELLED' and fields['RunTime']=='00:00:00' and fields['StartTime']=='Unknown',
                            'unused held worker retirement was not verified')
                if fields['JobState'] in ('PENDING','RUNNING','COMPLETING'):deps.append(a['job_id'])
            publish(path,{'reason':reason,'dependencies':sorted(set(deps+[os.environ['SLURM_JOB_ID']]))})
    intent=verified(path);jobs=pair(binding['config'],'finalize',phase='finalization',minutes=75)
    receipts=[]
    for candidate in jobs:
        reserve(binding,candidate)
        deps=intent['dependencies']+([receipts[0]['job_id']] if candidate['attempt'] else [])
        receipts.append(scheduler.submit(candidate,deps,hold=False))
    return publish(root/'FINALIZATION_LAUNCH.json',{'intent_sha256':intent['sha256'],'jobs':receipts})


def finalize(binding,c,deadline):
    from .delivery import catalogue,legacy_complete
    root=Path(binding['root']);bulk=Path(binding['config']['bulk_root'])
    output=None
    if (root/'FINAL_EXPORT.json').exists():
        proof=verified(root/'FINAL_EXPORT.json');output=proof['output']
        require(file_sha(Path(output['path'])/'paper_results_manifest.json')==proof['manifest_file_sha256'],
                'final export manifest changed')
    elif (root/'DESIGN.json').exists():
        from .export import export
        output=export(binding,verified(root/'DESIGN.json')['mode'])
    state={'collection_terminal':True,'required_results_complete':bool(output and output['required_results_complete']),
        'manuscript_integrated':False,'export':output,'legacy_results_complete':legacy_complete(binding),
        'native_artifacts':catalogue(binding),'intent':verified(root/'FINALIZATION_INTENT.json'),
        'source_binding_sha256':binding['sha256'],'remote_pushes':False,'canonical_manuscript_edits':False}
    terminal=publish(root/'TERMINAL.json',state)
    from .resources import accounting
    publish(root/'STATUS_INDEX.json',{'terminal_sha256':terminal['sha256'],'jobs':accounting(root),
        'source_binding':str(root/'BINDING.json'),'authorization':str(root/'RECOVERED_AUTHORIZATION.json'),
        'admission':str(root/'ADMISSION.json') if (root/'ADMISSION.json').exists() else None,
        'stages':[str(p) for p in sorted((root/'stages').glob('*/RESULT.json'))],
        'collection_terminal':True,'required_results_complete':state['required_results_complete'],
        'manuscript_integrated':False})
    return {'status':'complete','terminal_sha256':terminal['sha256'],**{k:state[k] for k in ('collection_terminal','required_results_complete','manuscript_integrated')}}
