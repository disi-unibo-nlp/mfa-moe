"""Bounded replacement within the existing CPU preparation ceiling."""
from pathlib import Path
import time
from moe_exp.final_routing.common import REPO,check_files,digest,file_sha,locked,publish,require,verified
from moe_exp.final_routing.scheduler import command,job_fields
from moe_exp.final_routing import workflow
from .adapter import Slurm,amendment

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def cpu_envelope(cards,acceptances,usage,new_cards):
    rows=[]
    for c in cards+new_cards:
        if c['budget_phase']!='preparation_cpu':continue
        key=digest(c);a=acceptances.get(key);u=usage.get(a['job_id']) if a else None
        if a:require(u is not None,'accepted preparation has no native accounting')
        terminal=u and u['state'] in TERMINAL
        seconds=u['seconds'] if terminal else c['minutes']*60
        require(0<=seconds<=c['minutes']*60+60,'unexpected CPU preparation runtime')
        rows.append({'card_sha256':key,'job_id':a['job_id'] if a else None,
            'state':u['state'] if u else 'PROPOSED','seconds':seconds,
            'basis':'terminal_actual' if terminal else 'full_time_limit',
            'maximum_seconds':c['minutes']*60})
    require(len({r['card_sha256'] for r in rows})==len(rows),'duplicate CPU envelope card')
    total=sum(r['seconds'] for r in rows)
    require(total<=8*3600,'original eight-hour CPU preparation allowance exhausted')
    return {'rows':rows,'aggregate_seconds':total,'limit_seconds':8*3600,
        'margin_inclusive_local_hours':total/3600*4*1.25,'limit_margin_inclusive_hours':40.,
        'original_allowance_reset':False}


def native_retirement(ids):
    raw=command(['sacct','-nP','-X','-j',','.join(ids),
        '--format=JobIDRaw,State,ElapsedRaw,Start,AllocTRES'])
    rows={}
    for line in raw.splitlines():
        jid,state,elapsed,start,tres=line.split('|')[:5]
        rows[jid]={'state':state.split()[0],'seconds':int(elapsed or 0),'start':start,'allocated_tres':tres}
    return rows


def unused_cancelled(row):
    return row['state']=='CANCELLED' and row['seconds']==0 and row['start']=='None' and not row['allocated_tres']


def plan(binding,proof):
    root=Path(binding['root']);old=verified(root/'waves/00/PLAN.json');cfg=binding['config']
    validate=next(c for c in old['cards'] if c['stage']=='validate')
    cpu=workflow.node(cfg,'cpu_price',depends=[digest(validate)],phase='preparation_cpu',minutes=120,
        repair='lfm_registry',automatic_recovery=False)
    # Reuse exact reader and legacy recovery obligations; neither is redrawn.
    prerequisites=[digest(c) for c in old['cards'] if c['stage'] in ('reader_pilot','legacy_price') and c['attempt']==1]
    require(len(prerequisites)==2,'original preparation dependencies differ')
    admit=workflow.node(cfg,'admit',depends=[digest(cpu),*prerequisites],phase='preparation_cpu',minutes=15)
    graph=workflow.protect(cfg,[cpu,admit],0,[digest(admit)],control_phase='finalization')
    return {**graph,'ops_r1_sha256':proof['sha256'],'replaces_cards':[
        digest(c) for c in old['cards'] if c['stage'] in ('admit','advance','guard')]}


def reserve(binding,graph):
    from moe_exp.final_routing.resources import accounting,snapshot,availability
    from moe_exp.final_routing.reservations import reserve_many
    root=Path(binding['root']);target=root/'OPS_R1_PREPARATION_ADMISSION.json'
    if target.exists():
        old=verified(target);require(old['graph_sha256']==verified(root/'OPS_R1_INITIAL_GRAPH.json')['sha256'],'reservation graph changed')
        reserve_many(binding,[c for c in graph['cards'] if c['budget_phase']!='preparation_cpu'])
        return old
    # The same lock serializes old and new reservation journals.
    with locked(root,'reservations'):
        cards=[workflow.payload(verified(p)) for p in (root/'cards').glob('*.json')]
        new_ids={digest(c) for c in graph['cards']}
        cards=[c for c in cards if digest(c) not in new_ids]
        acceptances={p.parent.name:verified(p) for p in (root/'submissions').glob('*/ACCEPTED.json')}
        usage=accounting(root);envelope=cpu_envelope(cards,acceptances,usage,graph['cards'])
        snap=snapshot();available=availability(binding['config'],snap)
        added=sum(c['minutes']/60*c['billing_units']*1.25 for c in graph['cards'])
        require(added<=available['available_hours'],'replacement exceeds original live entitlement')
        for fs in ('work','fast'):
            need=(10+binding['config']['budget']['preparation_output_gib'][fs])*1024**3
            require(snap['free_bytes'][fs]>need,'replacement storage reserve unavailable')
        value=publish(target,{'status':'admitted','graph_sha256':verified(root/'OPS_R1_INITIAL_GRAPH.json')['sha256'],
            'cpu_envelope':envelope,'snapshot':snap,'available':available,'additional_margin_inclusive_hours':added,
            'failed_attempts_preserved':{j:usage[j] for j in ('60643316','60643319')},
            'qualification_reader_allowances_unchanged':True,'automatic_cpu_recovery':False})
        for c in graph['cards']:
            if c['budget_phase']!='preparation_cpu':continue
            key=digest(c)
            publish(root/'reservations'/(key+'.json'),{'card_sha256':key,'phase':'preparation_cpu',
                'hours':c['minutes']/60*c['billing_units']*1.25,'includes_margin':True,'attempt':c['attempt'],
                'settlement_proof':str(target),'settlement_sha256':value['sha256'],
                'actual_plus_remaining_limit_hours':envelope['margin_inclusive_local_hours'],'limit_hours':40.})
    reserve_many(binding,[c for c in graph['cards'] if c['budget_phase']!='preparation_cpu'])
    return value


def launch(binding):
    from moe_exp.final_routing.cli import identity
    identity();root=Path(binding['root']);proof=amendment(binding)
    if (root/'OPS_R1_RELEASED.json').exists():return verified(root/'OPS_R1_RELEASED.json')
    check=verified(root/'OPS_R1_PREFLIGHT.json');require(check['status']=='complete','operational preflight incomplete')
    check_files(check['source_files'])
    for path in proof['files']:
        if Path(path).is_relative_to(REPO):
            relative=str(Path(path).relative_to(REPO))
            command(['git','ls-files','--error-unmatch',relative])
            command(['git','diff','--exit-code','HEAD','--',relative])
    if not (root/'OPS_R1_LOCAL_DELIVERY.json').exists():
        publish(root/'OPS_R1_LOCAL_DELIVERY.json',{'commit':command(['git','rev-parse','HEAD']),
            'files':proof['files'],'remote_pushes':False})
    if not (root/'OPS_R1_INITIAL_GRAPH.json').exists():publish(root/'OPS_R1_INITIAL_GRAPH.json',plan(binding,proof))
    graph=workflow.payload(verified(root/'OPS_R1_INITIAL_GRAPH.json'));scheduler=Slurm(binding['config'])
    if not (root/'OPS_R1_SCHEDULER_VALIDATION.json').exists():
        publish(root/'OPS_R1_SCHEDULER_VALIDATION.json',{'status':'complete',
            'checks':[scheduler.submit(graph['cards'][0],hold=True,test_only=True)]})
    reserve(binding,graph)
    by_id={digest(c):c for c in graph['cards']};receipts={}
    for c in graph['cards']:
        deps=[]
        for key in c.get('depends_on',[]):
            accepted=receipts.get(key) or workflow.accepted(root,key)
            require(accepted is not None,'replacement prerequisite lacks acceptance')
            deps.append(accepted['job_id'])
        receipts[digest(c)]=scheduler.submit(c,deps,hold=True)
    accepted_path=root/'OPS_R1_ACCEPTED.json'
    if not accepted_path.exists():publish(accepted_path,{'jobs':{key:r['job_id'] for key,r in receipts.items()},
        'graph_sha256':verified(root/'OPS_R1_INITIAL_GRAPH.json')['sha256'],'all_acceptances_verified':True})
    # New failure owners become active before any predecessor is retired.
    observed={}
    for key in graph['guards']+[graph['controller']]:
        jid=receipts[key]['job_id'];f=scheduler.readback(jid,by_id[key])['fields']
        if f['JobState']=='PENDING' and f.get('Reason')=='JobHeldUser':command(['scontrol','release',jid])
        observed[key]=scheduler.readback(jid,by_id[key]);require(observed[key]['fields'].get('Reason')!='JobHeldUser','new owner remains held')
    transfer=root/'OPS_R1_OWNERSHIP_TRANSFER.json'
    if not transfer.exists():
        old_ids=[workflow.accepted(root,key)['job_id'] for key in graph['replaces_cards']]
        require(not set(old_ids).intersection(binding['config']['protected_report_jobs']),'report owner entered transfer')
        before=native_retirement(old_ids)
        for jid in old_ids:
            if jid in before and unused_cancelled(before[jid]):continue
            f=job_fields(command(['scontrol','show','job','-o',jid]))
            require(f['JobState']=='PENDING' and f['Reason']=='JobHeldUser' and f['StartTime']=='Unknown' and f['RunTime']=='00:00:00','predecessor started or hold changed')
            command(['scancel',jid])
        after={}
        for _ in range(10):
            after=native_retirement(old_ids)
            if set(after)==set(old_ids) and all(unused_cancelled(r) for r in after.values()):break
            time.sleep(1)
        require(set(after)==set(old_ids) and all(unused_cancelled(r) for r in after.values()),'predecessor retirement lacks unused-allocation proof')
        publish(transfer,{'new_owners':observed,'before':before,'after':after,
            'replacements_accepted_and_released_before_retirement':True,
            'protected_reporting_jobs':binding['config']['protected_report_jobs'],'scientific_workers_unchanged':True})
    for key in graph['worker_cards']:
        jid=receipts[key]['job_id'];f=scheduler.readback(jid,by_id[key])['fields']
        if f['JobState']=='PENDING' and f.get('Reason')=='JobHeldUser':command(['scontrol','release',jid])
        require(scheduler.readback(jid,by_id[key])['fields'].get('Reason')!='JobHeldUser','replacement worker remains held')
    result=root/'OPS_R1_RELEASED.json'
    if result.exists():return verified(result)
    return publish(result,{'status':'released','jobs':{key:r['job_id'] for key,r in receipts.items()},
        'ops_r1_sha256':proof['sha256'],'base_scientific_binding_unchanged':binding['sha256'],
        'automatic_cpu_recovery':False,'failed_original_attempts_preserved':True})
