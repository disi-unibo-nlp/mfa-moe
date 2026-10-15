"""Human entrypoint and authenticated allocated-worker dispatch."""
from pathlib import Path
import argparse
import getpass
import json
import os
import socket
import subprocess
import time
from .common import MODELS, REPO, check_files, digest, file_sha, load, publish, require, verified
from .config import read_config,initialize,registry,sources
from .scheduler import Slurm,command,job_fields,duration_minutes
from .workflow import initial_graph,payload,submit_wave,release_wave


def identity():
    require(getpass.getuser()=='lmolfett' and 'leonardo' in socket.getfqdn(),'wrong user or cluster')


def ownership(config,snap):
    source=Path(config['source_campaign']);protected=set(config['protected_report_jobs'])
    old=[];coord=[]
    journals={str(load(p)['job_id']):p for p in (source/'submissions').glob('*/ACCEPTED.json')}
    for row in snap['jobs']:
        if row['job_id'] in protected:continue
        if row['name']=='rc3-coordination':coord.append(row['job_id'])
        if row['name']!='rc3-controller':continue
        require(row['job_id'] in journals,'live controller has no accepted ownership journal')
        receipt=load(journals[row['job_id']]);p=source/'cards'/(receipt['card_sha256']+'.json');c=load(p)
        require(c['stage']=='controller' and not c.get('ops_report_only'),'protected reporting controller differs')
        f=row['fields']
        require(f['JobState']=='PENDING' and f['RunTime']=='00:00:00' and f['StartTime']=='Unknown',
                'predecessor controller has started; reconcile its new handoff before transfer')
        old.append({'job_id':row['job_id'],'accepted_path':str(journals[row['job_id']]),
            'accepted_sha256_file':file_sha(journals[row['job_id']]),'card_path':str(p),'card_sha256_file':file_sha(p),
            'fields':f})
    return {'pending_controllers':old,'running_or_pending_coordination':coord,'protected_report_jobs':sorted(protected)}


def transfer(binding,plan):
    root=Path(binding['root']);path=root/'OWNERSHIP_TRANSFER.json'
    if path.exists():return verified(path)
    launch=verified(root/'waves/00/ACCEPTED.json');graph=verified(root/'waves/00/PLAN.json')
    scheduler=Slurm(binding['config']);by_hash={digest(c):c for c in graph['cards']}
    for ident in [graph['controller'],*graph['guards']]:scheduler.readback(launch['jobs'][ident],by_hash[ident])
    retired=[]
    for old in plan['pending_controllers']:
        require(old['job_id'] not in binding['config']['protected_report_jobs'],'reporting owner must remain')
        check_files({old['accepted_path']:old['accepted_sha256_file'],old['card_path']:old['card_sha256_file']})
        fields=job_fields(command(['scontrol','show','job','-o',old['job_id']]))
        require(fields['RunTime']=='00:00:00' and fields['StartTime']=='Unknown','predecessor started during handoff')
        if fields['JobState']=='PENDING':command(['scancel',old['job_id']])
        final=job_fields(command(['scontrol','show','job','-o',old['job_id']]))
        require(final['JobState']=='CANCELLED' and final['RunTime']=='00:00:00' and final['StartTime']=='Unknown',
                'pending predecessor retirement was not verified')
        retired.append({'job_id':old['job_id'],'fields':final})
    return publish(path,{'status':'complete','new_owner_jobs':[launch['jobs'][i] for i in [graph['controller'],*graph['guards']]],
        'all_replacements_accepted_before_retirement':True,'retired':retired,
        'coordination_preserved':plan['running_or_pending_coordination'],'reporting_preserved':plan['protected_report_jobs']})


def delivery(config):
    own=[p for p in sources(config) if Path(p).is_relative_to(REPO) and
         (Path(p).is_relative_to(REPO/'src/moe_exp/final_routing') or Path(p).is_relative_to(REPO/'tests/final_routing') or
          p in [str(REPO/'scripts/final_routing.py'),str(REPO/'scripts/slurm/final_routing.sbatch'),config['config_path']])]
    subprocess.run(['git','diff','--exit-code','HEAD','--',*[str(Path(p).relative_to(REPO)) for p in own]],
                   cwd=REPO,check=True,capture_output=True)
    for p in own:
        subprocess.run(['git','ls-files','--error-unmatch',str(Path(p).relative_to(REPO))],cwd=REPO,check=True,capture_output=True)
    revision=command(['git','rev-parse','HEAD'])
    return {'commit':revision,'files':{p:file_sha(p) for p in own},'remote_push':'user-owned'}


def launch(config):
    from .resources import snapshot,preparation_admission
    identity();root=Path(config['root'])
    validation=verified(root/'PRELAUNCH_VALIDATION.json')
    require(validation['status']=='complete','executable CPU path did not validate')
    check_files(validation['source_files'])
    local=delivery(config)
    if not (root/'LOCAL_DELIVERY.json').exists():publish(root/'LOCAL_DELIVERY.json',local)
    binding=initialize(config)
    if not (root/'LAUNCH_INTENT.json').exists():
        snap=snapshot();admission=preparation_admission(config,snap)
        require(admission['status']=='admitted',str(admission))
        own=ownership(config,snap)
        publish(root/'PREPARATION_ADMISSION.json',{'admission':admission,'snapshot':snap})
        publish(root/'LAUNCH_INTENT.json',{'ownership':own,'graph':initial_graph(config,own['running_or_pending_coordination']),
            'binding_sha256':binding['sha256'],'local_commit':local['commit']})
    intent=verified(root/'LAUNCH_INTENT.json');graph=intent['graph']
    require(intent['binding_sha256']==binding['sha256'],'launch source binding changed')
    scheduler=Slurm(config)
    if not (root/'SCHEDULER_VALIDATION.json').exists():
        checks=[]
        for c in [graph['cards'][0],next(c for c in graph['cards'] if c['stage']=='qualify')]:
            checks.append(scheduler.submit(c,test_only=True,hold=True))
        publish(root/'SCHEDULER_VALIDATION.json',{'status':'complete','checks':checks})
    accepted=submit_wave(binding,graph,release=False)
    # The durable failure owners must be live before a predecessor retires.
    for identity in graph['guards']+[graph['controller']]:
        jid=accepted['jobs'][identity]
        fields=job_fields(command(['scontrol','show','job','-o',jid]))
        if fields['JobState']=='PENDING' and fields.get('Reason')=='JobHeldUser':command(['scontrol','release',jid])
        require(job_fields(command(['scontrol','show','job','-o',jid])).get('Reason')!='JobHeldUser',
                'replacement failure owner remains held')
    transfer(binding,intent['ownership'])
    release_wave(binding,graph,accepted)
    return publish(root/'LAUNCH.json',{'status':'released','binding_sha256':binding['sha256'],
        'jobs':accepted['jobs'],'wave':0,'preparation_margin_inclusive_hours':1400,
        'production':'automatic after complete measured admission and receipt gates','remote_pushes':False})


def plan(config):
    root=Path(config['root']);assignments={};missing=[]
    for model in MODELS:
        assignments[model]={}
        for mode in ('full','fallback'):
            p=root/model/('ASSIGNMENTS-'+mode+'.json')
            if p.exists():
                v=verified(p);assignments[model][mode]={'path':str(p),'sha256':v['sha256'],
                    'assigned_cells':len(v['records']),'available_executions':v['available_executions']}
            else:
                assignments[model][mode]={'status':'requires allocated discovery preparation',
                    'maximum_assigned_cells':129*85+(129 if mode=='full' else 32)*84}
        for name in ('COUNT_NOMINATIONS.json','QUALIFIED.json'):
            if not (root/model/name).exists():missing.append(str(root/model/name))
    for name in ('LFM_CHECKPOINT_VERIFIED.json','CPU_PRICE.json','READER_PRICE.json','INHERITED_PRICES.json','ADMISSION.json'):
        if not (root/name).exists():missing.append(str(root/name))
    from .resources import snapshot,preparation_admission
    snap=snapshot()
    return {'read_only':True,'root':str(root),'source_campaign':config['source_campaign'],'assignments':assignments,
        'missing_prerequisites':missing,'proposed_preparation_graph':initial_graph(config),
        'preparation':preparation_admission(config,snap),
        'production_forecast':verified(root/'ADMISSION.json') if (root/'ADMISSION.json').exists() else
            {'status':'unpriced','requires':'all eight qualifications, whole-batch reader, inherited CPU and LFM CPU prices'},
        'storage_free_bytes':snap['free_bytes'],'remaining_account_hours':snap['remaining_hours'],
        'outstanding_reservation_hours':snap['outstanding_hours'],'checkpoint_download_complete':(root/'LFM_TRANSFER_COMPLETE.json').exists()}


def status(config):
    from .resources import accounting
    root=Path(config['root']);jobs=accounting(root)
    stages=[]
    for p in (root/'stages').glob('*/RESULT.json'):
        r=verified(p);stages.append({'card':p.parent.name,'status':r['status'],'job_id':r.get('job_id')})
    return {'root':str(root),'jobs':jobs,'stage_results':stages,
        'admission':verified(root/'ADMISSION.json') if (root/'ADMISSION.json').exists() else None,
        'terminal':verified(root/'TERMINAL.json') if (root/'TERMINAL.json').exists() else None,
        'remote_pushes':False,'manuscript_integrated':False}


def worker(binding_path,card_path):
    identity();require(os.environ.get('SLURM_JOB_ID'),'worker requires a real Slurm allocation')
    binding=verified(binding_path);c=payload(verified(card_path));root=Path(binding['root'])
    require(Path(card_path).resolve()==root/'cards'/(digest(c)+'.json'),'worker card path differs')
    check_files(binding['code_files'])
    require(file_sha(Path(binding['source_campaign'])/'CAMPAIGN.json')==binding['source_campaign_sha256'],'source campaign changed')
    acceptance=verified(root/'submissions'/digest(c)/'ACCEPTED.json')
    require(acceptance['job_id']==os.environ['SLURM_JOB_ID'],'allocation was not accepted for this card')
    fields=Slurm(binding['config']).readback(os.environ['SLURM_JOB_ID'],c)['fields']
    remaining=(c['minutes']-duration_minutes(fields['RunTime']))*60
    from .journal import run_stage
    from .execution import handler
    return run_stage(binding,c,handler,allocation_deadline=time.time()+remaining-60)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','submit','status','resume','export','worker'))
    parser.add_argument('--config',type=Path,default=REPO/'configs/final_routing.yaml')
    parser.add_argument('--binding',type=Path);parser.add_argument('--card',type=Path)
    args=parser.parse_args(argv)
    if args.command=='worker':
        require(args.binding and args.card,'worker requires bound card and source binding')
        value=worker(args.binding,args.card)
    else:
        config=read_config(args.config)
        if args.command=='plan':value=plan(config)
        elif args.command=='status':value=status(config)
        elif args.command in ('submit','resume'):value=launch(config)
        else:
            from .reservations import reserve
            from .workflow import node
            binding=initialize(config);root=Path(config['root'])
            require((root/'ALLOCATION.json').exists(),'export needs the allocated population inventory')
            source_state=digest(sorted(p.parent.name for p in (root/'stages').glob('*/RESULT.json')))
            c=node(config,'export',phase='finalization',minutes=30,source_state=source_state)
            reserve(binding,c);value=Slurm(config).submit(c)
    print(json.dumps(value,sort_keys=True))
    return 0 if value.get('status') not in ('execution_failed','upstream_unavailable','resource_limited','incomplete_results') else 1
