#!/usr/bin/env python3
"""Reconcile native cancelled-job accounting without changing the sealed workers."""
from pathlib import Path
import json
import time
from moe_exp.final_routing.common import REPO,check_files,digest,file_sha,publish,require,verified
from moe_exp.final_routing.cli import identity
from moe_exp.final_routing.scheduler import Slurm,command,job_fields
from moe_exp.final_routing.workflow import payload,release_wave


def accounting(ids):
    raw=command(['sacct','-nP','-X','-j',','.join(ids),
                 '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'])
    records={}
    for line in raw.splitlines():
        jid,state,code,elapsed,start,end,tres=line.split('|')[:7]
        require(jid in ids,'unexpected predecessor accounting identity')
        records[jid]={'job_id':jid,'state':state,'exit_code':code,'seconds':int(elapsed or 0),
                      'start':start,'end':end,'allocated_tres':tres}
    require(set(records)==set(ids),'predecessor accounting is incomplete')
    return records


def unstarted_terminal(row):
    return (row['state'].startswith('CANCELLED') and row['seconds']==0 and
            row['start']=='None' and not row['allocated_tres'])


def main():
    identity();root=REPO/'report/final-routing-20261015-r1'
    binding=verified(root/'BINDING.json');cfg=binding['config']
    local=verified(root/'LOCAL_DELIVERY.json');check_files(local['files'])
    graph=payload(verified(root/'waves/00/PLAN.json'));accepted=verified(root/'waves/00/ACCEPTED.json')
    intent=verified(root/'LAUNCH_INTENT.json');scheduler=Slurm(cfg)
    require(len(graph['cards'])==31 and accepted['all_acceptances_verified'],'preparation graph differs')
    cards={digest(c):c for c in graph['cards']}
    owners=graph['guards']+[graph['controller']]
    observed={key:scheduler.readback(accepted['jobs'][key],cards[key]) for key in owners}
    require(all(r['fields'].get('Reason')!='JobHeldUser' for r in observed.values()),'replacement owner is held')
    old=intent['ownership']['pending_controllers'];ids=[r['job_id'] for r in old]
    require(not set(ids).intersection(cfg['protected_report_jobs']),'reporting owner entered retirement')
    for row in old:
        check_files({row['accepted_path']:row['accepted_sha256_file'],row['card_path']:row['card_sha256_file']})
    before=accounting(ids);pending=[];control_before={}
    for jid,row in before.items():
        if unstarted_terminal(row):continue
        require(row['state']=='PENDING' and row['seconds']==0 and not row['allocated_tres'],
                'predecessor is not an unused pending allocation')
        f=job_fields(command(['scontrol','show','job','-o',jid]));control_before[jid]=f
        require(f['JobState']=='PENDING' and f['RunTime']=='00:00:00' and f['StartTime']=='Unknown',
                'pending predecessor started during handoff')
        pending.append(jid)
    if pending:
        command(['scontrol','hold',','.join(pending)])
        for jid in pending:
            f=job_fields(command(['scontrol','show','job','-o',jid]))
            require(f['JobState']=='PENDING' and f['Reason']=='JobHeldUser' and f['StartTime']=='Unknown',
                    'predecessor hold was not verified')
        command(['scancel',','.join(pending)])
    for attempt in range(10):
        after=accounting(ids)
        if all(unstarted_terminal(r) for r in after.values()):break
        time.sleep(1)
    require(all(unstarted_terminal(r) for r in after.values()),'unused predecessor retirement lacks terminal accounting')
    reports={j:job_fields(command(['scontrol','show','job','-o',j])) for j in cfg['protected_report_jobs']}
    require(all(f['JobState']=='PENDING' for f in reports.values()),'protected reporting state changed')
    publish(root/'OWNERSHIP_TRANSFER_RECONCILIATION.json',{'status':'complete',
        'script':str(Path(__file__).resolve()),'script_sha256':file_sha(__file__),
        'unchanged_worker_binding':binding['sha256'],'replacement_owners':observed,
        'predecessor_accounting_before':before,'predecessor_control_before':control_before,
        'predecessor_accounting_after':after,'protected_report_readbacks':reports,
        'observed_control_readback':verified(root/'OWNERSHIP_HANDOFF_HOLD.json'),
        'reason':'scontrol gives a cancelled pending job a synthetic StartTime equal to EndTime; sacct retains Start=None, zero elapsed and no allocated TRES',
        'scientific_sources_changed':False,'additional_allocations':0})
    publish(root/'OWNERSHIP_TRANSFER.json',{'status':'complete',
        'new_owner_jobs':[accepted['jobs'][key] for key in [graph['controller'],*graph['guards']]],
        'all_replacements_accepted_before_retirement':True,'retired':list(after.values()),
        'coordination_preserved':intent['ownership']['running_or_pending_coordination'],
        'reporting_preserved':intent['ownership']['protected_report_jobs'],
        'terminal_proof':'sacct Start=None; zero elapsed; no allocated TRES',
        'reconciliation':str(root/'OWNERSHIP_TRANSFER_RECONCILIATION.json')})
    release_wave(binding,graph,accepted)
    value=publish(root/'LAUNCH.json',{'status':'released','binding_sha256':binding['sha256'],
        'jobs':accepted['jobs'],'wave':0,'preparation_margin_inclusive_hours':1400,
        'production':'automatic after complete measured admission and receipt gates','remote_pushes':False})
    print(json.dumps({'status':value['status'],'jobs':len(value['jobs']),'validation_job':accepted['jobs'][graph['worker_cards'][0]]}))


if __name__=='__main__':main()
