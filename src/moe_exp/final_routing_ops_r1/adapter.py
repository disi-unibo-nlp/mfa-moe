"""Extend the historical registry in process; preserve sealed files and journals."""
from pathlib import Path
import os
import re
import sys
import time
from moe_exp.final_routing.common import REPO,check_files,digest,publish,require,verified,locked
from moe_exp.final_routing import scheduler as base_scheduler
from moe_exp.final_routing import workflow as base_workflow

ORIGINAL_MODELS=('gemma','glm','gpt','nemotron','qwen330b','qwen35','qwen36')
EXTENDED_MODELS=ORIGINAL_MODELS+('lfm25','lfm25-engineering')
_original_node=base_workflow.node
_original_advance=base_workflow.advance
_installed=None


def register_lfm():
    from moe_exp.final_routing.readouts import HISTORICAL
    sys.path.insert(0,str(HISTORICAL/'code'))
    from dp import s3_readout as sr
    require(tuple(sr.cm.MODELS) in (ORIGINAL_MODELS,EXTENDED_MODELS),'historical readout registry differs')
    sr.cm.MODELS=EXTENDED_MODELS
    return sr


def amendment(binding):
    value=verified(Path(binding['root'])/'OPS_R1_BINDING.json')
    require(value['base_binding_sha256']==binding['sha256'],'operational amendment has another scientific binding')
    check_files(value['files'])
    return value


class Slurm(base_scheduler.Slurm):
    def submit(self,card,dependencies=(),*,hold=False,test_only=False):
        proof=verified(self.root/'OPS_R1_BINDING.json')
        require(card.get('ops_r1_sha256')==proof['sha256'],'submission lacks the operational binding')
        cfg=self.config['slurm'];folder=self.root/'submissions'/digest(card);logs=self.root/'logs'
        logs.mkdir(parents=True,exist_ok=True,mode=0o700)
        args=['sbatch','--test-only' if test_only else '--parsable',
            '--account='+cfg['account'],'--partition='+cfg['partition'],'--qos='+cfg['qos'],
            '--nodes=1','--ntasks=1','--cpus-per-task='+str(card['cpus']),
            '--mem='+str(card['memory_gib'])+'G','--time='+str(card['minutes']),
            '--job-name='+card['job_name'],'--chdir='+str(REPO),
            '--output='+str(logs/'%x-%j.out'),'--error='+str(logs/'%x-%j.err')]
        if card['gpus']:args+=['--gres=gpu:'+str(card['gpus'])]
        if dependencies:
            require(all(re.fullmatch(r'\d+(?:_\d+)?',str(j)) for j in dependencies),'invalid dependency job ID')
            args+=['--dependency=afterany:'+':'.join(map(str,dependencies))]
        if hold:args.append('--hold')
        card_path=self.root/'cards'/(digest(card)+'.json');publish(card_path,card)
        args += [str(REPO/'scripts/slurm/final_routing_ops_r1.sbatch'),self.config['environment'],
                 str(self.root/'BINDING.json'),str(card_path)]
        if test_only:return {'test_only':True,'output':self.run(args),'argv':args}
        with locked(self.root,'submission'):
            if (folder/'INTENT.json').exists():
                require(verified(folder/'INTENT.json')['argv']==args,'submission dependencies or hold state changed')
            old=self.reconcile(folder,card)
            if old:return old
            publish(folder/'INTENT.json',{'card_sha256':digest(card),'argv':args,
                'date':time.strftime('%Y-%m-%d'),'epoch':time.time()})
            raw=self.run(args)
            require(re.fullmatch(r'\d+(;[\w.-]+)?',raw) is not None,'invalid sbatch acceptance')
            jid=raw.split(';')[0];publish(folder/'SBATCH.json',{'job_id':jid,'raw':raw})
            evidence=self.readback(jid,card)
            return publish(folder/'ACCEPTED.json',{'card_sha256':digest(card),**evidence})


def initial_plan(root):
    plan=verified(Path(root)/'OPS_R1_INITIAL_GRAPH.json')
    proof=verified(Path(root)/'OPS_R1_BINDING.json')
    require(plan['ops_r1_sha256']==proof['sha256'],'replacement graph has another binding')
    return plan


def advance(binding,card,deadline):
    if card['wave']!=0:return _original_advance(binding,card,deadline)
    root=Path(binding['root']);identity=digest(card)
    old=root/'controllers'/identity/'HANDOFF.json'
    if old.exists():return {'status':'complete','handoff':verified(old)}
    plan=initial_plan(root)
    require(identity==plan['controller'],'unrecognized initial controller')
    failures=[]
    for required in plan['terminal_cards']:
        path=root/'stages'/required/'RESULT.json'
        result=verified(path) if path.exists() else None
        if not result or result['status']!='complete':failures.append({'card':required,'result':result})
    if failures:
        final=base_workflow.queue_finalization(binding,{'wave':0,'failures':failures,'reason':'stage_failure','ops_r1_sha256':plan['ops_r1_sha256']})
        return {'status':'complete','handoff':base_workflow.handoff(root,identity,{'finalization':final})}
    path=root/'waves/01/PLAN.json'
    candidate=base_workflow.payload(verified(path)) if path.exists() else base_workflow.production_plan(binding,1)
    launch=base_workflow.submit_wave(binding,candidate)
    return {'status':'complete','handoff':base_workflow.handoff(root,identity,{'wave':1,'launch_sha256':launch['sha256']})}


def queue_finalization(binding,reason):
    """Preserve finite closeout; verify unused held jobs with native accounting."""
    from moe_exp.final_routing.reservations import reserve
    from .launch import native_retirement,unused_cancelled
    root=Path(binding['root']);path=root/'FINALIZATION_INTENT.json';scheduler=Slurm(binding['config'])
    command=base_scheduler.command;job_fields=base_scheduler.job_fields
    with locked(root,'finalization-intent'):
        if not path.exists():
            deps=[];retired={}
            live=set(command(['squeue','-h','-u','lmolfett','-o','%i']).splitlines())
            for p in (root/'submissions').glob('*/ACCEPTED.json'):
                accepted=verified(p);candidate=base_workflow.payload(verified(root/'cards'/(accepted['card_sha256']+'.json')))
                if candidate['stage'] in ('advance','guard','finalize'):continue
                jid=accepted['job_id']
                if jid not in live:continue
                fields=job_fields(command(['scontrol','show','job','-o',jid]))
                if fields['JobState']=='PENDING' and fields.get('Reason')=='JobHeldUser':
                    require(fields['RunTime']=='00:00:00' and fields['StartTime']=='Unknown','unused allocation already started')
                    command(['scancel',jid])
                    row=None
                    for _ in range(10):
                        row=native_retirement([jid]).get(jid)
                        if row and unused_cancelled(row):break
                        time.sleep(1)
                    require(row and unused_cancelled(row),'held worker retirement lacks native unused-allocation proof')
                    retired[jid]=row
                elif fields['JobState'] in ('PENDING','RUNNING','COMPLETING'):deps.append(jid)
            publish(path,{'reason':reason,'dependencies':sorted(set(deps+[os.environ['SLURM_JOB_ID']])),
                'unused_held_retirements':retired,'ops_r1_sha256':verified(root/'OPS_R1_BINDING.json')['sha256']})
    intent=verified(path);jobs=base_workflow.pair(binding['config'],'finalize',phase='finalization',minutes=75)
    receipts=[]
    for candidate in jobs:
        reserve(binding,candidate)
        deps=intent['dependencies']+([receipts[0]['job_id']] if candidate['attempt'] else [])
        receipts.append(scheduler.submit(candidate,deps,hold=False))
    return publish(root/'FINALIZATION_LAUNCH.json',{'intent_sha256':intent['sha256'],'jobs':receipts})


def install(binding):
    global _installed
    proof=amendment(binding)
    if _installed:
        require(_installed==proof['sha256'],'another operational amendment is already installed')
        return proof
    from moe_exp.final_routing import execution,journal
    original_handler=execution.handler;original_dependencies=journal.dependency_results
    def node(config,stage,model='all',**kwargs):
        return _original_node(config,stage,model,ops_r1_sha256=proof['sha256'],**kwargs)
    def handler(b,c,deadline):
        require(c.get('ops_r1_sha256')==proof['sha256'],'allocated card lacks the operational binding')
        if c['stage'] in ('cpu_price','lfm_readout'):
            register_lfm()
            registry_path=Path(b['root'])/'ops_r1'/('REGISTRY-'+digest(c)+'.json')
            publish(registry_path,{'card_sha256':digest(c),'ops_r1_sha256':proof['sha256'],
                'original_models':ORIGINAL_MODELS,'extended_models':EXTENDED_MODELS,
                'historical_sources_modified':False,'original_seven_indices_unchanged':True})
        return original_handler(b,c,deadline)
    def dependencies(root,c):
        if c.get('previous_wave')!=0:return original_dependencies(root,c)
        require(c.get('ops_r1_sha256')==proof['sha256'],'wave-zero substitution is unbound')
        rows=original_dependencies(root,{k:v for k,v in c.items() if k!='previous_wave'})
        for identity in initial_plan(root)['terminal_cards']:
            path=Path(root)/'stages'/identity/'RESULT.json'
            rows.append(verified(path) if path.exists() else {'status':'upstream_unavailable','missing_receipt':str(path)})
        return rows
    base_workflow.node=node;base_workflow.Slurm=Slurm;base_workflow.advance=advance
    base_workflow.queue_finalization=queue_finalization
    execution.handler=handler;journal.dependency_results=dependencies
    _installed=proof['sha256']
    return proof
