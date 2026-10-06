"""Bounded read-only Slurm reconciliation; writes dated private audit receipts."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import pwd
import os
import subprocess

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S=R/'steering-v1'
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
ARTIFACTS={
    'h14':S/'qualification/h14/resume-v1/H14.json',
    'h14-recovery':S/'qualification/h14/resume-v1-recovery/H14.json',
    'rk1c':R/'reasoning-kinematics/rk1c/results.json',
    'r3b':R/'forum/tests/r3_dynamics_validity/estimates.json',
    'dp-s3':R/'depth-paths/results/resume-v1/results.json',
    'r3d-readout-v3':R/'forum/tests/r3_context/readout-results.v3.json',
    'r3d-anticipation-retry':R/'forum/tests/r3_context/anticipation-results.json',
    'm9-prepare':S/'runs/m9-resume-v1/MANIFEST.json',
    'm10-prepare':S/'runs/m10-resume-v1/PROBE_MANIFEST.json',
    'x2-prefixes':S/'runs/x2-resume-v1/measurement-prefixes/PREFIXES.json',
    'x2-analysis':S/'runs/x2-resume-v1/analysis/estimates.json',
    'ordered-prefixes':S/'runs/ordered-qualification-v1/CPU_PREFIXES.json',
    'paper-package':REPO/'report/experimental-resume-v1/PAPER_SNAPSHOT.json',
    'paper-package-final':REPO/'report/experimental-resume-v1/PAPER_SNAPSHOT.json',
    'paper-routing-supplement':REPO/'report/experimental-resume-v1/ROUTING_PROFILE_SUPPLEMENT.json',
    'ordered-qualify':S/'runs/ordered-qualification-v1/results/QUALIFICATION.json',
    'ordered-qualify-retry':S/'runs/ordered-qualification-v1/results-retry/QUALIFICATION.json',
    'ordered-qualify-cpu-recovery':S/'runs/ordered-qualification-v1/results-cpu-recovery/QUALIFICATION.json',
    'ordered-qualify-cpu-recovery-v2':S/'runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json',
}


def main():
    if pwd.getpwuid(os.getuid()).pw_name!='lmolfett' or not os.uname().nodename.endswith('.leonardo.local'):
        raise ValueError('verified LEONARDO lmolfett identity required')
    receipts=[(p,json.loads(p.read_text())) for p in sorted((S/'runs/resume-v1').glob('*-submission.json'))]
    direct_path=REPO/'report/experimental-resume-v1/DIRECT_JOB_REGISTRY.json'
    if direct_path.is_file():
        direct=json.loads(direct_path.read_text())
        if direct.get('schema')!='experimental-resume-direct-slurm-jobs-v1':
            raise ValueError('unknown direct job registry schema')
        receipts.extend((direct_path,row) for row in direct['jobs'])
    if len({r['job_id'] for _,r in receipts})!=len(receipts):
        raise ValueError('duplicate job ID across resume receipts and direct registry')
    ids=[str(r['job_id']) for _,r in receipts]
    query=subprocess.run(['sacct','--array','-j',','.join(ids),'-n','-P','-o',
        'JobID,JobName%60,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%100'],text=True,capture_output=True,check=True).stdout
    parents={};steps=[]
    for row in query.splitlines():
        cols=row.split('|')
        if len(cols)<7:continue
        job,name,state,exit_code,elapsed,cpus,tres=cols[:7]
        record={'job_id':job,'job_name':name,'state':state,'exit_code':exit_code,
            'elapsed_seconds':int(elapsed or 0),'allocated_CPUs':int(cpus or 0),'allocated_TRES':tres}
        if '.' in job:steps.append(record)
        else:parents[job]=record
    jobs=[]
    for path,receipt in receipts:
        r=parents.get(str(receipt['job_id']))
        if r is None:raise ValueError('submitted job absent from sacct: '+str(receipt['job_id']))
        task=receipt['task'];gpu=(receipt.get('unit')=='GPU-h' or task.startswith('h14')
            or task.startswith('ordered-qualify') or task=='x2' or task.startswith('x2-nll'))
        count=0
        if gpu:
            for field in r['allocated_TRES'].split(','):
                if field.startswith('gres/gpu='):count=int(field.split('=')[1])
            if r['elapsed_seconds'] and count==0:raise ValueError('GPU count missing from accounting')
        else:count=r['allocated_CPUs']
        state=r['state'].split()[0];final=state in TERMINAL
        artifact=Path(receipt['artifact']) if receipt.get('artifact') else ARTIFACTS.get(task)
        if task.startswith('paper-'):
            matching=[p for p in (R/'paper/resume-v1').glob('*/FROZEN.json')
                if str(json.loads(p.read_text()).get('job_id'))==str(receipt['job_id'])]
            if len(matching)>1:raise ValueError('multiple snapshots from one paper job')
            artifact=matching[0] if matching else None
        if task=='x2':
            production=json.loads((S/'runs/x2-resume-v1/FROZEN.json').read_text())
            artifact=Path(production['output'])/'shard-0.status.json'
        if task.startswith('x2-nll'):
            production=json.loads((S/'runs/x2-resume-v1/FROZEN.json').read_text())
            version=(json.loads((S/'runs/resume-v1/NLL_RECOVERY_AMENDMENT.json').read_text())['new_addendum']
                if task=='x2-nll-recovery' else json.loads((S/'runs/resume-v1/NLL_UID_AMENDMENT.json').read_text())['new_addendum']
                if task=='x2-nll-v2' else json.loads((S/'runs/s2prop/FROZEN_ADDENDA.json').read_text())['snapshots']['nll'])
            artifact=S/'runs/x2-resume-v1/native-nll'/production['manifest_sha256']/version['tree_sha256']/'native-nll.json'
        found=artifact is not None and artifact.is_file()
        record={**r,'task':task,'receipt':str(path),'final':final,'unit':receipt['unit'],
            'actual_resource_hours':r['elapsed_seconds']*count/3600,
            'slurm_billing_hours':r['elapsed_seconds']*next((float(field.split('=')[1]) for field in r['allocated_TRES'].split(',') if field.startswith('billing=')),0)/3600,
            'artifact':str(artifact) if artifact else None,'artifact_present':found,
            'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest() if found else None,
            'verified_completion':final and state=='COMPLETED' and r['exit_code']=='0:0' and found}
        if found:
            content=json.loads(artifact.read_text())
            record['artifact_status']=content.get('status')
            if task.startswith('x3-cost-timing-only-pilot'):
                record['verified_completion'] &= content.get('status')=='complete' and content.get('n_done')==40 and content.get('undone')==0
            if task.startswith('counterfactual-routing-loss-qualification') or task.startswith('score-api-calibration'):
                record['verified_completion'] &= content.get('pass') is True
            if task=='x2':
                record['verified_completion'] &= content.get('status')=='complete' and content.get('n_done')==6630 and content.get('undone')==0
            if task=='h14-recovery' or task.startswith('ordered-qualify'):record['verified_completion'] &= content.get('pass') is True
            if task.startswith('paper-'):record['verified_completion'] &= str(content.get('job_id')) == str(receipt['job_id'])
        jobs.append(record)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    value={'schema':'resume-job-audit-v1','observed_utc':now,'host':os.uname().nodename,
        'user':pwd.getpwuid(os.getuid()).pw_name,'jobs':jobs,'steps':steps,
        'accounting':'elapsed allocated GPUs or CPUs; live jobs show accrued use, not final expenditure',
        'all_submitted_jobs_final':all(j['final'] for j in jobs)}
    token=now.replace(':','').replace('-','').replace('.','')
    out=S/'runs/resume-v1/audits';out.mkdir(exist_ok=True)
    (out/(token+'.json')).write_text(json.dumps(value,indent=1)+'\n')
    (REPO/'report/experimental-resume-v1/JOB_AUDIT.json').write_text(json.dumps(value,indent=1)+'\n')
    print(json.dumps({'observed_utc':now,'jobs':[{'task':j['task'],'id':j['job_id'],
        'state':j['state'],'hours':j['actual_resource_hours'],'artifact':j['artifact_present']} for j in jobs]}))


if __name__=='__main__':main()
