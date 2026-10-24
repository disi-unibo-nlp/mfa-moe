#!/usr/bin/env python3
"""One-shot authorized GPT attention recovery; budgeted and recorded per job."""
import json,subprocess
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
P=ROOT/'results/measurement-study/production-submissions.json'
d=json.loads(P.read_text())
env='--export=ALL,PRODUCTION_BUNDLE='+str(ROOT/'results/measurement-study/production-code-v4')
def save():P.write_text(json.dumps(d,indent=2)+'\n')
def submit(key,args,hours):
    if key in d['jobs']:return d['jobs'][key]['job_id']
    used=sum(j['maximum_gpu_hours'] for j in d['jobs'].values())+d.get('retired_budget_gpu_hours',0)
    if used+hours>360:raise RuntimeError('Budget cap exceeded')
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',env,*args]
    r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    job=int(r.stdout.strip().split(';')[0]);d['jobs'][key]=dict(job_id=job,command=cmd,maximum_gpu_hours=hours,verified=False,bundle='production-code-v4');save()
    info=subprocess.check_output(['scontrol','show','job',str(job)],text=True)
    assert f'JobId={job} ' in info and 'UserId=lmolfett(' in info
    d['jobs'][key]['verified']=True;save();print(key,job,flush=True);return job
if not d.get('gpt_attention_recovery_initialized'):
    for key in list(d['jobs']):
        if not key.startswith('gpt/'):continue
        old=d['jobs'][key];state=subprocess.check_output(['sacct','-X','-nP','-j',str(old['job_id']),'--format=JobID,State,ElapsedRaw,AllocTRES'],text=True).strip()
        if key=='gpt/verify':
            if '|FAILED|' not in state:raise RuntimeError('Unexpected verification state: '+state)
            d['retired_budget_gpu_hours']=d.get('retired_budget_gpu_hours',0)+old['maximum_gpu_hours']
        elif '|CANCELLED|0|' not in state:raise RuntimeError('Expected cancelled unallocated job: '+state)
        d.setdefault('superseded_jobs',[]).append(dict(key=key,**old,accounting=state,reason='GPT eager attention memory recovery'));del d['jobs'][key]
    d['gpt_attention_recovery_initialized']=True;save()
prep=submit('gpt/preparation-v4',['--job-name=mfa-plan-gpt-v4','sbatch/production_plan.sbatch','gpt'],0)
verify=submit('gpt/verify',[f'--dependency=afterok:{prep}','--qos=boost_qos_dbg','--time=00:30:00','--job-name=mfa-verify-gpt-v4','sbatch/production_capture.sbatch','gpt','0','verify'],1)
captures=[submit(f'gpt/capture/{i}',[f'--dependency=afterok:{verify}','--time=06:00:00',f'--job-name=mfa-capture-gpt-v4-{i}','sbatch/production_capture.sbatch','gpt',str(i)],12) for i in range(4)]
analysis=submit('gpt/analysis',['--dependency=afterok:'+':'.join(map(str,captures)),'--job-name=mfa-analyze-gpt-v4','sbatch/production_analyze.sbatch','gpt'],0)
submit('gpt/supplement',[f'--dependency=afterok:{analysis}','--job-name=mfa-classes-gpt-v4','sbatch/production_reporting.sbatch','gpt'],0)
next_job=d['jobs']['nemotron/verify']['job_id'];dep='afterany:'+':'.join(map(str,[verify,*captures]))
subprocess.run(['scontrol','update',f'JobId={next_job}','Dependency='+dep],check=True)
d['jobs']['nemotron/verify']['verified_dependency']=dep;save()
parents=[d['jobs'][s+'/'+stage]['job_id'] for s in ('gpt','nemotron','glm','qwen36','qwen330b') for stage in ('analysis','supplement')]
final=d['jobs']['final/report']['job_id'];subprocess.run(['scontrol','update',f'JobId={final}','Dependency=afterany:'+':'.join(map(str,parents))],check=True)
subprocess.run(['scontrol','release',str(next_job)],check=True)
for job in (next_job,final):
    info=subprocess.check_output(['scontrol','show','job',str(job)],text=True)
    print(next(line.strip() for line in info.splitlines() if 'Dependency=' in line),flush=True)
d['reserved_gpu_hours']=sum(j['maximum_gpu_hours'] for j in d['jobs'].values())+d.get('retired_budget_gpu_hours',0)
d['unreserved_gpu_hours']=360-d['reserved_gpu_hours'];save();print('Maximum incl failed verification:',d['reserved_gpu_hours'],flush=True)
