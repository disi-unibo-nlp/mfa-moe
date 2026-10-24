#!/usr/bin/env python3
"""One-shot submission of the user-authorized bounded collection DAG; no daemon."""
import json,os,re,subprocess
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
LEDGER=ROOT/'results/measurement-study/production-submissions.json'
PREP=json.loads((ROOT/'results/measurement-study/production-preparation-v2.json').read_text())
HOURS={'gpt':6,'qwen36':12,'nemotron':6,'glm':6,'qwen330b':8}
ORDER=['gpt','qwen36','nemotron','glm','qwen330b']
def publish(data):
    tmp=LEDGER.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(LEDGER)
def submit(key,args,gpu_hours,data):
    if key in data['jobs']:return data['jobs'][key]['job_id']
    if sum(x['maximum_gpu_hours'] for x in data['jobs'].values())+gpu_hours>360:raise RuntimeError('Authorized GPU cap exceeded')
    cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
    result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    if result.returncode:raise RuntimeError('Submission failed: '+result.stderr.strip())
    raw=result.stdout.strip()
    if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',raw):raise RuntimeError('Unexpected submission response: '+raw)
    job=int(raw.split(';')[0]);data['jobs'][key]=dict(job_id=job,command=cmd,maximum_gpu_hours=gpu_hours,verified=False)
    publish(data)
    info=subprocess.check_output(['scontrol','show','job',str(job)],text=True)
    if f'JobId={job} ' not in info or 'UserId=lmolfett(' not in info:raise RuntimeError('Submission verification failed')
    data['jobs'][key]['verified']=True;publish(data);print(json.dumps(dict(key=key,job_id=job,maximum_gpu_hours=gpu_hours)),flush=True)
    return job

def main():
    if os.getuid()!=133943:raise RuntimeError('Wrong account')
    if subprocess.check_output(['hostname'],text=True).strip()!='login02.leonardo.local':raise RuntimeError('Reverify orchestration host')
    data=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(authorization='report/measurement-study/automation-authorization.json',gpu_hours_cap=360,fast_gib_cap=200,preparation_jobs=PREP,concurrent_collection_gpu_jobs_cap=4,jobs={})
    previous=[]
    for source in ORDER:
        dependency='afterok:'+str(PREP[source])
        if previous:dependency+=',afterany:'+':'.join(map(str,previous))
        verify=submit(source+'/verify',[f'--dependency={dependency}','--time=00:30:00','--job-name=mfa-verify-'+source,'sbatch/production_capture.sbatch',source,'0','verify'],1,data)
        jobs=[]
        for shard in range(4):
            job=submit(source+f'/capture/{shard}',[f'--dependency=afterok:{verify}',f'--time={HOURS[source]:02d}:00:00',f'--job-name=mfa-capture-{source}-{shard}','sbatch/production_capture.sbatch',source,str(shard)],2*HOURS[source],data);jobs.append(job)
        submit(source+'/analysis',['--dependency=afterok:'+':'.join(map(str,jobs)),'--job-name=mfa-analyze-'+source,'sbatch/production_analyze.sbatch',source],0,data)
        previous=jobs
    data['reserved_gpu_hours']=sum(x['maximum_gpu_hours'] for x in data['jobs'].values());data['unreserved_gpu_hours']=360-data['reserved_gpu_hours'];publish(data)
    print(json.dumps(dict(status='submitted',jobs=len(data['jobs']),reserved_gpu_hours=data['reserved_gpu_hours'],unreserved_gpu_hours=data['unreserved_gpu_hours'])),flush=True)
if __name__=='__main__':main()
