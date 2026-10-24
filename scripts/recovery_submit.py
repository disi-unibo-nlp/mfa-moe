"""Submit one authorized recovery stage with durable receipt and read-back."""
import argparse, datetime, fcntl, getpass, json, re, socket, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'results/measurement-study/recovery-20260924.json'

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def check_budget(ledger, snapshot, amount, pool):
    if amount<0:raise ValueError('Nonnegative GPU reservation required')
    if snapshot['accounted_commitment_gpu_hours']+amount>snapshot['gpu_hours_cap']:
        raise RuntimeError('Campaign ceiling exceeded')
    observed={r['job_id']:r['gpu_hours'] for r in snapshot['jobs']}
    initial=sum(observed.get(r['job_id'],r['maximum_gpu_hours']) for r in ledger['jobs'].values() if r.get('budget_pool','initial')=='initial')
    if pool=='initial' and initial+amount>ledger['initial_reserved_gpu_hours']:
        raise RuntimeError('Initial recovery reserve exceeded; use authorized campaign pool for measured recovery')

def main():
    p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--gpu-hours',type=float,default=0);p.add_argument('--budget-pool',choices=['initial','campaign'],default='initial');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    assert getpass.getuser()=='lmolfett' and socket.gethostname().endswith('.leonardo.local')
    subprocess.run(['/usr/bin/python3.11','-B',str(ROOT/'scripts/reconcile_recovery_budget.py')],check=True)
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        d=json.loads(LEDGER.read_text())
        if a.key in d['jobs']:raise RuntimeError('Already submitted: '+a.key)
        snapshot=json.loads((LEDGER.parent/'recovery-accounting.json').read_text())
        if time.time()-datetime.datetime.fromisoformat(snapshot['at']).timestamp()>60:raise RuntimeError('Stale budget reconciliation')
        check_budget(d,snapshot,a.gpu_hours,a.budget_pool)
        args=a.args
        if args and args[0]=='--':args=args[1:]
        cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',*args]
        subprocess.run(['sbatch','--test-only',*cmd[2:]],cwd=ROOT,check=True)
        raw=subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
        if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',raw):raise RuntimeError('Unexpected submission receipt')
        job=raw.split(';')[0];record=dict(job_id=int(job),maximum_gpu_hours=a.gpu_hours,budget_pool=a.budget_pool,command=cmd,submitted_at=time.time(),verified=False)
        d['jobs'][a.key]=record;atomic(LEDGER,d)
        info=subprocess.check_output(['scontrol','show','job',job],text=True)
        if f'JobId={job} ' not in info or 'UserId=lmolfett(' not in info:raise RuntimeError('Job identity mismatch')
        record.update(verified=True,scontrol=info);atomic(LEDGER,d)
        print('VERIFIED_JOB',a.key,job,flush=True)
    # Foreground, bounded, same tool session. No daemon or new agent.
    subprocess.run(['/usr/bin/python3.11','-B',str(ROOT/'scripts/watch_slurm.py'),job,'--max-seconds','86400','--session-scope'],cwd=ROOT,check=True)
if __name__=='__main__':main()
