"""Reconcile actual finished allocations and maximum live reservations under 360 GPUh."""
import datetime,fcntl,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/measurement-study'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','REVOKED'}
def main():
    with (BASE/'recovery-20260924.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        production=json.loads((BASE/'production-submissions.json').read_text())
        recovery=json.loads((BASE/'recovery-20260924.json').read_text())
        jobs={str(r['job_id']):r for r in list(production['jobs'].values())+production.get('superseded_jobs',[])+list(recovery['jobs'].values()) if r['maximum_gpu_hours']>0}
        raw=subprocess.check_output(['sacct','-nPX','-j',','.join(jobs),'--format=JobIDRaw,User,State%40,ExitCode,ElapsedRaw,AllocTRES%120'],text=True)
        accounting={}
        for line in raw.splitlines():
            fields=line.split('|')
            if len(fields)>=6 and fields[0] in jobs:
                assert fields[1]=='lmolfett';accounting[fields[0]]=fields
        rows=[]
        for job,record in jobs.items():
            fields=accounting.get(job);maximum=record['maximum_gpu_hours']
            if fields and fields[2].split()[0] in TERMINAL:
                gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',fields[5])
                count=int(gpu[1]) if gpu else 0
                elapsed=int(fields[4]);charged=count*elapsed/3600
                if elapsed and not gpu and maximum>0 and fields[2].split()[0]!='CANCELLED':
                    charged=maximum;basis='missing allocation data; retain maximum'
                else:basis='observed terminal allocation'
            else:charged=maximum;basis='live or unavailable; retain full maximum'
            rows.append(dict(job_id=int(job),state=fields[2] if fields else 'accounting_unavailable',gpu_hours=charged,basis=basis,previous_maximum=maximum))
        pending_initial=max(0,recovery['initial_reserved_gpu_hours']-sum(x['maximum_gpu_hours']-x.get('refunded_unallocated_gpu_hours',0) for x in recovery['jobs'].values()))
        committed=sum(x['gpu_hours'] for x in rows)+pending_initial
        if committed>360:raise RuntimeError('Campaign ceiling exceeded')
        snapshot=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_hours_cap=360,accounted_commitment_gpu_hours=committed,remaining_gpu_hours=360-committed,reserved_unsubmitted_initial_gpu_hours=pending_initial,jobs=rows,
                      separate_prior_label_jobs=[58487704,58487708],note='Finished GPU allocations charged by observed elapsed time; unfinished jobs retain full requested maxima. Prior label resumes were explicitly separate; the new label retry is included.')
        path=BASE/'recovery-accounting.json';path.write_text(json.dumps(snapshot,indent=2)+'\n')
        recovery['remaining_unreserved_gpu_hours']=360-committed;recovery['accounting_snapshot']=str(path)
        (BASE/'recovery-20260924.json').write_text(json.dumps(recovery,indent=2)+'\n')
        print(json.dumps({k:v for k,v in snapshot.items() if k!='jobs'}))
if __name__=='__main__':main()
