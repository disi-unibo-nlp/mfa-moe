"""Artifact-gated, measured four-shard resource proposals; never submit speculatively."""
import argparse,json,math,os
from pathlib import Path
from measurement_study.production_plan import ROOT,storage_root,atomic,sha

def propose(shards,records,startup_seconds,remaining_gpu_hours):
    if not records or min(r['tokens'] for r in records)<=0:raise ValueError('Measured completed captures required')
    # Slowest measured completion rate protects against an optimistic short trace.
    seconds_per_token=max(r['seconds']/r['tokens'] for r in records)
    bytes_per_token=max(sum(f['bytes'] for f in r['files'])/r['tokens'] for r in records)
    jobs=[]
    for shard in shards:
        seconds=math.ceil(1.5*(startup_seconds+seconds_per_token*shard['expected_tokens'])/1800)*1800
        jobs.append(dict(shard=shard['shard'],seconds=seconds,gpus=2,gpu_hours=2*seconds/3600,
                         estimated_compressed_bytes=math.ceil(bytes_per_token*shard['expected_tokens']),
                         storage_cap_bytes=shard['artifact_cap_bytes']))
    total=sum(j['gpu_hours'] for j in jobs)
    fits=total<=remaining_gpu_hours and all(j['seconds']<=86400 and j['estimated_compressed_bytes']<=j['storage_cap_bytes'] for j in jobs)
    return dict(status='ready_for_scheduling' if fits else 'staged_exceeds_remaining_envelope',jobs=jobs,
                total_gpu_hours=total,remaining_gpu_hours=remaining_gpu_hours,seconds_per_token=seconds_per_token,
                compressed_bytes_per_token=bytes_per_token,startup_seconds=startup_seconds,
                sizing_rule='1.5 times startup plus measured runtime, each shard rounded upward to 30 minutes')

def main():
    p=argparse.ArgumentParser();p.add_argument('source',choices=['gemma','nemotron','qwen330b']);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    bundle=Path(__file__).resolve().parents[1]
    for rel,expected in json.loads((bundle/'SHA256.json').read_text()).items():
        if sha(bundle/rel)!=expected:raise ValueError('Frozen sizing code changed')
    recovery=json.loads((ROOT/'results/measurement-study/recovery-20260924.json').read_text())
    remaining=recovery.get('remaining_unreserved_gpu_hours',36)
    storage=storage_root(a.source);ready=storage/'manifests'/a.source/'ready.json'
    output=ROOT/'results/measurement-study/recovery-proposals'/f'{a.source}.json'
    if a.source=='gemma':
        prepared=json.loads(ready.read_text());out=Path(prepared['acceptance_path']).parent if prepared.get('acceptance_path') else storage/'capture/gemma'/str(prepared['verification_shard'])
        acceptance=json.loads((out/'acceptance.json').read_text());assert acceptance['status']=='passed' and acceptance['unchanged_resume']
        records=[json.loads(p.read_text()) for p in out.glob('*.complete.json')];startup=json.loads((out/'load.json').read_text())['seconds']
    elif a.source=='nemotron':
        out=ROOT/'results/measurement-study/recovery-storage/validation/nemotron-confirmation-v1';acceptance=json.loads((out/'acceptance.json').read_text());assert acceptance['status']=='passed'
        records=[dict(tokens=r['completion_tokens'],seconds=r['seconds'],files=r['files']) for r in acceptance['records']];startup=acceptance['startup_seconds']
    else:
        historical=ROOT/'results/measurement-study/long-storage-sizing/result.json'
        measure=next(m for m in json.loads(historical.read_text())['models'] if m['source']==a.source)
        records=[dict(tokens=measure['measured_tokens'],seconds=measure['measured_phase_seconds'],files=[dict(bytes=measure['packed_fixture_bytes'])])]
        startup=measure['measured_load_seconds'];acceptance=dict(status='historical_native_smoke_only',measurement_sha256=sha(historical))
    if not ready.exists():
        # Nemotron's inputs are already frozen and unchanged. Size from those inputs;
        # collection must first prepare manifests bound to the validated new bundle.
        prior=Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/measurement-study-v1')
        shards=[json.loads((prior/'manifests'/a.source/f'{i}.json').read_text()) for i in range(4)]
    else:shards=[json.loads((storage/'manifests'/a.source/f'{i}.json').read_text()) for i in range(4)]
    result=propose(shards,records,startup,remaining);result.update(source=a.source,acceptance=acceptance,submitted=False,
        requirement='Refresh accounting, require current native acceptance, bind new manifests, verify storage headroom, and attach every submitted job to this session')
    atomic(output,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
