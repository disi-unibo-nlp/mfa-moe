"""Reproduce each shard's first failed attempt and gate corrected native capture."""
import gc,io,json,os,socket,time
from pathlib import Path
from measurement_study.production_plan import ROOT,sha,atomic,read_location,digest
from measurement_study.production_capture import VLLMBackend,reduce_capture,pack_bytes
from measurement_study.native_selection import boundary_gaps

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):raise RuntimeError('GPU Slurm allocation required')
    import torch
    from moe_exp.schemas import TraceRecord
    if torch.cuda.device_count()!=2:raise RuntimeError('Exactly two allocated GPUs required')
    for rel,expected in json.loads((a.bundle/'SHA256.json').read_text()).items():
        if sha(a.bundle/rel)!=expected:raise ValueError('Frozen validation code changed')
    campaign=json.loads((a.bundle/'campaign.json').read_text())
    for path,expected in campaign['runtime_code_hashes'].items():
        if sha(path)!=expected:raise ValueError('Shared runtime changed')
    fixtures=json.loads((a.bundle/'nemotron-repro.json').read_text())
    out=ROOT/'results/measurement-study/recovery-storage/validation/nemotron';out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();backend=VLLMBackend(campaign['sources']['nemotron'],out);load=time.monotonic()-start
    records=[];failed=[]
    for fixture in fixtures:
        row=fixture['row'];manifest=fixture['manifest'];manifest['code_binding']=sha(a.bundle/'SHA256.json')
        manifest['binding']=digest({k:v for k,v in manifest.items() if k!='binding'})
        start=time.monotonic();trace=TraceRecord(**read_location(row['cohorts'][0]['locations'][0]))
        data=backend.capture(trace,'repro-'+row['id'])
        diagnostics=data['audit'][0]['selector_diagnostics']
        old=[r for r in diagnostics.values() if r['old_boundary_discrepancy'] < -1e-6]
        gaps=boundary_gaps(data['selection'],data['ids'])
        record=dict(shard=fixture['shard'],id=row['id'],input_tokens=row['total_input_tokens'],completion_tokens=row['completion_tokens'],old_mismatch_reproduced=bool(old),old_offenders=old,corrected_min_gap=float(gaps.min()),layers=manifest['spec']['layers'])
        try:
            packet,tensors=reduce_capture(row,data,backend,manifest)
            del data
            buf=io.BytesIO();torch.save(tensors,buf);del tensors
            files=[pack_bytes(out/(row['id']+'.pt.gz'),buf.getvalue(),2*1024**3)];del buf
            files.append(pack_bytes(out/(row['id']+'.json.gz'),json.dumps(packet,allow_nan=False).encode(),2*1024**3));del packet
            record.update(status='passed' if old else 'not_reproduced',files=files)
        except ValueError as exc:
            record.update(status='failed',error=str(exc));failed.append(row['id'])
        if not old:failed.append(row['id'])
        record['seconds']=time.monotonic()-start;records.append(record);gc.collect()
        atomic(out/f"shard-{fixture['shard']}.json",record)
    result=dict(status='passed' if not failed else 'failed',startup_seconds=load,records=records,failed=failed,bundle_sha256=sha(a.bundle/'SHA256.json'),threshold_unchanged=1e-6)
    atomic(out/'acceptance.json',result)
    if failed:raise RuntimeError('Native selector recovery has not passed; production remains gated')
if __name__=='__main__':main()
