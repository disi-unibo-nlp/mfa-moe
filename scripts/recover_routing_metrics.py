#!/usr/bin/env python3
"""Resume only interrupted bundles; freeze source identities and checkpoint each row."""
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import analyze_available_metrics as analysis
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'results/available-metrics/v1'
OUTPUT=ROOT/'results/available-metrics/recovery-v1'

def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n');temporary.replace(path)

def reduce_one(item):
    line,path=item
    try:return {'status':'complete','row':analysis.routing_features(json.loads(line),Path(path))}
    except (ValueError,KeyError,FileNotFoundError) as exc:return {'status':'rejected','error':str(exc)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Broad analysis requires Slurm')
    manifest=json.loads((ORIGINAL/'manifest.json').read_text())
    for file,digest in manifest['code_hashes'].items():
        if sha(Path(file).read_bytes())!=digest:raise RuntimeError('Frozen reducer changed: '+file)
    binding=sha(json.dumps({'manifest':manifest,'recovery_code':sha(Path(__file__).read_bytes())},sort_keys=True).encode())
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in manifest['jobs']:
            if task['kind']!='routing':continue
            original=json.loads((ORIGINAL/task['id']/'status.json').read_text())
            if original['status'].startswith('complete'):
                summaries.append({'id':task['id'],'origin':'original',**original});continue
            out=OUTPUT/task['id'];out.mkdir(parents=True,exist_ok=True)
            atomic(out/'status.json',{'status':'running','job_id':os.environ['SLURM_JOB_ID'],'binding':binding})
            rows=[];rejected=[];provenance=[];processed=0
            for info in task['traces']:
                path=Path(info['path']);st=path.stat()
                if st.st_size!=info['size'] or st.st_mtime_ns!=info['mtime_ns']:raise RuntimeError('Input changed: '+str(path))
                source_key=sha(str(path).encode())[:16];whole=hashlib.sha256()
                with path.open('rb') as stream:
                    pending=[]
                    def consume():
                        missing=[(line,str(path)) for index,line,key,cached in pending if cached is None]
                        results=iter(pool.map(reduce_one,missing,chunksize=1))
                        for index,line,key,cached in pending:
                            value=cached if cached is not None else next(results)
                            if cached is None:atomic(key,{'binding':binding,'line_sha256':sha(line),'result':value})
                            if value['status']=='complete':rows.append(value['row'])
                            else:rejected.append({'path':str(path),'line':index,'reason':value['error']})
                        pending.clear()
                    for index,line in enumerate(stream):
                        whole.update(line);key=out/'rows'/source_key/f'{index:08d}.json';cached=None
                        if key.exists():
                            record=json.loads(key.read_text())
                            if record['binding']!=binding or record['line_sha256']!=sha(line):raise RuntimeError('Stale row checkpoint')
                            cached=record['result']
                        pending.append((index,line,key,cached));processed+=1
                        if len(pending)>=64:
                            consume();print(task['id'],'checkpointed',processed,flush=True)
                    if pending:consume()
                provenance.append({'path':str(path),'sha256':whole.hexdigest()})
            analysis.write_rows(out/'rejected.jsonl',rejected);atomic(out/'provenance.json',provenance)
            result={'status':'complete' if not rejected else 'complete_with_rejections','cohorts':analysis.finish_groups(rows,out),'rejected':len(rejected),'binding':binding}
            atomic(out/'status.json',result);summaries.append({'id':task['id'],'origin':'recovery',**result})
            print(task['id'],result['status'],flush=True)
    atomic(OUTPUT/'routing-status.json',{'status':'complete' if all(x['status']=='complete' for x in summaries) else 'complete_with_rejections','jobs':summaries,'binding':binding})
if __name__=='__main__':main()
