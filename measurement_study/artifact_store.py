"""Lossless artifact packing and measured storage sizing; keeps source files."""
import argparse,gzip,hashlib,json,os,time
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')

def pack(source,destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Cannot overwrite source')
    before=source.stat();h=hashlib.sha256();destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix(destination.suffix+'.partial')
    try:
        with source.open('rb') as reader,temp.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=3,mtime=0) as writer:
                while block:=reader.read(1024*1024):h.update(block);writer.write(block)
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during packing')
        restored=hashlib.sha256()
        with gzip.open(temp,'rb') as reader:
            while block:=reader.read(1024*1024):restored.update(block)
        if restored.digest()!=h.digest():raise ValueError('Lossless round-trip failed')
        temp.replace(destination)
        return {'source':str(source),'packed':str(destination),'source_bytes':before.st_size,'packed_bytes':destination.stat().st_size,'sha256':h.hexdigest(),'round_trip_verified':True}
    finally:
        if temp.exists():temp.unlink()

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Artifact sizing requires an allocated CPU job')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/measurement-study/storage-sizing');a=p.parse_args()
    selected={'gpt':'smoke/gpt-58469979','glm':'smoke/glm-58469941','qwen330b':'smoke/qwen330b-58471217','qwen36':'native-replay/qwen36-58472802'}
    groups=[]
    for model,rel in selected.items():
        source=ROOT/'results/measurement-study'/rel;result=json.loads((source/'result.json').read_text());started=time.monotonic();files=[]
        for f in sorted((source/'forward').rglob('*')):
            if f.is_file():files.append(pack(f,a.output/model/(str(f.relative_to(source/'forward'))+'.gz')))
        tokens=result.get('completion_tokens') or sum(x['tokens'] for x in result['attempts'])
        groups.append({'model':model,'tokens':tokens,'source_bytes':sum(r['source_bytes'] for r in files),'packed_bytes':sum(r['packed_bytes'] for r in files),'pack_and_verify_seconds':time.monotonic()-started,'files':files})
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'result.json').write_text(json.dumps({'status':'lossless_storage_sizing_complete','groups':groups},indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='files'} for r in groups]),flush=True)
if __name__=='__main__':main()
