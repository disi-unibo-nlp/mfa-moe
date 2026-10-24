"""Enumerate, hash, and retire only explicitly authorized unshared weight files."""
import argparse, hashlib, json, os, shutil, stat, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub')
CANDIDATES={
 'gemma-bf16':('models--google--gemma-4-26B-A4B-it','4d7ae4984b7db7de8f8457170b3f1a419ee76d52'),
 'nemotron-bf16':('models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16','a9904d24bcc1d289a1950fa9d2b978c47cf903b9'),
 'qwen36-bf16':('models--unsloth--Qwen3.6-35B-A3B','2ab40a9acc6d567889ca4d4e59feb2da56121454'),
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()

def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def file_identity(path):
 s=Path(path).stat()
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1:raise ValueError('Weight must be owned, regular and unshared')
 return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)

def stage(output):
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Checksumming requires Slurm CPU allocation')
 if output.exists():raise ValueError('Retirement manifest already exists')
 records=[]
 for key,(model,revision) in CANDIDATES.items():
  snapshot=CACHE/model/'snapshots'/revision
  weights=sorted(snapshot.glob('*.safetensors'))
  if not weights:raise ValueError('No candidate weights')
  metadata=[]
  for p in sorted(snapshot.iterdir()):
   if p.is_file() and p not in weights:
    dest=output.parent/'metadata'/key/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    if sha(p)!=sha(dest):raise ValueError('Metadata copy mismatch')
    metadata.append(dict(original=str(p),archive=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
  for p in weights:
   if p.is_symlink() or p.resolve().parent!=snapshot.resolve():raise ValueError('Unexpected weight alias')
   identity=file_identity(p);h=sha(p)
   if file_identity(p)!=identity:raise ValueError('Weight changed while hashing')
   records.append(dict(model=key,revision=revision,path=str(p),sha256=h,identity=identity))
   print(json.dumps(dict(hashed=key,file=p.name,bytes=identity['size'])),flush=True)
  atomic(output.parent/(key+'-metadata.json'),dict(model=key,revision=revision,metadata=metadata))
 atomic(output,dict(status='hashed_ready_for_live_dependency_check',authorization='User approved retiring verified unused BF16 alternatives, then requested implementation',job_id=os.environ['SLURM_JOB_ID'],files=records,bytes=sum(r['identity']['size'] for r in records),created_at=time.time()))

def retire(manifest,audit):
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Retirement hash checks require Slurm CPU allocation')
 d=json.loads(manifest.read_text());a=json.loads(audit.read_text())
 if a['manifest_sha256']!=sha(manifest) or time.time()-a['at']>120:raise ValueError('Fresh manifest-bound dependency audit required')
 live=set(subprocess.check_output(['squeue','-h','-u','lmolfett','-o','%i'],text=True).split())-{os.environ.get('SLURM_JOB_ID','')}
 if not live.issubset(set(a['checked_job_ids'])):raise ValueError('An unaudited job appeared')
 if a.get('candidate_dependencies')!=[]:raise ValueError('Candidate still required')
 if set(r['model'] for r in d['files'])!=set(CANDIDATES):raise ValueError('Unexpected candidate set')
 for r in d['files']:
  model,revision=CANDIDATES[r['model']];p=Path(r['path'])
  if p.is_symlink() or p.parent!=CACHE/model/'snapshots'/revision or p.suffix!='.safetensors':raise ValueError('Outside authorized weight scope')
  if file_identity(p)!=r['identity']:raise ValueError('Weight changed after verification')
 receipt=manifest.with_name('retirement-receipt.json')
 if receipt.exists():raise ValueError('Retirement already attempted; inspect receipt')
 state=dict(status='retiring',manifest_sha256=sha(manifest),dependency_audit_sha256=sha(audit),files=[],bytes_retired=0)
 atomic(receipt,state)
 for r in d['files']:
  p=Path(r['path'])
  if file_identity(p)!=r['identity'] or sha(p)!=r['sha256']:raise ValueError('Weight changed immediately before unlink')
  p.unlink();state['files'].append(r);state['bytes_retired']+=r['identity']['size'];atomic(receipt,state)
 state.update(status='retired',finished_at=time.time());atomic(receipt,state)
 print(json.dumps(dict(status=state['status'],files=len(state['files']),bytes_retired=state['bytes_retired'])),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','retire']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--audit',type=Path);a=p.parse_args()
 if os.getuid()!=133943:raise RuntimeError('Wrong account')
 if a.action=='stage':stage(a.manifest)
 else:retire(a.manifest,a.audit)
if __name__=='__main__':main()
