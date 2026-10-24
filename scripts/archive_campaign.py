"""Inventory complete model results and verify durable WORK copies; never delete data."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
WORK=Path('/leonardo_work/IscrC_MIOSR/lmolfett')
BASE=ROOT/'results/measurement-study'
FAST=Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/measurement-study-v1')
ARCHIVE=BASE/'archive-v2'
SKIP={'runtime','__pycache__','.git','cache','tmp'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def tree(p):
 seen=set()
 for directory,dirs,files in os.walk(p,followlinks=True):
  q=Path(directory);st=q.stat();key=(st.st_dev,st.st_ino)
  if key in seen:dirs[:]=[];continue
  seen.add(key);dirs[:]=[x for x in dirs if x not in SKIP]
  for name in files:
   f=q/name
   if f.is_file() and not name.endswith(('.partial','.tmp','.lock')):yield f

def inventory(source):
 spec=json.loads((ROOT/'report/measurement-study/campaign.json').read_text())['sources'][source]
 paths=set();cohorts=set();records=[];manifests=[]
 for stage in ['production-analysis','production-supplement']:
  d=BASE/stage/source
  if json.loads((d/'result.json').read_text())['status']!='complete':raise ValueError('Analysis incomplete')
  paths.update(tree(d))
 for i in range(4):
  mp=FAST/'manifests'/source/f'{i}.json';rp=FAST/'capture'/source/str(i)/'result.json'
  m=json.loads(mp.read_text());r=json.loads(rp.read_text());manifests.append(m)
  expected={x['id'] for x in m['attempts']};ids=[x['id'] for x in r['records']]
  if r['status']!='complete_native_collection' or r['manifest_binding']!=m['binding'] or set(ids)!=expected or len(ids)!=len(expected) or sum(x['tokens'] for x in r['records'])!=m['expected_tokens']:raise ValueError('Capture coverage mismatch')
  records.extend(r['records']);paths.add(mp);paths.update(tree(rp.parent))
  for row in m['attempts']:
   for c in row['cohorts']:
    cohorts.add(c['id']);paths.add(Path(c['annotation']['path']))
    paths.update(Path(x['path']) for x in c['locations'])
 gen=Path(spec['data_root'])/'generation'/spec['generation_model'].replace('/','--')
 count=0
 for dataset in ['math500','aime24','aime25','amc23','minerva','olympiad']:
  d=json.loads((gen/dataset/'manifest.json').read_text())
  if d['status']!='complete':raise ValueError('Generation incomplete')
  count+=d['traces'];paths.update(tree(gen/dataset))
 paths.update(tree(Path(spec['sampling_root'])))
 available=ROOT/'results/available-metrics/v1/manifest.json';paths.add(available)
 for s in json.loads(available.read_text())['jobs']:
  if s['id'] not in cohorts:continue
  paths.update(Path(x['path']) for x in s['annotations']+s['traces'])
  for x in s['annotations']:
   part=Path(x['path']).parent.parent
   if part.name.startswith('part-'):paths.update(tree(part))
 completed=BASE/'completed-labels'/source
 if completed.exists():paths.update(tree(completed))
 # Preserve all immutable code/manifests; these already reside in WORK.
 for d in BASE.glob('*code*'):
  if d.is_dir():paths.update(tree(d))
 paths.update(p for p in BASE.glob('*.json') if p.name not in ['recovery-accounting.json','recovery-20260924.json'])
 for p in (BASE/'logs').glob('*'):
  if p.is_file() and source in p.name and not p.name.startswith('mfa-archive-'):paths.add(p)
 expected_hashes={str(Path(f['path']).resolve()):f['sha256'] for r in records for f in r['files']}
 aliases={}
 for p in paths:aliases.setdefault(p.resolve(),set()).add(str(p))
 files=[]
 for p,names in sorted(aliases.items(),key=lambda x:str(x[0])):
  if not p.is_file():raise ValueError('Missing archive dependency: '+str(p))
  if p.stat().st_uid!=os.getuid():raise ValueError('Archive source not owned by user')
  before=p.stat();h=sha(p);after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Archive source is changing')
  if str(p) in expected_hashes and h!=expected_hashes[str(p)]:raise ValueError('Accepted capture hash mismatch')
  durable=p.is_relative_to(WORK)
  dest=p if durable else ARCHIVE/'external'/str(p).lstrip('/')
  files.append(dict(source=str(p),destination=str(dest),aliases=sorted(names),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=h,copy_required=not durable))
 plan=dict(status='inventoried',source=source,generations=count,selected_attempts=len(records),cohorts=sorted(cohorts),files=files,total_bytes=sum(f['bytes'] for f in files),copy_bytes=sum(f['bytes'] for f in files if f['copy_required']),created_at=time.time(),manifest_bindings=[m['binding'] for m in manifests])
 out=ARCHIVE/source/'inventory.json';atomic(out,plan)
 (out.parent/'transfer-files.txt').write_text(''.join(f['source'].lstrip('/')+'\n' for f in files if f['copy_required']))
 print(json.dumps({k:v for k,v in plan.items() if k!='files'}),flush=True)

def copy_local(plan):
 # Optional explicitly approved alternative when the data mover is unavailable.
 for f in plan['files']:
  if not f['copy_required']:continue
  source=Path(f['source']);dest=Path(f['destination'])
  if not dest.is_relative_to(WORK):raise ValueError('Archive destination outside WORK')
  if dest.exists():
   if sha(dest)!=f['sha256']:raise ValueError('Existing archive file is corrupt')
   continue
  dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.partial')
  shutil.copy2(source,tmp)
  if tmp.stat().st_size!=f['bytes'] or sha(tmp)!=f['sha256']:raise ValueError('Archive copy changed')
  tmp.replace(dest)

def verify(path):
 plan=json.loads(path.read_text())
 for f in plan['files']:
  p=Path(f['destination'])
  if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise ValueError('Archive integrity failure: '+str(p))
 result=dict(status='archive_verified',source=plan['source'],inventory_sha256=sha(path),files=len(plan['files']),generations=plan['generations'],selected_attempts=plan['selected_attempts'],verified_at=time.time(),no_weights_required=True)
 atomic(path.with_name('acceptance.json'),result);print(json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','verify','copy-local']);p.add_argument('source',choices=['gpt','glm']);a=p.parse_args()
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Archive processing requires Slurm CPU allocation')
 if a.action=='inventory':inventory(a.source)
 else:
  path=ARCHIVE/a.source/'inventory.json'
  if a.action=='copy-local':copy_local(json.loads(path.read_text()))
  verify(path)
if __name__=='__main__':main()
