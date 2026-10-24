"""Verify saved Qwen3.5 inputs for native recovery without inference or downloads."""
import gzip,hashlib,json,os,time
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
BASE=ROOT/'results/measurement-study'
CACHE=ROOT.parent/'cache'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def location(info):
 p=Path(info['path']);s=p.stat()
 if (s.st_size,s.st_mtime_ns)!=(info['size'],info['mtime_ns']):raise ValueError('Source identity changed: '+str(p))
 with p.open('rb') as f:f.seek(info['byte_offset']);raw=f.read(info['line_bytes'])
 if hashlib.sha256(raw).hexdigest()!=info['line_sha256']:raise ValueError('Source content changed')
 r=json.loads(raw);t=r.get('metadata',{}).get('token_replay',{})
 prompt=t.get('prompt_token_ids');completion=t.get('completion_token_ids')
 exact=isinstance(prompt,list) and isinstance(completion,list) and bool(prompt) and bool(completion)
 if exact and not all(type(x) is int and x>=0 for x in prompt+completion):raise ValueError('Invalid saved token IDs')
 return r,dict(exact_token_ids=exact,input_tokens=len(prompt)+len(completion) if exact else None,completion_tokens=len(completion) if exact else None,token_ids_sha256=hashlib.sha256(json.dumps([prompt,completion],separators=(',',':')).encode()).hexdigest() if exact else None)

def main():
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Requires CPU Slurm allocation')
 bundle=Path(__file__).resolve().parent
 for name,expected in json.loads((bundle/'SHA256.json').read_text()).items():
  if sha(bundle/name)!=expected:raise ValueError('Frozen code changed')
 scope_path=bundle/'required-recovery.json';scope=json.loads(scope_path.read_text())
 joined=BASE/'source-join-v2/collection.json';d=json.loads(joined.read_text());out=BASE/'qwen35-recovery-audit-v1'
 if (out/'result.json').exists():raise ValueError('Audit already complete; preserve immutable result')
 models=[]
 for spec in scope['tracks']:
  for m in spec['generation_manifests']:
   if sha(m['path'])!=m['sha256']:raise ValueError('Generation manifest changed')
  rows=[r for r in d['attempts'] if r['model']==spec['model']]
  if len(rows)!=spec['labeled_trace_versions']:raise ValueError('Selected source coverage changed')
  records=[]
  for row in rows:
   seen=set();locations=[];annotations=[]
   for cohort in row['cohorts']:
    a=cohort['annotation'];raw=gzip.decompress(Path(a['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=a['sha256']:raise ValueError('Saved annotation changed')
    annotations.append(dict(cohort=cohort['id'],**a))
    for loc in cohort['locations']:
     key=(loc['path'],loc['byte_offset'],loc['line_sha256'])
     if key in seen:continue
     seen.add(key);trace,evidence=location(loc)
     if trace['model_id']!=spec['model']:raise ValueError('Source model mismatch')
     locations.append(dict(location=loc,**evidence))
   exact=[x for x in locations if x['exact_token_ids']]
   hashes={x['token_ids_sha256'] for x in exact}
   record=dict(id=row['id'],trace_sha256=row['trace_sha256'],dataset=row['dataset'],original_exact_replay_flag=row['exact_token_replay'],annotations=annotations,locations=locations,token_status='saved_exact_ids' if len(hashes)==1 else ('conflicting_saved_ids' if hashes else 'missing_saved_ids'),selected_location=exact[0] if len(hashes)==1 else None)
   records.append(record)
  model_cache=CACHE/'hf/hub'/('models--'+spec['model'].replace('/','--'))
  snapshots=[str(p.parent) for p in (model_cache/'snapshots').glob('*/config.json')]
  result=dict(source=spec['id'],generation_count=spec['saved_generations'],labeled_trace_versions=len(records),exact_token_versions=sum(r['token_status']=='saved_exact_ids' for r in records),missing_token_versions=sum(r['token_status']=='missing_saved_ids' for r in records),conflicting_token_versions=sum(r['token_status']=='conflicting_saved_ids' for r in records),checkpoint_configs=snapshots,native_capture_ready=False,native_gpu_jobs=[],status='requires_native_checkpoint_backend_validation_and_measured_sizing',records=records)
  atomic(out/(spec['id']+'.json'),result);models.append({k:v for k,v in result.items() if k!='records'})
  print(json.dumps(models[-1]),flush=True)
 result=dict(status='audit_complete_native_recovery_not_yet_ready',job_id=os.environ['SLURM_JOB_ID'],at=time.time(),scope_sha256=sha(scope_path),source_join_sha256=sha(joined),code_manifest_sha256=sha(bundle/'SHA256.json'),models=models,next_actions=['Resolve original native GPTQ/GGUF checkpoint revisions and tokenizer/backend identities','Retain missing/conflicting token cases as explicit gaps; no reconstructed IDs counted as exact','Preserve current labels and define any further labeling before judge retirement','Validate native capture on Slurm and measure budget before full submission'],gpu_hours_used=0)
 atomic(out/'result.json',result)
if __name__=='__main__':main()
