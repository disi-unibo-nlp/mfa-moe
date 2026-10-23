"""Content-bound, resumable source joins; execute broad scans on allocated CPUs."""
import argparse, gzip, hashlib, json, os, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def attempt_id(row, trace_sha):
    return digest([row['model_id'],row['dataset'],row['problem_id'],row.get('sample_id',0),trace_sha])

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,allow_nan=False)+'\n');temp.replace(path)

def annotation_sample(annotation, sample_id, key, legacy_samples):
    """Old annotations use a content hash; never assume their missing sample is 0."""
    if 'sample_id' in annotation:
        return annotation['sample_id']==sample_id
    values=legacy_samples.setdefault(key,set());values.add(sample_id)
    if len(values)>1:raise ValueError('Legacy annotation matches multiple sample identities')
    return True

def check_file(info):
    st=Path(info['path']).stat()
    if (st.st_size,st.st_mtime_ns)!=(info['size'],info['mtime_ns']):
        raise ValueError('Source changed: '+info['path'])

def join_cohort(spec,output):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from analyze_available_metrics import batches,TraceRecord,trace_digest,base_row
    out=Path(output)/spec['id'];binding=digest(spec);status=out/'status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old['binding']!=binding:raise ValueError('Resume source binding changed')
        if old['status']=='complete':return old
    annotations,provenance,conflicts=batches(spec)
    # Annotation maps omit model identity. Enforce one model slug per cohort.
    slugs={Path(f['path']).parent.parent.name for f in spec['traces']}
    if len(slugs)!=1:raise ValueError('Ambiguous cross-model annotation cohort')
    slug=next(iter(slugs));wanted={(d,p) for d,p,h in annotations};matched=set();attempts={};rejects=[];legacy_samples={}
    for info in spec['traces']:
        check_file(info);p=Path(info['path']);filehash=hashlib.sha256()
        with p.open('rb') as f:
            line_number=0
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                filehash.update(line);line_number+=1;r=json.loads(line)
                if (r['dataset'],r['problem_id']) not in wanted:continue
                if r['model_id'].replace('/','--')!=slug:raise ValueError('Source model differs from cohort slug')
                trace=TraceRecord(**r);sha=trace_digest(trace);key=(trace.dataset,trace.problem_id,sha)
                if key not in annotations:continue
                annotation=annotations[key]
                if not annotation_sample(annotation,trace.sample_id,key,legacy_samples):
                    rejects.append({'path':str(p),'line':line_number,'reason':'annotation sample identity mismatch'});continue
                core=base_row(trace);core.pop('features',None);aid=attempt_id(r,sha)
                replay=trace.metadata.get('token_replay',{})
                alignment=all(k in replay for k in ('prompt_token_ids','completion_token_ids'))
                location={**info,'line':line_number,'byte_offset':offset,'line_bytes':len(line),'line_sha256':hashlib.sha256(line).hexdigest()}
                if aid not in attempts:
                    annotation_path=out/'annotations'/f'{aid}.json.gz';annotation_path.parent.mkdir(parents=True,exist_ok=True)
                    raw=json.dumps(annotation,allow_nan=False).encode()
                    with gzip.open(annotation_path,'wb') as stream:stream.write(raw)
                    attempts[aid]={'id':aid,'trace_sha256':sha,'identity':core,'model':trace.model_id,'dataset':trace.dataset,
                        'completion_tokens':len(replay['completion_token_ids']) if alignment else core['completion_tokens'],
                        'exact_token_replay':alignment,'annotation':{'path':str(annotation_path),'sha256':hashlib.sha256(raw).hexdigest()},
                        'locations':[],'metadata_variants':[]}
                a=attempts[aid];a['locations'].append(location)
                if core!=a['identity'] and core not in a['metadata_variants']:a['metadata_variants'].append(core)
                matched.add(key)
        check_file(info);provenance.append({'path':str(p),'sha256':filehash.hexdigest()})
        print(json.dumps({'cohort':spec['id'],'source':str(p),'joined':len(attempts)}),flush=True)
    atomic(out/'attempts.json',list(attempts.values()));atomic(out/'provenance.json',provenance)
    atomic(out/'unmatched.json',[list(k) for k in annotations if k not in matched]);atomic(out/'rejected.json',rejects+conflicts)
    report={'status':'complete','binding':binding,'id':spec['id'],'attempts':len(attempts),'annotation_traces':len(annotations),
        'unmatched':len(annotations)-len(matched),'rejected':len(rejects),'label_conflicts':len(conflicts),
        'metadata_conflicts':sum(bool(a['metadata_variants']) for a in attempts.values())}
    atomic(status,report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--reuse-completed-from',type=Path);p.add_argument('--output',type=Path,default=ROOT/'results/measurement-study/source-join');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full source join requires a Slurm CPU allocation')
    manifest=ROOT/'results/available-metrics/v1/manifest.json';specs=[s for s in json.loads(manifest.read_text())['jobs'] if s['kind']=='labels']
    binding={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.reuse_completed_from:
        previous=json.loads((a.reuse_completed_from/'binding.json').read_text())
        if previous['manifest_sha256']!=binding['manifest_sha256']:raise ValueError('Cannot reuse a changed source manifest')
        if a.reuse_completed_from.resolve()==a.output.resolve():raise ValueError('Use a separate output for code upgrades')
        binding['reuse']={'root':str(a.reuse_completed_from.resolve()),'binding':previous}
    a.output.mkdir(parents=True,exist_ok=True);bp=a.output/'binding.json'
    if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('Resume source/code binding changed')
    atomic(bp,binding);reports=[]
    if a.reuse_completed_from:
        for spec in specs:
            prior=a.reuse_completed_from/spec['id'];status=prior/'status.json';target=a.output/spec['id']
            if not status.exists():continue
            value=json.loads(status.read_text())
            if value['status']!='complete':continue
            if value['binding']!=digest(spec):raise ValueError('Completed cohort source binding changed')
            for info in spec['traces']+spec['annotations']:check_file(info)
            if not (prior/'attempts.json').is_file():raise ValueError('Completed cohort lacks attempt artifact')
            if target.exists():
                if target.resolve()!=prior.resolve():raise ValueError('Existing output conflicts with reuse')
            else:target.symlink_to(prior.resolve(),target_is_directory=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(join_cohort,s,str(a.output)):s['id'] for s in specs}
        for f in as_completed(futures):
            reports.append(f.result());atomic(a.output/'progress.json',reports)
    merged={};logical=defaultdict(set)
    for spec in specs:
        for row in json.loads((a.output/spec['id']/'attempts.json').read_text()):
            aid=row['id'];core=row['identity'];logical[(row['model'],core['trace_id'])].add(aid)
            target=merged.setdefault(aid,{k:v for k,v in row.items() if k not in ('annotation','locations','metadata_variants')})
            target.setdefault('cohorts',[]).append({'id':spec['id'],'annotation':row['annotation'],'locations':row['locations'],
                'identity':core,'metadata_variants':row['metadata_variants']})
    groups=[]
    for model in sorted({r['model'] for r in merged.values()}):
        rs=[r for r in merged.values() if r['model']==model]
        groups.append({'model':model,'unique_content_attempts':len(rs),'completion_tokens':sum(r['completion_tokens'] or 0 for r in rs),
            'exact_replay_attempts':sum(r['exact_token_replay'] for r in rs),'benchmarks':sorted({r['dataset'] for r in rs})})
    result={'status':'source_join_complete_requires_capture_and_resource_acceptance','binding':binding,'cohorts':reports,
        'attempts':list(merged.values()),'model_groups':groups,
        'reused_logical_identities':[{'model':k[0],'trace_id':k[1],'content_attempts':sorted(v)} for k,v in logical.items() if len(v)>1],
        'submission_allowed':False,'pending_primary_labels':['glm','qwen330b']}
    atomic(a.output/'collection.json',result)
    print(json.dumps({k:result[k] for k in ('status','model_groups','reused_logical_identities')}),flush=True)
if __name__=='__main__':main()
