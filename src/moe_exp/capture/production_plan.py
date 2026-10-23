"""Frozen per-model input manifests and four balanced, resumable shards."""
import argparse, gzip, hashlib, json, os
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')
FAST=Path(os.environ.get('MEASUREMENT_STORAGE_ROOT', '/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/measurement-study-v1'))
SOURCES=('gpt','glm','qwen330b','qwen36','nemotron','gemma','qwen35')
HOURS={'gpt':6,'glm':6,'qwen330b':8,'qwen36':12,'nemotron':6}
CAP_GIB={'gpt':10,'glm':30,'qwen330b':35,'qwen36':50,'nemotron':15,'gemma':20}

def storage_root(source):
    override=os.environ.get('MEASUREMENT_STORAGE_ROOT')
    if override:return Path(override)
    if source in ('gemma','nemotron','qwen330b'):return ROOT/'results/measurement-study/recovery-storage'
    return FAST

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False)+'\n');tmp.replace(path)
def read_location(info):
    path=Path(info['path']);st=path.stat()
    if (st.st_size,st.st_mtime_ns)!=(info['size'],info['mtime_ns']):raise ValueError('Source changed: '+str(path))
    with path.open('rb') as f:f.seek(info['byte_offset']);raw=f.read(info['line_bytes'])
    if hashlib.sha256(raw).hexdigest()!=info['line_sha256']:raise ValueError('Source line checksum changed')
    result=json.loads(raw)
    if 'sentence_schema' in info:result.setdefault('metadata',{})['measurement_sentence_schema']=info['sentence_schema']
    if 'sentence_selection' in info:result.setdefault('metadata',{})['sentence_selection']=info['sentence_selection']
    return result
def bind_cohort_location(cohort, annotation):
    """Select the content-bound source copy with the annotation's sampling identity."""
    for i,loc in enumerate(cohort['locations']):
        raw=read_location(loc)
        if 'sentence_selection' in annotation and raw['metadata'].get('sentence_selection')!=annotation['sentence_selection']:
            continue
        if 'units' in annotation:
            from moe_exp.schemas import TraceRecord
            from moe_exp.capture.sentence_schema import infer_schema
            try:version=infer_schema(TraceRecord(**raw),annotation)
            except ValueError:continue
            loc['sentence_schema']=version
            raw.setdefault('metadata',{})['measurement_sentence_schema']=version
            cohort['sentence_schema']=version
        cohort['locations']=[loc]+cohort['locations'][:i]+cohort['locations'][i+1:]
        return raw
    raise ValueError('No source copy matches the explicit annotation sentence selection')

def normalize_annotation(trace, annotation):
    """Represent unlabeled sentences explicitly; never invent or overwrite labels."""
    import copy
    from moe_exp.capture.sentence_schema import install,infer_schema
    install();infer_schema(trace,annotation)
    from moe_exp.labeling.spans import sentence_spans, selected_sentence_indices, validate_annotation
    a=copy.deepcopy(annotation)
    # Compact batch joins omit this field; the content-bound source retains it.
    if 'sentence_selection' not in a:a['sentence_selection']=copy.deepcopy(trace.metadata.get('sentence_selection'))
    units=sentence_spans(trace)
    expected=[units[i] for i in selected_sentence_indices(trace,units)]
    if a.get('sentence_selection')!=trace.metadata.get('sentence_selection'):raise ValueError('Annotation selection mismatch')
    byindex={u['index']:u for u in a['units']}
    if len(byindex)!=len(a['units']) or set(byindex)-{u['index'] for u in expected}:raise ValueError('Invalid annotation unit identities')
    normalized=[]
    for unit in expected:
        saved=byindex.get(unit['index'])
        if saved is not None and any(saved.get(k)!=v for k,v in unit.items()):raise ValueError('Annotation offsets/text changed')
        saved=copy.deepcopy(saved) if saved is not None else dict(unit,status='unknown')
        if saved.get('status')=='unknown' or saved.get('label') is None:
            saved.pop('label',None);saved['status']='unknown';saved.setdefault('raw_completion',None)
            saved.setdefault('failure',dict(kind='unavailable_source_annotation',error_type='MissingLabelMetadata',message='No canonical label in the saved source cohort; original checkpoint remains unchanged',finish_reason=None))
        normalized.append(saved)
    a['units']=normalized;validate_annotation(trace,a);return a

def balanced(rows,n=4):
    shards=[[] for _ in range(n)];sizes=[0]*n
    for row in sorted(rows,key=lambda r:(-r['completion_tokens'],r['id'])):
        i=min(range(n),key=lambda j:(sizes[j],j));shards[i].append(row);sizes[i]+=row['completion_tokens']
    return shards

def main():
    global FAST
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--cohort-manifest',type=Path);p.add_argument('--outcomes',type=Path)
    p.add_argument('--annotations',type=Path,nargs='+',help='Shared annotation_batch JSON exports for selected cohort labels')
    p.add_argument('--outcome-manifest',type=Path);p.add_argument('--storage-root',type=Path)
    p.add_argument('--verification-receipt',type=Path,help='Accepted verify-only receipt used to fit the per-attempt runtime')
    p.add_argument('--cohort',choices=('A','B','all'),default='all')
    p.add_argument('--p90-seconds-per-token',type=float,help='Explicit measured per-token seconds (fitted P90 rate when no receipt is passed)')
    p.add_argument('--startup-seconds',type=float)
    p.add_argument('--per-attempt-seconds',type=float);p.add_argument('--bytes-per-token',type=float)
    p.add_argument('--verification-only',action='store_true');p.add_argument('--verification-seconds',type=int)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    if a.cohort_manifest is not None:
        from moe_exp.capture.campaign_v3_plan import build_manifests,publish_manifests
        if a.storage_root is None:
            p.error('v3 planning requires --storage-root')
        measured=any(v is not None for v in (a.p90_seconds_per_token,a.startup_seconds,a.per_attempt_seconds,a.bytes_per_token))
        if a.verification_only and measured:
            p.error('bootstrap verification is label-free and unmeasured; omit explicit measured runtime values')
        if not a.verification_only and not (a.verification_receipt is not None or measured):
            p.error('production planning requires --verification-receipt or explicit --startup-seconds/--per-attempt-seconds/--p90-seconds-per-token/--bytes-per-token')
        if a.verification_receipt is not None and measured:
            p.error('pass either --verification-receipt or explicit measured values, not both')
        if not a.verification_only and a.verification_receipt is None and any(v is None for v in (a.startup_seconds,a.per_attempt_seconds,a.p90_seconds_per_token,a.bytes_per_token)):
            p.error('explicit production sizing needs --startup-seconds, --per-attempt-seconds, --p90-seconds-per-token and --bytes-per-token')
        if a.verification_only and a.verification_seconds is None:
            a.verification_seconds=5*3600
        inventory=json.loads((a.bundle/'SHA256.json').read_text())
        for name,expected in inventory.items():
            if sha(a.bundle/name)!=expected:raise ValueError('Frozen bundle changed: '+name)
        campaign=json.loads((a.bundle/'campaign.json').read_text())
        manifests,verification,ready=build_manifests(cohort_path=a.cohort_manifest,outcomes_path=a.outcomes,
            outcome_manifest_path=a.outcome_manifest,spec=campaign['sources'][a.source],source=a.source,
            storage_root=a.storage_root,code_binding=sha(a.bundle/'SHA256.json'),selection=a.cohort,
            p90_seconds_per_token=a.p90_seconds_per_token,startup_seconds=a.startup_seconds,
            per_attempt_seconds=a.per_attempt_seconds,bytes_per_token=a.bytes_per_token,
            verification_receipt_path=a.verification_receipt,
            verification_only=a.verification_only,verification_seconds=a.verification_seconds,annotation_paths=a.annotations)
        destination=publish_manifests(a.storage_root,a.source,a.cohort,manifests,verification,ready)
        print(json.dumps(dict(status=ready['status'],manifest_dir=str(destination),attempts=ready['attempts'],shards=len(manifests))),flush=True)
        return
    if a.outcomes is not None or a.storage_root is not None or a.source=='qwen35':
        p.error('Explicit campaign-v3 inputs require --cohort-manifest; qwen35 has no legacy plan')
    FAST=storage_root(a.source)
    from moe_exp.capture.sentence_schema import install
    install()
    from moe_exp.schemas import TraceRecord
    from moe_exp.labeling.spans import trace_digest, validate_annotation
    from moe_exp.capture.dynamics.classes import saved_layout
    campaign=json.loads((a.bundle/'campaign.json').read_text());spec=campaign['sources'][a.source]
    for path,expected in campaign['runtime_code_hashes'].items():
        if sha(path)!=expected:raise ValueError('Reducer code changed: '+path)
    if a.source in ('glm','qwen330b'):
        from moe_exp.capture.join_collection import join_cohort
        path=ROOT/'results/measurement-study/completed-labels'/a.source/'input-manifest.json'
        label_spec=json.loads(path.read_text());joinroot=FAST/'source-joins'
        report=join_cohort(label_spec,joinroot)
        if report['unmatched'] or report['rejected'] or report['label_conflicts']:raise ValueError('Incomplete source join')
        rows=json.loads((joinroot/label_spec['id']/'attempts.json').read_text())
        for row in rows:row['cohorts']=[dict(id=label_spec['id'],annotation=row.pop('annotation'),locations=row.pop('locations'),identity=row['identity'],metadata_variants=row.pop('metadata_variants'))]
        source_binding=sha(path)
    else:
        path=ROOT/'results/measurement-study/source-join-v2/collection.json'
        if sha(path)!='eb07af23d96dfdec344f9e0ecb25910b0034ef3421d9e81497874ab992b20fa2':raise ValueError('Frozen source inventory changed')
        if a.source=='gemma':
            path=FAST/'source-joins/gemma-segmentation-v1/collection.json'
            receipt=json.loads(path.with_name('acceptance.json').read_text())
            if receipt['status']!='complete' or receipt['collection_sha256']!=sha(path):raise ValueError('Unaccepted Gemma source repair')
            if receipt['code_binding']!=sha(a.bundle/'SHA256.json'):raise ValueError('Gemma source repair code binding changed')
        rows=[r for r in json.loads(path.read_text())['attempts'] if r['model']==spec['generation_model']];source_binding=sha(path)
    tokenizer=None
    if a.source=='gemma':
        from transformers import AutoTokenizer
        from moe_exp.models.token_replay import validate_token_replay
        tokenizer=AutoTokenizer.from_pretrained(spec['checkpoint'],local_files_only=True)
    corpus=[];maximum=0
    for row in rows:
        if not row['exact_token_replay']:raise ValueError('Missing saved token IDs')
        first=None
        for cohort in row['cohorts']:
            ar=gzip.decompress(Path(cohort['annotation']['path']).read_bytes())
            if hashlib.sha256(ar).hexdigest()!=cohort['annotation']['sha256']:raise ValueError('Annotation changed')
            original=json.loads(ar);trace=TraceRecord(**bind_cohort_location(cohort,original))
            if trace_digest(trace)!=row['trace_sha256'] or trace.model_id!=spec['generation_model']:raise ValueError('Source identity mismatch')
            annotation=normalize_annotation(trace,original);validate_annotation(trace,annotation)
            if first is None:first=trace
        replay=first.metadata['token_replay'];layout=saved_layout(first)
        if tokenizer is not None:validate_token_replay(replay,tokenizer,first.cot_text)
        if layout['reasoning_tokens'] is None:raise ValueError('Missing saved offsets')
        n=len(replay['prompt_token_ids'])+len(replay['completion_token_ids']);maximum=max(maximum,n)
        row['total_input_tokens']=n;row['reasoning_tokens']=len(layout['reasoning_tokens'])
        corpus.append((first.dataset,first.problem_id,row['trace_sha256'],row['reasoning_tokens']))
    reference=dict(model_id=spec['generation_model'],mean_reasoning_tokens=sum(x[-1] for x in corpus)/len(corpus),traces=len(corpus),bins=10,corpus_sha256=digest(sorted(corpus)),rule='ceil(model_corpus_mean * i / bins); noncumulative; overflow after mean',limit_applied_to_reference=False)
    if maximum+1>49152:raise ValueError('Saved trace exceeds verified native context capacity; retain it for explicit recovery')
    base=dict(schema_version=1,source=a.source,spec=spec,source_binding=source_binding,code_binding=sha(a.bundle/'SHA256.json'),position_reference=reference,maximum_input_tokens=maximum,cohorts_separate=True)
    shards=balanced(rows)
    for i,rs in enumerate(shards):
        m={**base,'shard':i,'attempts':rs,'expected_attempts':len(rs),'expected_tokens':sum(r['completion_tokens'] for r in rs),'artifact_cap_bytes':CAP_GIB[a.source]*1024**3//4,'transient_cap_bytes':15*1024**3}
        m['binding']=digest(m);atomic(FAST/'manifests'/a.source/f'{i}.json',m)
    result={**base,'status':'ready','tokenizer_saved_ids_verified':tokenizer is not None,'verification_shard':max(range(4),key=lambda i:max(r['total_input_tokens'] for r in shards[i])),'sentence_schema_counts':{str(v):sum(c.get('sentence_schema',2)==v for r in rows for c in r['cohorts']) for v in (1,2)},'attempts':len(rows),'tokens':sum(r['completion_tokens'] for r in rows),'shards':[dict(index=i,attempts=len(rs),tokens=sum(r['completion_tokens'] for r in rs)) for i,rs in enumerate(shards)]}
    atomic(FAST/'manifests'/a.source/'ready.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('spec','position_reference')}),flush=True)
if __name__=='__main__':main()
