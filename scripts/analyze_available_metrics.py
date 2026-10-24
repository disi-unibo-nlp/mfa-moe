#!/usr/bin/env python3
"""CPU-only, snapshot-bound metrics for available annotation and routing cohorts.

Models, judge versions and sampling campaigns remain separate. Statistical units
are questions, with equal-weight means over available attempts within questions.
"""
import argparse, hashlib, json, math, os, re, subprocess, sys, traceback
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import t as student_t
from analyze_pilot_metrics import (TraceRecord, saved_layout, sentence_table, class_dynamics,
    summaries, keys, termination, digest, trace_digest, write_json, write_rows,
    bh_adjust, metric_values, METRICS, correlation, CLASSES)

REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DATA=REPO/'results/correlation_pipeline'
SCRATCH=Path('/leonardo_scratch/large/userexternal/lmolfett/mfa-moe')
STUDY=Path(__file__).resolve().parents[1]
OUT=None  # The command line must name a fresh campaign-v3 derivative root.
MODELS={'gpt':('gpt-oss-20b','openai--gpt-oss-20b'),
 'gemma':('gemma-nvfp4-nf4','nvidia--Gemma-4-26B-A4B-NVFP4'),
 'qwen36':('qwen36-35b-a3b-fp8','Qwen--Qwen3.6-35B-A3B-FP8'),
 'nemotron':('nemotron-nvfp4-dspark','nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'),
 'glm':('glm-4.7-flash','zai-org--GLM-4.7-Flash'),
 'qwen330b':('qwen3-30b-a3b','Qwen--Qwen3-30B-A3B')}


def file_info(p):
    s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)


def prepare():
    if OUT is None:raise ValueError('An explicit output root is required')
    if (OUT/'manifest.json').exists():raise ValueError('Existing input snapshot is immutable; choose a new output root')
    # rg avoids traversing the tensor contents and returns ignored result paths.
    paths=[Path(x) for x in subprocess.check_output(
        ['rg','--files','--no-ignore',str(DATA)],text=True).splitlines()]
    traces=[p for p in paths if p.name=='traces.jsonl' and '/parts/' not in str(p)
            and '/panels/' not in str(p) and '/code/' not in str(p)]
    jobs=[]; unavailable=[]
    for source,(sub,slug) in MODELS.items():
        for campaign in ('qwen38-mtp3-stratified','qwen38-mtp3-production'):
            root=SCRATCH/campaign/source
            files=sorted(root.glob('part-*/checkpoints/batch-*.json'))
            if not files:continue
            if campaign.endswith('stratified'):
                sd='sampling-100k-stratified-v2' if source=='gpt' else 'sampling-100k-stratified'
                ts=sorted((DATA/sub/'reasoning-vllm-v1'/sd/'generation'/slug).glob('*/traces.jsonl'))
            else:
                ts=sorted((DATA/sub/'generation'/slug).glob('*/traces.jsonl'))
                # Include the source's sampling variants for exact saved segmentation.
                ts+=sorted((DATA/sub/'reasoning-vllm-v1/sampling/generation'/slug).glob('*/traces.jsonl'))
                files+=sorted((SCRATCH/'qwen38-mtp3-production-v2-affected'/source).glob('part-*/checkpoints/batch-*.json'))
            jobs.append(dict(id=campaign+'-'+source,kind='labels',format='batch',
                annotations=[file_info(p) for p in files],traces=[file_info(p) for p in ts]))
        if not any(j['id'].endswith('-'+source) for j in jobs):
            unavailable.append(dict(model=source,status='no_completed_batch_labels_at_snapshot'))
    # Retain historical label campaigns separately; do not count backups twice.
    roots=sorted({p.parents[3] for p in paths if p.parent.name=='shards' and p.suffix=='.json'
                  and 'annotations' in str(p) and 'backup' not in str(p)})
    roots+=sorted({p.parents[2] for p in paths if p.name=='annotations.jsonl'
                  and '/panels/' not in str(p)})
    for root in sorted(set(roots)):
        af=[p for p in paths if p.is_relative_to(root) and
            ((p.parent.name=='shards' and p.suffix=='.json') or p.name=='annotations.jsonl')]
        if not af:continue
        slugs={p.parent.parent.name if p.name=='annotations.jsonl' else p.parents[2].name for p in af}
        ts=[p for p in traces if p.parent.parent.name in slugs]
        jobs.append(dict(id='legacy-'+digest(str(root))[:12],kind='labels',format='annotation',
            source=str(root),annotations=[file_info(p) for p in af],traces=[file_info(p) for p in ts]))
    for p in paths:
        if p.name=='traces_with_routing.jsonl' and '/code/' not in str(p):
            jobs.append(dict(id='routing-'+digest(str(p))[:12],kind='routing',traces=[file_info(p)]))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    code=[Path(__file__),Path(__file__).with_name('analyze_pilot_metrics.py')]
    # Bind the modules actually imported, including a frozen bundle's runtime.
    code += [Path(module.__file__) for name,module in list(sys.modules.items())
             if name.startswith('moe_exp.') and getattr(module,'__file__',None)]
    m=dict(schema_version=1,jobs=jobs,unavailable=unavailable,
      code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
      resource_envelope='Two CPU-only debug jobs; 8 CPUs and 32 GiB each; 30 minutes each; zero GPUs; at most 8 allocated CPU-hours total.',
      limitations=['Snapshot of currently persisted labels; pending labeling jobs are excluded.',
       'Separate campaigns overlap and must never be summed as independent observations.',
       'Sampling is outcome/length dependent; these are exploratory selected-sample associations.',
       'Question means use available attempts; not full benchmark avg@32.',
       'No transitions across gaps; observed run lengths remain censored.',
       'Routing availability is source-specific; saved scalar summaries cannot reconstruct missing distributions.'])
    write_json(OUT/'manifest.json',m);print(json.dumps({'jobs':len(jobs),'kinds':dict(Counter(x['kind'] for x in jobs)), 'unavailable':unavailable}),flush=True)


def read_bytes(info):
    p=Path(info['path']);s=p.stat()
    if s.st_size!=info['size'] or s.st_mtime_ns!=info['mtime_ns']:raise ValueError('Input changed since snapshot: '+str(p))
    return p.read_bytes()


def batches(spec):
    annotations={}; provenance=[]; conflicts=[]
    for info in spec['annotations']:
        raw=read_bytes(info);p=Path(info['path']);provenance.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
        data=json.loads(raw) if p.suffix=='.json' else None
        docs=([data] if data is not None else [json.loads(l) for l in raw.splitlines() if l])
        for doc in docs:
            if spec['format']=='batch':
                classifier={k:v for k,v in doc['binding']['config'].items() if k not in ('base_url','source_plan_sha256')}
                for r in doc['records']:
                    ident=r['identity'];key=(ident['dataset'],ident['problem_id'],ident['trace_sha256'])
                    a=annotations.setdefault(key,dict(schema_version=1,trace_sha256=ident['trace_sha256'],
                        dataset=ident['dataset'],problem_id=ident['problem_id'],sample_id=ident['sample_id'],
                        classifier=classifier,units={}))
                    if a['classifier']!=classifier:raise ValueError('Mixed classifiers in batch cohort')
                    u={**r['unit'],'label':r.get('label')}
                    if r.get('status')=='unknown':u['status']='unknown'
                    idx=u['index'];old=a['units'].get(idx)
                    # Explicit repaired segmentation overrides old records only in repaired source.
                    if old is not None and old!=u and 'production-v2-affected' not in str(p):
                        conflicts.append(dict(identity=ident,reason='conflicting labels'))
                        u={**u,'label':None,'status':'unknown'}
                    a['units'][idx]=u
            else:
                key=(doc['dataset'],doc['problem_id'],doc['trace_sha256'])
                old=annotations.get(key)
                if old is not None and old!=doc:raise ValueError('Conflicting annotation copies')
                annotations[key]=doc
    if spec['format']=='batch':
        for a in annotations.values():a['units']=sorted(a['units'].values(),key=lambda u:u['index'])
    return annotations,provenance,conflicts


def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def base_row(t):
    correct=t.is_correct
    # Unknown outcomes remain unknown; a missing answer is not an adjudication.
    count=t.metadata.get('usage',{}).get('completion_tokens')
    if count is None:count=len(t.metadata.get('token_replay',{}).get('completion_token_ids',[])) or None
    # Prompt identity also groups repeated question IDs with identical prompt text.
    question=t.prompt or str(t.generation_messages)
    return {**keys(t),'question_id':digest([t.dataset,' '.join(question.split())]),
        'source_question_id':keys(t)['question_id'],'is_correct':correct,'completion_tokens':count,
        'capped':termination(t)=='generation_cap','features':{}}


def label_features(t,a):
    a=dict(a)
    if 'sentence_selection' not in a:a['sentence_selection']=t.metadata.get('sentence_selection')
    ss=sentence_table(t,a,saved_layout(t))
    # Preserve gaps: unknown classes never form transitions or zero occupancy.
    for sentence in ss:
        if sentence.get('label') not in CLASSES:sentence['label']=None
        if sentence.get('previous_class') not in CLASSES:sentence['previous_class']=None
    d=class_dynamics(ss);row=base_row(t);f=row['features']
    counts=Counter(s['label'] for s in ss if s['label'] in CLASSES);n=sum(counts.values())
    row.update(sentences=len(ss),labeled_sentences=n,unknown_sentences=len(ss)-n,
        class_fraction_denominator=n,coverage=n/len(ss) if ss else None,
        classifier=a.get('classifier'),transitions=len(d['transitions']))
    tr=Counter((x['source'],x['destination']) for x in d['transitions'])
    for c in CLASSES:
        f['class_fraction/'+c]=counts[c]/n if n else None
        den=sum(tr[c,dst] for dst in CLASSES);change=den-tr[c,c]
        if den:
            ps=[tr[c,dst]/den for dst in CLASSES]
            f['dynamics/'+c+'/transition_entropy']=-sum(p*math.log(p) for p in ps if p)
            f['dynamics/'+c+'/stay_rate']=tr[c,c]/den
            for dst in CLASSES:f['next_sentence/'+c+'/'+dst]=tr[c,dst]/den
        if change:
            for dst in CLASSES:
                if dst!=c:f['class_change/'+c+'/'+dst]=tr[c,dst]/change
        runs=[r for r in d['runs'] if r['label']==c and r['entry_observed']]
        for metric in ('sentences','tokens'):
            vals=[r[metric] for r in runs if r[metric] is not None]
            if vals:f['dynamics/'+c+'/observed_run_'+metric]=float(np.mean(vals))
        for metric in ('return_sentences','return_tokens'):
            vals=[r[metric] for r in d['returns'] if r['label']==c and r[metric] is not None]
            if vals:f['dynamics/'+c+'/'+metric]=float(np.mean(vals))
    for event in ('backtracking_steps','contradiction_steps','self_correction_steps'):
        f['existing_events/'+event]=float(bool(getattr(t.step_labels,event)))
    # Summaries include triples, predecessor conditioning and censor-aware exit hazards.
    return row,d


def grouped(rows,name):
    by=defaultdict(list)
    for r in rows:
        if r['is_correct'] is not None and name in r['features'] and finite(r['features'][name]):
            by[r['question_id']].append(r)
    x=[];y=[];c=[];within=[];support=[]
    for q,rs in sorted(by.items()):
        x.append(np.mean([r['features'][name] for r in rs]));y.append(np.mean([r['is_correct'] for r in rs]))
        c.append([np.mean([math.log(max(1,r['completion_tokens'])) for r in rs]) if all(r['completion_tokens'] is not None for r in rs) else np.nan,
                  np.mean([r['capped'] for r in rs])])
        yes=[r['features'][name] for r in rs if r['is_correct']];no=[r['features'][name] for r in rs if not r['is_correct']]
        if yes and no:within.append(float(np.mean(yes)-np.mean(no)))
        support.append(any(r['features'][name]!=0 for r in rs))
    return np.array(x),np.array(y),np.array(c),within,support


def stats(x,y,c=None):
    if c is not None:
        ok=np.isfinite(c).all(1);x=x[ok];y=y[ok];c=c[ok]
    n=len(x);result=dict(n=n)
    if n<10 or np.std(x)<1e-12 or np.std(y)<1e-12:
        return dict(result,status='insufficient_support')
    rng=np.random.default_rng(42);rs=[];rank=0
    if c is not None:
        c=np.c_[np.ones(n),c];rank=np.linalg.matrix_rank(c)-1
        xr=x-c@np.linalg.lstsq(c,x,rcond=None)[0];yr=y-c@np.linalg.lstsq(c,y,rcond=None)[0]
    else:xr=x;yr=y
    r=float(correlation(xr,yr))
    for _ in range(20):
        ix=rng.integers(0,n,(50,n));xx=x[ix];yy=y[ix]
        if c is not None:
            cc=c[ix];pinv=np.linalg.pinv(cc)
            xx-=np.einsum('bnk,bk->bn',cc,np.einsum('bkn,bn->bk',pinv,xx))
            yy-=np.einsum('bnk,bk->bn',cc,np.einsum('bkn,bn->bk',pinv,yy))
        rs.extend(correlation(xx,yy).tolist())
    rs=np.array(rs);rs=rs[np.isfinite(rs)];df=n-rank-2
    return dict(result,status='descriptive',r=r if math.isfinite(r) else None,
        p=float(2*student_t.sf(abs(r)*math.sqrt(df/max(1e-15,1-r*r)),df)) if df>0 and math.isfinite(r) else None,
        confidence_interval=np.quantile(rs,[.025,.975]).tolist() if len(rs) else None,bootstrap=1000)


def comparisons(rows,out):
    results=[]
    for name in sorted({n for r in rows for n in r['features']}):
        x,y,c,within,support=grouped(rows,name)
        for adj,controls in [('none',None),('log_length_and_cap',c)]:
            z=stats(x,y,controls);z.update(metric=name,family=name.split('/')[0],adjustment=adj,
                mean_metric=float(x.mean()) if len(x) else None,mean_accuracy=float(y.mean()) if len(y) else None,
                positive_questions=sum(support),within_question_mixed_outcome_n=len(within),
                within_question_correct_minus_wrong=float(np.mean(within)) if within else None)
            if name.startswith(('next_sentence/','class_change/')) and (sum(support)<10 or sum(y>0)<10 or sum(y<1)<10):
                z['p']=None;z['status']='insufficient_edge_or_outcome_support'
            results.append(z)
    families=defaultdict(list)
    for z in results:families[z['family'],z['adjustment']].append(z)
    for group in families.values():
        for z,q in zip(group,bh_adjust([z.get('p') for z in group])):z['q_bh']=q
    write_rows(out/'correlations.jsonl',results)
    return dict(tests=len(results),significant_bh=sum(z.get('q_bh') is not None and z['q_bh']<.05 for z in results))


def finish_groups(rows,out):
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['dataset'])].append(r)
    summaries_out=[]
    for (model,dataset),rs in sorted(groups.items()):
        d=out/dataset;write_rows(d/'attempts.jsonl',rs)
        stat=comparisons(rs,d)
        summary=dict(model=model,dataset=dataset,attempts=len(rs),questions=len({r['question_id'] for r in rs}),
            scored_attempts=sum(r['is_correct'] is not None for r in rs),capped=sum(r['capped'] for r in rs),
            labeled_sentences=sum(r.get('labeled_sentences',0) for r in rs),
            adjacent_labeled_transitions=sum(r.get('transitions',0) for r in rs),**stat)
        write_json(d/'summary.json',summary);summaries_out.append(summary)
    return summaries_out


def labels_job(spec,out):
    annotations,prov,conflicts=batches(spec);rows=[];seen=set();errors=[];tablefiles={}
    wanted={(k[0],k[1]) for k in annotations}
    # Each cohort has its own files, so tables from other campaigns cannot collide.
    for name in ('transitions','runs','motifs','returns'):
        tablefiles[name]=(out/(name+'.jsonl')).open('w')
    aggregate={k:[] for k in tablefiles}
    try:
        for info in spec['traces']:
            p=Path(info['path']);st=p.stat()
            if st.st_size!=info['size'] or st.st_mtime_ns!=info['mtime_ns']:raise ValueError('Trace source changed')
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for line in stream:
                    h.update(line);r=json.loads(line)
                    if (r['dataset'],r['problem_id']) not in wanted:continue
                    t=TraceRecord(**r);key=(t.dataset,t.problem_id,trace_digest(t))
                    if key in seen or key not in annotations:continue
                    try:row,d=label_features(t,annotations[key])
                    except ValueError as e:
                        errors.append(dict(trace=list(key),source=str(p),reason=str(e)));continue
                    seen.add(key);rows.append(row)
                    for name,values in d.items():
                        for v in values:tablefiles[name].write(json.dumps(v)+'\n')
                        aggregate[name].extend(values)
            prov.append(dict(path=str(p),sha256=h.hexdigest()))
            print(spec['id'],'joined',len(rows),flush=True)
    finally:
        for f in tablefiles.values():f.close()
    # These are explicitly pooled descriptive counts; inference uses question means.
    for name,rs in summaries(aggregate).items():write_rows(out/(name+'.jsonl'),rs)
    del aggregate
    write_rows(out/'rejected.jsonl',errors+conflicts)
    write_rows(out/'unmatched.jsonl',[dict(identity=list(k)) for k in annotations if k not in seen])
    write_json(out/'provenance.json',prov)
    report=finish_groups(rows,out)
    return dict(status='complete',cohorts=report,annotation_traces=len(annotations),joined=len(seen),
        unmatched=len(annotations)-len(seen),rejected_candidates=len(errors),conflicts=len(conflicts),
        routing_metrics='unavailable in labels-only inputs; see independent routing cohorts')


def routing_features(record,path):
    t=TraceRecord(**record);row=base_row(t);f=row['features'];meta=t.metadata
    row['layers']=t.model_logs.layer_indices;row['forward_provenance']=meta.get('forward_provenance')
    cf=meta.get('correlation_features',{})
    for k,v in cf.get('values',{}).items():
        if finite(v):f['saved_full/'+k]=v
    views=meta.get('correlation_views')
    if views:
        if views['trace_sha256']!=trace_digest(t):raise ValueError('Stale saved views')
        for scope in views['scopes']:
            for k,v in scope['values'].items():
                if finite(v):f['saved_'+scope['view']+'/'+scope['name']+'/'+k]=v
    dyn=meta.get('routing_dynamics');windows=[]
    if dyn:
        if dyn['trace_sha256']!=trace_digest(t) or dyn['forward_provenance']!=meta.get('forward_provenance'):
            raise ValueError('Stale routing dynamics')
        windows=dyn['windows'];row['new_metric_scope']='saved_reasoning_windows'
    else:
        from moe_exp.correlation_pipeline.dynamics.common import artifact_path
        from moe_exp.correlation_pipeline.dynamics.routing import reduce_arrays
        selected=artifact_path(path,t.model_logs.selected_experts)
        if selected:
            import torch
            ids=torch.load(selected,map_location='cpu',weights_only=True)
            if isinstance(ids,dict):raise ValueError('Unsupported selected tensor container')
            # Legacy traces lack exact completion alignment: preserve their saved scope.
            windows=reduce_arrays(ids,layers=t.model_logs.layer_indices,model=t.model_id,
                semantics={'num_experts':cf['num_experts']})
            row['new_metric_scope']='legacy_saved_token_scope_not_verified_reasoning_only'
    buckets=defaultdict(list)
    for w in windows:
        values=metric_values(w)
        for metric,value in zip(METRICS,values):
            if np.isfinite(value):buckets[(w['layer'],w['window'],metric)].append(float(value))
    all_layers=defaultdict(list)
    for (layer,width,metric),vals in buckets.items():
        v=float(np.mean(vals));f[f'new_layer/L{layer}/W{width}/{metric}']=v
        all_layers[(width,metric)].append(v)
    for (width,metric),vals in all_layers.items():f[f'new_global/W{width}/{metric}']=float(np.mean(vals))
    row['windows']=len(windows)
    row['availability']=dict(full_probabilities=bool(dyn),executed_mixture=bool(dyn),
        selected_ids=bool(windows),hidden_scalar_metrics=any(k.startswith('hidden_') for k in cf.get('values',{})),
        weak_boundary='requires calibration-only threshold; not fitted on these outcomes')
    return row


def routing_job(spec,out):
    rows=[];errors=[];prov=[]
    for info in spec['traces']:
        p=Path(info['path']);s=p.stat()
        if s.st_size!=info['size'] or s.st_mtime_ns!=info['mtime_ns']:raise ValueError('Routing input changed')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for i,line in enumerate(stream):
                h.update(line)
                try:rows.append(routing_features(json.loads(line),p))
                except (ValueError,KeyError,FileNotFoundError) as e:errors.append(dict(line=i,reason=str(e)))
                if i%20==0:print(spec['id'],'routing',i,flush=True)
        prov.append(dict(path=str(p),sha256=h.hexdigest()))
    write_rows(out/'rejected.jsonl',errors);write_json(out/'provenance.json',prov)
    return dict(status='complete' if not errors else 'complete_with_rejections',cohorts=finish_groups(rows,out),rejected=len(errors))


def run_one(spec,output_root=None):
    from measurement_study.production_analyze import claim_output,checked_complete,outcome_fingerprint
    root=Path(output_root) if output_root is not None else OUT
    if root is None:raise ValueError('An explicit output root is required')
    out=root/spec['id']
    # This standalone path audits historical cohorts; production_analyze owns
    # the corrected v3 sidecar join and must be used for scientific v3 results.
    paths=[Path(info['path']) for kind in ('traces','annotations') for info in spec.get(kind,[])]
    binding=dict(schema_version=3,spec=spec,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        outcome_binding=outcome_fingerprint(spec.get('outcome_binding')),
        code_hashes={str(Path(module.__file__).resolve()):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for name,module in list(sys.modules.items()) if getattr(module,'__file__',None)
                     and (name.startswith('moe_exp.') or name in ('analyze_pilot_metrics','analyze_available_metrics'))})
    binding['code_hashes'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    completed=checked_complete(out/'status.json',binding) if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete' else None
    if completed is not None:return dict(id=spec['id'],**completed)
    claim_output(out,binding)
    try:
        write_json(out/'status.json',dict(status='running',job_id=os.environ['SLURM_JOB_ID'],analysis_binding=binding))
        result=labels_job(spec,out) if spec['kind']=='labels' else routing_job(spec,out)
    except Exception as e:
        result=dict(status='failed',error=str(e),traceback=traceback.format_exc())
    result.update(analysis_binding=binding,outcome_scope='legacy_stored_outcomes_diagnostic_only')
    result['output_hashes']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*')
                             if p.is_file() and p.name not in ('status.json','analysis-binding.json')}
    write_json(out/'status.json',result);print(spec['id'],result['status'],flush=True)
    return dict(id=spec['id'],**result)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','labels','routing']);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    from measurement_study.production_analyze import fresh_output_root
    OUT=fresh_output_root(a.output_root)
    if a.mode=='prepare':prepare();return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Broad analysis requires Slurm')
    manifest=json.loads((OUT/'manifest.json').read_text())
    for file,sha in manifest['code_hashes'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:raise ValueError('Code changed after snapshot: '+file)
    jobs=[j for j in manifest['jobs'] if j['kind']==a.mode]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[f.result() for f in as_completed([pool.submit(run_one,j,str(OUT)) for j in jobs])]
    write_json(OUT/(a.mode+'-status.json'),dict(status='complete' if all(r['status'].startswith('complete') for r in results) else 'failed',jobs=results))
    if any(r['status']=='failed' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
