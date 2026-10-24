"""CPU-only completion of class dynamics and the common temporal metric inventory.

Reads immutable production packets. No new generation, labeling, or GPU work.
"""
import argparse, copy, gzip, itertools, json, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from measurement_study.production_plan import ROOT, FAST, SOURCES, sha, atomic
from measurement_study.contract import CLASSES, TEMPORAL


def window_values(window):
    values=dict(window)
    for lag in (1,4,16,64):
        a,b=values.get(f'overlap_lag{lag}'),values.get(f'overlap_shuffled_lag{lag}')
        values[f'overlap_excess_lag{lag}']=a-b if a is not None and b is not None else None
    for family in ('full','mixture'):
        a,b=values.get(family+'_jsd'),values.get(family+'_shuffled_jsd')
        values[family+'_jsd_excess']=a-b if a is not None and b is not None else None
    return values


def class_features(sentences, tables):
    """Zero observed counts are zero; unobserved conditioning sets stay absent."""
    f={};labeled=[s for s in sentences if s['label'] in CLASSES]
    total=sum(s['tokens'] for s in labeled)
    for c in CLASSES:
        if total:f['class_token_fraction/'+c]=sum(s['tokens'] for s in labeled if s['label']==c)/total
    for n in (2,3):
        counts=Counter(tuple(r['pattern']) for r in tables['motifs'] if len(r['pattern'])==n);den=sum(counts.values())
        if den:
            for pattern in itertools.product(CLASSES,repeat=n):f[f'motif_{n}/'+('/'.join(pattern))]=counts[pattern]/den
    contexts=Counter();edges=Counter()
    for r in tables['transitions']:
        if r['previous_class']:
            key=(r['previous_class'],r['source']);contexts[key]+=1;edges[key+(r['destination'],)]+=1
    for key,n in contexts.items():
        for dest in CLASSES:f['previous_conditioned/'+('/'.join(key+(dest,)))]=edges[key+(dest,)]/n
    return f


def temporal_features(windows,sentences,layers):
    """Endpoint contrasts exclude unknown endpoints from BOTH comparison arms."""
    import numpy as np
    labels={token:s['label'] for s in sentences if s['label'] in CLASSES for token in s['token_indices']}
    all_values=defaultdict(list);byclass=defaultdict(list)
    for w in windows:
        values=window_values(w)
        for metric in TEMPORAL:
            value=values.get(metric)
            if isinstance(value,(int,float)) and np.isfinite(value):
                key=(w['layer'],w['window'],metric);all_values[key].append(value)
                label=labels.get(w['end_token']-1)
                if label is not None:byclass[key+(label,)].append(value)
    f={};global_values=defaultdict(list);contrasts={};global_contrasts=defaultdict(list)
    for (layer,width,metric),values in all_values.items():
        value=float(np.mean(values));f[f'temporal_complete/L{layer}/W{width}/{metric}']=value
        global_values[width,metric].append(value)
        for c in CLASSES:
            own=byclass.get((layer,width,metric,c),[])
            other=[v for d in CLASSES if d!=c for v in byclass.get((layer,width,metric,d),[])]
            if own and other:
                delta=float(np.mean(own)-np.mean(other));name=f'{c}/L{layer}/W{width}/{metric}'
                contrasts[name]=dict(value=delta,class_windows=len(own),other_labeled_windows=len(other))
                global_contrasts[c,width,metric].append(delta)
    for (width,metric),v in global_values.items():f[f'temporal_complete_global/W{width}/{metric}']=float(np.mean(v))
    for (c,width,metric),v in global_contrasts.items():f[f'class_routing_contrast/{c}/W{width}/{metric}']=float(np.mean(v))
    return f,contrasts


def analyze_group(args):
    path,out=args
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import analyze_available_metrics as m
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[json.loads(l) for l in Path(path).open()]
    results=[]
    for name in sorted({k for r in rows for k in r['features']}):
        x,y,c,within,support=m.grouped(rows,name)
        for adjustment,controls in [('none',None),('log_length_and_cap',c)]:
            z=m.stats(x,y,controls);z.update(metric=name,family=name.split('/')[0],adjustment=adjustment,
                mean_metric=float(x.mean()) if len(x) else None,positive_questions=sum(support),
                within_question_mixed_outcome_n=len(within),within_question_correct_minus_wrong=float(np.mean(within)) if within else None)
            if name.startswith(('motif_','previous_conditioned/','class_routing_contrast/')) and (sum(support)<10 or sum(y>0)<10 or sum(y<1)<10):
                z['p']=None;z['status']='insufficient_pattern_or_outcome_support'
            results.append(z)
    families=defaultdict(list)
    for z in results:families[z['family'],z['adjustment']].append(z)
    for group in families.values():
        for z,q in zip(group,m.bh_adjust([z.get('p') for z in group])):z['q_bh']=q
    m.write_rows(out/'correlations.jsonl',results)
    result=dict(status='complete',input_sha256=sha(path),attempts=len(rows),tests=len(results));atomic(out/'result.json',result);return result


def main():
    global FAST
    from measurement_study.sentence_schema import install
    install()
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--workers',type=int,default=6);p.add_argument('--worker-smoke',type=Path);a=p.parse_args()
    from measurement_study.production_plan import storage_root
    FAST=storage_root(a.source)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    if a.worker_smoke is not None:
        a.worker_smoke.mkdir(parents=True,exist_ok=True)
        fixture=a.worker_smoke/'empty.jsonl';fixture.write_text('')
        with ProcessPoolExecutor(max_workers=1) as pool:
            result=pool.submit(analyze_group,(str(fixture),str(a.worker_smoke/'worker'))).result()
        if result['status']!='complete':raise RuntimeError('Multiprocessing smoke failed')
        atomic(a.worker_smoke/'result.json',result);return
    import numpy as np
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import sentence_table,saved_layout,class_dynamics,summaries
    from measurement_study.production_analyze import splits
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import analyze_available_metrics as m
    from measurement_study.artifact_layout import result_dir,require_result_variant,capture_binding
    out=result_dir(a.source,'production-supplement');out.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads((FAST/'manifests'/a.source/f'{i}.json').read_text()) for i in range(4)]
    replay_variant=require_result_variant(a.source,manifests)
    binding=capture_binding(FAST,a.source,manifests)
    byid={r['id']:r for manifest in manifests for r in manifest['attempts']};folds=splits(list(byid.values()))
    records=[]
    for manifest in manifests:
        path=FAST/'capture'/a.source/str(manifest['shard'])/'result.json';result=json.loads(path.read_text())
        if result['status']!='complete_native_collection' or result['manifest_binding']!=manifest['binding']:raise ValueError('Incomplete or changed capture')
        records.extend(result['records'])
    if len(records)!=len(byid) or {r['id'] for r in records}!=set(byid):raise ValueError('Incomplete attempt coverage')
    # Thresholds were fitted by the primary analysis on disjoint calibration questions.
    calpath=result_dir(a.source,'production-analysis')/'weak-boundary-calibration.json';cal=json.loads(calpath.read_text())
    if set(cal['question_ids'])!={q for q,f in folds.items() if f=='calibration'}:raise ValueError('Calibration identity mismatch')
    grouped=defaultdict(list);tables=defaultdict(lambda:{k:[] for k in ('transitions','runs','motifs','returns')});class_contrasts=defaultdict(list)
    layers=manifests[0]['spec']['layers'];coverage=defaultdict(Counter)
    for record in records:
        info=record['files'][1]
        if sha(info['path'])!=info['sha256']:raise ValueError('Changed metrics artifact')
        with gzip.open(info['path'],'rt') as f:packet=json.load(f)
        identity=byid[record['id']]['identity'];fold=folds[identity['question_id']];windows=packet['trace']['metadata']['routing_dynamics']['windows']
        for w in windows:
            threshold=cal['thresholds'].get(str(w['layer']))
            if fold!='calibration' and threshold is not None:w['weak_boundary_fraction']=float(np.mean(np.asarray(w['boundary_gaps'])<=threshold))
            for metric,value in window_values(w).items():
                if metric in TEMPORAL and isinstance(value,(int,float)) and np.isfinite(value):coverage[f"L{w['layer']}/W{w['window']}"][metric]+=1
        for cohort,variant in packet['cohorts'].items():
            trace=TraceRecord(**variant['source_trace']);ss=sentence_table(trace,variant['annotation'],saved_layout(trace));d=class_dynamics(ss)
            row=m.base_row(trace);row['features']=class_features(ss,d);features,contrasts=temporal_features(windows,ss,layers);row['features'].update(features)
            row['labeled_sentences']=sum(s['label'] in CLASSES for s in ss);row['transitions']=len(d['transitions']);grouped[cohort,trace.dataset].append(row)
            class_contrasts[cohort].append(dict(id=record['id'],dataset=trace.dataset,question_id=row['question_id'],values=contrasts))
            for name,values in d.items():
                for v in values:v['question_id']=row['question_id']
                tables[cohort][name].extend(values)
        print(json.dumps({'source':a.source,'supplement_attempt':record['id']}),flush=True)
    jobs=[]
    for (cohort,dataset),rs in grouped.items():
        path=out/'compact'/cohort/(dataset+'.jsonl');m.write_rows(path,rs);jobs.append((str(path),str(out/'associations'/cohort/dataset)))
    for cohort,table in tables.items():
        for name,rs in summaries(table).items():m.write_rows(out/'class-dynamics'/cohort/(name+'.jsonl'),rs)
        m.write_rows(out/'class-contrasts'/cohort/'attempts.jsonl',class_contrasts[cohort])
    atomic(out/'coverage.json',{'layers':layers,'temporal':dict(coverage),'expected_temporal_metrics':list(TEMPORAL),'undefined_policy':'Missing history, unknown endpoint labels and absent conditioning groups remain unavailable; no zero imputation'})
    del grouped,tables,class_contrasts
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(analyze_group,j) for j in jobs]):results.append(f.result());atomic(out/'progress.json',dict(completed=len(results),expected=len(jobs)))
    atomic(out/'result.json',dict(status='complete',source=a.source,replay_variant=replay_variant,input_binding=binding,analysis_code_binding=sha(Path(__file__).resolve().parents[1]/'SHA256.json'),attempts=len(records),groups=results,scope=['all common temporal metrics','class token occupancy','second-order transitions','bigrams and trigrams','censor-aware exit hazards','endpoint class routing contrasts on each actual routed layer'],pooled_dynamics_counts='descriptive only; association tests average attempts within question',limitations=['Retrospective class labels; exploratory associations','Endpoint contrasts omit unknown endpoints from both arms','No causal interpretation']))
if __name__=='__main__':main()
