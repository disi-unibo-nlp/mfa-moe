#!/usr/bin/env python3
"""Checkpointed, CPU-only descriptive analysis of immutable pilot routing summaries."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr, t as student_t
from moe_exp.schemas import TraceRecord
from moe_exp.correlation_pipeline.dynamics.classes import saved_layout, sentence_table, class_dynamics, summaries
from moe_exp.correlation_pipeline.dynamics.common import CLASSES, keys, termination, digest, write_json, write_rows
from moe_exp.correlation_pipeline.dynamics.evaluation import bh_adjust
from moe_exp.correlation_pipeline.spans import trace_digest

METRICS = ('selection_entropy','boundary_gap_mean','full_local_entropy',
 'full_marginal_entropy','full_entropy_difference','full_jsd','full_shuffled_jsd',
 'mixture_local_entropy','mixture_marginal_entropy','mixture_entropy_difference',
 'mixture_jsd','mixture_shuffled_jsd','mixture_effective_experts','set_turnover',
 'overlap_lag1','overlap_lag4','overlap_lag16','overlap_lag64',
 'overlap_excess_lag1','overlap_excess_lag4','overlap_excess_lag16','overlap_excess_lag64',
 'full_jsd_excess','mixture_jsd_excess')
LANDMARKS = (256,512,1024,2048)


def metric_values(window):
    out = dict(window)
    for lag in (1,4,16,64):
        a,b = window.get(f'overlap_lag{lag}'),window.get(f'overlap_shuffled_lag{lag}')
        out[f'overlap_excess_lag{lag}'] = a-b if a is not None and b is not None else None
    for family in ('full','mixture'):
        a,b = window.get(family+'_jsd'),window.get(family+'_shuffled_jsd')
        out[family+'_jsd_excess'] = a-b if a is not None and b is not None else None
    return np.array([out.get(k) if out.get(k) is not None else np.nan for k in METRICS],float)


def add(bucket,key,values):
    if key not in bucket:
        bucket[key]=[np.zeros(len(values)),np.zeros(len(values),dtype=int)]
    finite=np.isfinite(values)
    bucket[key][0][finite]+=values[finite]
    bucket[key][1][finite]+=1


def mean(pair):
    return np.divide(pair[0],pair[1],out=np.full(len(pair[0]),np.nan),where=pair[1]>0)


def compact(record):
    trace=TraceRecord(**record)
    data=trace.metadata['routing_dynamics']
    if data['trace_sha256']!=trace_digest(trace): raise ValueError('Stale routing data')
    if data['annotation_sha256']!=digest(trace.metadata['reasoning_annotation']): raise ValueError('Stale annotation')
    if data['forward_provenance']!=trace.metadata['forward_provenance']: raise ValueError('Checkpoint mismatch')
    if trace.model_logs.layer_indices!=list(range(40)): raise ValueError('Missing layers')
    layout=saved_layout(trace)
    sentences=sentence_table(trace,layout=layout)
    labels=np.full(layout['token_count'],-1,dtype=int)
    for s in sentences:
        if s['label'] in CLASSES: labels[s['token_indices']]=CLASSES.index(s['label'])
    accum,by_class={},{}
    for w in data['windows']:
        values=metric_values(w); width=w['window']; layer=w['layer']; end=w['end_token']
        add(accum,('full',layer,width),values)
        for mark in LANDMARKS:
            if end<=mark: add(accum,(str(mark),layer,width),values)
        label=labels[end-1]
        if label>=0: add(by_class,(CLASSES[label],layer,width),values)
    features={}
    for (scope,layer,width), pair in accum.items():
        # Prefix analyses exclude traces ending before the landmark.
        if scope!='full' and layout['token_count']<int(scope): continue
        for name,value in zip(METRICS,mean(pair)):
            if np.isfinite(value): features[f'{scope}/L{layer}/W{width}/{name}']=float(value)
    for scope in ('full',*(str(x) for x in LANDMARKS)):
        if scope!='full' and layout['token_count']<int(scope): continue
        for width in (64,256):
            vals=[mean(accum[(scope,l,width)]) for l in range(40) if (scope,l,width) in accum]
            if vals:
                for name,value in zip(METRICS,np.nanmean(vals,axis=0)):
                    if np.isfinite(value): features[f'{scope}/all/W{width}/{name}']=float(value)
    # Within-question, current endpoint-class versus other endpoint classes.
    contrasts={}
    for label in CLASSES:
        for width in (64,256):
            pairs=[]
            for layer in range(40):
                a=by_class.get((label,layer,width)); total=accum.get(('full',layer,width))
                if a is not None and total is not None:
                    other=[total[0]-a[0],total[1]-a[1]]
                    pairs.append(mean(a)-mean(other))
            if pairs:
                for name,value in zip(METRICS,np.nanmean(pairs,axis=0)):
                    if np.isfinite(value): contrasts[f'{label}/W{width}/{name}']=float(value)
    counts=Counter(s['label'] for s in sentences)
    for label in CLASSES:
        features['class_sentence_fraction/'+label]=counts[label]/max(1,len(sentences))
        features['class_token_fraction/'+label]=float(np.sum(labels==CLASSES.index(label))/max(1,np.sum(labels>=0)))
    tables=class_dynamics(sentences)
    run_lengths=[x['sentences'] for x in tables['runs'] if x['entry_observed']]
    features['class_dynamics/mean_run_sentences']=float(np.mean(run_lengths)) if run_lengths else 0.
    features['class_dynamics/class_change_fraction']=sum(x['source']!=x['destination'] for x in tables['transitions'])/max(1,len(tables['transitions']))
    features['length/log_completion_tokens']=float(np.log(max(1,layout['token_count'])))
    correct=trace.is_correct
    if correct is None and trace.gold_answer and not trace.model_answer: correct=False
    return {**keys(trace),'is_correct':correct,'completion_tokens':layout['token_count'],
      'reasoning_tokens':len(layout['reasoning_tokens']),'capped':termination(trace)=='generation_cap',
      'windows':len(data['windows']),'sentences':len(sentences),'classes':dict(counts),
      'features':features,'class_contrasts':contrasts,'transitions':tables['transitions'],
      'runs':tables['runs'],'forward_provenance':trace.metadata['forward_provenance']}


def correlation(x,y):
    x=x-x.mean(axis=-1,keepdims=True); y=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((x*x).sum(axis=-1)*(y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-14)


def statistics(x,y,controls=None,bootstrap=1000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<10 or min(np.sum(y==0),np.sum(y==1))<3 or np.std(x)<1e-12:
        return {'status':'insufficient_support','n':len(x)}
    rng=np.random.default_rng(42)
    ix=rng.integers(0,len(x),(bootstrap,len(x)))
    xb,yb=x[ix],y[ix]
    rank=0
    if controls is not None:
        c=np.c_[np.ones(len(x)),np.asarray(controls,float)]
        rank=np.linalg.matrix_rank(c)-1
        x=x-c@np.linalg.lstsq(c,x,rcond=None)[0]
        y=y-c@np.linalg.lstsq(c,y,rcond=None)[0]
        cb=c[ix]; pinv=np.linalg.pinv(cb)
        xb=xb-np.einsum('bnk,bk->bn',cb,np.einsum('bkn,bn->bk',pinv,xb))
        yb=yb-np.einsum('bnk,bk->bn',cb,np.einsum('bkn,bn->bk',pinv,yb))
    r=float(correlation(x,y))
    reps=correlation(xb,yb); reps=reps[np.isfinite(reps)]
    df=len(x)-rank-2
    p=float(2*student_t.sf(abs(r)*math.sqrt(df/max(1e-15,1-r*r)),df)) if df>0 and math.isfinite(r) else None
    return {'status':'descriptive','n':len(x),'r':r if math.isfinite(r) else None,
      'p':p,'confidence_interval':np.quantile(reps,[.025,.975]).tolist() if len(reps) else None,
      'valid_bootstraps':len(reps),'bootstrap':bootstrap}


def analyze(rows,out):
    scored=[r for r in rows if r['is_correct'] is not None]
    if len({r['question_id'] for r in scored})!=len(scored):
        raise ValueError('This pilot analyzer requires one attempt per distinct question')
    # Layer scans use full traces only; prefix scans use all-layer means to limit searches.
    names=sorted({k for r in scored for k in r['features'] if k.startswith(('full/','class_','length/')) or '/all/' in k})
    results=[]
    for number,name in enumerate(names):
        use=[r for r in scored if name in r['features']]
        x=[r['features'][name] for r in use]; y=[int(r['is_correct']) for r in use]
        family=('layer_full' if name.startswith('full/L') else 'global_full' if name.startswith('full/') else
                'prefix' if '/all/' in name else 'class' if name.startswith('class_') else 'length')
        raw=statistics(x,y)
        raw.update(metric=name,family=family,adjustment='none',correct=sum(y),incorrect=len(y)-sum(y))
        raw['spearman_r']=float(spearmanr(x,y).statistic) if len(x)>2 and np.std(x)>1e-12 and np.std(y)>0 else None
        results.append(raw)
        if not name.startswith('length/') and not family=='prefix':
            c=[[np.log(r['completion_tokens']),int(r['capped'])] for r in use]
            adjusted=statistics(x,y,c)
            adjusted.update(metric=name,family=family,adjustment='log_length_and_cap',correct=sum(y),incorrect=len(y)-sum(y))
            results.append(adjusted)
        if number%100==0:
            write_json(out/'progress.json',dict(stage='correlations',features_done=number,features_total=len(names)))
            print('correlations',number,'/',len(names),flush=True)
    families=defaultdict(list)
    for r in results: families[(r['family'],r['adjustment'])].append(r)
    for group in families.values():
        for r,q in zip(group,bh_adjust([r.get('p') for r in group])): r['q_bh']=q
    write_rows(out/'correlations.jsonl',results)
    contrasts=[]
    for name in sorted({k for r in rows for k in r['class_contrasts']}):
        x=np.array([r['class_contrasts'][name] for r in rows if name in r['class_contrasts']])
        rng=np.random.default_rng(42)
        samples=x[rng.integers(0,len(x),(1000,len(x)))].mean(1)
        p=min(1.,2*(min(np.sum(samples<=0),np.sum(samples>=0))+1)/1001)
        contrasts.append(dict(metric=name,questions=len(x),mean_within_question_difference=float(x.mean()),
            confidence_interval=np.quantile(samples,[.025,.975]).tolist(),p=float(p) if len(x)>=10 else None,
            status='descriptive' if len(x)>=10 else 'insufficient_support'))
    for r,q in zip(contrasts,bh_adjust([r['p'] for r in contrasts])): r['q_bh']=q
    write_rows(out/'class_metric_contrasts.jsonl',contrasts)
    tables={'transitions':[v for r in rows for v in r['transitions']],
            'runs':[v for r in rows for v in r['runs']],'motifs':[],'returns':[]}
    for name,records in summaries(tables).items(): write_rows(out/(name+'.jsonl'),records)
    summary=dict(status='complete',questions=len(rows),scored=len(scored),correct=sum(r['is_correct'] for r in scored),
        capped=sum(r['capped'] for r in rows),completion_tokens=sum(r['completion_tokens'] for r in rows),
        sentences=sum(r['sentences'] for r in rows),windows=sum(r['windows'] for r in rows),
        correlation_tests=len(results),class_contrasts=len(contrasts),
        limitations=['MATH-500 pilot only, 64 length-stratified questions; one attempt per question.',
          'NF4 replay of FP8 generations; no claim of numerical equivalence.',
          'Full-trace and class associations are retrospective and noncausal.',
          'Length/cap adjustment is a sensitivity check, not an online predictor or causal identification.',
          'Windows are averaged within each question; 640800 windows are not independent samples.',
          'Endpoint-class contrasts compare trailing windows whose endpoint belongs to that class; windows can cross class boundaries.',
          'Prefix results condition on the trace reaching each completion-token landmark; no future routing is used.',
          'No expert-specific rankings or nested predictive validation in this compact analysis.',
          'Bootstrap intervals resample questions 1000 times; correlation p-values use the conventional t reference.',
          'BH correction is separate for global/full, layer/full, prefixes, classes and length, and raw versus adjusted families.'])
    write_json(out/'summary.json',summary)
    selected=sorted([r for r in results if r['family']=='global_full' and r['adjustment']=='none' and r.get('r') is not None],key=lambda r:-abs(r['r']))
    lines=['# Pilot metric correlations','',json.dumps(summary,indent=2),'',
      '| Metric | Questions | r | 95% question-bootstrap CI | BH q |','|---|---:|---:|---|---:|']
    for r in selected:
        lines.append(f"| {r['metric']} | {r['n']} | {r['r']:+.3f} | {r['confidence_interval']} | {r['q_bh']:.4g} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    show=selected[:16][::-1]
    for i,r in enumerate(show):
        lo,hi=r['confidence_interval']; ax.plot([lo,hi],[i,i],color='tab:blue'); ax.scatter([r['r']],[i],color='tab:blue')
    ax.set_yticks(range(len(show)),[r['metric'].replace('full/all/','') for r in show]); ax.axvline(0,color='gray',ls='--')
    ax.set_xlabel('Correlation with correctness (question bootstrap 95% CI)')
    ax.set_title('Qwen3.6 MATH-500 pilot: retrospective NF4 routing associations')
    fig.tight_layout();fig.savefig(out/'correlations.png',dpi=180);fig.savefig(out/'correlations.pdf');plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Analysis must run in Slurm')
    path=args.root/'forward/unsloth--Qwen3.6-35B-A3B/math500/traces_with_routing.jsonl'
    acceptance=json.loads((args.root/'acceptance.json').read_text())
    if acceptance['status']!='complete': raise ValueError('Pilot replay not accepted')
    args.output.mkdir(parents=True,exist_ok=True)
    binding=dict(input_sha256=acceptance['forward_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics=list(METRICS),bootstrap=1000,seed=42)
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['binding']!=binding: raise ValueError('Changed analysis contract')
    write_json(manifest,dict(status='running',binding=binding))
    h=hashlib.sha256();rows=[]
    with path.open('rb') as f:
        for i,line in enumerate(f):
            h.update(line)
            checkpoint=args.output/f'trace-{i:03d}.json'
            if checkpoint.exists(): row=json.loads(checkpoint.read_text())
            else:
                row=compact(json.loads(line));write_json(checkpoint,row)
            rows.append(row)
            print('reduced trace',i+1,'windows',row['windows'],flush=True)
            write_json(args.output/'progress.json',dict(stage='reduce',traces=len(rows),expected=acceptance['traces']))
    if h.hexdigest()!=binding['input_sha256'] or len(rows)!=acceptance['traces']: raise ValueError('Input acceptance changed')
    result=analyze(rows,args.output)
    write_json(manifest,dict(status='complete',binding=binding,summary=result))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
