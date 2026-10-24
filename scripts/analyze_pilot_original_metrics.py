#!/usr/bin/env python3
"""Complete original routing metrics and class dynamics using saved pilot artifacts."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,os,math
from pathlib import Path
import numpy as np
from analyze_pilot_metrics import statistics,CLASSES,TraceRecord,saved_layout,sentence_table,class_dynamics,keys,write_json,write_rows,bh_adjust,digest,trace_digest


def extract(record,base):
    t=TraceRecord(**record);v=t.metadata['correlation_views']
    if v['trace_sha256']!=trace_digest(t): raise ValueError('Stale view')
    if v['annotation_sha256']!=digest(t.metadata['reasoning_annotation']): raise ValueError('Stale view annotations')
    f={};contrasts={}
    full=next(x for x in v['scopes'] if x['view']=='full')
    for scope in v['scopes']:
        if scope['view'] not in ('full','class'): continue
        for name,value in scope['values'].items():
            # Confidence is an affine transform of entropy; do not double-test both.
            if value is None or not math.isfinite(value) or name.startswith('router_entropy'): continue
            f[f"{scope['view']}/{scope['name']}/{name}"]=value
            if scope['view']=='class' and name.endswith('_mean_layers'):
                # Valid within-question class minus all-reasoning contrast (not class-vs-rest).
                fv=full['values'].get(name)
                if fv is not None: contrasts[f"{scope['name']}/{name}"]=value-fv
    s=sentence_table(t,layout=saved_layout(t));d=class_dynamics(s)
    for label in CLASSES:
        trs=[x for x in d['transitions'] if x['source']==label]
        if trs:
            counts=Counter(x['destination'] for x in trs);n=len(trs)
            f[f'dynamics/{label}/stay_rate']=counts[label]/n
            f[f'dynamics/{label}/transition_entropy']=-sum((c/n)*np.log(c/n) for c in counts.values())
        runs=[x for x in d['runs'] if x['label']==label and x['entry_observed']]
        if runs:
            f[f'dynamics/{label}/observed_run_sentences']=float(np.mean([x['sentences'] for x in runs]))
            f[f'dynamics/{label}/observed_run_tokens']=float(np.mean([x['tokens'] for x in runs]))
        returns=[x for x in d['returns'] if x['label']==label]
        if returns:
            f[f'dynamics/{label}/return_interval_sentences']=float(np.mean([x['return_sentences'] for x in returns]))
    # Event metadata is retrospective and does not establish causal or prospective utility.
    events={k:bool(getattr(t.step_labels,k)) for k in ('backtracking_steps','contradiction_steps','self_correction_steps')}
    return {k:base[k] for k in ('trace_id','question_id','is_correct','completion_tokens','capped')} | dict(features=f,contrasts=contrasts,events=events)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    a.output.mkdir(parents=True,exist_ok=True)
    expected=json.loads((a.root/'acceptance.json').read_text())['forward_sha256']
    binding=dict(source_sha256=expected,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics_sha256=hashlib.sha256(Path(__file__).with_name('analyze_pilot_metrics.py').read_bytes()).hexdigest())
    manifest=a.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['binding']!=binding:raise ValueError('Changed contract')
    write_json(manifest,dict(status='running',binding=binding))
    h=hashlib.sha256();rows=[]
    path=a.root/'forward/unsloth--Qwen3.6-35B-A3B/math500/traces_with_routing.jsonl'
    with path.open('rb') as stream:
        for i,line in enumerate(stream):
            h.update(line);saved=a.output/f'trace-{i:03d}.json'
            if saved.exists():row=json.loads(saved.read_text())
            else:
                base=json.loads((a.base/f'trace-{i:03d}.json').read_text());record=json.loads(line)
                if keys(TraceRecord(**record))['trace_id']!=base['trace_id']:raise ValueError('Trace join mismatch')
                row=extract(record,base);write_json(saved,row)
            rows.append(row);print('original metrics reduced',i+1,flush=True)
    if h.hexdigest()!=expected or len(rows)!=64:raise ValueError('Input changed')
    results=[]
    for j,name in enumerate(sorted({k for r in rows for k in r['features']})):
        use=[r for r in rows if r['is_correct'] is not None and name in r['features']]
        x=[r['features'][name] for r in use];y=[int(r['is_correct']) for r in use]
        family=('legacy_class_layer' if name.startswith('class/') and '_l' in name and not name.endswith('layers') else
                'legacy_class_global' if name.startswith('class/') else 'dynamics' if name.startswith('dynamics/') else
                'legacy_global' if name.endswith('layers') else 'legacy_layer')
        for adj,c in [('none',None),('log_length_and_cap',[[np.log(r['completion_tokens']),int(r['capped'])] for r in use])]:
            z=statistics(x,y,c);z.update(metric=name,family=family,adjustment=adj,correct=sum(y),incorrect=len(y)-sum(y));results.append(z)
        if j%100==0:print('original correlations',j,flush=True)
    groups=defaultdict(list)
    for r in results:groups[(r['family'],r['adjustment'])].append(r)
    for group in groups.values():
        for r,q in zip(group,bh_adjust([r.get('p') for r in group])):r['q_bh']=q
    write_rows(a.output/'correlations.jsonl',results)
    contrasts=[]
    for name in sorted({k for r in rows for k in r['contrasts']}):
        x=np.array([r['contrasts'][name] for r in rows if name in r['contrasts']])
        rng=np.random.default_rng(42);samples=x[rng.integers(0,len(x),(1000,len(x)))].mean(1)
        p=min(1.,2*(min(np.sum(samples<=0),np.sum(samples>=0))+1)/1001)
        contrasts.append(dict(metric=name,questions=len(x),mean_class_minus_full=float(x.mean()),
          confidence_interval=np.quantile(samples,[.025,.975]).tolist(),p=float(p) if len(x)>=10 else None))
    for r,q in zip(contrasts,bh_adjust([r['p'] for r in contrasts])):r['q_bh']=q
    write_rows(a.output/'class_metric_contrasts.jsonl',contrasts)
    # Conservative joint correction across original and new global correctness searches.
    combined=results+[json.loads(l) for l in (a.base/'correlations.jsonl').open()]
    for adj in ('none','log_length_and_cap'):
        selected=[r for r in combined if r['adjustment']==adj and r['family'] in ('legacy_global','global_full')]
        for r,q in zip(selected,bh_adjust([r.get('p') for r in selected])):r['q_joint_global']=q
    write_rows(a.output/'combined_correlations.jsonl',combined)
    summary=dict(status='complete',questions=len(rows),tests=len(results),
       available_original_metrics=['normalized router confidence','top-1 probability margin','selected-expert probability mass',
        'selected/unselected boundary margin','top-k/non-top-k mean gap','top-1 switch rate','top-k overlap'],
       unavailable_original_metrics=['hidden-state norm','hidden trajectory distance','hidden/router geometry'],
       note='All original metrics retain exact saved definitions and class adjacency. Entropy is represented by confidence, not separately tested.',
       limitations=['Observed class-run lengths can be censored; use saved censor-aware hazards for durations.',
        'Within-question original class contrasts are class minus all reasoning, not class versus rest.',
        'Class-specific correctness correlations condition on the class being present.',
        'No causal conclusions; no held-out predictive validation in this analysis.'])
    write_json(a.output/'summary.json',summary)
    write_json(manifest,dict(status='complete',binding=binding,summary=summary))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
