"""Generate model-specific cache tables and paired uncertainty from audited sidecars."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL='oss'


def percentile(v,q):
    v=sorted(v);x=(len(v)-1)*q;i=int(x)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)


def main():
    traces=json.loads((ROOT/f'report/{MODEL}_cache_100_traces.json').read_text())
    quality=json.loads((ROOT/f'report/{MODEL}_cache_100_rescoring.json').read_text())
    original=json.loads((ROOT/traces['run']/'analysis/summary.json').read_text())
    groups={c:{r['id']:r for r in traces['rows'] if r['condition']==c} for c in ['baseline','guided']}
    assert groups['baseline'].keys()==groups['guided'].keys()
    buckets=defaultdict(list)
    for k,r in groups['baseline'].items():buckets[r['dataset']].append(k)
    rng=random.Random(42)
    samples=[[rng.choice(ids) for ids in buckets.values() for _ in ids] for _ in range(5000)]
    summaries=[];effects={}
    conditions=[('baseline_lru','baseline','lru'),('baseline_frequency_pins','baseline','pins'),('guided_target_pins','guided','pins'),('guided_lru','guided','lru')]
    for name,c,mode in conditions:
        for budget in sorted(next(iter(groups['baseline'].values()))['counts'],key=int):
            rows=list(groups[c].values())
            counts={k:sum(r['counts'][budget][mode][k] for r in rows) for k in rows[0]['counts'][budget][mode]}
            saved=next(r for r in original['summaries'] if r['condition']==name and r['capacity_per_layer']==int(budget))
            for k,v in counts.items():assert saved[k]==v,(name,budget,k)
            summaries.append(saved)
    keys=list(groups['baseline'])
    for budget in sorted(next(iter(groups['baseline'].values()))['counts'],key=int):
        def metric(ids,kind):
            b=sum(groups['baseline'][k]['counts'][budget]['lru']['total_loads'] for k in ids)
            g=sum(groups['guided'][k]['counts'][budget]['pins']['total_loads'] for k in ids)
            if kind=='per_token':
                b/=sum(groups['baseline'][k]['tokens']-1 for k in ids)
                g/=sum(groups['guided'][k]['tokens']-1 for k in ids)
            return 100*(g/b-1)
        effects[budget]={}
        for kind in ['per_answer','per_token']:
            vals=[metric(ids,kind) for ids in samples]
            effects[budget][kind]=dict(change_percent=metric(keys,kind),ci95=[percentile(vals,.025),percentile(vals,.975)])
    result=dict(summaries=summaries,effects=effects,comparison=quality['comparison'],
                bootstrap='5000 paired dataset-stratified problem resamples; Python random seed 42; percentile; no multiple-testing correction',
                verified_against_original_cache_summary=True)
    (ROOT/f'report/{MODEL}_cache_100_findings.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[r'\begin{table}[ht]',r'\centering\small',r'\caption{GPT-OSS, 100 held-out problems: simulated decode-only expert loads. Every response starts cold; initial pin loads are included. Slots are per layer.}',r'\label{tab:oss-cache-100-loads}',r'\begin{tabular}{llrrr}',r'\toprule',r'Policy & Slots & Loads/answer & Loads/token & Hit rate (\%) \\',r'\midrule']
    labels={'baseline_lru':'Baseline LRU','baseline_frequency_pins':'Baseline frequency pins','guided_target_pins':'Guided target pins','guided_lru':'Guided LRU'}
    for r in summaries:
        lines.append(f"{labels[r['condition']]} & {r['capacity_per_layer']} & {r['loads_per_answer']:,.1f} & {r['loads_per_decode_token']:.2f} & {100*r['hit_fraction']:.2f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/f'report/{MODEL}_cache_100_table.tex').write_text(('\n'.join(lines)+'\n').replace('GPT-OSS', {'oss':'GPT-OSS','qwen':'Qwen','gemma':'Gemma'}[MODEL]).replace('tab:oss-cache-100-loads', f'tab:{MODEL}-cache-100-loads'))
    print(json.dumps(effects,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['oss','qwen','gemma'],default='oss')
    MODEL=parser.parse_args().model
    RUN=ROOT/f'results/moe_cache_streaming/{MODEL}_100_20260925T071519Z'
    main()
