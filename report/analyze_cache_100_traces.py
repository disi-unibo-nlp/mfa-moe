"""Parallel per-answer cache analysis for multi-model cache findings.
PYTHONPATH=src python3 report/analyze_cache_100_traces.py
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from moe_exp.moe_cache_streaming.cache import simulate
from moe_exp.moe_cache_streaming.run import load_run
ROOT=Path(__file__).resolve().parents[1]
MODEL='oss'
RUN=ROOT/f'results/moe_cache_streaming/{MODEL}_100_20260925T071519Z'


def task(item):
    condition, r, policy, run_path, layer_ids = item
    p=Path(run_path)/condition/r['routing']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['routing_sha256']
    with gzip.open(p,'rt') as f: trace=json.load(f)
    assert set(trace)==set(layer_ids)
    totals={b:{mode:defaultdict(int) for mode in ['lru','pins']} for b in [policy['top_k']*i for i in [2,4,8]]}
    for layer, sequence in trace.items():
        assert len(sequence)==r['generated_token_count']-1
        assert all(len(s)==policy['top_k'] and all(0<=i<policy['num_experts'] for i in s) for s in sequence)
        entry=policy['layers'].get(layer)
        pins=[]
        if entry:
            targets=[i for i,s in enumerate(entry['scores']) if s>0]
            pins=targets if condition=='guided' else [e['expert'] for e in sorted(entry['experts'],key=lambda e:(-e['frequency_mass'],e['expert']))[:len(targets)]]
        for budget in totals:
            lru=simulate(sequence,budget)
            pinned=simulate(sequence,budget,pins) if pins else lru
            for mode,counts in [('lru',lru),('pins',pinned)]:
                for k,v in counts.items():totals[budget][mode][k]+=v
    return dict(condition=condition,id=r['id'],dataset=r['input']['dataset'],tokens=r['generated_token_count'],counts=totals)


def main():
    jobs=[]
    for c in ['baseline','guided']:
        m,records=load_run(RUN/c)
        jobs.extend((c,r,m['policy'],str(RUN),m['routing']['layers']) for r in records)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[]
        for r in pool.map(task,jobs,chunksize=1):
            rows.append(r)
            if len(rows)%20==0:print(len(rows),flush=True)
    result=dict(run=str(RUN.relative_to(ROOT)),rows=rows,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'src/moe_exp/moe_cache_streaming/cache.py',RUN/'baseline/manifest.json',RUN/'guided/manifest.json']})
    (ROOT/f'report/{MODEL}_cache_100_traces.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in ['baseline','guided']:
        rr=[r for r in rows if r['condition']==c]
        for b in sorted(rr[0]['counts']):
            for mode in ['lru','pins']:
                loads=sum(r['counts'][b][mode]['total_loads'] for r in rr)
                print(c,b,mode,loads/len(rr),loads/sum(r['tokens']-1 for r in rr),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['oss','qwen','gemma'],default='oss')
    MODEL=parser.parse_args().model
    RUN=ROOT/f'results/moe_cache_streaming/{MODEL}_100_20260925T071519Z'
    main()
