#!/usr/bin/env python3
"""Build a complete labeled-attempt inventory on a Slurm CPU allocation.

Includes every analyzed cohort plus pending primary GLM/Qwen3 label plans.
No collection job is submitted by this command.
"""
import argparse,hashlib,json,os
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def pack(rows,max_tokens=200000):
    shards=[];current=[];tokens=0
    for row in sorted(rows,key=lambda r:(r['dataset'],r['id'])):
        n=row.get('completion_tokens')
        if n is None:raise ValueError('Cannot size attempt without token count')
        if current and (tokens+n>max_tokens or current[0]['dataset']!=row['dataset']):
            shards.append({'attempt_ids':[r['id'] for r in current],'completion_tokens':tokens,'dataset':current[0]['dataset']});current=[];tokens=0
        current.append(row);tokens+=n
    if current:shards.append({'attempt_ids':[r['id'] for r in current],'completion_tokens':tokens,'dataset':current[0]['dataset']})
    return shards

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/measurement-study/collection-plan');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full cohort inventory requires Slurm')
    campaign=json.loads((ROOT/'report/measurement-study/campaign.json').read_text())
    available=ROOT/'results/available-metrics/v1';manifest=json.loads((available/'manifest.json').read_text())
    attempts={};provenance={};issues=[];cohorts=[]
    for job in manifest['jobs']:
        if job['kind']!='labels':continue
        status=json.loads((available/job['id']/'status.json').read_text())
        if status['status']!='complete':issues.append({'cohort':job['id'],'status':status['status']})
        count=0
        for file in sorted((available/job['id']).glob('*/attempts.jsonl')):
            provenance[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            with file.open() as stream:
                for line in stream:
                    r=json.loads(line);identity=(r['model'],r['trace_id']);key=digest(identity)
                    core={k:r.get(k) for k in ['model','dataset','problem_id','sample_id','trace_id','completion_tokens','capped','is_correct','question_id']}
                    if key in attempts:
                        if attempts[key]['identity']!=core:issues.append({'attempt':key,'error':'conflicting compact attempt metadata'})
                    else:attempts[key]={'id':key,'identity':core,'model':r['model'],'dataset':r['dataset'],'completion_tokens':r['completion_tokens'],'cohorts':[],'requires_exact_source_join':True}
                    attempts[key]['cohorts'].append(job['id']);count+=1
        cohorts.append({'id':job['id'],'attempt_memberships':count,'source_trace_files':job['traces'],'annotation_files':len(job['annotations'])})
    pending=[]
    for source in ['glm','qwen330b']:
        spec=campaign['sources'][source];path=Path(spec['sampling_root'])/'sampling_manifest.json';m=json.loads(path.read_text());provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        members=[r for r in m['completions'] if r['selected_sentences']>0]
        pending.append({'source':source,'model':spec['generation_model'],'sampling_manifest':str(path),'members':members,'labels_status':'pending_existing_jobs'})
    by_model=defaultdict(list)
    for row in attempts.values():by_model[row['model']].append(row)
    groups=[]
    for model,rows in sorted(by_model.items()):
        sized=[r for r in rows if isinstance(r['completion_tokens'],int) and r['completion_tokens']>0]
        sized_ids={r['id'] for r in sized}
        for row in rows:
            if row['id'] not in sized_ids:issues.append({'attempt':row['id'],'error':'missing token cost or empty completion; retain in denominator'})
        groups.append({'model':model,'unique_labeled_attempts':len(rows),'completion_tokens':sum(r['completion_tokens'] for r in sized),'planned_shards':pack(sized)})
    result={'status':'prepared_requires_source_join_and_acceptance','campaign_sha256':digest(campaign),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cohorts':cohorts,'attempts':list(attempts.values()),'model_groups':groups,'pending_primary_labels':pending,'issues':issues,'provenance':provenance,'submission_allowed':False,'required_gates':['exact generation source join','all-model native capture','long-trace memory and throughput','complete required metrics','calibration-only thresholds','approved GPU/CPU/storage cap'],'benchmark_summary':'question-balanced within benchmark, equal weight across all six benchmarks including AIME'}
    result['sha256']=digest(result);write(a.output/'collection.json',result)
    print(json.dumps({'models':[{k:v for k,v in g.items() if k!='planned_shards'} for g in groups],'cohorts':len(cohorts),'pending_models':len(pending),'issues':len(issues),'sha256':result['sha256']}),flush=True)
if __name__=='__main__':main()
