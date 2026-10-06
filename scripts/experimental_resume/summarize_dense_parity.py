"""Family-clustered uncertainty for the completed direct seven-class LLM audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SOURCE = ROOT / ('steering-v1/runs/routing-control-v1/dense-judge-parity/'
                 'results-8e939bee-2341bac1/PARITY.json')
OUT = REPO / 'report/experimental-resume-v1/DENSE_JUDGE_PARITY_AUDIT.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    audit = json.loads(SOURCE.read_text())
    if audit.get('sha256') != digest({k:v for k,v in audit.items() if k!='sha256'}):
        raise ValueError('direct judge audit seal differs')
    output = subprocess.run(['sacct','-j',audit['job_id'],'-n','-P','-o','JobID,State,ExitCode,ElapsedRaw,AllocTRES'],
                            check=True,text=True,capture_output=True).stdout
    parents = [line.split('|') for line in output.splitlines() if line.startswith(audit['job_id']+'|')]
    if len(parents)!=1 or parents[0][1:3]!=['COMPLETED','0:0'] or 'gres/gpu=2' not in parents[0][4]:
        raise ValueError('judge allocation not verified as completed on two GPUs')
    records = audit['records']
    if len(records)!=200:
        raise ValueError('fixture population changed')
    groups = defaultdict(list)
    classes = Counter()
    matched = Counter()
    for row in records:
        groups[row['family']].append(row)
        classes[row['historical_label']]+=1
        matched[row['historical_label']]+=int(row['new_label']==row['historical_label']
                                              and row['finish_reason']=='stop')
    families=sorted(groups)
    def rates(items):
        n=len(items)
        return (sum(r['new_label'] is not None and r['finish_reason']=='stop' for r in items)/n,
                sum(r['new_label']==r['historical_label'] and r['finish_reason']=='stop' for r in items)/n)
    observed=rates(records)
    rng=random.Random(20261001)
    boot=[]
    for _ in range(5000):
        sample=[r for family in (rng.choice(families) for _ in families) for r in groups[family]]
        boot.append(rates(sample))
    intervals=[]
    for coordinate in range(2):
        values=sorted(x[coordinate] for x in boot)
        intervals.append([values[124],values[4874]])
    result={'schema':'dense-judge-parity-audit-v1','population':'200 class-balanced historical sparse-label fixtures from discovery families',
            'model_intervention':'new direct Qwen3.8-27B vLLM prompt versus historical sparse class labels',
            'source_sha256':audit['sha256'],'job_id':audit['job_id'],'job_state':parents[0][1],
            'job_exit_code':parents[0][2],'elapsed_seconds':int(parents[0][3]),
            'actual_GPU_hours':int(parents[0][3])*2/3600,
            'families':len(families),'assigned_fixtures':len(records),
            'parsed_stop_coverage':observed[0],'coverage_family_cluster_ci95':intervals[0],
            'historical_agreement':observed[1],'agreement_family_cluster_ci95':intervals[1],
            'per_class':{name:{'n':classes[name],'agree':matched[name],'agreement':matched[name]/classes[name]}
                         for name in sorted(classes)},
            'confusion':audit['confusion'],
            'multiplicity_family':'operational qualification; two descriptive intervals, no scientific rejection claim',
            'method':'5,000 percentile bootstrap replicates of 48 discovery families, seed 20261001; item-weighted ratio',
            'status':'QUALIFIED_FOR_EXPLORATORY_DENSE_CLASSES_ONLY',
            'limits':'Both raters are model-derived; historical labels are not human truth. No substantive-check or online-detector validity follows.'}
    result['sha256']=digest(result)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps({'coverage':observed[0],'agreement':observed[1],
                      'coverage_ci95':intervals[0],'agreement_ci95':intervals[1],
                      'GPU_hours':result['actual_GPU_hours']}))


if __name__=='__main__':
    main()
