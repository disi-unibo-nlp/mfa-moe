"""Rejoin Gemma original/repaired sentence versions as separate provenance cohorts."""
import argparse,copy,json,os
from pathlib import Path
from measurement_study.production_plan import ROOT,atomic,sha
from measurement_study.join_collection import join_cohort

MIXED='qwen38-mtp3-production-gemma'

def split_spec(spec):
    groups={1:[],2:[]}
    for info in spec['annotations']:
        version=2 if '/qwen38-mtp3-production-v2-affected/' in info['path'] else 1
        groups[version].append(info)
    if not all(groups.values()):raise ValueError('Both original and explicit repaired sources required')
    return [dict(spec,id=MIXED+f'-sentence-v{v}',annotations=files) for v,files in groups.items()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm CPU allocation required')
    for name,h in json.loads((a.bundle/'SHA256.json').read_text()).items():
        if sha(a.bundle/name)!=h:raise ValueError('Frozen recovery bundle changed: '+name)
    oldpath=ROOT/'results/measurement-study/source-join-v2/collection.json'
    if sha(oldpath)!='eb07af23d96dfdec344f9e0ecb25910b0034ef3421d9e81497874ab992b20fa2':raise ValueError('Original source inventory changed')
    model='nvidia/Gemma-4-26B-A4B-NVFP4'
    rows={r['id']:copy.deepcopy(r) for r in json.loads(oldpath.read_text())['attempts'] if r['model']==model}
    expected={aid for aid,r in rows.items() if any(c['id']==MIXED for c in r['cohorts'])}
    for row in rows.values():row['cohorts']=[c for c in row['cohorts'] if c['id']!=MIXED]
    sourcepath=ROOT/'results/available-metrics/v1/manifest.json'
    source=next(s for s in json.loads(sourcepath.read_text())['jobs'] if s['id']==MIXED)
    out=ROOT/'results/measurement-study/recovery-storage/source-joins/gemma-segmentation-v1'
    reports=[];seen=set()
    for spec in split_spec(source):
        report=join_cohort(spec,out);reports.append(report)
        if report['unmatched'] or report['rejected'] or report['label_conflicts']:raise ValueError('Incomplete source recovery')
        for row in json.loads((out/spec['id']/'attempts.json').read_text()):
            aid=row['id'];seen.add(aid)
            if aid not in expected or row['trace_sha256']!=rows[aid]['trace_sha256']:raise ValueError('Recovered source identity changed')
            rows[aid]['cohorts'].append(dict(id=spec['id'],annotation=row['annotation'],locations=row['locations'],identity=row['identity'],metadata_variants=row['metadata_variants'],annotation_version=spec['id'].rsplit('-v',1)[-1]))
    if seen!=expected or any(not r['cohorts'] for r in rows.values()):raise ValueError('Recovered attempt coverage changed')
    collection=out/'collection.json'
    atomic(collection,dict(attempts=list(rows.values()),cohorts=reports,original_collection_sha256=sha(oldpath),source_manifest_sha256=sha(sourcepath),policy='Preserve original and repaired annotations as separate sentence-version cohorts; never merge sentence indices across versions.'))
    atomic(out/'acceptance.json',dict(status='complete',collection_sha256=sha(collection),code_binding=sha(a.bundle/'SHA256.json'),attempts=len(rows),split_attempts=len(seen),cohorts=reports))
    print(json.dumps(dict(status='complete',attempts=len(rows),cohorts=reports)),flush=True)
if __name__=='__main__':main()
