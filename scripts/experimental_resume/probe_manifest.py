"""Freeze M10/M8 matched prefixes and common M9 family folds; replay remains gated."""
import hashlib
import json
import os
from pathlib import Path

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal_write(path,body):
    value={**body,'sha256':digest(body)}
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('immutable preparation differs')
    path.write_text(json.dumps(value,separators=(',',':'))+'\n');return value


def prepare():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('probe preparation requires CPU Slurm')
    path=R/'steering-v1/runs/m9-resume-v1/MANIFEST.json';m9=json.loads(path.read_text())
    if digest({k:v for k,v in m9.items() if k!='sha256'})!=m9['sha256']:raise ValueError('M9 manifest changed')
    family=json.loads((REPO/'report/experimental-resume-v1/family-freeze.json').read_text())
    qfamily={q:f for f,qs in family['new_parent_pools']['families'].items() for q in qs}
    questions=sorted({r['question'] for r in m9['rows']})
    if len(questions)!=200 or len(m9['rows'])!=400:raise ValueError('original M9 assignment changed')
    families=sorted({qfamily[q] for q in questions},key=lambda f:digest(['M9-M10-fold-v1',20261001,f]))
    family_fold={f:i%5 for i,f in enumerate(families)};qfold={q:family_fold[qfamily[q]] for q in questions}
    out=R/'steering-v1/runs/m10-resume-v1';out.mkdir(exist_ok=True)
    folds=seal_write(out/'JOINT_FOLDS.json',{'schema':'M9-M10-joint-folds-v1','M9_manifest':m9['sha256'],
        'family_freeze':family['sha256'],'outer':5,'inner':3,'seed':20261001,'questions':qfold,
        'family_ids':{q:qfamily[q] for q in questions},
        'learned_transforms':'layer, lambda, PCA, projections and text transforms fit on training families only',
        'populations':'M9 all assigned; M10 eligible-prefix survivors; same fixed family fold'})
    selected=[]
    for r in m9['rows']:
        p=r['prepared']
        if p['status']!='generate_closure':continue
        original=p['prefix_presence_start'];prompt=p['prompt_token_ids'][:original+m9['cut']]
        if len(prompt)!=original+m9['cut']:raise ValueError('M9 source prefix incomplete')
        selected.append({'uid':digest(['M10-cut-v1',r['uid']]),'M9_uid':r['uid'],'question':r['question'],
            'seed_k':r['seed_k'],'prompt_token_ids':prompt,'original_prompt_len':original,
            'prefix_len':m9['cut'],'family':qfamily[r['question']],'outer_fold':qfold[r['question']],
            'PAC_label':'PENDING complete blinded M9 grading'})
    order=sorted(questions,key=lambda q:hashlib.sha256(('forum-v1|'+q).encode()).hexdigest())[:100]
    seed0={r['question']:r for r in m9['rows'] if r['seed_k']==0};repeats=[]
    from moe_steer import manifests as M
    native=M.load_manifest(R/'steering-v1/manifests/x1-v1.json')
    if native['sha256']!=m9['native_manifest_sha256']:raise ValueError('M9 native prompt manifest differs')
    for q in order:
        r=seed0[q];prefix=r['native_endpoint_token_ids'][:8192];original=native['questions'][q]['prompt_token_ids']
        repeats.append({'uid':digest(['M8-two-launch-v1',q]),'question':q,'seed_k':0,
            'prompt_token_ids':original+prefix,'prefix_len':len(prefix),'original_prompt_len':len(original),
            'family':qfamily[q],'outer_fold':qfold[q],
            'native_finished_before_repeat_cap':r['native_endpoint_natural_stop'] and len(r['native_endpoint_token_ids'])<=8192})
    body={'schema':'M10-M8-matched-prefix-v1','M9_manifest':m9['sha256'],'joint_fold_sha256':folds['sha256'],
        'family_freeze':family['sha256'],'layers_one_indexed':[20,30,40],'layers_absolute':[19,29,39],
        'cut':14336,'M10_rows':selected,'M8_rows':repeats,'M8_launches':2,'M8_cap':8192,
        'repeat_question_order':order,'primary_PCA_max_components':8,'secondary_PCA_max_components':64,
        'routing_top_k':8,'targets':['hidden_vector','hidden_scalar_norm_step_distance','router'],
        'baseline':'strengthened M7 original problem and emitted past text plus prefix position; no future completion fields',
        'multiplicity':'three primary P contrasts Holm .05','noise_controls':20,
        'advancement':'vector P <= -.02, adjusted CI < 0, held-out probe-score ICC >= .7; complete cells required',
        'reliability':'scalar and family-held-out probe-score ICC(A,1); membership agreement cannot substitute',
        'online_replay_allowlist':['prompt_token_ids','prefix_len','original_prompt_len'],
        'generation':'none; only M9 supplies generated-closure outcomes','GPU_hour_ceiling':3.,'CPU_core_hour_ceiling':12.,
        'execution_status':'HOLD_PAC_LABELS_HOOK_QUALIFICATION_AND_COMPLETE_TWO_LAUNCH_PRICE',
        'code':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            REPO/'src/moe_exp/routing_control/reliability.py']},'prepare_job_id':os.environ['SLURM_JOB_ID'],
        'maximum_replayed_prefix_tokens':sum(len(r['prompt_token_ids']) for r in selected)+2*sum(len(r['prompt_token_ids']) for r in repeats)}
    value=seal_write(out/'PROBE_MANIFEST.json',body)
    print(json.dumps({'path':str(out/'PROBE_MANIFEST.json'),'sha256':value['sha256'],
        'M10_prefixes':len(selected),'M8_prefixes':len(repeats),'execution':'HOLD'}))


if __name__=='__main__':prepare()
