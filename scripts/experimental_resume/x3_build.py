"""Legacy X3 builder and complete maximum-workload projection; never submits jobs.

Production requires saved, sealed G3 selection, 96 prospectively frozen prefix
eligibility records, retained-native nonfire UIDs and all-in stage pricing.
The separate causal routing-action study has launch priority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S=ROOT/'steering-v1'


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verified(path):
    v=json.loads(Path(path).read_text());body={k:x for k,x in v.items() if k!='sha256'}
    if digest(body)!=v.get('sha256'):raise ValueError('unsealed or changed X3 input '+str(path))
    return v


def pinned_order(split):
    ids=[f"{r['dataset']}|{r['source_problem_id']}" for r in split['questions'] if r.get('subsplit')=='dev-disc']
    if len(ids)!=96:raise ValueError('the original 96 dev-disc questions must remain intact')
    return sorted(ids,key=lambda q:hashlib.sha256(('forum-v1|'+q).encode()).hexdigest())


def projection(signs,e4,prefill_tokens=None,label_GPU_h=None,nll_GPU_h=None,retries_GPU_h=None):
    arms=1+signs*(3 if e4 else 2)
    cap=64*2*arms*32768+32*2*arms*1024
    # Escalation is separate, never pooled. Include it prospectively to authorize it.
    escalation=56*2*3*1024
    cost=json.loads((S/'runs/x2prep/cost_model.json').read_text())
    rate=cost['inputs']['x1']['steady_tok_per_s']*cost['constants']['pessimistic_derate']
    base=cap/rate*2/3600+838*2/3600
    overhead={'prefill_GPU_h':None if prefill_tokens is None else prefill_tokens/cost['inputs']['q10']['prefill_tps']*2/3600,
        'labeling_GPU_h':label_GPU_h,'native_NLL_GPU_h':nll_GPU_h,'retries_GPU_h':retries_GPU_h,
        'prefix_preparation_GPU_h':0.,'cold_loads_GPU_h':838*2/3600,
        'other_overhead_GPU_h':None}
    return {'signs':signs,'E4':e4,'questions':96,'seeds':[0,1],'arms_per_question':arms,
        'maximum_decode_tokens_before_prefix_subtraction':cap,
        'generation_historical_pessimistic_GPU_h':base,
        'escalation_maximum_decode_tokens':escalation,
        'escalation_additional_decode_and_load_GPU_h':escalation/rate*2/3600+838*2/3600,
        'dev_spare':'56 questions; separate 3-arm analysis, no force escalation; only efficacy-only failure',
        'all_in_components':overhead,'ceiling_GPU_h':12.,
        'status':'HOLD_FULL_PRICE_UNKNOWN','timing_scope':'historical scenario, not qualified context-matched X3 timing',
        'E4_execution':'requires independently qualified forced first pulse plus three later onset opportunities; no unqualified E4 manifest'}


def build(args):
    from moe_steer import manifests as M, policies as P, engine
    from moe_steer.spec import seal
    from x2_build import e_policy, assert_no_confirm
    world=M.load_world();split=M.load_split();order=pinned_order(split)
    selected=verified(args.selection);eligible=verified(args.eligibility);price=verified(args.price)
    if selected.get('schema')!='G3-dose-selection-v1' or selected.get('native_nll_parity_pass') is not True:
        raise ValueError('valid native NLL and sealed registered G3 selection required')
    cells=selected['selected']
    if not cells or len(cells)>2 or len({c['sign'] for c in cells})!=len(cells):
        raise ValueError('no G3 passing sign, or invalid selection; legacy X3 remains stopped')
    rows={r['question']:r for r in eligible['rows']}
    if set(rows)!=set(order):raise ValueError('all 96 assigned questions, including nonfires, are required')
    if price.get('status')!='PASS_COMPLETE_STAGE' or price.get('all_in_GPU_h',float('inf'))>12.:
        raise ValueError('complete X3 projection, including grading, loads and retries, must fit 12 GPU-h')
    if price.get('new_study_priority_resolved') is not True:
        raise ValueError('new action-study feasibility/discovery has priority over X3 launch')
    if args.e4:raise ValueError('E4 is projected but its independent execution qualification is incomplete')
    tree=engine.code_tree_sha256()
    if tree!=args.expect_tree:raise ValueError('unexpected frozen cap runtime')
    target=[];random=[]
    for c in cells:
        if c.get('G3_pass') is not True:raise ValueError('selected cell did not pass registered G3')
        p=e_policy(world.inputs,c['scope'],(c['operator'],c['sign'],c['magnitude']),P.landmark_schedule(256))
        target.append(p)
        for k in (0,1):
            base=e_policy(world.inputs,c['scope'],(c['operator'],c['sign'],c['random_magnitudes'][str(k)]),P.landmark_schedule(256))
            random.append(P.matched_random(base,world.inputs,k))
    sham=P.sham(world.inputs.scope('ALL'),P.landmark_schedule(256))
    table=P.build_table([*target,*random,sham]);requests=[];retained=[]
    first64=set(order[:64]);name=args.name
    for q in order:
        row=rows[q];prefix=row.get('prefix_len');long=q in first64
        if not row['eligible'] or (long and prefix>=32768):
            if set(row.get('native_uids',{}))!={'0','1'}:
                raise ValueError('nonfire needs both verified retained-native seed UIDs')
            retained.append({'question':q,'native_uids':row['native_uids'],'assigned_arms':1+2*len(cells),
                'endpoint_cap':32768 if long else 1024,'semantic_occupancy':0,'reason':row.get('reason','finished before pulse / budget')})
            continue
        if type(prefix) is not int or prefix<4097:raise ValueError('invalid registered onset prefix')
        cap=32768-prefix if long else 1024
        parent={'trace_ref':row['trace_ref'],'prefix_len':prefix}
        for k in (0,1):
            pairs=[('N',sham.name)]
            for c,p in zip(cells,target):
                pairs.append(('E+' if c['sign']>0 else 'E-',p.name))
                base=e_policy(world.inputs,c['scope'],(c['operator'],c['sign'],c['random_magnitudes'][str(k)]),P.landmark_schedule(256))
                pairs.append(('M+' if c['sign']>0 else 'M-',P.matched_random(base,world.inputs,k).name))
            for arm,policy in pairs:
                r=M.make_request(name,table,world.infos[q],arm=arm,policy_name=policy,seed_k=k,
                    parent=parent,landmark=prefix,expected_len=cap,max_new_tokens=cap)
                if r['max_tokens']!=cap:raise ValueError('the same-total-budget cap did not reach the request')
                requests.append(r)
    assert_no_confirm(order,split,world.infos)
    if not requests:raise ValueError('no eligible firing requests; report zero opportunities without GPU submission')
    manifest=M.build_manifest(name,'x3',table,requests,world.infos,n_shards=1,seals=world.seals,
        code_tree=tree,lexicon_path=world.inputs.lexicon_path,vocab_path=world.vocab_path,
        routed={'target_policies':None,'window':64,'after_pulse':1024},expect_fingerprint=args.fingerprint)
    M.validate_manifest(manifest)
    companion=seal({'schema':'legacy-X3-enrollment-v1','manifest_sha256':manifest['sha256'],
        'original_96_order':order,'first64':order[:64],'first64_sha256':digest(order[:64]),
        'retained_nonfires':retained,'selection_sha256':selected['sha256'],
        'eligibility_sha256':eligible['sha256'],'price_sha256':price['sha256'],
        'endpoint':'fixed 512-token Explore occupancy, excludes trigger; absorbing termination; unknown labels >5% holds interpretation',
        'multiplicity':'G4 gate .05/3; six claim contrasts .05/6 and 99.167% intervals',
        'maximum_decode_tokens':sum(r['max_tokens'] for r in manifest['requests'])})
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/(name+'.json')).write_text(json.dumps(manifest,indent=1)+'\n')
    (args.out/'ENROLLMENT.json').write_text(json.dumps(companion,indent=1)+'\n')
    print(json.dumps({'manifest_sha256':manifest['sha256'],'requests':len(requests),'retained_questions':len(retained)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',action='store_true');p.add_argument('--e4',action='store_true')
    p.add_argument('--selection',type=Path);p.add_argument('--eligibility',type=Path);p.add_argument('--price',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--name',default='x3-resume-v1')
    p.add_argument('--expect-tree');p.add_argument('--fingerprint')
    a=p.parse_args()
    if a.project:
        split=json.loads((S/'manifests/split-v1.json').read_text());order=pinned_order(split)
        v={'schema':'legacy-X3-projections-v1','first64':order[:64],'first64_sha256':digest(order[:64]),
            'scenarios':[projection(signs,e4) for signs in (1,2) for e4 in (False,True)],
            'production_builder':'implemented with mandatory sealed selection/eligibility/full-price gates; no production manifest yet'}
        destination=Path(__file__).resolve().parents[2]/'report/experimental-resume-v1/X3_PROJECTIONS.json'
        destination.write_text(json.dumps(v,indent=1)+'\n');print(json.dumps({'path':str(destination),'launch':'HOLD'}))
    else:
        if any(getattr(a,n) is None for n in ('selection','eligibility','price','out','expect_tree')):
            p.error('production requires selection, eligibility, complete price, out and expect-tree')
        build(a)
