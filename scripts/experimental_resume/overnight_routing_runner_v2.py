"""Source-bound operator routing on a common horizon, gated by exact v2 GPU qualification."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import overnight_routing_runner_v1 as common
import overnight_routing_qualify_v2 as qualification
import run_boundary_micro_screen as base

ACTIVE_SHARD = None
WORKER = Path(__file__).with_name('overnight_routing_worker_v2.py')


def qualified_result():
    value=qualification.validate_result(base.sealed(qualification.RESULT))
    common.require(all(base.file_sha(path)==sha for path,sha in value['code_files'].items()),
                   'qualified inherited worker/base source changed')
    return value


def normalized(design):
    """Accept already-frozen diagnostic C and new per-transition common-horizon designs."""
    transitions=design.get('transitions',list(common.TRANSITIONS))
    horizon=design.get('horizon',256 if design.get('mode')=='native_operator' else 1024)
    common.require(horizon in (256,1024),'unqualified horizon')
    actions=[]
    for value in design['actions']:
        action=dict(value)
        if 'bias' in action:
            action.update(kind='bias',sign=1,magnitude=action.pop('bias'))
        common.require(action['sign']==1 and
                       ((action['kind']=='bias' and action['magnitude'] in (.5,1.)) or
                        (action['kind']=='force' and action['magnitude']==0.) or
                        (action['kind']=='reweight' and action['magnitude']==1.)),
                       'unqualified operator or magnitude')
        actions.append(action)
    arms_by=design.get('arms_by_transition')
    if arms_by is None:
        arms_by={transition:[{**arm,'policies':({transition:arm['policies'][transition]} if arm['role']!='native' else {}),
                              'slots':([0] if arm['role']!='native' and not arm['slots'] else arm['slots'])}
                            for arm in design['arms']] for transition in transitions}
    common.require(set(arms_by)==set(transitions),'arms missing a frozen transition')
    lookup={a['name']:a for a in actions}
    common.require(len(lookup)==len(actions),'duplicate action names')
    from overnight_routing_worker_v2 import OperatorPulse
    from moe_exp.routing_control.design import digest
    table=common.build_policy_table(actions)
    for transition,arms in arms_by.items():
        names={a['name'] for a in arms}
        common.require(4<=len(arms)<=16 and len(names)==len(arms) and
                       {'native','native_duplicate'}<=names and
                       sum(a['role']=='native' for a in arms)==2 and
                       {'target','random'}<={a['role'] for a in arms},'unqualified arm set')
        for arm in arms:
            common.require(set(arm)=={'name','role','policies','slots'},'unknown arm fields')
            if arm['role']=='native':
                common.require(not arm['policies'] and not arm['slots'],'native arm edits routing');continue
            common.require(arm['role'] in ('target','random') and
                           set(arm['policies'])=={transition} and arm['slots'] in ([0],[0,512]),
                           'invalid operator policy assignment')
            common.require(horizon==1024 or arm['slots']==[0],'second pulse outside256horizon')
            for index in range(4):
                names=[n.replace('{random_set}',str(index)) for n in arm['policies'][transition]]
                common.require(all(n in lookup and lookup[n]['transition']==transition for n in names),
                               'policy target belongs to another transition')
                common.require(all(('{random_set}' in n)==(arm['role']=='random')
                                   for n in arm['policies'][transition]),'random assignment is not balanced selector')
                body={'action_policy_names':names,'slots':arm['slots'],'horizon':1024}
                OperatorPulse.load({**body,'sha256':digest(body)},table,names[0])
    common.require(design.get('analysis_seed')==20261004 and design.get('planned_contrasts'),
                   'freeze contrast family and analysis seed')
    scopes=design.get('analysis_scopes',['all',*transitions])
    common.require(scopes==['all',*transitions],'analysis scope differs')
    return transitions,horizon,actions,arms_by


def scoped_contrasts(design,arms_by):
    result=[]
    contrasts=design.get('planned_contrasts_scoped',design['planned_contrasts'])
    for contrast in contrasts:
        if isinstance(contrast,dict):
            scope=contrast.get('scope',contrast.get('transition'))
            left=contrast.get('left',contrast.get('a'));right=contrast.get('right',contrast.get('b'))
            entries=[{'scope':scope,'left':left,'right':right}]
        else:
            entries=[{'scope':scope,'left':contrast[0],'right':contrast[1]}
                     for scope in design['analysis_scopes']]
        for entry in entries:
            scope=entry['scope'];eligible=list(arms_by) if scope=='all' else [scope]
            common.require(scope=='all' or scope in arms_by,'unknown contrast scope')
            common.require(entry['left']!=entry['right'] and all(
                {entry['left'],entry['right']}<={a['name'] for a in arms_by[t]} for t in eligible),
                'contrast contains arms not assigned in this population')
            common.require(entry not in result,'duplicate planned contrast')
            result.append(entry)
    return result


def enrollment(design):
    path=Path(design.get('source_enrollment_path',common.SOURCE)).resolve()
    source=base.sealed(path)
    family=base.sealed(base.FAMILY_FREEZE)
    transitions=design.get('transitions',list(common.TRANSITIONS))
    rows=[dict(r) for r in source['rows'] if r['transition'] in transitions]
    rows.sort(key=lambda r:(transitions.index(r['transition']),base.digest(['overnight-v2-enrollment',r['uid']])))
    common.require(rows and len({r['uid'] for r in rows})==len(rows) and
                   len({(r['family'],r['transition']) for r in rows})==len(rows), 'missing/duplicate source starts')
    if path==common.SOURCE.resolve():
        pool=set(family['new_parent_pools']['parent_pools']['discovery'])
    else:
        common.require(source['schema']=='extension-overnight-enrollment-v1' and
                       source['status']=='READY_EXACT_NATIVE_PREFIXES' and
                       source['family_freeze_sha256']==family['sha256'], 'unrecognized fresh enrollment receipt')
        pool=set(source['family_pool'])
    excluded=set(family['new_parent_pools']['parent_pools']['mechanism'])|set(
        family['new_parent_pools']['parent_pools']['utility'])
    common.require(not pool & excluded and {r['family'] for r in rows}<=pool,
                   'enrollment overlaps prior mechanism/utility or leaves its frozen pool')
    for row in rows:
        common.require(row.get('canonical_question') and row['prompt_ids'] and
                       1<=len(row['prefix_ids'])<=8192 and all(type(i)is int and i>=0
                        for i in row['prompt_ids']+row['prefix_ids']) and
                       all(base.digest(row[k])==row[k+'_sha256'] for k in ('prompt_ids','prefix_ids')),
                       'invalid exact native prefix or token hash')
    return path,source,rows


def schedules(rows,arms_by):
    sets={};orders={}
    for transition,arms in arms_by.items():
        members=sorted({r['family'] for r in rows if r['transition']==transition})
        for seed in (0,1):
            ranked=sorted(members,key=lambda f:base.digest(['overnight-v2-random',transition,seed,f]))
            for index,family in enumerate(ranked):sets[f'{family}|{transition}|{seed}']=(index+seed)%4
        names=[a['name'] for a in arms]
        blocks=sorted(((r['uid'],seed) for r in rows if r['transition']==transition for seed in (0,1)),
                      key=lambda x:base.digest(['overnight-v2-order',transition,*x]))
        for index,(uid,seed) in enumerate(blocks):
            shift=index%len(names);orders[f'{uid}|{seed}']=names[shift:]+names[:shift]
    return sets,orders


def workload(rows,arms_by,horizon):
    counts={'expected_requests':0,'expected_prefill_tokens':0,'maximum_decode_tokens':0,'maximum_context_tokens':0}
    for row in rows:
        n=2*len(arms_by[row['transition']]);prefix=len(row['prompt_ids'])+len(row['prefix_ids'])
        counts['expected_requests']+=n;counts['expected_prefill_tokens']+=n*prefix
        counts['maximum_decode_tokens']+=n*horizon
        counts['maximum_context_tokens']=max(counts['maximum_context_tokens'],prefix+horizon)
    return counts


def projection(rows,arms_by,horizon,max_wall_seconds,rows_per_shard=0):
    ref=base.sealed(common.REFERENCE);load=ref['cold_load_seconds']+ref['shutdown_seconds'];reserve=900
    common.require(max_wall_seconds>=3600 and rows_per_shard>=0,'invalid full-stage pricing limits')
    usable=max_wall_seconds-load-reserve;shards=[];first=0;work=0.
    for index,row in enumerate(rows):
        count=workload([row],arms_by,horizon)
        next_work=ref['repeat_factor']*(count['expected_prefill_tokens']/ref['serial_prefill_stress_tokens_per_second']+
                                     count['maximum_decode_tokens']/ref['serial_decode_stress_tokens_per_second'])
        common.require(next_work<=usable,'one full matched prefix block cannot fit wall limit')
        if work and (work+next_work>usable or row['transition']!=rows[first]['transition'] or
                     rows_per_shard and index-first>=rows_per_shard):
            shards.append({'start_row':first,'end_row':index,'transition':rows[first]['transition'],
                           'estimated_work_seconds':work,'estimated_wall_seconds':work+load+reserve})
            first=index;work=0.
        work+=next_work
    if work:shards.append({'start_row':first,'end_row':len(rows),'transition':rows[first]['transition'],
                          'estimated_work_seconds':work,'estimated_wall_seconds':work+load+reserve})
    loads=len(shards)+max(1,math.ceil(len(shards)*(ref['repeat_factor']-1)))
    seconds=sum(s['estimated_work_seconds'] for s in shards)+loads*load+len(shards)*reserve
    return {'schema':'overnight-routing-price-v2','reference_price_sha256':ref['sha256'],'shards':shards,
            'gpus_per_job':2,'max_wall_seconds':max_wall_seconds,'cold_loads_including_reserve':loads,
            'repeat_factor':ref['repeat_factor'],'preemption_reserve_seconds_per_job':reserve,
            'estimated_complete_gpu_hours':2*seconds/3600,'status':'PASS_COMPLETE_STAGE_GENERATION_ONLY',
            'scope':'All assigned generations, full caps and prefix preparation,25%repeat work,cold/recovery loads,shutdown and900sreserve perjob. Blind ratings priced separately.'}


def prepare(design_path,manifest_path,price_path,rows_per_shard=0,max_wall_seconds=8100):
    design=base.sealed(design_path);transitions,horizon,actions,arms_by=normalized(design)
    source_path,source,rows=enrollment(design)
    qual=qualified_result()
    sets,orders=schedules(rows,arms_by);price=projection(rows,arms_by,horizon,max_wall_seconds,rows_per_shard)
    frame_path=design.get('source_frame_path',source.get('source_frame_path'))
    if frame_path is None and source_path==common.SOURCE.resolve():
        # Existing full-prefix discovery start frame used by the v1 measurements.
        from build_overnight_blind_frame_v1 import SOURCE
        frame_path=str(SOURCE)
    elif frame_path is None and source['schema']=='extension-overnight-enrollment-v1':
        from prepare_mechanism_extension_220_v1 import FRAME
        frame_path=str(FRAME)
    common.require(frame_path is not None,'explicit blinded start-context frame required')
    frame=base.sealed(frame_path)
    if source['schema']=='extension-overnight-enrollment-v1':
        common.require(source.get('frame_sha256',source.get('source_frame_sha256'))==frame['sha256'],
                       'extension enrollment frame differs from blinded start-context source')
    codes={str(Path(__file__).resolve()):base.file_sha(__file__),str(WORKER):base.file_sha(WORKER),
           str(Path(common.__file__).resolve()):base.file_sha(common.__file__),
           str(Path(base.__file__).resolve()):base.file_sha(base.__file__),
           str(Path(qualification.__file__).resolve()):base.file_sha(qualification.__file__)}
    for filename in ('overnight_routing_entry_v1.py','diagnose_mechanism_validation_v3.py',
                     'overnight_routing_run_v2.sbatch'):
        path=Path(__file__).with_name(filename);codes[str(path)]=base.file_sha(path)
    allarms={arm['name']:arm for arms in arms_by.values() for arm in arms}
    body={'schema':'overnight-routing-manifest-v2','design_path':str(Path(design_path).resolve()),
          'design_sha256':design['sha256'],'source_enrollment_path':str(source_path),'source_enrollment_sha256':source['sha256'],
          'source_frame_path':str(Path(frame_path).resolve()),'source_frame_sha256':frame['sha256'],
          'family_freeze_sha256':base.sealed(base.FAMILY_FREEZE)['sha256'],
          'qualification_sha256':qual['sha256'],'qualification_manifest_sha256':qual['qualification_manifest_sha256'],
          'base_tree_sha256':base.REQUIRED_BASE_TREE,'qualified_worker_sha256':base.file_sha(WORKER),
          'engine_profile':common.PROFILE,'code_files':codes,'rows':rows,'seeds':[0,1],
          'horizon':horizon,'pulse_width':256,'mode':'operator_pulses_v2','actions':actions,
          'arms_by_transition':arms_by,'arms':list(allarms.values()),'transitions':transitions,
          'random_set_by_family_transition_seed':sets,'arm_order_by_uid_seed':orders,
          'planned_contrasts':design['planned_contrasts'],'analysis_scopes':design.get('analysis_scopes',['all',*transitions]),
          'planned_contrasts_scoped':scoped_contrasts(design,arms_by),
          'analysis_seed':20261004,'bootstrap_replicates':50000,
          'shards':price['shards'],'rows_per_shard':rows_per_shard,'max_wall_seconds':max_wall_seconds,
          **workload(rows,arms_by,horizon),'claim_limit':design.get('claim_limit',
          'Exploratory family-clustered operator comparison. Fixed prefix and horizon only; no original-prompt utility or independent confirmation claim.')}
    manifest=common.write_once(manifest_path,body)
    price=common.write_once(price_path,{**price,'manifest_sha256':manifest['sha256']})
    return manifest,price


def validate(manifest,driver_path):
    common.require(manifest['schema']=='overnight-routing-manifest-v2' and
                   manifest['engine_profile']==common.PROFILE and manifest['seeds']==[0,1] and
                   manifest['pulse_width']==256 and os.environ.get('VLLM_BATCH_INVARIANT','0')=='0',
                   'unqualified schema/profile/pulse')
    codes=manifest['code_files']
    common.require(codes.get(str(Path(__file__).resolve()))==base.file_sha(__file__) and
                   codes.get(str(Path(driver_path).resolve()))==base.file_sha(driver_path) and
                   all(base.file_sha(p)==s for p,s in codes.items()),'generation source changed')
    qual=qualified_result()
    common.require(manifest['qualification_sha256']==qual['sha256'] and
                   manifest['qualification_manifest_sha256']==qual['qualification_manifest_sha256'] and
                   manifest['qualified_worker_sha256']==base.file_sha(WORKER) and
                   manifest['base_tree_sha256']==base.REQUIRED_BASE_TREE and
                   manifest['family_freeze_sha256']==base.sealed(base.FAMILY_FREEZE)['sha256'],
                   'operator qualification changed')
    design=base.sealed(manifest['design_path']);transitions,horizon,actions,arms_by=normalized(design)
    source_path,source,rows=enrollment(design)
    common.require(manifest['design_sha256']==design['sha256'] and
                   manifest['source_enrollment_path']==str(source_path) and
                   manifest['source_enrollment_sha256']==source['sha256'] and rows==manifest['rows'] and
                   actions==manifest['actions'] and arms_by==manifest['arms_by_transition'] and
                   horizon==manifest['horizon'],'source enrollment or design changed')
    common.require(manifest['planned_contrasts_scoped']==scoped_contrasts(design,arms_by) and
                   manifest['planned_contrasts']==design['planned_contrasts'] and
                   manifest['analysis_seed']==20261004 and manifest['bootstrap_replicates']==50000,
                   'frozen analysis family changed')
    common.require(base.sealed(manifest['source_frame_path'])['sha256']==manifest['source_frame_sha256'],
                   'blinded start-context frame changed')
    sets,orders=schedules(rows,arms_by)
    common.require(sets==manifest['random_set_by_family_transition_seed'] and orders==manifest['arm_order_by_uid_seed'],
                   'random/control/position schedule changed')
    common.require(all(manifest[k]==v for k,v in workload(rows,arms_by,horizon).items()),'workload counts changed')
    price=projection(rows,arms_by,horizon,manifest['max_wall_seconds'],manifest['rows_per_shard'])
    common.require(price['shards']==manifest['shards'],'shard allocation changed')
    arms=arms_by[manifest['shards'][ACTIVE_SHARD]['transition']] if ACTIVE_SHARD is not None else manifest['arms']
    return rows,actions,arms


def request_metadata(manifest,rows,arms_unused=None):
    for row in rows:
        arms={a['name']:a for a in manifest['arms_by_transition'][row['transition']]}
        prompt=row['prompt_ids']+row['prefix_ids']
        for seed in manifest['seeds']:
            random_set=manifest['random_set_by_family_transition_seed'][f"{row['family']}|{row['transition']}|{seed}"]
            for position,name in enumerate(manifest['arm_order_by_uid_seed'][f"{row['uid']}|{seed}"]):
                arm=arms[name];policies=[p.replace('{random_set}',str(random_set)) for p in arm['policies'].get(row['transition'],[])]
                uid='overnight-v2|'+base.digest([manifest['sha256'],row['uid'],seed,name])[:24]
                yield row,{'uid':uid,'prefix_uid':row['uid'],'family':row['family'],'transition':row['transition'],
                           'question':row['canonical_question'],'seed':seed,'arm':name,'role':arm['role'],
                           'policy':policies[0] if policies else 'zero','policies':policies,'slots':arm['slots'],
                           'random_set':random_set if arm['role']=='random' else None,
                           'execution_position':position,'prompt_len':len(prompt),'prompt_sha256':base.digest(prompt)}


def build_requests(manifest,rows,arms,world,table):
    from moe_steer import engine,qualify as Q
    from moe_exp.routing_control.design import digest
    cases=[]
    for row,meta in request_metadata(manifest,rows):
        prompt=row['prompt_ids']+row['prefix_ids'];info=world.infos.get(row['canonical_question'])
        common.require(info is not None and info['prompt_token_ids']==row['prompt_ids'] and
                       engine.THINK_END_ID not in row['prefix_ids'],'exact native prompt/closure changed')
        extra=Q.steer_extra(table,meta['uid'],meta['policy'],len(row['prompt_ids']),prefix_len=len(row['prefix_ids']),restore_presence=True)
        if meta['role']!='native':
            template={'action_policy_names':meta['policies'],'slots':meta['slots'],'horizon':1024}
            extra['steer']['meta']={'routing_control':{**template,'sha256':digest(template)}}
        sampling=Q.card_params(manifest['horizon'],Q.crn(info,meta['seed']),extra,presence=0.,routed_start=len(prompt)-1)
        cases.append((Q.QReq(meta['uid'],prompt,sampling),meta))
    common.require(len(cases)==manifest['expected_requests'] and len({m['uid'] for _,m in cases})==len(cases),
                   'missing/duplicate assignment')
    offset=0
    for row in rows:
        names={a['name'] for a in manifest['arms_by_transition'][row['transition']]}
        for seed in manifest['seeds']:
            block=cases[offset:offset+len(names)];offset+=len(names)
            common.require(len({tuple(req.prompt) for req,_ in block})==1 and
                           len({req.sampling['seed'] for req,_ in block})==1 and
                           {meta['arm'] for _,meta in block}==names and
                           all(meta['prefix_uid']==row['uid'] and meta['seed']==seed for _,meta in block),
                           'same-prefix same-seed arm block differs')
    if ACTIVE_SHARD is not None:
        shard=manifest['shards'][ACTIVE_SHARD];chosen={r['uid'] for r in rows[shard['start_row']:shard['end_row']]}
        cases=[c for c in cases if c[1]['prefix_uid'] in chosen]
    return cases


def audit_operator_dose(result,manifest,original_audit):
    report=original_audit(result,manifest)
    if result['error'] or result['role']=='native':return report
    actions={a['name']:a for a in manifest['actions']}
    for rank in ('0','1'):
        for policy,values_by_layer in result['action_dose'][rank]['dose'].items():
            action=actions[policy]
            for layer,targets in action['experts']:
                values=values_by_layer[str(layer)]
                if action['kind']=='force':
                    common.require(values['actual_target_hits']==values['active_rows']*len(targets),
                                   'force-in did not execute all targeted experts')
                if action['kind']=='reweight':
                    common.require(values['membership_changes']==0 and
                                   values['native_target_hits']==values['actual_target_hits'],
                                   'reweight changed same-hidden-state native selected expert identities')
    return {**report,'operator_semantics':'force membership and reweight unchanged selection checked'}


def seal_shard(out,manifest,index):
    original=common.request_metadata;original_audit=common.audit_output_dose
    common.request_metadata=request_metadata
    common.audit_output_dose=lambda result,m:audit_operator_dose(result,m,original_audit)
    try:
        return common.seal_shard(out,manifest,index)
    finally:
        common.request_metadata=original;common.audit_output_dose=original_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','price','design','out','overlay'):p.add_argument('--'+key,type=Path,required=key=='manifest')
    for key in ('prepare','cpu-preflight','seal-shard','seal-stage'):p.add_argument('--'+key,action='store_true')
    p.add_argument('--shard-index',type=int);p.add_argument('--rows-per-shard',type=int,default=0)
    p.add_argument('--max-wall-seconds',type=int,default=8100);args=p.parse_args()
    if args.prepare:
        common.require(args.design and args.price,'prepare requires design and price')
        m,price=prepare(args.design,args.manifest,args.price,args.rows_per_shard,args.max_wall_seconds)
        print(json.dumps({'manifest_sha256':m['sha256'],'requests':m['expected_requests'],'shards':len(m['shards']),
                          'gpu_hours':price['estimated_complete_gpu_hours']}));return
    m=base.sealed(args.manifest);rows,actions,arms=validate(m,base.__file__)
    if args.seal_shard or args.seal_stage:
        indices=range(len(m['shards'])) if args.seal_stage else [args.shard_index]
        receipts=[seal_shard(args.out,m,i) for i in indices]
        if args.seal_stage:
            counts=Counter()
            for receipt in receipts:counts.update(receipt['counts'])
            common.require(counts['assigned']==m['expected_requests'],'full stage incomplete')
            common.write_once(args.out/'STAGE_COMPLETION.json',{'schema':'overnight-routing-stage-completion-v2',
                'manifest_sha256':m['sha256'],'counts':dict(counts),'shard_completion_sha256s':[r['sha256'] for r in receipts],
                'status':'COMPLETE_UNGRADED_DISCOVERY_GENERATION'})
        print(json.dumps({'status':'PASS_COMPLETION','shards':len(receipts)}));return
    from moe_steer import engine,manifests as M
    if args.cpu_preflight:
        table=common.build_policy_table(actions);cases=build_requests(m,rows,arms,M.load_world(),table)
        from overnight_routing_worker_v2 import OperatorPulse
        for request,meta in cases:
            if meta['role']!='native':OperatorPulse.load(request.sampling['extra_args']['steer']['meta']['routing_control'],table,meta['policy'])
        print(json.dumps({'status':'PASS_CPU_PREFLIGHT','requests':len(cases),'shards':len(m['shards'])}));return
    common.require(args.out and args.overlay and args.shard_index is not None and
                   0<=args.shard_index<len(m['shards']),'generation requires output,overlay,shard')
    global ACTIVE_SHARD
    ACTIVE_SHARD=args.shard_index
    args.batch_size=len(m['arms_by_transition'][m['shards'][ACTIVE_SHARD]['transition']])
    args.out=args.out/f'shard-{ACTIVE_SHARD:03d}'
    original_kwargs=engine.engine_kwargs;original_build=engine.build_llm
    base.validate_manifest,base.build_requests,base.build_policy_table=validate,build_requests,common.build_policy_table
    engine.engine_kwargs=lambda *a,**kw:original_kwargs(*a,**{**kw,'max_num_seqs':1,'enforce_eager':True})
    def build(kwargs,*a,**kw):
        return original_build({**kwargs,'worker_extension_cls':'overnight_routing_worker_v2.OvernightWorkerExtension'},*a,**kw)
    engine.build_llm=build
    base.run(args)


if __name__=='__main__':main()
