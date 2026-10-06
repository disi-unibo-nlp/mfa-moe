"""Exact serial/eager GPU qualification for bias, force-in and gate reweight pulses."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time
import traceback

import overnight_routing_runner_v1 as common
import run_boundary_micro_screen as base

DOC = common.DOC
MANIFEST = DOC/'OVERNIGHT_OPERATOR_QUAL_MANIFEST_v2.json'
RESULT = DOC/'OVERNIGHT_OPERATOR_QUAL_RESULT_v2.json'
WORKER = Path(__file__).with_name('overnight_routing_worker_v2.py')
RUN_ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')


def prepared():
    source = base.sealed(common.SOURCE)
    dictionary = base.sealed(DOC/'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json')
    previous = base.sealed(common.QUALIFICATION)
    fixtures = []
    for transition in common.TRANSITIONS:
        selected = sorted((r for r in source['rows'] if r['transition'] == transition),
                          key=lambda r: base.digest(['overnight-operator-qual-v2',r['uid']]))[:2]
        common.require(len(selected) == 2, 'qualification needs two fixed starts per transition')
        fixtures.extend(selected)
    actions = []
    for target in dictionary['target_templates']:
        for kind, magnitude in (('bias',1.),('force',0.),('reweight',1.)):
            actions.append({'name': target['transition']+'_'+kind, 'transition':target['transition'],
                            'experts':target['experts'], 'kind':kind, 'sign':1,'magnitude':magnitude})
    plan = []
    for i,row in enumerate(fixtures):
        for role in ('native','native_duplicate','bias','force','reweight'):
            plan.append({'fixture':i,'role':role,'horizon':1024,'slots':[] if role.startswith('native') else [0],
                         'closed':False,'preempt':False})
    plan.extend([
        {'fixture':0,'role':'bias','horizon':1024,'slots':[0,512],'closed':False,'preempt':False},
        {'fixture':1,'role':'force','horizon':1024,'slots':[0],'closed':False,'preempt':True},
        {'fixture':2,'role':'reweight','horizon':1024,'slots':[0],'closed':False,'preempt':True},
        {'fixture':3,'role':'force','horizon':128,'slots':[0],'closed':True,'preempt':False},
        {'fixture':3,'role':'reweight','horizon':128,'slots':[0],'closed':True,'preempt':False},
        *[{'fixture':0,'role':kind,'horizon':256,'slots':[0],'closed':False,'preempt':False}
          for kind in ('bias','force','reweight')]])
    reference = base.sealed(common.REFERENCE)
    prefill = sum(len(fixtures[x['fixture']]['prompt_ids']) +
                  len(fixtures[x['fixture']]['prefix_ids']) + int(x['closed']) for x in plan)
    decode = sum(x['horizon'] for x in plan)
    seconds = (reference['repeat_factor'] * (prefill/reference['serial_prefill_stress_tokens_per_second'] +
               decode/reference['serial_decode_stress_tokens_per_second']) + reference['cold_load_seconds'] +
               reference['shutdown_seconds'] + 900)
    codes = {**previous['code_files'], str(Path(common.__file__).resolve()):base.file_sha(common.__file__),
             str(Path(__file__).resolve()):base.file_sha(__file__), str(WORKER):base.file_sha(WORKER)}
    for filename in ('overnight_routing_entry_v1.py','diagnose_mechanism_validation_v3.py',
                     'overnight_routing_qualify_v2.sbatch'):
        path=Path(__file__).with_name(filename);codes[str(path)]=base.file_sha(path)
    return {'schema':'overnight-routing-qualification-manifest-v2','engine_profile':common.PROFILE,
            'base_tree_sha256':base.REQUIRED_BASE_TREE,'source_manifest_sha256':source['sha256'],
            'source_qualification_sha256':previous['sha256'],'family_freeze_sha256':source['family_freeze_sha256'],
            'source_dictionary_sha256':dictionary['sha256'],'fixtures':fixtures,'actions':actions,'plan':plan,
            'code_files':codes,'maximum_requests':len(plan),'maximum_decode_tokens':decode,
            'prefill_tokens':prefill,'estimated_wall_seconds':seconds,'requested_wall_seconds':7200,
            'estimated_gpu_hours':2*seconds/3600,'gpus':2,'generation_only':True,
            'checks':['exact top8 IDs and token alignment','inactive native routing and isolated neighbors',
                      'pulse0 and repeat512 boundaries','closure disables both operators',
                      'force target membership','reweight native membership and finite gate dose',
                      'both TP rank dose agreement','preemption/recompute for force and reweight',
                      '256 and1024 sampler horizons','source-bound UID and assignment completeness']}


def validate_manifest(value):
    expected=prepared()
    common.require(value == {**expected,'sha256':base.digest(expected)}, 'qualification source/assignment changed')
    common.require(value['estimated_wall_seconds'] <= value['requested_wall_seconds'], 'qualification price exceeds complete job')
    return value


def validate_result(value):
    manifest = validate_manifest(base.sealed(MANIFEST))
    common.require(value['schema']=='overnight-routing-qualification-result-v2' and value['pass'] is True and
                   value['qualification_manifest_sha256']==manifest['sha256'] and
                   value['code_files']==manifest['code_files'] and value['engine_profile']==common.PROFILE and
                   value['requests']==manifest['maximum_requests'] and
                   len(value['checks'])==manifest['maximum_requests'] and
                   all(x['pass'] is True for x in value['checks']) and value['kernel_checks_pass'],
                   'native operator qualification is not passing/exact')
    for path,sha in value['artifacts'].items():
        common.require(base.file_sha(path)==sha,'qualification raw artifact changed')
    return value


class PreemptingDriver:
    def __init__(self, inner):
        self.inner=inner; self.count=0; self.events=[]
    def __getattr__(self,name):
        return getattr(self.inner,name)
    def step(self):
        self.count+=1
        if self.count in (120,300):
            active=self.inner.in_flight()
            self.events.append({'call':self.count,'in_flight':active,
                'reset_ok':bool(self.inner.llm.reset_prefix_cache(reset_running_requests=True)) if active else False})
        return self.inner.step()


def kernel_checks(table):
    """Check exact force/reweight formulas on allocated CUDA, including inactive rows."""
    import torch
    from moe_steer import ops
    compiled=table.compile();tables=ops.DeviceTables.from_compiled(compiled,'cuda',max_tokens=2)
    act=torch.tensor([True,False],device='cuda')
    for policy in table.policies:
        if policy.operator.kind not in ('force','reweight'):continue
        index=table.index_of(policy.name);pidx=torch.tensor([index,index],device='cuda',dtype=torch.int32)
        for layer,targets in policy.targets.experts:
            h=tables.h_of_layer(layer)
            if policy.operator.kind=='force':
                for dtype in (torch.float32,torch.bfloat16):
                    logits=torch.arange(256,device='cuda',dtype=torch.float32).repeat(2,1).to(dtype)
                    edited=ops.steer_logits(logits,act,pidx,h,tables)
                    common.require(torch.equal(edited[1],logits[1]),'CUDA force modified inactive row')
                    selected=edited[0].topk(8).indices.tolist()
                    common.require(set(targets)<=set(selected),'CUDA force failed native top8 membership')
            else:
                ids=list(targets)+[e for e in range(256) if e not in targets][:8-len(targets)]
                ids=torch.tensor([ids,ids],device='cuda',dtype=torch.int32)
                weights=torch.tensor([[.05,.1,.15,.2,.1,.15,.1,.15]]*2,device='cuda')
                changed=ops.reweight(weights,ids,act,pidx,h,tables)
                expected=weights[0].clone();expected[:len(targets)]*=torch.exp(torch.tensor(1.,device='cuda'))
                expected*=weights[0].sum()/expected.sum()
                common.require(torch.allclose(changed[0],expected,rtol=1e-6,atol=1e-7) and
                               torch.equal(changed[1],weights[1]),'CUDA reweight formula or inactive parity differs')
    return True


def run(args):
    common.require(os.environ.get('SLURM_JOB_ID') and not socket.gethostname().startswith('login'),
                   'qualification requires authorized GPU Slurm')
    manifest=validate_manifest(base.sealed(MANIFEST))
    from moe_steer import engine,manifests as M,qualify as Q
    from moe_exp.routing_control.design import digest
    import numpy as np
    worker_prep=base.sealed(base.WORKER_PREP)
    overlay=Path(worker_prep['overlay'])
    out=args.out or RUN_ROOT/f"overnight-operator-qual-v2-{manifest['sha256'][:16]}"
    out.mkdir(parents=True,exist_ok=False)
    table=common.build_policy_table(manifest['actions'])
    base.atomic_json(out/'policy-table.json',table.sealed())
    world=M.load_world(); deadline=Q.Deadline.from_env(); started=time.time(); driver=None
    result={'schema':'overnight-routing-qualification-result-v2','pass':False,
            'qualification_manifest_sha256':manifest['sha256'],'code_files':manifest['code_files'],
            'engine_profile':common.PROFILE,'job_id':os.environ['SLURM_JOB_ID'],'checks':[],'requests':0}
    try:
        deadline.require(1100,'cold load and complete qualification prefix')
        result['kernel_checks_pass']=kernel_checks(table)
        fingerprint=engine.fingerprint(); tele=out/'telemetry';tele.mkdir()
        env=engine.engine_env(out/'policy-table.json',tele,expect_fingerprint=fingerprint['combined'])
        env['PYTHONPATH']=os.pathsep.join((str(overlay),str(Path(__file__).parent),env['PYTHONPATH']))
        base.prepare_worker_import_path(engine,env,overlay)
        kwargs=engine.engine_kwargs(plugin=True,max_num_seqs=1,enforce_eager=True,return_routed_experts=True)
        kwargs.update(worker_extension_cls='overnight_routing_worker_v2.OvernightWorkerExtension',
                      gpu_memory_utilization=.80,long_prefill_token_threshold=1024)
        model=engine.build_llm(kwargs);driver=Q.QDriver(model,plugin=True)
        harness=Q.Harness(driver,tele,deadline=deadline,abort_margin=120.)
        raw=[]; arrays={}; pending=[]
        # Five-arm neighbor blocks followed by individual recovery/closure/cap checks.
        groups=[list(range(i,i+5)) for i in range(0,20,5)]+[[i] for i in range(20,len(manifest['plan']))]
        for group in groups:
            reqs=[]; metas=[]
            for index in group:
                plan=manifest['plan'][index]; row=manifest['fixtures'][plan['fixture']]
                prefix=list(row['prefix_ids'])+([engine.THINK_END_ID] if plan['closed'] else [])
                prompt=row['prompt_ids']+prefix; role=plan['role']; native=role.startswith('native')
                policy='zero' if native else row['transition']+'_'+role
                uid='overnight-qual-v2|'+base.digest([manifest['sha256'],index])[:24]
                extra=Q.steer_extra(table,uid,policy,len(row['prompt_ids']),prefix_len=len(prefix),restore_presence=True)
                policies=[] if native else [policy]*len(plan['slots'])
                if not native:
                    template={'action_policy_names':policies,'slots':plan['slots'],'horizon':1024}
                    extra['steer']['meta']={'routing_control':{**template,'sha256':digest(template)}}
                info=world.infos[row['canonical_question']]
                common.require(info['prompt_token_ids']==row['prompt_ids'],'exact fixture prompt differs')
                reqs.append(Q.QReq(uid,prompt,Q.card_params(plan['horizon'],Q.crn(info,0),extra,presence=0.,routed_start=len(prompt)-1)))
                metas.append({**plan,'uid':uid,'policy':policy,'policies':policies,'transition':row['transition'],
                              'role':'native' if native else 'target','operator':role,
                              'prompt_sha256':base.digest(prompt)})
            common.require(len({r.sampling['seed'] for r in reqs})==1,'paired fixture seed differs')
            wrapper=PreemptingDriver(driver) if metas[0]['preempt'] else driver
            harness.driver=wrapper
            base.atomic_json(out/f'assignment-{group[0]:03d}.json',{'requests':metas,'manifest_sha256':manifest['sha256']})
            batch=harness.run(reqs,cap_in_flight=len(reqs));harness.flush()
            telemetry,_,_=harness.telemetry()
            for meta in metas:
                outcome=batch.outcomes.get(meta['uid']); reasons=[]
                if outcome is None or outcome.error:
                    result['checks'].append({'uid':meta['uid'],'pass':False,'reasons':['missing/error outcome']})
                    continue
                route=np.asarray(outcome.routed) if outcome.routed is not None else None
                if route is None or route.shape!=(len(outcome.tokens),40,8):
                    reasons.append('routed shape differs')
                else:
                    arrays[meta['uid']]=route
                    if not ((route>=0)&(route<256)).all() or not (np.diff(np.sort(route,axis=-1),axis=-1)>0).all():
                        reasons.append('invalid native top8')
                record={**meta,'tokens':outcome.tokens,'error':False,'finish':outcome.finish,
                        'stop_reason':outcome.stop_reason,'routed_present':route is not None,
                        'inactive_native_checks':{},'action_dose':{}}
                for rank in (0,1):
                    receipt=telemetry.get(rank,{}).get(meta['uid'])
                    if receipt is None:
                        reasons.append('missing rank telemetry');continue
                    record['inactive_native_checks'][str(rank)]=receipt.get('inactive_native_checks')
                    if meta['role']=='native':
                        if receipt.get('cpu_active_rows')!=0:reasons.append('native neighbor edited')
                    else:
                        record['action_dose'][str(rank)]={'rows':receipt.get('ordered_action_rows'),
                            'segments':receipt.get('ordered_segments'),'dose':receipt.get('ordered_action_dose')}
                    if meta['preempt'] and (receipt.get('preemptions',0)<1 or receipt.get('recompute_rows',0)<1):
                        reasons.append('missing force/reweight preemption and recompute')
                if meta['closed']:
                    for entry in record['action_dose'].values():
                        if any(entry['rows'].values()) or any(entry['segments'].values()):reasons.append('closed prefix edited')
                    try:
                        common.audit_output_dose({**record,'role':'native','policies':[],'slots':[]},
                                                 {'horizon':meta['horizon'],'actions':manifest['actions']})
                    except Exception as error:reasons.append(str(error))
                else:
                    try:
                        common.audit_output_dose(record,{'horizon':meta['horizon'],'actions':manifest['actions']})
                    except Exception as error:
                        reasons.append(str(error))
                if meta['slots']==[0,512] and not meta['closed']:
                    stop=next((i+1 for i,t in enumerate(outcome.tokens) if t==engine.THINK_END_ID),len(outcome.tokens))
                    if stop<768:reasons.append('second complete pulse not exercised')
                if meta['operator'] in ('force','reweight') and not meta['closed']:
                    targets={str(layer):ids for layer,ids in next(a for a in manifest['actions'] if a['name']==meta['policy'])['experts']}
                    for entry in record['action_dose'].values():
                        for layer,ids in targets.items():
                            values=entry['dose'][meta['policy']][layer]
                            if meta['operator']=='force' and values['actual_target_hits']!=values['active_rows']*len(ids):
                                reasons.append('force target membership failed')
                            if meta['operator']=='force' and route is not None:
                                n=int(values['active_rows'])
                                if any(not np.any(route[:n,int(layer),:]==expert,axis=-1).all() for expert in ids):
                                    reasons.append('executed force top8 differs from telemetry')
                            if meta['operator']=='reweight' and (values['membership_changes']!=0 or
                                values['actual_target_hits']!=values['native_target_hits']):
                                reasons.append('reweight changed same-state native expert IDs')
                if meta['preempt'] and not any(x['reset_ok'] and x['in_flight'] for x in wrapper.events):
                    reasons.append('preemption reset not exercised')
                raw.append(record)
                result['checks'].append({'uid':meta['uid'],'operator':meta['operator'],'slots':meta['slots'],
                    'closed':meta['closed'],'preempt':meta['preempt'],'horizon':meta['horizon'],
                    'tokens':len(outcome.tokens),'pass':not reasons,'reasons':reasons})
            print(json.dumps({'qualified_group':group,'pass_so_far':all(c['pass'] for c in result['checks'])}),flush=True)
        base.atomic_json(out/'RAW_RESULTS.json',raw)
        with (out/'ROUTED.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
        result.update(requests=len(raw),pass_=len(raw)==manifest['maximum_requests'] and all(c['pass'] for c in result['checks']),
                      runtime_fingerprint=fingerprint,artifacts={str(out/'RAW_RESULTS.json'):base.file_sha(out/'RAW_RESULTS.json'),
                      str(out/'ROUTED.npz'):base.file_sha(out/'ROUTED.npz')})
        result['pass']=result.pop('pass_')
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-5000:])
    result['elapsed_seconds']=time.time()-started
    common.write_once(out/'QUALIFICATION.json',result)
    if result['pass']:common.write_once(RESULT,result)
    print(json.dumps({'pass':result['pass'],'result':str(out/'QUALIFICATION.json'),'error':result.get('error')}),flush=True)
    Q.finish_child(0 if result['pass'] else 3,driver)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--cpu-preflight',action='store_true');p.add_argument('--out',type=Path);args=p.parse_args()
    if args.prepare:
        value=common.write_once(MANIFEST,prepared());validate_manifest(value)
        print(json.dumps({k:value[k] for k in ('maximum_requests','maximum_decode_tokens','estimated_wall_seconds','estimated_gpu_hours','sha256')}));return
    if args.cpu_preflight:
        value=validate_manifest(base.sealed(MANIFEST));common.build_policy_table(value['actions'])
        print(json.dumps({'status':'PASS_OPERATOR_QUAL_CPU_PREFLIGHT','requests':value['maximum_requests']}));return
    run(args)


if __name__=='__main__':main()
