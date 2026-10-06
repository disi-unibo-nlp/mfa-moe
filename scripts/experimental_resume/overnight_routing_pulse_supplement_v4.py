"""Additive, fixed-fixture qualification of the previously unexercised second pulse.

The failed v2/v3 receipts remain immutable. No semantic outcomes select this
engineering fixture. Exactly three requests use the unchanged worker and sampler.
"""
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
import adjudicate_overnight_operator_qualification_v3 as adjudication
import overnight_routing_dose_audit_v3 as corrected

common.audit_output_dose = corrected.audit_output_dose

DOC = common.DOC
MANIFEST = DOC/'OVERNIGHT_PULSE_SUPPLEMENT_MANIFEST_v4.json'
RESULT = DOC/'OVERNIGHT_PULSE_SUPPLEMENT_RESULT_v4.json'
WORKER = Path(__file__).with_name('overnight_routing_worker_v2.py')
RUN_ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')


def prior_evidence():
    """Require the exact remaining coverage gap, with all 27 other cases accepted."""
    original, runner, _ = adjudication.modules()
    manifest = original.validate_manifest(base.sealed(original.MANIFEST))
    receipt = base.sealed(adjudication.RESULT)
    failed = [(i, c['reasons']) for i, c in enumerate(receipt['checks']) if not c['pass']]
    common.require(receipt['schema'] == 'overnight-operator-qualification-adjudication-v3' and
        receipt['status'] == 'FAIL_REMAINING_QUALIFICATION_CHECKS' and receipt['pass'] is False and
        receipt['qualification_manifest_sha256'] == manifest['sha256'] and
        receipt['requests'] == 28 and len(receipt['checks']) == 28 and
        failed == [(20, ['second complete pulse not exercised'])] and
        receipt['kernel_checks_pass'] is True and receipt['gpu_reexecution'] is False,
        'supplement requires exactly the single unexercised second-pulse case')
    common.require(all(base.file_sha(p) == sha for p, sha in receipt['code_files'].items()) and
        all(base.file_sha(p) == sha for p, sha in receipt['artifacts'].items()) and
        base.sealed(Path(receipt['original_result_path']))['sha256'] == receipt['original_result_sha256'],
        'original qualification/adjudication evidence changed')
    raw_path = next(Path(p) for p in receipt['artifacts'] if Path(p).name == 'RAW_RESULTS.json')
    raw = json.loads(raw_path.read_text())
    common.require(len(raw) == 28, 'original raw cases incomplete')
    return manifest, receipt, raw


def choose_fixture(raw):
    # Engineering coverage selection only: first original fixture whose five
    # requests all exercised 1024 steps without emitted reasoning closure.
    from moe_steer.engine import THINK_END_ID
    eligible = [i for i in range(4) if all(len(r['tokens']) == 1024 and
                THINK_END_ID not in r['tokens'] for r in raw[i*5:i*5+5])]
    common.require(bool(eligible), 'no existing fixed long engineering fixture')
    common.require(eligible[0] == 2, 'frozen existing long fixture differs')
    return 2


def prepared():
    original, receipt, raw = prior_evidence()
    selected = choose_fixture(raw)
    fixtures = [original['fixtures'][selected]]
    actions = original['actions']
    plan = [
        {'fixture': 0, 'role': 'native', 'horizon': 1024, 'slots': [], 'closed': False, 'preempt': False},
        {'fixture': 0, 'role': 'bias', 'horizon': 1024, 'slots': [0, 512], 'closed': False, 'preempt': False},
        {'fixture': 0, 'role': 'native_duplicate', 'horizon': 1024, 'slots': [], 'closed': False, 'preempt': False}]
    reference = base.sealed(common.REFERENCE)
    prefill = 3 * (len(fixtures[0]['prompt_ids']) + len(fixtures[0]['prefix_ids']))
    decode = 3072
    seconds = (reference['repeat_factor'] * (prefill/reference['serial_prefill_stress_tokens_per_second'] +
               decode/reference['serial_decode_stress_tokens_per_second']) + reference['cold_load_seconds'] +
               reference['shutdown_seconds'] + 900)
    codes = {**receipt['code_files']}
    for name in ('overnight_routing_pulse_supplement_v4.py', 'overnight_routing_pulse_entry_v4.py',
                 'overnight_routing_pulse_supplement_v4.sbatch'):
        path = Path(__file__).with_name(name); codes[str(path.resolve())] = base.file_sha(path)
    return {'schema': 'overnight-pulse-supplement-manifest-v4', 'engine_profile': common.PROFILE,
            'base_tree_sha256': base.REQUIRED_BASE_TREE,
            'original_qualification_manifest_sha256': original['sha256'],
            'original_adjudication_path': str(adjudication.RESULT),
            'original_adjudication_sha256': receipt['sha256'],
            'original_fixture_index': selected,
            'selection': 'First existing original fixture whose five runs reached1024 with no reasoning closure; fixed index2, no further search.',
            'fixtures': fixtures, 'actions': actions, 'plan': plan,
            'code_files': codes, 'maximum_requests': 3, 'maximum_decode_tokens': decode,
            'prefill_tokens': prefill, 'reference_price_sha256': reference['sha256'],
            'estimated_wall_seconds': seconds, 'requested_wall_seconds': 3600,
            'estimated_gpu_hours': 2*seconds/3600, 'gpus': 2, 'generation_only': True,
            'checks': ['exact512 edited rows across full256-token pulses at0 and512',
                       'native inactive neighbors before/after targeted request',
                       'TP dose agreement, native top8, untouched layers, closure and caps',
                       'same exact worker, CRN seed and sampler; prior27 accepted cases preserved'],
            'interpretation': 'Engineering coverage supplement only; no semantic effect estimate or replacement of either failed receipt.'}


def validate_manifest(value):
    expected=prepared()
    common.require(value == {**expected,'sha256':base.digest(expected)}, 'qualification source/assignment changed')
    common.require(value['estimated_wall_seconds'] <= value['requested_wall_seconds'], 'qualification price exceeds complete job')
    return value


def validate_result(value):
    manifest = validate_manifest(base.sealed(MANIFEST))
    common.require(value['schema']=='overnight-pulse-supplement-result-v4' and value['pass'] is True and
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
    out=args.out or RUN_ROOT/f"overnight-pulse-supplement-v4-{manifest['sha256'][:16]}"
    out.mkdir(parents=True,exist_ok=False)
    table=common.build_policy_table(manifest['actions'])
    base.atomic_json(out/'policy-table.json',table.sealed())
    world=M.load_world(); deadline=Q.Deadline.from_env(); started=time.time(); driver=None
    result={'schema':'overnight-pulse-supplement-result-v4','pass':False,
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
        groups=[list(range(3))]
        for group in groups:
            reqs=[]; metas=[]
            for index in group:
                plan=manifest['plan'][index]; row=manifest['fixtures'][plan['fixture']]
                prefix=list(row['prefix_ids'])+([engine.THINK_END_ID] if plan['closed'] else [])
                prompt=row['prompt_ids']+prefix; role=plan['role']; native=role.startswith('native')
                policy='zero' if native else row['transition']+'_'+role
                uid='overnight-pulse-v4|'+base.digest([manifest['sha256'],index])[:24]
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
        print(json.dumps({'status':'PASS_PULSE_SUPPLEMENT_CPU_PREFLIGHT','requests':value['maximum_requests']}));return
    run(args)


if __name__=='__main__':main()
