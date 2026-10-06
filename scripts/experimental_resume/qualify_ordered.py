"""One bounded engine session for ordered-pulse mechanics, never semantic discovery.

All prefixes, model loads, prefill, decoding, capture and recovery count against the
separate study's qualification line. Fixed engineering targets are
not proposed causal actions. Failed or short fixtures are never replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')


class PreemptingDriver:
    """The public reset-prefix-cache recovery call used in the saved H14 check."""
    def __init__(self,inner,calls):
        self.inner=inner;self.calls=set(calls);self.count=0;self.events=[]
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self):
        self.count+=1
        if self.count in self.calls:
            flying=self.inner.in_flight()
            self.events.append({'call':self.count,'in_flight':flying,
                'ok':bool(self.inner.llm.reset_prefix_cache(reset_running_requests=True)) if flying else False})
        return self.inner.step()


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('ordered-worker qualification requires GPU Slurm')
    from moe_steer import engine,manifests as M,policies as P,qualify as Q
    from moe_steer.spec import TargetSet,Schedule
    from moe_exp.routing_control.design import digest
    import numpy as np
    args.out.mkdir(parents=True,exist_ok=False)
    receipts=json.loads(args.binding.read_text())
    for path,expected in receipts['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('qualification source changed')
    deadline=Q.Deadline.from_env();started=time.time();driver=None
    result={'schema':'ordered-worker-qualification-v1','pass':False,'job_id':os.environ['SLURM_JOB_ID'],
        'worker_code_digest':receipts['sha256'],'base_tree':engine.code_tree_sha256(),
        'scope':'engineering only; candidate parser is separate and semantic transition detector remains unqualified',
        'criteria':{'maximum_fixture_requests':17,'pulse_slots':[0,512],'pulse_length':256,
            'horizon':1024,'min_evaluable_ordered_per_batch':3,'preempt_calls':[110,254,514,700],
            'inactive_expert_and_weight_mismatches':0,'neighbor_active_rows':0,
            'counts':'CPU and device per-action/per-layer rows agree on both ranks; closure clips both pulses',
            'recovery':'at least one preemption and recomputation for every ordered batch-B request'},
        'semantic_detector_qualified':False,'runtime_price_qualified_for_other_contexts':False}
    try:
        world=M.load_world()
        family=json.loads((REPO/'report/experimental-resume-v1/family-freeze.json').read_text())
        parents=family['new_parent_pools']['parent_pools']['discovery']
        allowed={q for f in parents for q in family['new_parent_pools']['families'][f]}
        qfamily={q:f for f,qs in family['new_parent_pools']['families'].items() for q in qs}
        if args.fixtures is not None:
            from types import SimpleNamespace
            prepared=json.loads(args.fixtures.read_text())
            if digest({k:v for k,v in prepared.items() if k!='sha256'})!=prepared['sha256']:
                raise ValueError('CPU qualification fixture seal differs')
            if prepared['family_freeze_sha256']!=family['sha256'] or len(prepared['rows'])!=4:
                raise ValueError('CPU qualification fixtures have another family freeze or count')
            if len({r['family'] for r in prepared['rows']})!=4 or any(r['question'] not in allowed or r['family']!=qfamily[r['question']] or len(r['completion_ids'])!=2048 for r in prepared['rows']):
                raise ValueError('CPU qualification fixtures violate fixed discovery-family support')
            traces=[SimpleNamespace(**r) for r in prepared['rows']]
            result['CPU_fixture_sha256']=prepared['sha256']
        else:
            fixtures=Q.Fixtures(world)
            candidates=[a for a in fixtures.attempts() if a['question'] in allowed and a['n_reasoning']>=8192]
            candidates.sort(key=lambda a:hashlib.sha256(('ordered-qual-v1|'+str(a['attempt_id'])).encode()).hexdigest())
            picked=[];seen=set()
            for a in candidates:
                f=qfamily[a['question']]
                if f in seen:continue
                picked.append(a);seen.add(f)
                if len(picked)==4:break
            if len(picked)!=4:raise ValueError('fewer than four discovery-family engineering prefixes; no replacement pool')
            traces=[fixtures.load_trace(a) for a in picked]
        targets=[TargetSet('qual-A','CUSTOM',((23,(3,)),(24,(7,))), 'fixed engineering fixture'),
                 TargetSet('qual-B','CUSTOM',((23,(11,)),(24,(19,))), 'fixed engineering fixture')]
        actions=[P.bias(targets[0],1,.5,Schedule('always')),P.bias(targets[1],1,1.,Schedule('always'))]
        table=P.build_table(actions);table_path=args.out/'policy-table.json'
        table_path.write_text(json.dumps(table.sealed(),indent=1)+'\n')
        tele=args.out/'telemetry';tele.mkdir()
        result['pre_runtime_setup_seconds']=time.time()-started
        # Fingerprinting imports torch: the 950-second cold-phase allowance
        # includes that import and belongs before it, not after it.
        deadline.require(950,'cold runtime import/load plus both fixed qualification batches')
        fingerprint_start=time.time()
        fingerprint=engine.fingerprint()
        result['runtime_fingerprint_seconds']=time.time()-fingerprint_start
        env=engine.engine_env(table_path,tele,expect_fingerprint=fingerprint['combined'])
        env['PYTHONPATH']=os.pathsep.join((str(Path(args.overlay)),env['PYTHONPATH']))
        kwargs=engine.engine_kwargs(plugin=True,max_num_seqs=48,return_routed_experts=True)
        kwargs['worker_extension_cls']='moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension'
        kwargs.update(gpu_memory_utilization=.80,long_prefill_token_threshold=1024)
        build_start=time.time();model=engine.build_llm(kwargs,env)
        build_seconds=time.time()-build_start;driver=Q.QDriver(model,plugin=True)
        harness=Q.Harness(driver,tele,deadline=deadline,abort_margin=30.)
        manifests=[];all_rows=[];checks=[];batches=[]
        for tag in ('A','B'):
            reqs=[];meta=[]
            for i,trace in enumerate(traces):
                names=tuple(a.name for a in (actions if i%2==0 else actions[::-1]))
                for role in ('ordered','native'):
                    uid=f'ordered-qual-v1|{tag}|{i}|{role}'
                    prefix=trace.completion_ids[:2048];prompt=trace.prompt_ids+prefix
                    first=names[0] if role=='ordered' else 'zero'
                    extra=Q.steer_extra(table,uid,first,len(trace.prompt_ids),prefix_len=len(prefix),restore_presence=True)
                    if role=='ordered':
                        template={'action_policy_names':list(names),'slots':[0,512],'horizon':1024}
                        extra['steer']['meta']={'routing_control':{**template,'sha256':digest(template)}}
                    sampling=Q.card_params(1024,Q.crn(world.infos[trace.question]),extra,
                        presence=0.,routed_start=len(prompt)-1)
                    reqs.append(Q.QReq(uid,prompt,sampling))
                    meta.append({'uid':uid,'role':role,'question':trace.question,'family':qfamily[trace.question],
                        'names':names,'prompt_len':len(prompt),'prompt_sha256':digest(prompt)})
            if tag=='A':
                trace=traces[0];prefix=trace.completion_ids[:2048]+[engine.THINK_END_ID]
                prompt=trace.prompt_ids+prefix;uid='ordered-qual-v1|A|closed'
                names=tuple(a.name for a in actions)
                extra=Q.steer_extra(table,uid,names[0],len(trace.prompt_ids),prefix_len=len(prefix),restore_presence=True)
                template={'action_policy_names':list(names),'slots':[0,512],'horizon':1024}
                extra['steer']['meta']={'routing_control':{**template,'sha256':digest(template)}}
                reqs.append(Q.QReq(uid,prompt,Q.card_params(128,Q.crn(world.infos[trace.question]),extra,
                    presence=0.,routed_start=len(prompt)-1)))
                meta.append({'uid':uid,'role':'closed','question':trace.question,'family':qfamily[trace.question],
                    'names':names,'prompt_len':len(prompt),'prompt_sha256':digest(prompt)})
            if tag=='B':harness.driver=PreemptingDriver(driver,(110,254,514,700))
            deadline.require(180,'fixed qualification batch '+tag)
            batch=harness.run(reqs);harness.flush();rank_records,_,_=harness.telemetry()
            evaluable=0;raw={}
            for item in meta:
                out=batch.outcomes.get(item['uid'])
                if out is None or out.error:checks.append({'uid':item['uid'],'pass':False,'reason':'missing/error'});continue
                routed=np.asarray(out.routed) if out.routed is not None else None
                reasons=[]
                if routed is None or routed.shape!=(len(out.tokens),40,8):reasons.append('routed shape')
                else:raw[item['uid']]=routed
                stop=next((i+1 for i,t in enumerate(out.tokens) if t==engine.THINK_END_ID),len(out.tokens))
                stop=min(stop,len(out.tokens))
                expected={name:0 for name in item['names']}
                segments={name:[] for name in item['names']}
                if item['role']=='ordered':
                    for slot,name in zip((0,512),item['names']):
                        end=min(slot+256,stop)
                        if end>slot:expected[name]+=end-slot;segments[name].append([slot,end])
                    evaluable+=int(stop>=768)
                for rank in (0,1):
                    record=rank_records.get(rank,{}).get(item['uid'])
                    if record is None:reasons.append('missing rank '+str(rank));continue
                    off=record.get('inactive_native_checks',{})
                    if not off or any(v['expert_identity_mismatches'] or v['weight_mismatches'] for v in off.values()):
                        reasons.append('inactive routing differs rank '+str(rank))
                    if item['role']=='native':
                        if record['cpu_active_rows']!=0:reasons.append('native neighbor active')
                        continue
                    if record.get('ordered_action_rows')!=expected or record.get('ordered_segments')!=segments:
                        reasons.append('pulse order/boundary/count rank '+str(rank))
                    doses=record.get('ordered_action_dose',{})
                    for name,count in expected.items():
                        if name not in doses or any(v['active_rows']!=count for v in doses.get(name,{}).values()):
                            reasons.append('device dose rows rank '+str(rank))
                    if tag=='B' and (record['preemptions']<1 or record['recompute_rows']<1):
                        reasons.append('no recovery rank '+str(rank))
                checks.append({'uid':item['uid'],'pass':not reasons,'reasons':reasons,'tokens':len(out.tokens),
                    'active_horizon':stop,'expected_action_rows':expected,'expected_segments':segments})
                all_rows.append({**item,'tokens':out.tokens,'finish':out.finish})
            if evaluable<3:checks.append({'batch':tag,'pass':False,'reason':'insufficient complete pulse windows','evaluable':evaluable})
            with (args.out/f'arrays-{tag}.npz').open('wb') as stream:np.savez_compressed(stream,**raw)
            events=getattr(harness.driver,'events',[])
            batches.append({'batch':tag,'seconds':batch.seconds,'steps':batch.steps,'evaluable':evaluable,
                'events':events,'requests':len(reqs),'prefill_tokens':sum(len(r.prompt) for r in reqs),
                'decode_tokens':sum(len(o.tokens) for o in batch.outcomes.values())})
            manifests.extend(meta)
        result.update(pass_=all(c['pass'] for c in checks),checks=checks,batches=batches,
            cold_build_seconds=build_seconds,fixture_plan=manifests,family_freeze=family['sha256'],
            final_worker_status=harness.status(),vllm_fingerprint=fingerprint,
            fixtures_note='Discovery families only; native future length screens engineering evaluability only, never a study detector or scientific enrollment rule.')
        result['pass']=result.pop('pass_')
        Q.atomic_json(args.out/'raw-results.json',all_rows)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';result['traceback']=traceback.format_exc()[-4000:]
    result['elapsed_driver_seconds']=time.time()-started
    result={**result,'sha256':digest(result)}
    Q.atomic_json(args.out/'QUALIFICATION.json',result)
    print(json.dumps({'pass':result['pass'],'path':str(args.out/'QUALIFICATION.json'),'error':result.get('error')}),flush=True)
    Q.finish_child(0 if result['pass'] else 3,driver)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--fixtures',type=Path)
    run(p.parse_args())
