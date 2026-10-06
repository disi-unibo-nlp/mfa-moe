"""M9 single-cut closure: amended resource gate for the frozen experiment.

The original manifest, assignments, sampler tree and closure driver remain
immutable. This launcher changes only the versioned complete-stage resource
gate and binds output receipts to its own code and amendment seals.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket

R=Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S=R/'steering-v1'


def helper(name):
    path=REPO/'src/moe_exp/routing_control'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('closure_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('native prefix preparation requires CPU Slurm')
    from moe_steer import manifests as M, results as RS, engine
    from transformers import AutoTokenizer
    C=helper('closure');split=M.load_split();world=M.load_world()
    legacy_parser_path=R/'forum/tests/text/code/forum_text/parser.py'
    parser_spec=importlib.util.spec_from_file_location('M9_legacy_parser',legacy_parser_path)
    legacy_parser=importlib.util.module_from_spec(parser_spec);parser_spec.loader.exec_module(legacy_parser)
    common_path=legacy_parser_path.with_name('common.py')
    if 'EMISSION_TEMPLATE = "Final answer: {candidate}"' not in common_path.read_text():
        raise ValueError('registered integer emission template changed')
    native=M.load_manifest(S/'manifests/x1-v1.json')
    questions={r['question'] for r in native['requests']
        if world.infos[r['question']]['split']=='dev' and world.infos[r['question']]['pool']=='L'}
    if len(questions)!=200:raise ValueError('M9 retains the original 200 dev-L questions')
    wanted={r['uid']:r for r in native['requests'] if r['question'] in questions and r['seed_k'] in (0,1)}
    if len(wanted)!=400:raise ValueError('M9 retains both seeds for all 200 questions')
    tokenizer=AutoTokenizer.from_pretrained(engine.snapshot_path(),local_files_only=True)
    rows=[];input_hashes={};seen=set()
    for shard in range(native['n_shards']):
        source=S/'runs/x1'/f'shard-{shard}.results.jsonl.gz'
        h=hashlib.sha256()
        with source.open('rb') as stream:
            while block:=stream.read(1<<20):h.update(block)
        input_hashes[str(source)]=h.hexdigest()
        for record in RS.read_shard_records(S/'runs/x1',shard):
            uid=record['uid']
            if uid not in wanted:continue
            if uid in seen:raise ValueError('duplicate native UID')
            RS.validate_result(record,wanted[uid]);seen.add(uid)
            if record['manifest_sha256']!=native['sha256']:raise ValueError('native source manifest changed')
            q=wanted[uid]['question'];prompt=native['questions'][q]['prompt_token_ids']
            ids=record['completion_token_ids'];natural=RS.is_natural_stop(record['finish_reason'],ids[-1] if ids else None)
            prepared=C.prepare_closure(prompt,ids,native_natural_stop=natural,
                encode=lambda s:tokenizer.encode(s,add_special_tokens=False))
            row={'uid':digest(['M9-single-cut-v1',uid]),'question':q,'seed_k':wanted[uid]['seed_k'],
                'seed':wanted[uid]['seed'],'native_uid':uid,'prepared':prepared,
                'native_endpoint_token_ids':ids[:C.TOTAL_BUDGET],
                'native_endpoint_natural_stop':natural and len(ids)<=C.TOTAL_BUDGET,
                'native_tokens_charged':min(len(ids),C.TOTAL_BUDGET),
                'native_full_reference_uid':uid,
                'integer_comparator':C.prepare_integer_comparator(ids,native_natural_stop=natural,
                    decode=lambda ts:tokenizer.decode(ts,skip_special_tokens=False),
                    encode=lambda text:tokenizer.encode(text,add_special_tokens=False),
                    candidates=legacy_parser.candidates),
                'reasoning_already_closed_at_cut':248069 in ids[:C.CUT]}
            rows.append(row)
    if seen!=set(wanted):raise ValueError('missing native assigned requests; never replace them')
    rows.sort(key=lambda row:hashlib.sha256(row['uid'].encode()).hexdigest())
    code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
        Path(inspect.getfile(C.prepare_closure)),REPO/'src/moe_exp/routing_control/receipts.py',
        legacy_parser_path,common_path]}
    body={'schema':'M9-single-cut-manifest-v1','native_manifest_sha256':native['sha256'],
        'native_inputs':input_hashes,'split_sha256':split['sha256'],'code':code,'code_tree':engine.code_tree_sha256(),
        'cut':C.CUT,'total_budget':C.TOTAL_BUDGET,'closure_text':C.CLOSURE_TEXT,'rows':rows,
        'counts':{status:sum(r['prepared']['status']==status for r in rows) for status in sorted({r['prepared']['status'] for r in rows})},
        'max_decode_tokens':sum(r['prepared'].get('max_new_tokens',0) for r in rows),
        'prefill_tokens':sum(len(r['prepared'].get('prompt_token_ids',[])) for r in rows),
        'integer_comparator_counts':{status:sum(r['integer_comparator']['status']==status for r in rows)
            for status in sorted({r['integer_comparator']['status'] for r in rows})},
        'integer_parser_limitation':'fixed legacy integer-only parser retains its documented incomplete-lookahead bug; new controller adapter is separate',
        'online_input_allowlist':['prepared.prompt_token_ids','prepared.max_new_tokens','prepared.prefix_presence_start','seed'],
        'native_and_integer_comparison_fields':'post-hoc references only, excluded from generation inputs',
        'grading':'strict math_verify plus frozen J1 on every required answer form; blinded bundle; missing judgments stay incomplete',
        'CPU_prepare_job_id':os.environ['SLURM_JOB_ID'],'inference':'HOLD replay qualification and complete generation+grading price'}
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    value={**body,'sha256':digest(body)}
    if args.manifest.exists() and json.loads(args.manifest.read_text())!=value:raise ValueError('manifest already frozen differently')
    args.manifest.write_text(json.dumps(value,separators=(',',':'))+'\n')
    print(json.dumps({'path':str(args.manifest),'sha256':value['sha256'],'counts':value['counts'],
        'max_decode_tokens':value['max_decode_tokens'],'prefill_tokens':value['prefill_tokens'],'inference':'HOLD'}))


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('closure inference requires allocated GPU compute nodes')
    from moe_steer import engine, results as RS
    value=json.loads(args.manifest.read_text())
    if digest({k:v for k,v in value.items() if k!='sha256'})!=value['sha256']:
        raise ValueError('closure manifest changed')
    if engine.code_tree_sha256()!=value['code_tree']:raise ValueError('closure sampler tree changed')
    for p,expected in value['code'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise ValueError('closure code changed')
    if not args.replay or not args.price:raise ValueError('replay and complete-price gate artifacts required')
    replay=json.loads(args.replay.read_text());price=json.loads(args.price.read_text())
    for gate in (replay,price):
        if digest({k:v for k,v in gate.items() if k!='sha256'})!=gate.get('sha256'):
            raise ValueError('unsealed closure gate artifact')
        if gate.get('manifest_sha256')!=value['sha256']:raise ValueError('closure gate is for another manifest')
    if (replay.get('pass') is not True or
        replay.get('schema')!='M9-deterministic-isolation-adjudication-v1' or
        replay.get('sampling_mode')!='deterministic_greedy_isolation_only' or
        replay.get('stochastic_batch_identity')!='FAILED_PREVIOUS_CHECK_UNRESOLVED' or
        not all(replay.get(k) is True for k in ('identical_prefix','prefix_presence','closure_injection','batch_isolation'))):
        raise ValueError('the matched-prefix and injected-closure replay must qualify before inference')
    source_price=price.get('source_price',{})
    if (not isinstance(source_price,dict) or
        digest({k:v for k,v in source_price.items() if k!='sha256'})!=source_price.get('sha256') or
        price.get('schema')!='M9-complete-stage-amendment-v1' or
        price.get('status')!='PASS_COMPLETE_STAGE_AMENDED' or
        price.get('source_price_sha256')!=source_price.get('sha256') or
        source_price.get('status')!='AMENDMENT_REQUIRED_BEFORE_PRODUCTION' or
        source_price.get('manifest_sha256')!=value['sha256'] or
        source_price.get('replay_sha256')!=replay['sha256'] or
        price.get('generation_replay_ceiling_GPU_h',0)<source_price.get('generation_replay_loads_GPU_h',float('inf')) or
        price.get('grading_ceiling_GPU_h',0)<source_price.get('grading_GPU_h',float('inf')) or
        price.get('driver_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('M9 amended complete-stage price or launcher binding is invalid')
    kwargs=engine.engine_kwargs(plugin=False,max_num_seqs=48)
    kwargs['logits_processors']=['moe_steer.logits:PrefixPresencePenalty']
    kwargs.update(gpu_memory_utilization=.80,long_prefill_token_threshold=1024)
    env=engine.engine_env(None,None,plugin=False)
    for key in tuple(os.environ):
        if key.startswith('STEER_') and key not in env:os.environ.pop(key)
    engine.apply_env(env)
    from vllm import LLM,SamplingParams
    receipts=helper('receipts')
    binding={'manifest_sha256':value['sha256'],'replay_sha256':replay['sha256'],
             'price_sha256':price['sha256'],'driver_sha256':price['driver_sha256']}
    with receipts.ReceiptStore(args.out,binding,[r['uid'] for r in value['rows']]) as store:
        attempt=store.attempt();done=store.read()
        receipts.atomic_json(attempt/'START.json',{'job_id':os.environ['SLURM_JOB_ID'],'already_complete':len(done)})
        pending=[]
        for row in value['rows']:
            if row['uid'] in done:continue
            p=row['prepared']
            if p['status']=='incomplete_prefix':
                store.put({'uid':row['uid'],'status':'FAILED_SOURCE_PREFIX','operational_correct':False,
                    'tokens_charged':p['tokens_charged']})
            elif p['status'].startswith('retain_native'):
                store.put({'uid':row['uid'],'status':p['status'],
                    'completion_token_ids':row['native_endpoint_token_ids'],
                    'natural_stop':row['native_endpoint_natural_stop'],'tokens_charged':row['native_tokens_charged']})
            else:pending.append(row)
        model=LLM(**kwargs) if pending else None
        for start in range(0,len(pending),48):
            block=pending[start:start+48];prompts=[];params=[]
            for row in block:
                p=row['prepared'];prompts.append({'prompt_token_ids':p['prompt_token_ids']})
                params.append(SamplingParams(**engine.card_sampling(p['max_new_tokens'],row['seed'],presence_penalty=0.,
                    extra_args={'steer':{'prefix_presence':{'penalty':1.5,'prefix_start':p['prefix_presence_start']}}})))
            outputs=model.generate(prompts,params,use_tqdm=False)
            if len(outputs)!=len(block):raise ValueError('incomplete closure batch')
            for row,output in zip(block,outputs):
                p=row['prepared'];completion=output.outputs[0];ids=list(completion.token_ids)
                if list(output.prompt_token_ids)!=p['prompt_token_ids'] or len(ids)>p['max_new_tokens']:
                    raise ValueError('closure prefix or continuation budget changed')
                natural=RS.is_natural_stop(completion.finish_reason,ids[-1] if ids else None)
                if not natural and len(ids)!=p['max_new_tokens']:raise ValueError('closure stopped before cap without EOS')
                cumulative=p['prompt_token_ids'][p['prefix_presence_start']:] + ids
                store.put({'uid':row['uid'],'status':'GENERATED','completion_token_ids':cumulative,
                    'new_answer_token_ids':ids,'natural_stop':natural,'tokens_charged':len(cumulative),
                    'injection_tokens':p['injection_tokens'],'finish_reason':completion.finish_reason})
        result=store.read()
        receipts.atomic_json(args.out/'closure-results.json',{'binding':binding,'records':list(result.values()),
            'grading_status':'PENDING blinded strict+J1; no utility result before complete grading'})
        receipts.atomic_json(attempt/'END.json',{'status':'COMPLETE_GENERATION','job_id':os.environ['SLURM_JOB_ID']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--replay',type=Path);p.add_argument('--price',type=Path);args=p.parse_args()
    if args.command=='run' and args.out is None:p.error('run requires out')
    prepare(args) if args.command=='prepare' else run(args)
