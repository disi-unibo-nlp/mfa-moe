"""Frozen J1 runtime reuse, exact complete pricing and CPU/GPU attachment chain."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess

import dispatch_overnight_readers_v1 as shared
import dispatch_generated_dense_v1 as verify
import submit_fresh_veto_sensitivity_v2 as dependencies

DOC=shared.DOC
PLAN=DOC/'UTILITY_J1_CHAIN_PLAN_v1.json'
ENVELOPE=DOC/'UTILITY_J1_MAX384_ENVELOPE_v1.json'
REUSE=DOC/'UTILITY_J1_RUNTIME_REUSE_AUDIT_v1.json'
SCRIPTS=Path(__file__).parent
STAGE=shared.RUNS.parents[1]
BASE=STAGE/'code/s1-9a61e32f48c04c24'
LEGACY=BASE/'sbatch/steer_j1.sbatch'
HIST_LOG=STAGE/'provenance/vllm-j1-59068750.log'
HIST_STDOUT=STAGE/'logs/st-gpu-8-59068750.out'
HIST_ITEMS=STAGE/'runs/x1/score/items.jsonl'
MODEL=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
OUTCOME_PLAN=DOC/'UTILITY_OUTCOME_MEASUREMENT_PLAN_v3.json'
MODES=('attach','prepare','dispatch','finalize')
WRAPPERS={mode:SCRIPTS/('utility_j1_'+mode+'_v1.sbatch') for mode in MODES}
require=shared.reader.require
sealed=shared.base.sealed
sha=shared.base.file_sha
digest=shared.base.digest


def outcome_module():
    import utility_outcomes_v3 as outcomes
    return outcomes


def code_files():
    outcomes=outcome_module()
    paths=[Path(__file__),Path(shared.__file__),Path(verify.__file__),Path(dependencies.__file__),
           Path(outcomes.__file__),Path(outcomes.__file__).with_suffix('.sbatch'),LEGACY,*WRAPPERS.values()]
    return {str(p.resolve()):sha(p) for p in paths}


def history():
    log=HIST_LOG.read_text()
    rates=[tuple(map(float,m.groups())) for m in re.finditer(r'Avg prompt throughput: ([0-9.]+) tokens/s, Avg generation throughput: ([0-9.]+) tokens/s',log)]
    prefill=[x for x,y in rates if x>0];decode=[y for x,y in rates if y>0]
    stdout=HIST_STDOUT.read_text()
    require('judge_ready seconds=736' in stdout and '{"items": 228, "todo": 228}' in stdout and 'J1_COMPLETE' in stdout and
            len(decode)==80 and min(decode)==13.9 and statistics.median(decode)==114.1,
            'historical J1 timing source differs')
    return {'job_id':'59068750','observed_parent_state':'COMPLETED','observed_exit_code':'0:0',
        'observed_parent_wall_seconds':1552,'observed_GPUs':2,'observed_GPU_hours':2*1552/3600,
        'observed_items':228,'observed_votes':684,'observed_ready_seconds':736,
        'observed_active_plus_shutdown_seconds':1552-736,
        'median_prompt_tokens_per_second':statistics.median(prefill),
        'minimum_positive_prompt_tokens_per_second':min(prefill),
        'median_output_tokens_per_second':statistics.median(decode),
        'minimum_positive_output_tokens_per_second':min(decode),
        'source_files':{str(p):sha(p) for p in (HIST_LOG,HIST_STDOUT,HIST_ITEMS)},
        'interpretation':'Logged aggregate rates from the same historical server profile; median is a planning assumption, not a guaranteed lower bound. Minimum positive interval is a pessimistic sensitivity, not expected performance. Original parent success and server-step cancellation are separately preserved.'}


def binding():
    reuse=sealed(REUSE);measurement=sealed(OUTCOME_PLAN)
    require(reuse['sha256']=='54f714af623e82c05cedd83f9cb927fcec43f6f6cb960c543bef8f2488c170a3' and
            reuse['wrapper_sha256']==sha(LEGACY) and reuse['snapshot_path']==str(BASE), 'frozen J1 reuse sources changed')
    require(all(sha(path)==value for path,value in measurement['code_files'].items()),'utility v3 measurement source changed')
    require(measurement['sha256']=='19314ebda38fb2b51bca615e931637470639128498568b841904184657964195','utility v3 measurement plan pin differs')
    return {'schema':'utility-j1-chain-plan-v1','reuse_audit_sha256':reuse['sha256'],
        'outcome_measurement_plan_sha256':measurement['sha256'],'code_files':code_files(),
        'legacy_profile':reuse['server'],'client_profile':reuse['client'],'history':history(),
        'max_shard_wall_seconds':21600,'max_complete_items_per_shard':48,'work_allowance':1.25,
        'load_seconds_projection':736,'shutdown_seconds_projection':240,'per_shard_reserve_seconds':900,
        'cold_readiness_limit_seconds':2400,'transport_timeout_seconds':1800,
        'transport_attempts_per_vote':3,'votes':3,'max_tokens_per_vote':8192,
        'transport_sleep_seconds_per_vote':30,
        'credential_policy':'Whitelist basic module/runtime environment only, then add exact nonsecret STEER path/tree/provenance pins. No inherited HF, token, credential, provider or agent variables reach the logging wrapper.',
        'pricing':'Exact current prompt IDs, three votes, three possible transport attempts per vote, all capped decode and each shard load/shutdown/reserve. Historical item-rate forecast, median-rate full-cap projection, and min-positive-rate pessimistic projection remain distinct. Client timeout envelope is reported separately and <=6h per <=48-item shard.',
        'retry_policy':'Only unchanged client transport retries are automatic. No new generation or repeated judgment on semantic grounds. Interrupted physical jobs require explicit reconciliation and a new receipt.',
        'resource_scope':'CPU preparation/pricing and final analysis each2cores/2h; one-shot attachments each2cores/10min. GPU jobs use exact legacy16CPU/120GiB/2GPU wrapper, at most6h each; actual sacct allocation cost is authoritative.'}


def validate_plan():
    plan=sealed(PLAN)
    require(plan['binding']==binding(),'frozen utility J1 chain sources changed')
    return plan


def exact_ids(value):
    if isinstance(value,Mapping):value=value.get('input_ids')
    if hasattr(value,'tolist'):value=value.tolist()
    if isinstance(value,(tuple,list)) and len(value)==1 and isinstance(value[0],(tuple,list)):value=value[0]
    require(isinstance(value,(list,tuple)) and value and all(type(x)is int and x>=0 for x in value),'invalid exact judge prompt IDs')
    return list(value)


def messages(item,judge):
    return [{'role':'system','content':judge.SYSTEM},{'role':'user','content':
        f"Problem:\n{item['problem']}\n\nReference answer:\n{item['gold']}\n\nCandidate:\n{item['candidate']}"}]


def safe_environment(env, additions=None):
    # The exact legacy wrapper logs every HF_* and STEER_* variable. None is inherited.
    allowed={'PATH','HOME','USER','LOGNAME','SHELL','LANG','LC_ALL','LC_CTYPE','TERM','PWD','TZ',
        'LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','PKG_CONFIG_PATH',
        'MODULEPATH','MODULESHOME','MODULES_CMD','MODULESBEGINENV','LOADEDMODULES','_LMFILES_',
        'LMOD_CMD','LMOD_DIR','LMOD_SETTARG_CMD','LMOD_VERSION','MODULES_USE_COMPAT_VERSION',
        'CUDA_HOME','CUDA_PATH','BASH_FUNC_module%%','BASH_FUNC_ml%%','BASH_FUNC_switchml%%'}
    result={k:v for k,v in env.items() if k in allowed}
    if additions:
        require(set(additions)<={'STEER_CODE','STEER_EXPECT_TREE','STEER_PROVENANCE_DIR'},'unapproved submission environment additions')
        result.update(additions)
    return shared.submission_environment(result)


def price_rows(rows, settings):
    hist=settings['history'];repeat=settings['work_allowance'];votes=settings['votes'];attempts=settings['transport_attempts_per_vote'];cap=settings['max_tokens_per_vote']
    load=settings['load_seconds_projection'];shutdown=settings['shutdown_seconds_projection'];reserve=settings['per_shard_reserve_seconds']
    def estimates(block):
        prompt=sum(r['prompt_tokens'] for r in block)*votes*attempts
        decode=len(block)*votes*attempts*cap
        median=repeat*(prompt/hist['median_prompt_tokens_per_second']+decode/hist['median_output_tokens_per_second'])
        pessimistic=repeat*(prompt/hist['minimum_positive_prompt_tokens_per_second']+decode/hist['minimum_positive_output_tokens_per_second'])
        client=math.ceil(len(block)/48)*votes*(attempts*settings['transport_timeout_seconds']+settings['transport_sleep_seconds_per_vote'])
        return {'exact_prompt_tokens_all_attempts':prompt,'max_decode_tokens_all_attempts':decode,
            'median_full_cap_work_seconds':median,'pessimistic_full_cap_work_seconds':pessimistic,
            'median_full_cap_wall_seconds':load+shutdown+reserve+median,
            'pessimistic_full_cap_wall_seconds':load+shutdown+reserve+pessimistic,
            'client_timeout_envelope_wall_seconds':settings['cold_readiness_limit_seconds']+shutdown+reserve+client,
            'historical_item_rate_forecast_wall_seconds':load+shutdown+reserve+repeat*hist['observed_active_plus_shutdown_seconds']*len(block)/hist['observed_items']}
    shards=[];block=[]
    for row in rows:
        require(type(row['prompt_tokens'])is int and row['prompt_tokens']>0 and row['prompt_tokens']+cap<=24576,'judge prompt exceeds exact frozen context')
        proposed=block+[row];cost=estimates(proposed)
        if block and (len(proposed)>settings['max_complete_items_per_shard'] or
                      max(cost['median_full_cap_wall_seconds'],cost['client_timeout_envelope_wall_seconds'])>settings['max_shard_wall_seconds']):
            shards.append({'rows':block,**estimates(block)});block=[row]
        else:block=proposed
        single=estimates(block)
        require(max(single['median_full_cap_wall_seconds'],single['client_timeout_envelope_wall_seconds'])<=settings['max_shard_wall_seconds'],'complete item cannot fit six-hour shard; revise proposal before submission')
    if block:shards.append({'rows':block,**estimates(block)})
    for index,shard in enumerate(shards):
        shard.update(index=index,item_ids=[r['item_id'] for r in shard.pop('rows')],requested_wall_seconds=settings['max_shard_wall_seconds'])
    return {'schema':'utility-j1-complete-price-v1','items':len(rows),'votes':votes*len(rows),
        'max_transport_attempts':votes*attempts*len(rows),'shards':shards,'cold_loads':len(shards),'shutdowns':len(shards),
        'historical_item_rate_forecast_GPU_h':sum(s['historical_item_rate_forecast_wall_seconds'] for s in shards)*2/3600,
        'median_full_cap_all_attempts_projected_GPU_h':sum(s['median_full_cap_wall_seconds'] for s in shards)*2/3600,
        'pessimistic_minimum_positive_rate_projected_GPU_h':sum(s['pessimistic_full_cap_wall_seconds'] for s in shards)*2/3600,
        'client_timeout_envelope_GPU_h':sum(s['client_timeout_envelope_wall_seconds'] for s in shards)*2/3600,
        'requested_allocation_GPU_hour_ceiling':sum(s['requested_wall_seconds'] for s in shards)*2/3600,
        'status':'NO_J1_ITEMS_CPU_ONLY' if not rows else 'PASS_COMPLETE_MEDIAN_PROJECTION_AND_CLIENT_TIMEOUT_ENVELOPES',
        'interpretation':'Median and historical views are forecasts, not runtime guarantees. Pessimistic min-positive interval full-cap projection is a separate sensitivity and may exceed requested wall; the unchanged client still exits after bounded transport attempts, potentially with UNCERTAIN judgments. No automatic semantic retry. All scheduled allocation, including failed/uncertain judgments, is counted.'}


def read_items(prep,root):
    path=root/'preparation/items.jsonl'
    require(sha(path)==prep['files']['items']['file_sha256'],'prepared items file changed')
    with path.open() as stream:
        items=[json.loads(line) for line in stream if line.strip()]
    require(len(items)==prep['J1_items'] and len({i['item_id'] for i in items})==len(items),'duplicate/missing blind J1 items')
    return items


def stage_paths(index_path,out_parent):
    index=sealed(index_path)
    require(index['schema']=='utility-production-reconciled-index-v1','unsupported utility index')
    parent=Path(out_parent).resolve()
    require(str(parent).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/'),'grading output leaves owned work tree')
    return index,parent/('outcomes-v3-'+index['sha256'][:16])


def prepare(index_path,out_parent):
    plan=validate_plan();index,root=stage_paths(index_path,out_parent)
    root.mkdir(parents=True,exist_ok=True)
    stage_binding={'schema':'utility-j1-stage-binding-v1','chain_plan_sha256':plan['sha256'],
        'index_path':str(index_path.resolve()),'index_sha256':index['sha256']}
    shared.save(root/'BINDING.json',stage_binding)
    outcomes=outcome_module();prep_path=root/'preparation/GRADE_PREP.json'
    if not prep_path.exists():outcomes.prepare(index_path,root/'preparation')
    prep=sealed(prep_path)
    require(prep['index_sha256']==index['sha256'] and prep['measurement_plan_sha256']==plan['binding']['outcome_measurement_plan_sha256'],'grading preparation rebound')
    items=read_items(prep,root)
    score,_=outcomes.scoring_modules();judge=score.answer_equivalence
    score.assert_blind(items,'utility J1 exact price')
    tokenizer=None
    if items:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rows=[]
    for item in items:
        ids=exact_ids(tokenizer.apply_chat_template(messages(item,judge),tokenize=True,add_generation_prompt=True,enable_thinking=True))
        rows.append({'item_id':item['item_id'],'prompt_tokens':len(ids),'prompt_ids_sha256':digest(ids),
                     'messages_sha256':digest(messages(item,judge)),
                     'vote_seeds':[int(item['item_id'][:8],16)%1000000+i for i in range(3)]})
    price=price_rows(rows,plan['binding'])
    envelope=sealed(ENVELOPE)
    require(envelope['chain_plan_sha256']==plan['sha256'] and len(items)<=envelope['maximum_items'] and
            price['requested_allocation_GPU_hour_ceiling']<=envelope['requested_allocation_GPU_hour_ceiling'],
            'exact grading price exceeds the separately funded generic envelope')
    (root/'j1').mkdir(exist_ok=True)
    by_id={i['item_id']:i for i in items}
    for shard in price['shards']:
        directory=root/'j1'/('shard-'+str(shard['index']).zfill(3));directory.mkdir(exist_ok=True)
        path=directory/'items.jsonl'
        content=''.join(json.dumps(by_id[item_id],ensure_ascii=False)+'\n' for item_id in shard['item_ids'])
        if path.exists():require(path.read_text()==content,'existing J1 shard input changed')
        else:path.write_text(content)
        shard.update(items_path=str(path),items_file_sha256=sha(path),verdicts_path=str(directory/'verdicts.jsonl'))
    value=shared.save(root/'J1_PRICE.json',{**price,'plan_sha256':plan['sha256'],'grade_preparation_sha256':prep['sha256'],
        'items_file_sha256':prep['files']['items']['file_sha256'],'prompt_records':rows,'max384_envelope_sha256':envelope['sha256'],
        'tokenizer_files':{str(p):sha(p) for p in (MODEL/'tokenizer.json',MODEL/'tokenizer_config.json') if p.exists()},
        'source_history':plan['binding']['history'],'model_snapshot':str(MODEL)})
    print(json.dumps({'grade_root':str(root),'price_sha256':value['sha256'],'items':len(items),'shards':len(price['shards']),
        'median_full_cap_all_attempts_projected_GPU_h':price['median_full_cap_all_attempts_projected_GPU_h']}),flush=True)


def priced(root):
    plan=validate_plan();bound=sealed(root/'BINDING.json');prep=sealed(root/'preparation/GRADE_PREP.json');price=sealed(root/'J1_PRICE.json')
    require(bound['chain_plan_sha256']==price['plan_sha256']==plan['sha256'] and
            bound['index_sha256']==prep['index_sha256'] and price['grade_preparation_sha256']==prep['sha256'], 'grading stage/price binding differs')
    envelope=sealed(ENVELOPE)
    require(price['max384_envelope_sha256']==envelope['sha256'] and envelope['chain_plan_sha256']==plan['sha256'] and
            price['items']<=envelope['maximum_items'] and price['requested_allocation_GPU_hour_ceiling']<=envelope['requested_allocation_GPU_hour_ceiling'], 'grading envelope changed or exceeded')
    items=read_items(prep,root)
    expected=price_rows(price['prompt_records'],plan['binding'])
    require([r['item_id'] for r in price['prompt_records']]==[i['item_id'] for i in items] and
            all(price[k]==v for k,v in expected.items() if k!='shards') and len(price['shards'])==len(expected['shards']) and
            all(all(actual[k]==v for k,v in original.items()) and sha(actual['items_path'])==actual['items_file_sha256']
                for actual,original in zip(price['shards'],expected['shards'],strict=True)), 'complete price or exact shard inputs differ')
    require(all(sha(p)==h for p,h in price['tokenizer_files'].items()),'judge tokenizer changed')
    return plan,prep,price,items


def dispatch(root):
    plan,prep,price,items=priced(root)
    directory=root/'dispatch';directory.mkdir(exist_ok=True)
    common={'schema':'utility-j1-dispatch-binding-v1','plan_sha256':plan['sha256'],
            'grade_preparation_sha256':prep['sha256'],'price_sha256':price['sha256']}
    with (directory/'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify.live_account_snapshot(directory,'LIVE_ACCOUNT-'+os.environ['SLURM_JOB_ID']+'.json')
        jobs=[]
        env=safe_environment(os.environ,{'STEER_CODE':str(BASE),'STEER_EXPECT_TREE':sealed(BASE/'MANIFEST.json')['tree_sha256'],
             'STEER_PROVENANCE_DIR':str(root/'j1/provenance')})
        for shard in price['shards']:
            name='utility-j1-'+str(shard['index']).zfill(3)
            job=shared.submit(directory,name,['--time=06:00:00','--job-name='+name,str(LEGACY),shard['items_path'],shard['verdicts_path']],env,
                              {**common,'shard':shard['index'],'items_file_sha256':shard['items_file_sha256'],'wrapper_sha256':sha(LEGACY)})
            verify.ensure_verified(directory,name,job);jobs.append(job)
        # Submission receipts remain authoritative on resume; successful parents can be purged.
        args=[];proofs=[]
        for shard,job in zip(price['shards'],jobs,strict=True):
            dep,proof=dependencies.dependency(job,lambda s=shard: {'validated_items':len(valid_verdicts(Path(s['verdicts_path']),s['item_ids']))})
            if dep:args.append(job)
            proofs.append(proof)
        name='utility-j1-finalize';receipt=directory/(name+'.json')
        if receipt.exists():
            saved=sealed(receipt);require(all(saved['binding'].get(k)==v for k,v in common.items()),'prior finalizer changed');job=saved['job_id']
        else:
            job=shared.submit(directory,name,[*(['--dependency=afterok:'+':'.join(args)] if args else []),str(WRAPPERS['finalize']),'--grade-root',str(root)],
                safe_environment(os.environ),{**common,'parent_jobs':jobs,'dependency_proofs':proofs})
        verify.ensure_verified(directory,name,job)
        result=shared.save(directory/'CHAIN.json',{'schema':'utility-j1-dispatch-chain-v1',**common,'J1_jobs':jobs,'finalize_job':job,'grade_root':str(root)})
    print(json.dumps(result),flush=True)


def valid_verdicts(path,item_ids):
    outcomes=outcome_module();score,_=outcomes.scoring_modules();judge=score.answer_equivalence
    with path.open() as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    require(len(rows)==len(item_ids) and len({r['item_id'] for r in rows})==len(rows) and
            {r['item_id'] for r in rows}==set(item_ids),'J1 output misses/duplicates/adds item IDs')
    for row in rows:
        votes=row['votes']
        require(isinstance(votes,list) and len(votes)==3 and all(isinstance(v,str) and (v in {'EQUIVALENT','NOT_EQUIVALENT','UNCERTAIN','UNPARSED','TRUNCATED'} or
                re.fullmatch(r'ERROR:[A-Za-z_][A-Za-z_0-9]*',v)) for v in votes) and
                row['parser_version']==judge.PARSER_VERSION==2 and row['prompt_version']==judge.PROMPT_VERSION==1 and
                row['sampler']==judge.JUDGE_SAMPLER,'J1 vote/parser/prompt/sampler contract differs')
        top,n=Counter(votes).most_common(1)[0]
        expected=top if n>=2 and top in ('EQUIVALENT','NOT_EQUIVALENT') else 'UNCERTAIN'
        require(row['verdict']==expected and type(row['unanimous'])is bool and row['unanimous']==(n==3) and
                row['truncated_votes']==sum(v=='TRUNCATED' for v in votes),'J1 vote majority or truncation record differs')
    return rows


def allocation(job):
    result=subprocess.run(['sacct','-X','-nP','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],check=True,capture_output=True,text=True)
    rows=[line.strip().rstrip('|').split('|') for line in result.stdout.splitlines() if line.strip()]
    exact=[r for r in rows if r[0]==job]
    require(len(exact)==1 and exact[0][1:3]==['COMPLETED','0:0'],'J1 parent allocation did not complete successfully')
    row=exact[0];match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',row[4])
    require(match is not None and int(match.group(1))==2,'J1 allocation GPU count differs')
    return {'job_id':job,'state':row[1],'exit_code':row[2],'elapsed_seconds':int(row[3]),'GPUs':2,
            'allocated_GPU_hours':2*int(row[3])/3600,'sacct_stdout':result.stdout}


def finalize(root):
    plan,prep,price,items=priced(root)
    chain=sealed(root/'dispatch/CHAIN.json')
    require(chain['price_sha256']==price['sha256'] and len(chain['J1_jobs'])==len(price['shards']),'grading job chain differs')
    rows=[];cost=[]
    for shard,job in zip(price['shards'],chain['J1_jobs'],strict=True):
        provenance=sealed(root/'j1/provenance'/(job+'.json'))
        require(provenance['schema']=='steer-provenance-v1' and str(provenance['job_id'])==job and
                provenance['mode']=='j1' and provenance['dry_run'] is False and provenance['live_code'] is False and
                provenance['tree_sha256']==sealed(BASE/'MANIFEST.json')['tree_sha256'] and
                provenance['launcher_sha256']==sha(LEGACY) and provenance['items_sha256']==shard['items_file_sha256'] and
                provenance['args']==[shard['items_path'],shard['verdicts_path']] and
                provenance['versions']=={'vllm':'0.29.0+cu129','torch':'2.13.0+cu129'},
                'actual J1 runtime provenance differs from the frozen legacy wrapper/profile')
        rows.extend(valid_verdicts(Path(shard['verdicts_path']),shard['item_ids']))
        cost.append({**allocation(job),'runtime_provenance_sha256':provenance['sha256']})
    require(len({r['item_id'] for r in rows})==len(rows)==len(items),'whole J1 stage incomplete')
    by_id={r['item_id']:r for r in rows};ordered=[by_id[i['item_id']] for i in items]
    verdicts=root/'verdicts.jsonl';text=''.join(json.dumps(r)+'\n' for r in ordered)
    if verdicts.exists():require(verdicts.read_text()==text,'existing merged verdicts differ')
    else:verdicts.write_text(text)
    shared.save(root/'J1_COMPLETION.json',{'schema':'utility-j1-stage-completion-v1','plan_sha256':plan['sha256'],
        'grade_preparation_sha256':prep['sha256'],'price_sha256':price['sha256'],'items':len(items),
        'verdicts_file_sha256':sha(verdicts),'job_allocations':cost,'actual_J1_allocated_GPU_hours':sum(r['allocated_GPU_hours'] for r in cost),
        'cost_scope':'J1 allocation only. Add to generation/side-reader allocation totals once; do not add forecast or token-derived cost again.',
        'per_shard_verdicts':{s['verdicts_path']:sha(s['verdicts_path']) for s in price['shards']}})
    outcome_module().analyze(root/'preparation/GRADE_PREP.json',verdicts,root/'analysis')


def cpu_once(directory,name,args,binding,parent=None,artifact_check=None):
    path=directory/(name+'.json')
    if path.exists():
        old=sealed(path);require(all(old['binding'].get(k)==v for k,v in binding.items()),'CPU attachment rebound')
        verify.ensure_verified(directory,name,old['job_id']);return old['job_id']
    dep,proof=dependencies.dependency(parent,artifact_check) if parent else ([],None)
    job=shared.submit(directory,name,[*dep,*args],safe_environment(os.environ),{**binding,'parent_job':parent,'dependency_proof':proof})
    verify.ensure_verified(directory,name,job);return job


def attach(chain_dir,out_parent):
    plan=validate_plan();chain_dir=Path(chain_dir).resolve();parent=Path(out_parent).resolve() if out_parent else None
    initial=sealed(chain_dir/'INITIAL_CHAIN.json')
    require(initial['schema']=='utility-production-initial-chain-v1','utility production initial chain differs')
    directory=chain_dir/('j1-attachment-'+plan['sha256'][:12]);directory.mkdir(exist_ok=True)
    common={'schema':'utility-j1-production-attachment-v1','plan_sha256':plan['sha256'],'initial_chain_sha256':initial['sha256']}
    with (directory/'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify.live_account_snapshot(directory,'LIVE_ACCOUNT-'+os.environ['SLURM_JOB_ID']+'.json')
        final_path=chain_dir/'FINAL_CHAIN.json'
        if not final_path.exists():
            recovering=(chain_dir/'RECOVERY_CHAIN.json').exists()
            source=chain_dir/('RECOVERY_CHAIN.json' if recovering else 'INITIAL_CHAIN.json');chain=sealed(source)
            name='utility-j1-wait-recovery' if recovering else 'utility-j1-wait-initial'
            def done():
                candidates=[final_path] if recovering else [final_path,chain_dir/'RECOVERY_CHAIN.json']
                found=[p for p in candidates if p.exists()]
                require(found,'completed production accounting produced no final/recovery receipt')
                return {str(p):sealed(p)['sha256'] for p in found}
            job=cpu_once(directory,name,[str(WRAPPERS['attach']),'--chain-dir',str(chain_dir),*(['--out-parent',str(parent)] if parent else [])],
                {**common,'accounting_receipt_sha256':chain['sha256']},str(chain['account_job']),done)
            print(json.dumps({'next_attachment_job':job}),flush=True);return
        final=sealed(final_path);index_path=Path(final['reconciled_index'])
        index=sealed(index_path)
        require(final['schema']=='utility-production-final-chain-v1' and
                final['reconciled_index_sha256']==index['sha256'] and index['production_manifest_sha256']==initial['manifest_sha256'],
                'final production index/manifest binding differs')
        parent=parent or Path(initial['output'])
        _,root=stage_paths(index_path,parent)
        bind={**common,'final_chain_sha256':final['sha256'],'index_sha256':index['sha256'],'grade_root':str(root)}
        prep=cpu_once(directory,'utility-j1-prepare',[str(WRAPPERS['prepare']),'--index',str(index_path),'--out-parent',str(parent)],bind)
        def prepared():
            price=sealed(root/'J1_PRICE.json');return {'price_sha256':price['sha256'],'grade_preparation_sha256':price['grade_preparation_sha256']}
        dispatcher=cpu_once(directory,'utility-j1-dispatch',[str(WRAPPERS['dispatch']),'--grade-root',str(root)],bind,prep,prepared)
        shared.save(directory/'CHAIN.json',{'schema':'utility-j1-preparation-chain-v1',**bind,'prepare_job':prep,'dispatcher_job':dispatcher})
        print(json.dumps({'prepare_job':prep,'dispatcher_job':dispatcher,'grade_root':str(root)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=(*MODES,'freeze-plan'))
    p.add_argument('--index',type=Path);p.add_argument('--out-parent',type=Path);p.add_argument('--grade-root',type=Path);p.add_argument('--chain-dir',type=Path)
    args=p.parse_args()
    if args.mode=='freeze-plan':
        value=shared.save(PLAN,{'schema':'utility-j1-chain-plan-envelope-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),'binding':binding()})
        price=price_rows([{'item_id':format(i,'024x'),'prompt_tokens':16384} for i in range(384)],value['binding'])
        envelope=shared.save(ENVELOPE,{'schema':'utility-j1-max384-envelope-v1','maximum_items':384,
            'maximum_prompt_tokens_per_item':16384,'maximum_complete_shards':len(price['shards']),
            'outcome_measurement_plan_sha256':value['binding']['outcome_measurement_plan_sha256'],
            'chain_plan_sha256':value['sha256'],**{k:price[k] for k in ('requested_allocation_GPU_hour_ceiling','median_full_cap_all_attempts_projected_GPU_h',
                'historical_item_rate_forecast_GPU_h','pessimistic_minimum_positive_rate_projected_GPU_h','client_timeout_envelope_GPU_h')},
            'complete_generic_price':price,'scope':'Resource envelope only:384 possible blind items with maximum supported prompts. Placeholder item IDs are not assignments. Exact later prompts/items are priced before GPU submission; no semantic candidate selection.'})
        print(json.dumps({'plan_sha256':value['sha256'],'envelope_sha256':envelope['sha256']}));return
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and os.environ.get('SLURM_JOB_PARTITION')=='lrd_all_viz' and
            getpass.getuser()=='lmolfett' and not socket.gethostname().startswith('login'),'J1 orchestration/analysis requires CPU Slurm step')
    if args.mode=='prepare':
        require(args.index and args.out_parent,'prepare needs index and out-parent');prepare(args.index,args.out_parent)
    elif args.mode in ('dispatch','finalize'):
        require(args.grade_root,'mode needs grade-root');globals()[args.mode](args.grade_root)
    else:
        require(args.chain_dir,'attach needs production chain-dir');attach(args.chain_dir,args.out_parent)


if __name__=='__main__':main()
