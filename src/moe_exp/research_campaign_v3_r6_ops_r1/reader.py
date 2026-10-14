"""Versioned reader lifecycle correction; original scientific helpers are reused."""
from pathlib import Path
import os
import sys
import time
from moe_exp.research_campaign_v3_r6.contracts import SEED, allocated, check_files, digest, load, publish, require
from moe_exp.research_campaign_v3_r6.reader_contract import ReaderUnits, token_ids
from moe_exp.research_campaign_v3_r6.reader_engine import parse_output, recover_raw
from moe_exp.research_campaign_v3_r6.lifecycle import cleanup, deadline_before


def abort_external(engine, uids):
    """Original assignment UIDs are external vLLM request IDs."""
    return engine.abort_request(uids, internal=False)


def annotate(campaign, model, rows, deadline):
    from .policy import check_binding
    check_binding(campaign)
    allocated(gpu=True)
    from moe_exp.research_campaign_v3_r6.annotation import Renderer
    from transformers import AutoTokenizer
    root=Path(campaign['root']);units=ReaderUnits(root/model/'annotations');a=campaign['annotator']
    check_files(campaign['code_files']);check_files(campaign.get('reader_runtime_files',{}))
    renderer=Renderer(campaign);llm=None
    try:
        tokenizer=AutoTokenizer.from_pretrained(a['path'],local_files_only=True)
        recover_raw(campaign,model,rows,tokenizer,renderer)
        pending=[r for r in rows if units.state(r['uid']) is None]
        if not pending:return {'status':'reused','units':len(rows)}
        # Every input and context allowance is validated before loading weights
        # or writing a scientific submission intent.
        prepared=[]
        for row in pending:
            deadline_before(deadline,120,'reader input preparation')
            conversation=renderer.request('render',inputs=row['inputs'])['conversation']
            prompt=token_ids(tokenizer,conversation,a['config'])
            folder=units.root/'lifecycle'/row['uid']
            publish(folder/'PREPARED.json',{'uid':row['uid'],'assignment_sha256':digest(row),
                'prompt_sha256':digest(prompt),'input_tokens':len(prompt),'engine_admitted':False,
                'output_allowance':min(a['config']['max_tokens'],a['native_context']-len(prompt))})
            if len(prompt)>=a['native_context']:
                if units.begin(row,os.environ['SLURM_JOB_ID']):
                    units.commit(row,{'status':'upstream_unavailable','label':None,'unknown':True,
                        'reason':'complete original annotation context exceeds original reader context',
                        'input_tokens':len(prompt),'engine_admitted':False,'draw_started':False})
                continue
            prepared.append((row,prompt))
        if not prepared:return {'status':'complete','units':len(rows),'all_contexts_unavailable':True}
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import RequestOutputKind
        prepared=[(row,prompt,SamplingParams(**a['decoding'],
            max_tokens=min(a['config']['max_tokens'],a['native_context']-len(prompt)),
            seed=int(digest([SEED,'reader',row['uid']])[:8],16),skip_special_tokens=False))
            for row,prompt in prepared]
        for row,prompt,params in prepared:
            params.output_kind=RequestOutputKind.FINAL_ONLY
        start=time.monotonic()
        llm=LLM(model=a['path'],tokenizer=a['path'],tensor_parallel_size=2,dtype='bfloat16',
            max_model_len=a['native_context'],max_num_seqs=64,enforce_eager=True,
            enable_prefix_caching=True,async_scheduling=False,enable_chunked_prefill=True,
            max_num_batched_tokens=8192,gpu_memory_utilization=.90,kv_cache_dtype='bfloat16',
            attention_config={'backend':'FLASH_ATTN'},language_model_only=True,generation_config='vllm')
        load_seconds=time.monotonic()-start;tokenizer=llm.get_tokenizer();engine=llm.llm_engine
        require(type(engine.engine_core).__name__=='InprocClient','reader admission requires the bound in-process engine')
        for offset in range(0,len(prepared),a['config']['batch_size']):
            deadline_before(deadline,60,'original reader batch')
            batch=[(r,p,params) for r,p,params in prepared[offset:offset+a['config']['batch_size']]
                   if units.begin(r,os.environ['SLURM_JOB_ID'])]
            if not batch:continue
            require(not engine.has_unfinished_requests(),'neighboring reader request still active')
            batch_id=digest([r['uid'] for r,p,params in batch]);lookup={r['uid']:(r,p) for r,p,params in batch}
            tick=time.monotonic();outputs={};partials={}
            try:
                for row,prompt,params in batch:
                    folder=units.root/'lifecycle'/row['uid']
                    publish(folder/'SUBMISSION_INTENT.json',{'uid':row['uid'],'job_id':os.environ['SLURM_JOB_ID'],
                        'prompt_sha256':digest(prompt),'batch_sha256':batch_id})
                    engine.add_request(row['uid'],{'prompt_token_ids':prompt},params)
                    publish(folder/'ADMITTED.json',{'uid':row['uid'],'job_id':os.environ['SLURM_JOB_ID'],
                        'prompt_sha256':digest(prompt),'batch_sha256':batch_id,'execution_may_have_started':True})
                while engine.has_unfinished_requests():
                    deadline_before(deadline,0,'assigned reader execution')
                    for response in engine.step():
                        require(response.request_id in lookup and len(response.outputs)==1,'reader response attribution differs')
                        row,prompt=lookup[response.request_id];output=response.outputs[0]
                        partials[row['uid']]=list(output.token_ids)
                        if not response.finished:continue
                        require(response.prompt_token_ids==prompt,'reader prompt attribution differs')
                        raw=publish(units.root/'lifecycle'/row['uid']/'OUTPUT.json',{
                            'uid':row['uid'],'assignment_sha256':digest(row),'prompt_sha256':digest(prompt),
                            'token_ids':list(output.token_ids),'finish_reason':output.finish_reason,
                            'elapsed_seconds':time.monotonic()-tick,'load_seconds':load_seconds,
                            'batch_sha256':batch_id,'batch_uids':list(lookup)})
                        outputs[row['uid']]=raw
                require(len(outputs)==len(batch),'reader batch lacks terminal original outputs')
                publish(units.root/'batches'/(batch_id+'.json'),{'batch_sha256':batch_id,'batch_size':len(batch),
                    'seconds':max(v['elapsed_seconds'] for v in outputs.values()),'committed_outputs_only':True})
                for row,prompt,params in batch:
                    units.commit(row,parse_output(campaign,row,tokenizer,renderer,outputs[row['uid']]))
            except BaseException as exc:
                # Complete raw outputs are available for deterministic parsing.
                # Others remain unknown; admitted requests are never redrawn.
                for row,prompt,params in batch:
                    if row['uid'] in outputs:continue
                    if (units.state(row['uid']) or {}).get('status')=='started':
                        units.commit(row,{'status':'execution_failed','label':None,'unknown':True,
                            'error_type':type(exc).__name__,'error':str(exc)[:2000],
                            'retained_partial_token_ids':partials.get(row['uid'],[]),
                            'engine_admitted':(units.root/'lifecycle'/row['uid']/'ADMITTED.json').exists()})
                raise
    finally:
        actions=[]
        if llm:
            if llm.llm_engine.has_unfinished_requests():
                actions.append(('abort_reader',lambda:abort_external(llm.llm_engine,[r['uid'] for r in rows])))
            actions.append(('reader_shutdown',llm.llm_engine.engine_core.shutdown))
        actions.append(('renderer_shutdown',renderer.close))
        cleanup(actions,sys.exception())
    return {'status':'complete','units':len(rows),'original_assignments_retained':True}
