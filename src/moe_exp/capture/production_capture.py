"""Exact saved-token capture, all routed layers and every separate label cohort."""
import argparse, copy, gc, gzip, hashlib, io, json, math, os, socket, time
from pathlib import Path
from moe_exp.capture.production_plan import ROOT,FAST,SOURCES,digest,sha,atomic,read_location,normalize_annotation

def pack_bytes(path,raw,limit):
    """Atomic, lossless output; fail before publishing beyond the shard cap."""
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    try:
        with tmp.open('wb') as f:
            with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=3,mtime=0) as z:z.write(raw)
        if tmp.stat().st_size>limit:raise RuntimeError('Shard artifact storage cap reached')
        h=hashlib.sha256()
        with gzip.open(tmp,'rb') as f:
            while block:=f.read(1024*1024):h.update(block)
        if h.hexdigest()!=hashlib.sha256(raw).hexdigest():raise RuntimeError('Lossless read-back mismatch')
        tmp.replace(path)
        return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),raw_sha256=h.hexdigest())
    finally:tmp.unlink(missing_ok=True)

def load_annotation(cohort):
    info=cohort['annotation']
    if 'inline' in info:
        if digest(info['inline'])!=info.get('sha256'):raise ValueError('Inline annotation changed')
        # Manifest loading checks bound input files once, not once per attempt.
        # Return a fresh value so normalization cannot mutate the frozen manifest.
        return json.loads(json.dumps(info['inline'],allow_nan=False))
    raw=gzip.decompress(Path(info['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Annotation changed')
    return json.loads(raw)

class HFBackend:
    def __init__(self,spec,out):
        from moe_exp.models.loader import load_model_and_tokenizer
        from moe_exp.models.router_adapters import available_router_layers
        from moe_exp.provenance import forward_provenance
        self.spec=spec
        self.model,self.tokenizer=load_model_and_tokenizer(spec['checkpoint'],quantization='none',revision=spec['revision'],local_files_only=True,offload_folder=str(out/'offload'))
        if 'disk' in getattr(self.model,'hf_device_map',{}).values():raise ValueError('Unbudgeted disk weight offload')
        if available_router_layers(self.model)!=spec['layers']:raise ValueError('Changed native layer topology')
        q=getattr(self.model,'hf_quantizer',None)
        if spec['precision']=='mxfp4':
            if q is None or q.quantization_config.dequantize:raise ValueError('Native MXFP4 required')
        elif q is not None:raise ValueError('Unexpected quantization')
        self.provenance=forward_provenance(self.model,self.tokenizer,revision=spec['revision'],local_files_only=True)
        if spec['generation_model']=='openai/gpt-oss-20b':
            from moe_exp.capture.gpt_attention import install_gpt_query_chunking
            self.provenance.update(install_gpt_query_chunking(self.model,128))
    def validate_attention(self,trace,out):
        import torch
        from transformers.models.gpt_oss import modeling_gpt_oss as implementation
        chunked=implementation.eager_attention_forward
        try:
            implementation.eager_attention_forward=chunked._measurement_native
            native=self.capture_single(trace,'native-differential')
        finally:implementation.eager_attention_forward=chunked
        bounded=self.capture_single(trace,'bounded-differential')
        if not torch.equal(native['ids'].sort(-1).values,bounded['ids'].sort(-1).values):raise ValueError('Query chunking changed native selected expert sets on validation trace')
        evidence={}
        for name in ('router','hidden'):
            x=native[name].float();y=bounded[name].float();error=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-12))
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not math.isfinite(error) or error>0.005:raise ValueError('Query chunking differential error: '+name)
            evidence[name+'_relative_l2']=error
        native_weights=native['weights'].gather(-1,native['ids'].argsort(-1));bounded_weights=bounded['weights'].gather(-1,bounded['ids'].argsort(-1))
        if not torch.isfinite(native_weights).all() or not torch.isfinite(bounded_weights).all() or not torch.allclose(native_weights,bounded_weights,rtol=0.01,atol=0.001):raise ValueError('Query chunking changed native weights beyond rounding tolerance')
        atomic(out/'attention-differential.json',dict(status='passed',native_selected_sets_identical=True,completion_tokens=len(trace.metadata['token_replay']['completion_token_ids']),**evidence))

    def capture(self,trace,key):
        import torch
        if self.spec['generation_model']=='openai/gpt-oss-20b' and (self.spec.get('force_chunked') or len(trace.metadata['token_replay']['prompt_token_ids'])+len(trace.metadata['token_replay']['completion_token_ids'])>32768):
            from moe_exp.capture.hf_chunked_capture import capture_chunked
            return capture_chunked(self.model,self.tokenizer,trace,self.spec['layers'],self.spec.get('prefill_chunk_size',512))
        return self.capture_single(trace,key)

    def capture_single(self,trace,key):
        import torch
        from moe_exp.models.inference import extract_logs_single_pass
        from moe_exp.models.router_adapters import decoder_layers
        details={}
        router,hidden=extract_logs_single_pass(model=self.model,tokenizer=self.tokenizer,problem=trace.prompt,cot_text=trace.cot_text,system_prompt=trace.system_prompt,messages=trace.generation_messages,extract_hidden_states=True,layer_indices=self.spec['layers'],token_replay=trace.metadata['token_replay'],routing_details=details)
        sem=details['semantics'];ids=details['selected_experts'];weights=details['expert_weights']
        if not sem.get('native_selection') or sem.get('executed_weights')!='native_expert_input_weights':raise ValueError('Native dispatch capture required')
        selection=router
        if self.spec['generation_model']=='zai-org/GLM-4.7-Flash':
            c=self.model.config
            if c.n_group!=1 or c.topk_group!=1:raise ValueError('Unsupported grouped selection')
            bias=torch.stack([decoder_layers(self.model)[l].mlp.gate.e_score_correction_bias.detach().float().cpu() for l in self.spec['layers']])
            selection=router.float().exp()+bias[:,None,:]
            sem={**sem,'selection_scores':'native_sigmoid_plus_bias','deterministic_topk':True}
        return dict(router=router,hidden=hidden,ids=ids,weights=weights,selection=selection,semantics=sem,audit={'native_dispatch':True,'backend':'hf','layers':self.spec['layers']})

class VLLMBackend:
    def __init__(self,spec,out,transient_cap_bytes=15*1024**3):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.spec,self.out=spec,out
        self.transient_cap_bytes=transient_cap_bytes
        self.buffer_cap_bytes=spec.get('buffer_cap_bytes')
        self.tokenizer=AutoTokenizer.from_pretrained(spec['checkpoint'],revision=spec['revision'],local_files_only=True)
        nemo='Nemotron' in spec['generation_model']
        kwargs=dict(model=spec['checkpoint'],revision=spec['revision'],tokenizer_revision=spec['revision'],dtype='bfloat16',tensor_parallel_size=2,max_num_seqs=1,enforce_eager=True,enable_prefix_caching=False,enable_expert_parallel=False,enable_eplb=False,speculative_config=None,async_scheduling=False,max_model_len=int(spec.get('max_input_tokens',49151))+1,max_num_batched_tokens=1024,gpu_memory_utilization=.80,generation_config='vllm',enable_chunked_prefill=True,kernel_config={'moe_backend':'marlin','linear_backend':'marlin'},worker_extension_cls='moe_exp.capture.vllm_capture.MeasurementWorkerExtension')
        if spec.get('source')=='glm' or spec['generation_model']=='zai-org/GLM-4.7-Flash':
            kwargs['kernel_config']={'moe_backend':'triton'}
            kwargs['kv_cache_dtype']='bfloat16'
        if spec.get('source')=='qwen35' or 'GPTQ' in spec['generation_model']:
            kwargs['quantization']='gptq_marlin'
        if nemo:kwargs.update(mamba_backend='triton',mamba_cache_mode='align',kv_cache_dtype='bfloat16',attention_config={'backend':'TRITON_ATTN'})
        else:kwargs['language_model_only']=True
        from moe_exp.capture.gemma_cache_policy import cache_options
        kwargs.update(cache_options(spec,out))
        self.llm=LLM(**kwargs)
        if spec['generation_model'].startswith('nvidia/Gemma'):
            from vllm import SamplingParams
            generated=self.llm.generate([{'prompt_token_ids':self.tokenizer.encode('Compute 2 + 2.',add_special_tokens=True)}],SamplingParams(max_tokens=8,temperature=0),use_tqdm=False)
            atomic(out/'short-generation.json',dict(status='complete',token_ids=list(generated[0].outputs[0].token_ids),text=generated[0].outputs[0].text))
            if not generated[0].outputs[0].token_ids:raise RuntimeError('Gemma short generation empty')
        configured=self.llm.collective_rpc('measurement_configure',kwargs={'spec':spec});self.configured=configured;atomic(out/'backend.json',{'workers':configured,'engine':kwargs})
        self.provenance=dict(checkpoint=spec['generation_model'],revision=spec['revision'],precision=spec['precision'],kv_cache_dtype=kwargs.get('kv_cache_dtype','auto'),replay_variant=spec.get('replay_variant','primary'),historical_kv_cache_dtype_status=spec.get('historical_kv_cache_dtype_status','unverified'),backend='vllm-native-'+('triton' if spec['generation_model']=='zai-org/GLM-4.7-Flash' else 'marlin'),hidden_observation_point=spec.get('hidden_observation_point','decoder_layer_input_reconstructed_residual'),capture_code_sha256=sha(Path(__file__).with_name('vllm_capture.py')))
    def capture(self,trace,key):
        import torch
        from vllm import SamplingParams
        from moe_exp.models.token_replay import validate_token_replay
        replay=trace.metadata['token_replay'];prompt,completion=validate_token_replay(replay,self.tokenizer,trace.cot_text);tokens=prompt+completion
        from moe_exp.capture.capture_buffers import capture_memory_budget
        budget=capture_memory_budget(len(completion),len(self.spec['layers']),int(self.spec['hidden_size']),int(self.spec.get('num_experts',self.spec['experts'])),int(self.spec['top_k']))
        if budget['transient_bytes']>self.transient_cap_bytes:raise ValueError('Manifest transient budget is too small for trace')
        self.llm.collective_rpc('measurement_register',kwargs={'key':key,'tokens':tokens,'prompt_length':len(prompt),'max_buffer_bytes':budget['worker_bytes']})
        engine=self.llm.llm_engine
        internal=engine.add_request(key,{'prompt_token_ids':tokens},SamplingParams(max_tokens=1,temperature=0,ignore_eos=True,detokenize=False))
        if internal!=key:raise RuntimeError('Request ID randomization is active')
        while engine.has_unfinished_requests():engine.step()
        raw=self.out/'transient.pt'
        reports=self.llm.collective_rpc('measurement_snapshot',kwargs={'key':key,'output':str(raw),'max_bytes':self.transient_cap_bytes})
        if len(reports)!=2 or {r['rank'] for r in reports}!={0,1} or reports[0]['hashes']!=reports[1]['hashes']:raise RuntimeError('TP capture mismatch')
        if any(r['tokens']!=len(completion) or r['layers']!=self.spec['layers'] for r in reports):raise RuntimeError('Missing token/layer coverage')
        atomic(self.out/(key+'.native-audit.json'),{'workers':reports})
        self.llm.collective_rpc('measurement_release',kwargs={'key':key})
        if raw.stat().st_size>self.transient_cap_bytes:raise RuntimeError('Transient storage cap exceeded')
        data=torch.load(raw,map_location='cpu',weights_only=True);raw.unlink()
        nemo=('Nemotron' in self.spec['generation_model'] or self.spec['generation_model']=='zai-org/GLM-4.7-Flash')
        data['semantics']=dict(native_selection=True,executed_weights='native_expert_input_weights',full_distribution='normalized_sigmoid_affinity' if nemo else 'softmax_probability',deterministic_topk=True,selection_scores='native_sigmoid_plus_bias' if nemo else 'router_signal',num_experts=data['router'].shape[-1])
        data['audit']=dict(native_dispatch=True,layers=self.spec['layers'],workers=reports,routed_output_scaling=self.configured[0].get('routed_output_scaling',{}))
        return data

def counts_for_tokens(ids,indices,experts):
    import torch
    return [torch.bincount(layer[indices].long().reshape(-1),minlength=experts).tolist() for layer in ids]

def expert_class_counts(trace,annotation,ids,experts):
    from moe_exp.correlation_pipeline.dynamics.classes import sentence_table,saved_layout
    from moe_exp.capture.contract import CLASSES
    sentences=sentence_table(trace,annotation,saved_layout(trace));groups={c:set() for c in CLASSES}
    for s in sentences:
        if s['label'] in groups and s['token_indices'] is not None:groups[s['label']].update(s['token_indices'])
    all_tokens=sorted(set().union(*groups.values()))
    return dict(population='labeled reasoning tokens; class gaps excluded',tokens=len(all_tokens),counts=counts_for_tokens(ids,all_tokens,experts),classes={c:dict(tokens=len(ix),counts=counts_for_tokens(ids,sorted(ix),experts)) for c,ix in groups.items()})

def marker_counts(trace,ids,experts):
    import re
    offsets=trace.metadata['token_replay']['completion_offsets'];out={}
    for marker in ('<think>','</think>','Alternatively'):
        spans=[m.span() for m in re.finditer(re.escape(marker),trace.cot_text)]
        indices=[i for i,(a,b) in enumerate(offsets) if b>a and any(b>x and a<y for x,y in spans)]
        out[marker]=dict(occurrences=len(spans),aligned_tokens=len(indices),counts=counts_for_tokens(ids,indices,experts),status='observed' if spans else 'unavailable_no_occurrence')
    return out

def reduce_capture(row,data,backend,manifest):
    import torch
    from moe_exp.schemas import TraceRecord
    from moe_exp.labeling.spans import trace_digest, validate_annotation
    from moe_exp.correlation_pipeline.features import compute_layer_features,json_safe_features,FEATURE_SCHEMA_VERSION
    from moe_exp.correlation_pipeline.views import compute_views
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.dynamics.routing import reduce_arrays,probabilities
    from moe_exp.correlation_pipeline.dynamics.common import DEFAULT_CONFIG
    from moe_exp.correlation_pipeline.dynamics import SCHEMA_VERSION
    from moe_exp.capture.contract import LEGACY
    spec=manifest['spec'];layers=spec['layers'];router=data['router'];hidden=data['hidden'];ids=data['ids'];weights=data['weights'];scores=data['selection'];experts=router.shape[-1]
    if tuple(ids.shape)!=(len(layers),row['completion_tokens'],spec['top_k']):raise ValueError('Incomplete dispatched IDs')
    if hidden.shape[:2]!=ids.shape[:2] or weights.shape!=ids.shape:raise ValueError('Incomplete native signals')
    from moe_exp.capture.native_selection import require_native_boundary
    gaps=require_native_boundary(scores,ids)
    values=json_safe_features(compute_layer_features(router,hidden,ids,max_geometry_tokens=128,layer_indices=layers))
    if any(values.get(f'{metric}_l{layer:02d}') is None for layer in layers for metric in LEGACY):raise ValueError('Missing legacy metric')
    features=dict(schema_version=FEATURE_SCHEMA_VERSION,config={'max_geometry_tokens':128},token_count=row['completion_tokens'],layer_indices=layers,num_layers=len(layers),num_experts=experts,top_k=ids.shape[-1],values=values)
    first=TraceRecord(**read_location(row['cohorts'][0]['locations'][0]));layout=saved_layout(first)
    class Probabilities:
        shape=router.shape
        def __getitem__(self,index):return probabilities(router[index].float().numpy())
    sem={**data['semantics'],'num_experts':experts,'deterministic_topk':True}
    windows=reduce_arrays(ids,layers=layers,model=first.model_id,probabilities_array=Probabilities(),weights=weights,selection_scores=scores,semantics=sem,token_indices=layout['reasoning_tokens'])
    if any(w['boundary_gap_mean'] is None for w in windows):raise ValueError('Missing native boundary gaps')
    variants={}
    for cohort in row['cohorts']:
        trace=TraceRecord(**read_location(cohort['locations'][0]))
        annotation=normalize_annotation(trace,load_annotation(cohort)) if cohort.get('annotation') else None
        if trace_digest(trace)!=row['trace_sha256'] or saved_layout(trace)['reasoning_tokens']!=layout['reasoning_tokens']:raise ValueError('Cohort changes replay or reasoning scope')
        trace.metadata.pop('reasoning_annotation',None)
        if annotation is not None:
            validate_annotation(trace,annotation);trace.metadata['reasoning_annotation']=annotation
        views=compute_views(trace,backend.tokenizer,router,hidden,ids,layers,modes=['full','position']+(['class'] if annotation is not None else []),reference=manifest['position_reference'],max_geometry_tokens=128)
        variants[cohort['id']]=dict(source_trace=trace.model_dump(mode='json'),original_annotation_sha256=cohort.get('annotation',{}).get('sha256'),annotation=annotation,views=views,expert_class=expert_class_counts(trace,annotation,ids,experts) if annotation is not None else None)
    first.metadata['forward_provenance']=backend.provenance
    first.metadata['correlation_features']=features
    first.metadata['routing_dynamics']=dict(schema_version=SCHEMA_VERSION,config=dict(DEFAULT_CONFIG),trace_sha256=row['trace_sha256'],annotation_sha256=None,forward_provenance=backend.provenance,semantics=sem,windows=windows)
    first.model_logs.layer_indices=layers
    # Outcomes are joined at analysis time from the explicit sidecar; the packet
    # carries only native capture provenance and never a planning-time label.
    packet=dict(schema_version=2,weights_storage_dtype='float16',id=row['id'],trace=first.model_dump(mode='json'),cohorts=variants,native_dispatch_audit=data['audit'],marker_counts=marker_counts(first,ids,experts),native_expert_counts=counts_for_tokens(ids,list(range(ids.shape[1])),experts),manifest_binding=manifest['binding'])
    if not torch.isfinite(weights).all() or (weights<0).any() or (weights>torch.finfo(torch.float16).max).any():raise ValueError('Native weights cannot be stored as finite nonnegative fp16')
    tensors=dict(ids=ids.to(torch.int16),weights=weights.to(torch.float16),native_boundary_gaps=gaps.float(),reasoning_tokens=torch.tensor(layout['reasoning_tokens'],dtype=torch.int32))
    return packet,tensors

def main():
    global FAST
    from moe_exp.capture.sentence_schema import install
    install()
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);p.add_argument('--shard',type=int,required=True);p.add_argument('--verify-only',action='store_true');p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    from moe_exp.capture.production_plan import storage_root
    FAST=storage_root(a.source)
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):raise RuntimeError('GPU Slurm allocation required')
    import torch
    from moe_exp.schemas import TraceRecord
    if torch.cuda.device_count()!=2:raise RuntimeError('Exactly two allocated GPUs required')
    code=json.loads((a.bundle/'SHA256.json').read_text())
    for rel,expected in code.items():
        if sha(a.bundle/rel)!=expected:raise ValueError('Frozen code changed: '+rel)
    m=json.loads((FAST/'manifests'/a.source/f'{a.shard}.json').read_text())
    if digest({k:v for k,v in m.items() if k!='binding'})!=m['binding'] or sha(a.bundle/'SHA256.json')!=m['code_binding']:raise ValueError('Frozen manifest mismatch')
    campaign=json.loads((a.bundle/'campaign.json').read_text())
    for file,expected in campaign['runtime_code_hashes'].items():
        if sha(file)!=expected:raise ValueError('Reducer changed')
    out=FAST/'capture'/a.source/str(a.shard);out.mkdir(parents=True,exist_ok=True)
    completed=[];remaining=[];used=0
    for row in m['attempts']:
        marker=out/(row['id']+'.complete.json')
        if marker.exists():
            done=json.loads(marker.read_text())
            if done['manifest_binding']!=m['binding']:raise ValueError('Incompatible resume')
            for f in done['files']:
                if sha(f['path'])!=f['sha256']:raise ValueError('Corrupted saved attempt')
                used+=f['bytes']
            completed.append(done)
        else:remaining.append(row)
    verification_rows=[]
    if a.verify_only:
        longest=max(m['attempts'],key=lambda r:r['total_input_tokens'])
        shorts=[r for r in m['attempts'] if 256<=r['completion_tokens']<=4096 and r['id']!=longest['id']]
        if not shorts or longest['total_input_tokens']!=m['maximum_input_tokens']:raise ValueError('Validation must include a short completion and corpus longest input')
        verification_rows=[longest,min(shorts,key=lambda r:r['total_input_tokens'])]
        wanted={r['id'] for r in verification_rows};remaining=[r for r in remaining if r['id'] in wanted]
    if remaining:
        start=time.monotonic();backend=HFBackend(m['spec'],out) if a.source in ('gpt','glm','qwen330b') else VLLMBackend(m['spec'],out,transient_cap_bytes=m['transient_cap_bytes'])
        atomic(out/'load.json',dict(seconds=time.monotonic()-start,source=a.source,job=os.environ['SLURM_JOB_ID'],provenance=backend.provenance))
        if a.verify_only and a.source=='gpt':
            short=min((r for r in m['attempts'] if r['total_input_tokens']<=4096 and r['completion_tokens']>=256),key=lambda r:r['total_input_tokens'])
            backend.validate_attention(TraceRecord(**read_location(short['cohorts'][0]['locations'][0])),out)
        # Longest complete input first: validates actual corpus memory before the rest.
        for row in sorted(remaining,key=lambda r:(-r['total_input_tokens'],r['id'])):
            start=time.monotonic();trace=TraceRecord(**read_location(row['cohorts'][0]['locations'][0]))
            data=backend.capture(trace,'capture-'+row['id'])
            packet,tensors=reduce_capture(row,data,backend,m);del data
            tensor_bytes=io.BytesIO();torch.save(tensors,tensor_bytes);del tensors
            tensor_record=pack_bytes(out/(row['id']+'.pt.gz'),tensor_bytes.getvalue(),m['artifact_cap_bytes']-used);used+=tensor_record['bytes'];del tensor_bytes
            packet['tensor_artifact']=tensor_record
            packet_record=pack_bytes(out/(row['id']+'.json.gz'),json.dumps(packet,allow_nan=False).encode(),m['artifact_cap_bytes']-used);used+=packet_record['bytes'];del packet
            done=dict(id=row['id'],manifest_binding=m['binding'],files=[tensor_record,packet_record],tokens=row['completion_tokens'],seconds=time.monotonic()-start,job=os.environ['SLURM_JOB_ID'])
            atomic(out/(row['id']+'.complete.json'),done);completed.append(done)
            atomic(out/'progress.json',dict(completed=len(completed),expected=m['expected_attempts'],tokens=sum(r['tokens'] for r in completed),artifact_bytes=used))
            print(json.dumps({'completed':len(completed),'expected':m['expected_attempts'],'last_seconds':done['seconds']}),flush=True)
    if a.verify_only:
        atomic(out/'verification.json',dict(status='passed',manifest_binding=m['binding'],completed=len(completed),verified_ids=[r['id'] for r in verification_rows],maximum_input_tokens=max(r['total_input_tokens'] for r in verification_rows),all_routed_layers=m['spec']['layers'],exact_saved_token_coverage=True));return
    if len(completed)!=m['expected_attempts'] or sum(r['tokens'] for r in completed)!=m['expected_tokens']:raise RuntimeError('Incomplete planned coverage')
    atomic(out/'result.json',dict(status='complete_native_collection',source=a.source,shard=a.shard,manifest_binding=m['binding'],attempts=len(completed),tokens=sum(r['tokens'] for r in completed),artifact_bytes=used,records=completed))
if __name__=='__main__':
    import sys
    if '--manifest' in sys.argv:
        from moe_exp.capture.campaign_v3_capture import main as v3_main
        raise SystemExit(v3_main())
    raise SystemExit('Use a new v3 bundle and explicit --manifest; legacy accepted outputs are immutable.')
