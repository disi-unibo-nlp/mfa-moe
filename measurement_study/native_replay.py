#!/usr/bin/env python3
"""Replay saved exact tokens through native vLLM weights and existing reducers."""
import argparse, hashlib, json, os, socket, time
from pathlib import Path
ROOT=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False)+'\n');tmp.replace(path)

def choose(spec):
    short=[];long=None;slug=spec['generation_model'].replace('/','--')
    for dataset in spec['datasets']:
        p=Path(spec['sampling_root'])/'generation'/slug/dataset/'traces.jsonl'
        with p.open() as f:
            for line in f:
                r=json.loads(line);replay=r['metadata'].get('token_replay',{});n=len(replay.get('completion_token_ids',[]))
                if 256<=n<=4096 and len(short)<2:short.append((dataset,r))
                if 8192<=n<=32768 and long is None:long=(dataset,r)
                if len(short)==2 and long is not None:return short+[long]
    raise ValueError('Need two short and one long exact saved completion')

def main():
    p=argparse.ArgumentParser();p.add_argument('source',choices=['qwen36','nemotron','gemma']);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):raise RuntimeError('Authorized GPU Slurm step required')
    campaign=json.loads(Path(os.environ.get('MEASUREMENT_CAMPAIGN',str(ROOT/'report/measurement-study/campaign.json'))).read_text());spec=campaign['sources'][a.source]
    if not spec.get('checkpoint') or not spec.get('revision'):raise RuntimeError('Pinned native checkpoint must be resolved before replay')
    for file,expected in campaign['runtime_code_hashes'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:raise RuntimeError('Reducer changed after preparation')
    out=ROOT/'results/measurement-study/native-replay'/f"{a.source}-{os.environ['SLURM_JOB_ID']}";out.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'phase':'prepare_exact_replay','source':a.source}),flush=True)
    selected=choose(spec);slug=spec['generation_model'].replace('/','--')
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    from moe_exp.correlation_pipeline.labels_export import SOURCES
    from moe_exp.correlation_pipeline.views import compute_views,position_reference
    from moe_exp.correlation_pipeline.features import compute_layer_features,json_safe_features,FEATURE_SCHEMA_VERSION
    from moe_exp.correlation_pipeline.dynamics.routing import reduce_trace,reduce_arrays,probabilities
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.schemas import TraceRecord
    from transformers import AutoTokenizer
    import torch
    adapt_source(a.source,SOURCES[a.source]['label_root'],Path(spec['sampling_root']),out/'annotations')
    annotations={}
    for dataset in {d for d,r in selected}:
        for line in (out/'annotations'/slug/dataset/'annotations.jsonl').open():
            r=json.loads(line);annotations[(dataset,r['problem_id'],r['sample_id'])]=r
        rows=[r for d,r in selected if d==dataset];path=out/'generation'/slug/dataset/'traces.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tokenizer=AutoTokenizer.from_pretrained(spec['checkpoint'],local_files_only=True)
    reference=position_reference(list((out/'generation'/slug).glob('*/traces.jsonl')),tokenizer,spec['generation_model'],10)
    from vllm import LLM,SamplingParams
    kwargs=dict(model=spec['checkpoint'],revision=spec['revision'],tokenizer_revision=spec['revision'],dtype='bfloat16',
        tensor_parallel_size=2,max_num_seqs=8,enforce_eager=True,enable_prefix_caching=False,
        enable_expert_parallel=False,enable_eplb=False,speculative_config=None,async_scheduling=False,
        max_model_len=49152,max_num_batched_tokens=1024,gpu_memory_utilization=.80,generation_config='vllm',
        enable_chunked_prefill=True,kernel_config={'moe_backend':'marlin','linear_backend':'marlin'},
        worker_extension_cls='measurement_study.vllm_capture.MeasurementWorkerExtension')
    if a.source in ('qwen36','gemma'):kwargs['language_model_only']=True
    else:kwargs.update(mamba_backend='triton',mamba_cache_mode='align',kv_cache_dtype='bfloat16',attention_config={'backend':'TRITON_ATTN'})
    print(json.dumps({'phase':'load_native_model','source':a.source,'kv_cache_dtype':kwargs.get('kv_cache_dtype','auto')}),flush=True)
    start=time.monotonic();llm=LLM(**kwargs);load_seconds=time.monotonic()-start
    configured=llm.collective_rpc('measurement_configure',kwargs={'spec':spec});write(out/'backend.json',{'workers':configured,'spec':spec,'engine':kwargs})
    completed=[]
    for index,(dataset,row) in enumerate(selected):
        trace=TraceRecord(**row);replay=row['metadata']['token_replay'];tokens=replay['prompt_token_ids']+replay['completion_token_ids'];key=f'measure-{os.environ["SLURM_JOB_ID"]}-{index}'
        llm.collective_rpc('measurement_register',kwargs={'key':key,'tokens':tokens,'prompt_length':len(replay['prompt_token_ids'])})
        started=time.monotonic();engine=llm.llm_engine
        internal=engine.add_request(key,{'prompt_token_ids':tokens},SamplingParams(max_tokens=1,temperature=0,ignore_eos=True,detokenize=False))
        if internal!=key:raise RuntimeError('Request ID randomization was not disabled')
        while engine.has_unfinished_requests():engine.step()
        raw=out/'transient'/f'{index}.pt'
        reports=llm.collective_rpc('measurement_snapshot',kwargs={'key':key,'output':str(raw)})
        if len(reports)!=2 or {r['rank'] for r in reports}!={0,1} or reports[0]['hashes']!=reports[1]['hashes']:raise RuntimeError('TP native capture mismatch')
        if any(r['tokens']!=len(replay['completion_token_ids']) or r['layers']!=spec['layers'] for r in reports):raise RuntimeError('Missing token or layer coverage')
        llm.collective_rpc('measurement_release',kwargs={'key':key})
        data=torch.load(raw,map_location='cpu',weights_only=True);forward_seconds=time.monotonic()-started
        trace.metadata['reasoning_annotation']=annotations[(dataset,trace.problem_id,trace.sample_id)]
        trace.metadata['forward_provenance']={'checkpoint':spec['generation_model'],'revision':spec['revision'],'precision':spec['precision'],'backend':'vllm-0.29.0+cu129-native-marlin','hidden_observation_point':('decoder_layer_input' if a.source=='gemma' else 'decoder_layer_input_reconstructed_residual'),'capture_code_sha256':hashlib.sha256(Path(__file__).with_name('vllm_capture.py').read_bytes()).hexdigest()}
        ids,weights,router,hidden=data['ids'],data['weights'],data['router'],data['hidden'];layers=spec['layers']
        values=json_safe_features(compute_layer_features(router,hidden,ids,max_geometry_tokens=128,layer_indices=layers))
        from measurement_study.contract import LEGACY
        missing=[f'{name}_l{layer:02d}' for layer in layers for name in LEGACY if values.get(f'{name}_l{layer:02d}') is None]
        if missing:raise RuntimeError('Missing legacy metrics: '+str(missing[:10]))
        trace.metadata['correlation_features']={'schema_version':FEATURE_SCHEMA_VERSION,'config':{'max_geometry_tokens':128},'token_count':len(replay['completion_token_ids']),'layer_indices':layers,'num_layers':len(layers),'num_experts':router.shape[-1],'top_k':ids.shape[-1],'values':values}
        trace.metadata['correlation_views']=compute_views(trace,tokenizer,router,hidden,ids,layers,modes=['full','class','position'],reference=reference,max_geometry_tokens=128)
        semantics={'native_selection':True,'executed_weights':'native_expert_input_weights','weight_semantics':('selected_probability_times_learned_per_expert_scale' if a.source=='gemma' else 'native_router_weights'),'full_distribution':'softmax_probability' if a.source!='nemotron' else 'normalized_sigmoid_affinity','deterministic_topk':True,'selection_scores':'router_signal' if a.source!='nemotron' else 'native_sigmoid_plus_bias','num_experts':router.shape[-1]}
        dynamics=reduce_trace(trace,router,ids,weights,layers,semantics)
        if a.source=='nemotron':
            class ProbabilitySlices:
                shape=router.shape
                def __getitem__(self,index):return probabilities(router[index].float().numpy())
            dynamics['windows']=reduce_arrays(ids,layers=layers,model=trace.model_id,probabilities_array=ProbabilitySlices(),weights=weights,selection_scores=data['selection'],semantics=semantics,token_indices=saved_layout(trace)['reasoning_tokens'])
        if not dynamics['windows'] or any(w['boundary_gap_mean'] is None for w in dynamics['windows']):raise RuntimeError('Missing temporal native selection gaps')
        trace.metadata['routing_dynamics']=dynamics;trace.metadata['native_dispatch_audit']=reports
        tensor_dir=out/'forward'/slug/dataset/'tensors';tensor_dir.mkdir(parents=True,exist_ok=True)
        expert_file=tensor_dir/f'{index}-experts.pt';weight_file=tensor_dir/f'{index}-weights.pt'
        torch.save(ids.to(torch.int16),expert_file);torch.save(weights,weight_file)
        trace.model_logs.layer_indices=layers;trace.model_logs.selected_experts=str(expert_file);trace.model_logs.expert_weights=str(weight_file)
        path=out/'forward'/slug/dataset/f'trace-{index}.json';write(path,trace.model_dump(mode='json'))
        completed.append({'index':index,'dataset':dataset,'problem_id':trace.problem_id,'sample_id':trace.sample_id,'tokens':len(replay['completion_token_ids']),'forward_seconds':forward_seconds,'raw_capture_bytes':raw.stat().st_size,'trace_sha256':trace_digest(trace),'path':str(path)})
        write(out/'progress.json',{'status':'running','completed':completed});raw.unlink();del data,router,hidden,weights,ids
    result={'status':'native_capture_and_long_trace_passed','source':a.source,'layers':spec['layers'],'load_seconds':load_seconds,'attempts':completed,'remaining_gates':['complete corpus coverage','calibration-only weak-boundary thresholds','metric statistical acceptance'],'artifact_bytes':sum(p.stat().st_size for p in (out/'forward').rglob('*') if p.is_file())}
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
