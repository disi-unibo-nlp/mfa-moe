#!/usr/bin/env python3
"""Bounded, native-checkpoint extraction smoke; never starts a production sweep."""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
from measurement_study.contract import extraction_options,LEGACY,routed_layers
BASE=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe')
ROOT=BASE/'routing-study'

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def prepare_fixture(s,out,include_long=False):
    slug=s['generation_model'].replace('/','--');chosen=[];long=None
    # Two bounded complete traces; never truncate or relabel an existing attempt.
    for dataset in s['datasets']:
        with (Path(s['sampling_root'])/'generation'/slug/dataset/'traces.jsonl').open() as f:
            for line in f:
                row=json.loads(line);replay=row['metadata'].get('token_replay',{})
                n=len(replay.get('completion_token_ids',[]))
                if 256<=n<=4096 and len(chosen)<2:chosen.append((dataset,row,n))
                if include_long and 8192<=n<=32768 and long is None:long=(dataset,row,n)
                if len(chosen)==2 and (not include_long or long is not None):break
        if len(chosen)==2 and (not include_long or long is not None):break
    if len(chosen)!=2 or (include_long and long is None):raise RuntimeError('Required short/long exact-replay traces are unavailable')
    if include_long:chosen.append(long)
    datasets=sorted({d for d,_,_ in chosen})
    for dataset in datasets:
        path=out/'generation'/slug/dataset/'traces.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        source_manifest=Path(s['sampling_root'])/'generation'/slug/dataset/'manifest.json'
        original=json.loads(source_manifest.read_text())
        if original.get('target_model_id')!=s['forward_model']:raise ValueError('Source checkpoint identity mismatch')
        path.write_text(''.join(json.dumps(r)+'\n' for d,r,_ in chosen if d==dataset))
        write(path.with_name('manifest.json'),dict(target_model_id=original['target_model_id'],
            dataset=dataset,status='bounded_replay_fixture',traces=sum(d==dataset for d,_,_ in chosen),
            source_manifest=str(source_manifest),source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return slug,chosen,datasets

def main():
    p=argparse.ArgumentParser();p.add_argument('source',choices=['gpt','glm','qwen330b']);p.add_argument('--long',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not socket.gethostname().startswith('lrdn'):
        raise RuntimeError('Authorized Slurm GPU allocation required')
    campaign=json.loads(Path(os.environ.get('MEASUREMENT_CAMPAIGN',str(ROOT/'report/measurement-study/campaign.json'))).read_text());s=campaign['sources'][a.source]
    for file,expected in campaign['runtime_code_hashes'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Inspected replay code changed: '+file)
    if s['native_backend_readiness']!='debug_validation_ready':raise RuntimeError('Backend is not ready for GPU validation')
    out=ROOT/'results/measurement-study/smoke'/f"{a.source}-{os.environ['SLURM_JOB_ID']}"
    from moe_exp.correlation_pipeline.forward_adapter import adapt_source
    from moe_exp.correlation_pipeline.labels_export import SOURCES
    from moe_exp.correlation_pipeline.extract import build_parser,extract_all
    from moe_exp.models.loader import load_model_and_tokenizer
    from moe_exp.correlation_pipeline.forward_audit import audit_bundle
    from moe_exp.correlation_pipeline.provenance import code_provenance
    from moe_exp.models.router_adapters import available_router_layers
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('Two allocated GPUs required')
    labels_available=a.source=='gpt'
    if labels_available:
        source=SOURCES[a.source]
        adapt_source(a.source,source['label_root'],Path(s['sampling_root']),out/'annotations')
    slug,chosen,datasets=prepare_fixture(s,out,include_long=a.long)
    start=time.monotonic()
    model,tokenizer=load_model_and_tokenizer(s['checkpoint'],quantization='none',revision=s['revision'],
                                            local_files_only=True,offload_folder=str(out/'offload'))
    quantizer=getattr(model,'hf_quantizer',None)
    if a.source=='gpt':
        if quantizer is None or quantizer.quantization_config.dequantize:raise RuntimeError('Native MXFP4 changed precision')
    elif quantizer is not None:raise RuntimeError('Unexpected quantization for native BF16 checkpoint')
    if available_router_layers(model)!=s['layers']:raise RuntimeError('Loaded routed layers differ from manifest')
    load_seconds=time.monotonic()-start
    options=extraction_options(s,out/'generation',out/'annotations',out/'forward',datasets)
    if not labels_available:options.remove('class')
    args=build_parser().parse_args(options)
    from moe_exp.correlation_pipeline.dynamics import routing
    original_reducer=routing.reduce_trace
    if a.source=='glm':
        from measurement_study.native_gaps import glm_reducer
        routing.reduce_trace=glm_reducer(model)
    started=time.monotonic()
    try:execution=extract_all(args,model_and_tokenizer=(model,tokenizer))
    finally:routing.reduce_trace=original_reducer
    elapsed=time.monotonic()-started
    failures=[];count=0
    for d in datasets:
        path=out/'forward'/slug/d/'traces_with_routing.jsonl'
        audit=audit_bundle(path,require_replay=True);write(out/f'audit-{d}.json',audit)
        if audit['status']!='complete':failures.append('native artifact audit')
        for line in path.open():
            r=json.loads(line);count+=1;meta=r['metadata'];features=meta.get('correlation_features',{}).get('values',{})
            if r['model_logs']['layer_indices']!=s['layers']:failures.append('layer coverage')
            for layer in s['layers']:
                for metric in LEGACY:
                    if features.get(f'{metric}_l{layer:02d}') is None:failures.append(f'{metric}/L{layer}')
            dyn=meta.get('routing_dynamics',{});sem=dyn.get('semantics',{})
            if not sem.get('native_selection') or sem.get('executed_weights')!='native_expert_input_weights':failures.append('native routing signals')
            if not dyn.get('windows'):failures.append('temporal windows')
            elif any(w['boundary_gap_mean'] is None for w in dyn['windows']):failures.append('native boundary gaps')
    payload=dict(status=('native_capture_and_long_trace_passed' if a.long else 'capture_smoke_passed') if not failures and count==len(chosen) else 'failed',
        source=a.source,precision=s['precision'],labels_available=labels_available,code=code_provenance(),layers=s['layers'],traces=count,
        load_seconds=load_seconds,forward_seconds=elapsed,completion_tokens=sum(n for _,_,n in chosen),
        artifact_bytes=sum(p.stat().st_size for p in (out/'forward').rglob('*') if p.is_file()),
        max_cuda_allocated=[torch.cuda.max_memory_allocated(i) for i in range(2)],failures=sorted(set(failures)),
        remaining_gates=['long-trace memory','full corpus coverage','calibration-only weak-boundary thresholds',
                         'all-model native backend capture','complete metric and statistical acceptance'],execution=execution)
    write(out/'result.json',payload)
    if payload['status']=='failed':raise RuntimeError('Smoke failed: '+str(payload['failures']))
    print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
