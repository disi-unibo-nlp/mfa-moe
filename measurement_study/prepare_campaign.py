#!/usr/bin/env python3
"""Prepare a read-only cache audit and explicit missing-data collection manifest."""
import hashlib,json
from pathlib import Path
from measurement_study.contract import *
BASE=Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe')
OUT=BASE/'routing-study/report/measurement-study'
CACHE=BASE/'cache/hf/hub'
SOURCES={
 'gpt':('openai/gpt-oss-20b','gpt-oss-20b'),
 'gemma':('nvidia/Gemma-4-26B-A4B-NVFP4','gemma-nvfp4-nf4'),
 'qwen36':('Qwen/Qwen3.6-35B-A3B-FP8','qwen36-35b-a3b-fp8'),
 'nemotron':('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4','nemotron-nvfp4-dspark'),
 'glm':('zai-org/GLM-4.7-Flash','glm-4.7-flash'),
 'qwen330b':('Qwen/Qwen3-30B-A3B','qwen3-30b-a3b')}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources={}
    old=json.loads((BASE/'repo/results/correlation_pipeline/labeled-forward-v1/plan.json').read_text())
    for key,(model,sub) in SOURCES.items():
        configs=list((CACHE/('models--'+model.replace('/','--'))/'snapshots').glob('*/config.json'))
        if key=='gemma':
            transfer_path=OUT/'gemma-transfer.json';transfer=json.loads(transfer_path.read_text())
            native=Path(transfer['destination']);receipt_path=native/'transfer-receipt.json'
            if receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                if receipt['status']!='verified' or receipt['manifest_sha256']!=sha(transfer_path):raise ValueError('Unverified Gemma transfer receipt')
                if receipt['revision']!=transfer['revision']:raise ValueError('Gemma checkpoint revision mismatch')
                if any(not (native/f['name']).is_file() or (native/f['name']).stat().st_size!=f['bytes'] for f in transfer['files']):raise ValueError('Gemma transfer file set changed')
                configs=[native/'config.json']

        item=dict(generation_model=model,forward_model=model,quantization='none',datasets=list(DATASETS),
            data_root=str(BASE/'repo/results/correlation_pipeline'/sub),
            labels_status='pending_existing_jobs' if key in ('glm','qwen330b') else 'available_stratified_labels',
            required_action='collect all native routing and hidden measurements; analyze joined labels')
        if len(configs)==1:
            p=configs[0];c=json.loads(p.read_text());t=c.get('text_config',c)
            quant=c.get('quantization_config',t.get('quantization_config')) or {}
            layers=routed_layers(c)
            item.update(revision=p.parent.name,checkpoint=str(p.parent),config_sha256=sha(p),layers=layers,
                router_layers=len(layers),decoder_layers=t.get('num_hidden_layers'),
                precision=quant.get('quant_method','bfloat16'),
                experts=t.get('num_experts',t.get('n_routed_experts',t.get('num_local_experts'))),
                top_k=t.get('num_experts_per_tok',t.get('top_k_experts')))
            idx=p.parent/'model.safetensors.index.json'
            shards=sorted(set(json.loads(idx.read_text())['weight_map'].values())) if idx.exists() else ['model.safetensors']
            item['missing_weight_shards']=[n for n in shards if not (p.parent/n).is_file()]
            item['checkpoint_bytes']=sum((p.parent/n).stat().st_size for n in shards if (p.parent/n).is_file())
            if key=='gemma':
                if quant.get('quant_algo')!='NVFP4':raise ValueError('Gemma checkpoint is not native NVFP4')
                item.update(precision='nvfp4',experts=t['num_experts'],top_k=t['top_k_experts'],
                    historical_generation_revision_status='unverified',topology_requires_native_checkpoint_verification=False,
                    hidden_observation_point='decoder_layer_input_before_input_layernorm',
                    native_dispatch_weights='selected_softmax_renormalized_times_learned_per_expert_scale')
            item['cpu_interface']='existing transformer native-router adapter'
            item['native_backend_readiness']=('debug_validation_ready' if key in ('gpt','glm','qwen330b')
                else 'requires_native_vllm_replay_capture_validation')
        else:
            item.update(native_backend_readiness='native_checkpoint_not_resolved',layers=None,revision=None)
            # Topology is informational only; never substitute this checkpoint for execution.
            if key=='gemma':
                topology=next((CACHE/'models--google--gemma-4-26B-A4B-it/snapshots').glob('*/config.json'))
                item.update(layers=routed_layers(json.loads(topology.read_text())),
                    topology_reference=str(topology),topology_requires_native_checkpoint_verification=True)
        if key in old['sources']:
            s=old['sources'][key]
            item.update(sampling_root=s['sampling_root'],available_primary_attempts=s['traces'],
                        available_primary_reasoning_tokens=s['reasoning_tokens'])
        else:
            sampling=BASE/'repo/results/correlation_pipeline'/sub/'reasoning-vllm-v1/sampling-100k-stratified'
            item['sampling_root']=str(sampling)
        sources[key]=item
    ledger_path=BASE/'routing-study/results/measurement-study/gpu-budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    status='validation_submitted' if ledger and ledger.get('jobs') else 'prepared_not_submitted'
    manifest=dict(schema_version=1,status=status,sources=sources,
        required_recovery_sources=[dict(id='qwen35_gptq',model='Qwen3.5 GPTQ-Int4',required=True,status='staged_requires_native_recovery',action='resolve original quantized checkpoint; reconcile labels and exact token IDs; validate and size all-layer recovery'),
                           dict(id='qwen35_gguf',model='Qwen3.5 GGUF UD-Q4-K-XL',required=True,status='staged_requires_native_recovery',action='resolve original GGUF backend and exact token IDs; native all-layer capture required; keep HF 4-bit replay separate')],
        required_signals=list(SIGNALS),legacy_metrics=list(LEGACY),temporal_metrics=list(TEMPORAL),class_metrics=list(CLASS_METRICS),
        windows=[64,256],stride=64,lags=[1,4,16,64],shuffle_replicates=16,bootstrap_replicates=1000,
        precision_policy='Use generation checkpoint. Any alternate precision is a separately named sensitivity analysis, never a replacement.',
        layer_policy='Every actual routed layer at its absolute decoder depth; preserve native expert count, top-k and shared branches.',
        scope='Exact saved completion tokens, reasoning segments only; full/class/position views; old and new metrics.',
        completeness_policy='Missing checkpoint, signal, layer, label, calibration or applicable metric remains unfinished; no silent feature intersection.',
        calibration_policy='Question-grouped calibration-only native boundary threshold; no final-test outcomes used to fit thresholds.',
        proposed_debug_budget=dict(max_jobs=8,gpus_per_job=2,cpus_per_job=8,ram_gib_per_job=96,
            max_minutes_per_job=30,max_concurrent_jobs=2,max_gpu_hours=8,
            purpose='Native replay capture validation and measured throughput/storage sizing; no automatic production sweep.'),
        production_budget_status='Not requested until measured per-model throughput, long-trace memory, and bytes/token are available.',
        prepared_first_job='sbatch/measurement_capture_smoke.sbatch gpt',
        first_job_scope='Two existing labeled completions, 256..4096 completion tokens, all layers, hidden and temporal features; native MXFP4.',
        code_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    runtime=BASE/'repo/src/moe_exp'
    files=[runtime/p for p in ('correlation_pipeline/extract.py','correlation_pipeline/features.py',
        'correlation_pipeline/views.py','correlation_pipeline/spans.py','correlation_pipeline/forward_adapter.py',
        'correlation_pipeline/labels_export.py','correlation_pipeline/annotation_batch.py',
        'models/loader.py','models/router_adapters.py','models/routing_extraction.py')]
    files+=list((runtime/'correlation_pipeline/dynamics').glob('*.py'))
    manifest['runtime_code_hashes']={str(p):sha(p) for p in files}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'campaign.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Complete cross-model measurement campaign','',
      f'Status: {status}. Existing CPU results are preliminary. Submission IDs and resource reservations are in results/measurement-study/gpu-budget.json.','',
      '| Model | Routed layers | Native replay readiness |','|---|---:|---|']
    for k,s in sources.items():lines.append(f"| {s['generation_model']} | {len(s['layers']) if s['layers'] is not None else 'unresolved'} | {s['native_backend_readiness']} |")
    lines+=['','Every source requires the same metric inventory in campaign.json. Counts refer to routed layers, not total decoder blocks.',
      'Native routing distributions differ (softmax versus normalized sigmoid affinities). Report semantics and normalized supplements; do not pretend these are identical selection rules.',
      'Native selection boundary diagnostics must account for bias/group restrictions; missing adapters are engineering work, not permission to substitute logit top-k.',
      'Unknown or sparse labels break class histories. Completing routing does not fill unlabeled sentences.',
      'Qwen3.5 GPTQ and GGUF are required separate recovery tracks; incomplete native capture keeps campaign coverage unfinished.',
      '', 'Validation envelope: at most eight two-GPU, 30-minute debug jobs (8 GPU-hours total), at most two running or pending debug jobs per user. The user authorized this envelope; backend readiness is checked before each submission.',
      'First concrete job: native GPT-OSS all-layer replay of two saved labeled completions, including hidden-state summaries and the new temporal reducers.',
      'Production replay will use normal QoS and a separate measured capped proposal. Validation submissions are recorded in the budget ledger.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{'layers':len(s['layers']) if s['layers'] is not None else None,'readiness':s['native_backend_readiness']} for k,s in sources.items()},indent=2))

if __name__=='__main__':main()
