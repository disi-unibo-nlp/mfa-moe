"""Seal the terminal unsupported-GDN outcome of the separate BI engine test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
JOB = '59197533'
GPU_RUN = ROOT / 'runs/routing-control-v1/batch-invariant-qual-db182a1debc7465c'
LOG = ROOT / f'logs/st-batch-inv-qual-v1-{JOB}.out'
ERR = ROOT / f'logs/st-batch-inv-qual-v1-{JOB}.err'
MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_BATCH_INVARIANT_QUAL_MANIFEST_v1.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_BATCH_INVARIANT_QUAL_FAILURE_v1.json'
MODEL_CONFIG = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
                    'models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/'
                    '95a723d08a9490559dae23d0cff1d9466213d989/config.json')
SELECTOR = Path('/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/lib/'
                'python3.11/site-packages/vllm/v1/attention/selector.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('seal changed: ' + str(path))
    return value


def main():
    job_line = subprocess.check_output([
        'sacct', '-j', JOB, '--format=JobID,State,ExitCode,ElapsedRaw', '-n', '-P'],
        text=True).splitlines()[0].split('|')
    if job_line != [JOB, 'FAILED', '3:0', '856']:
        raise ValueError('GPU job final state differs: ' + repr(job_line))
    dependent = subprocess.check_output([
        'sacct', '-j', '59198296', '--format=JobID,State,ElapsedRaw', '-n', '-P'],
        text=True).splitlines()[0].split('|')
    if dependent != ['59198296', 'CANCELLED', '0']:
        raise ValueError('dependent CPU audit status differs')
    manifest, binding = sealed(MANIFEST), sealed(GPU_RUN / 'BINDING.json')
    if binding['manifest_sha256'] != manifest['sha256']:
        raise ValueError('failed output binding changed')
    if (GPU_RUN / 'SUMMARY.json').exists() or list(GPU_RUN.glob('batch-*.json')):
        raise ValueError('unexpected inference receipts from unsupported engine')
    log = LOG.read_text(errors='replace')
    if ('Using HUMMING Fp8 MoE backend' not in log or
        'VLLM batch_invariant mode is not supported for GDN_ATTN.' not in log):
        raise ValueError('expected Humming selection/GDN incompatibility absent')
    config = json.loads(MODEL_CONFIG.read_text())
    layers = config['text_config']['layer_types'] if 'text_config' in config else config['layer_types']
    if 'linear_attention' not in layers:
        raise ValueError('GDN model layer evidence changed')
    if 'not mamba_attn_backend.supports_batch_invariance()' not in SELECTOR.read_text():
        raise ValueError('installed vLLM GDN rejection guard changed')
    body = {'schema': 'routing-batch-invariant-qualification-failure-v1',
            'job_id': JOB, 'final_state': 'FAILED', 'exit_code': '3:0',
            'elapsed_seconds': 856, 'gpu_count': 2,
            'charged_gpu_hour_upper_bound': 2 * 856 / 3600,
            'manifest_sha256': manifest['sha256'],
            'binding_sha256': binding['sha256'],
            'driver_sha256': file_sha(REPO / 'scripts/experimental_resume/run_batch_invariant_qual.py'),
            'launcher_sha256': file_sha(REPO / 'scripts/experimental_resume/run_batch_invariant_qual_v1.sbatch'),
            'stdout_sha256': file_sha(LOG), 'stderr_sha256': file_sha(ERR),
            'model_config_sha256': file_sha(MODEL_CONFIG),
            'vllm_selector_sha256': file_sha(SELECTOR),
            'selected_moe_backend': 'HUMMING Fp8 MoE',
            'blocking_backend': 'GDN_ATTN',
            'root_cause': 'installed vLLM 0.29 rejects global VLLM_BATCH_INVARIANT=1 for Qwen3.6 GDN attention during worker initialization',
            'inference_requests_completed': 0,
            'dependent_cpu_audit_job_id': '59198296',
            'dependent_cpu_audit_state': 'CANCELLED',
            'dependent_cpu_audit_elapsed_seconds': 0,
            'decision': 'Do not rerun global batch-invariant flag for this model/wheel; pursue separate serial/eager native-repeat qualification with original MARLIN engine.'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing failure receipt differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'gpu_hours_upper': body['charged_gpu_hour_upper_bound']}))


if __name__ == '__main__':
    main()
