"""Price the complete frozen M9 closure and grading stages after replay passes.

Read-only Slurm accounting is required. This script never submits production.
If the registered two-GPU-hour generation/replay ceiling is exceeded, it emits
an amendment-required result instead of a false production pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
MANIFEST = R / 'steering-v1/runs/m9-resume-v1/MANIFEST.json'
COST = R / 'steering-v1/runs/x2prep/cost_model.json'
EXPECTED_MANIFEST = '63772b263e855692ecc5a37b10f634f96b30fa51e7e54bf22f5026f16e43024b'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed M9 input seal: ' + str(path))
    return value


def gpu_hours(job_id, *, state='COMPLETED', exit_code='0:0'):
    raw = subprocess.run(['sacct', '-n', '-P', '-j', str(job_id), '-o',
                          'JobID,State,ExitCode,ElapsedRaw,AllocTRES%100'],
                         check=True, text=True, capture_output=True).stdout
    parent = next((line.split('|') for line in raw.splitlines() if line.split('|')[0] == str(job_id)), None)
    if parent is None or parent[1] != state or parent[2] != exit_code:
        raise ValueError('replay Slurm job state or exit code differs')
    fields = dict(part.split('=', 1) for part in parent[4].split(',') if '=' in part)
    if int(fields.get('gres/gpu', 0)) != 2:
        raise ValueError('M9 replay must use exactly two allocated GPUs')
    return int(parent[3]) * 2 / 3600


def price(replay):
    manifest = sealed(MANIFEST)
    if manifest['sha256'] != EXPECTED_MANIFEST or replay['manifest_sha256'] != manifest['sha256'] or replay['pass'] is not True:
        raise ValueError('M9 manifest or replay gate not qualified')
    if (replay.get('schema') != 'M9-deterministic-isolation-adjudication-v1' or
        replay.get('sampling_mode') != 'deterministic_greedy_isolation_only' or
        replay.get('stochastic_batch_identity') != 'FAILED_PREVIOUS_CHECK_UNRESOLVED'):
        raise ValueError('M9 replay adjudication scope differs')
    if not all(replay.get(k) is True for k in ('identical_prefix','prefix_presence','closure_injection','batch_isolation')):
        raise ValueError('replay lacks a required engineering pass')
    measured_replay = gpu_hours(replay['job_id'])
    failed_preflight = [
        {'job_id': 59116442, 'GPU_h': gpu_hours(59116442, state='FAILED', exit_code='1:0')},
        {'job_id': 59117029, 'GPU_h': gpu_hours(59117029, state='FAILED', exit_code='3:0')},
    ]
    failed_replay = sum(row['GPU_h'] for row in failed_preflight)
    cost = json.loads(COST.read_text())
    decode_tps = cost['inputs']['x1']['steady_tok_per_s'] * cost['constants']['pessimistic_derate']
    prefill_tps = cost['inputs']['q10']['prefill_tps'] * cost['constants']['pessimistic_derate']
    # Same-manifest recovery is checkpointed; the 25% allowance covers repeated
    # suffix work, with two full cold loads and shutdown allowances.
    decode_s = manifest['max_decode_tokens'] / decode_tps * 1.25
    prefill_s = manifest['prefill_tokens'] / prefill_tps * 1.25
    load_s = 2 * 838
    shutdown_s = 2 * 196
    production_h = 2 * (decode_s + prefill_s + load_s + shutdown_s) / 3600
    generation_replay_h = failed_replay + measured_replay + production_h
    # Prior J1 job 59068750: 228 items, 1552 s allocated, 736 s until ready.
    # Bound all 239 generated answers plus 28 deterministic integer emissions,
    # even if every strict result requires J1. Retained/fallback native answers
    # reuse the scored X1 record when naturally stopped by B16; caps count wrong.
    items_max = manifest['counts']['generate_closure'] + manifest['integer_comparator_counts']['integer_emission']
    if items_max != 267:
        raise ValueError('M9 maximum new J1 item count changed')
    grading_s = items_max * (1552 - 736) / 228 * 2 + 2 * 736 + 2 * 196
    grading_h = 2 * grading_s / 3600
    body = {'schema': 'M9-complete-stage-price-v1', 'manifest_sha256': manifest['sha256'],
            'replay_sha256': replay['sha256'], 'replay_job_id': replay['job_id'],
            'replay_actual_GPU_h': measured_replay, 'failed_preflight_GPU_h': failed_replay,
            'failed_preflight_jobs': failed_preflight,
            'stochastic_batch_identity': 'FAILED_PREVIOUS_CHECK_UNRESOLVED',
            'generation_replay_loads_GPU_h': generation_replay_h,
            'generation_components': {'decode_seconds': decode_s, 'prefill_seconds': prefill_s,
                                      'two_cold_loads_seconds': load_s, 'two_shutdown_allowances_seconds': shutdown_s,
                                      'two_GPU_production_hours': production_h,
                                      'decode_tokens_max': manifest['max_decode_tokens'],
                                      'prefill_tokens': manifest['prefill_tokens'],
                                      'repeat_work_factor': 1.25},
            'grading_items_max': items_max, 'grading_GPU_h': grading_h,
            'grading_basis': 'J1 job 59068750: 228 items in 1552 allocated seconds, ready at 736 seconds; doubled per-item time, two loads and shutdown allowances',
            'registered_generation_ceiling_GPU_h': 2.,
            'registered_shared_grading_remaining_GPU_h': 1.1377777777777778,
            'status': ('PASS_COMPLETE_STAGE' if generation_replay_h <= 2. and grading_h <= 1.1377777777777778
                       else 'AMENDMENT_REQUIRED_BEFORE_PRODUCTION'),
            'note': 'Broad current user authorization permits a measured resource amendment. Greedy request isolation does not erase the failed stochastic identity check.'}
    return {**body, 'sha256': digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = price(sealed(args.replay))
    if args.out.exists():
        if sealed(args.out) != result:
            raise ValueError('existing M9 price differs')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'path': str(args.out), 'status': result['status'],
                      'generation_replay_GPU_h': result['generation_replay_loads_GPU_h'],
                      'grading_GPU_h': result['grading_GPU_h']}))


if __name__ == '__main__':
    main()
