"""One-shot authorized Slurm dispatch after exact blind-reader pricing.

Each submission has a durable attempted receipt before sbatch. An interrupted
unresolved attempt fails closed rather than silently submitting duplicate work.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import run_boundary_micro_screen as base
import rate_overnight_semantics_v1 as reader

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
RUNS = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')


def save(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('existing dispatch receipt differs: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=1)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    return value


def submission_environment(env):
    """Keep application settings; let the new allocation establish placement.

    sbatch called inside srun otherwise exports parent CPU/GPU/rank bindings.
    Remove scheduler options too: resource requests come from reviewed arguments
    and batch directives, never from a parent allocation's environment.
    This copy does not mutate the running dispatcher's own Slurm environment.
    """
    prefixes = ('SLURM_', 'SRUN_', 'SBATCH_', 'PMI_', 'PMIX_')
    device_keys = {'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
                   'ROCR_VISIBLE_DEVICES', 'ZE_AFFINITY_MASK'}
    return {key: value for key, value in env.items()
            if not key.startswith(prefixes) and key not in device_keys}


def submit(directory, name, arguments, env, binding):
    receipt = directory / f'{name}.json'
    attempted = directory / f'{name}.attempt.json'
    if receipt.exists():
        old = base.sealed(receipt)
        if old['binding'] != binding or old['arguments'] != arguments:
            raise ValueError('existing job receipt has different inputs')
        return old['job_id']
    if attempted.exists():
        raise RuntimeError('unresolved prior sbatch attempt; reconcile exact job before retry: ' + str(attempted))
    child_env = submission_environment(env)
    removed_keys = sorted(set(env) - set(child_env))
    subprocess.run(['sbatch', '--test-only', *arguments], check=True, env=child_env, capture_output=True, text=True)
    save(attempted, {'schema': 'overnight-submit-attempt-v1', 'name': name,
                     'binding': binding, 'arguments': arguments, 'time_ns': time.time_ns(),
                     'dispatcher_job_id': os.environ.get('SLURM_JOB_ID'),
                     'removed_parent_environment_keys': removed_keys})
    result = subprocess.run(['sbatch', '--parsable', *arguments], check=True, env=child_env, capture_output=True, text=True)
    identifier = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'[0-9]+', identifier):
        raise RuntimeError('unrecognized sbatch response; reconcile attempted receipt')
    record = {'schema': 'overnight-submit-receipt-v1', 'name': name, 'binding': binding,
              'arguments': arguments, 'job_id': identifier, 'submission': result.stdout.strip(),
              'removed_parent_environment_keys': removed_keys}
    save(receipt, record)
    verified = subprocess.run(['scontrol', 'show', 'job', identifier], check=True, capture_output=True, text=True)
    if f'JobId={identifier}' not in verified.stdout or 'UserId=lmolfett(' not in verified.stdout:
        raise RuntimeError('exact submitted job did not verify')
    save(directory / f'{name}.verified.json', {'schema': 'overnight-job-verification-v1',
                                              'job_id': identifier, 'scontrol': verified.stdout})
    print(json.dumps({'submitted': name, 'job_id': identifier}), flush=True)
    return identifier


def paths(lane):
    if lane not in ('A', 'B'):
        raise ValueError('this immutable dispatcher supports qualified v1A/B only')
    manifest_path = DOC / f'OVERNIGHT_DISCOVERY_{lane}_MANIFEST_v1.json'
    manifest = base.sealed(manifest_path)
    tag = manifest['sha256'][:16]
    return manifest_path, manifest, RUNS / f'overnight-semantic-v1-{tag}', RUNS / f'overnight-readers-v1-{tag}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', required=True, choices=('A', 'B'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('dispatcher requires its authorized one-shot CPU Slurm job')
    manifest_path, manifest, measurement, ratings = paths(args.lane)
    frame, price = base.sealed(measurement / 'BLIND_FRAME.json'), base.sealed(measurement / 'READER_PRICE.json')
    reader.validate(frame, price)
    envelope = base.sealed(DOC / f'OVERNIGHT_DISCOVERY_{args.lane}_ENVELOPE_v1.json')
    if (frame['generation_manifest_sha256'] != manifest['sha256'] or
            envelope['manifest_sha256'] != manifest['sha256'] or
            price['estimated_complete_gpu_hours'] > envelope['semantic_measurement']['measurement_gpu_hour_ceiling']):
        raise ValueError('manifest binding or complete reader envelope differs')
    directory = DOC / ('overnight-submissions-' + manifest['sha256'][:16])
    directory.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OVERNIGHT_MANIFEST=str(manifest_path), OVERNIGHT_MEASUREMENT_OUT=str(measurement),
               OVERNIGHT_BLIND_FRAME=str(measurement / 'BLIND_FRAME.json'),
               OVERNIGHT_READER_PRICE=str(measurement / 'READER_PRICE.json'), OVERNIGHT_READER_OUT=str(ratings),
               OVERNIGHT_ANALYSIS_OUT=str(DOC / f'OVERNIGHT_DISCOVERY_{args.lane}_ANALYSIS_v1'))
    wrappers = [REPO / 'scripts/experimental_resume/rate_overnight_semantics_v1.sbatch',
                REPO / 'scripts/experimental_resume/analyze_overnight_semantics_v1.sbatch']
    for path in wrappers:
        subprocess.run(['bash', '-n', str(path)], check=True)
    binding = {'manifest_sha256': manifest['sha256'], 'frame_sha256': frame['sha256'],
               'price_sha256': price['sha256'], 'envelope_sha256': envelope['sha256'],
               'wrapper_hashes': {str(p): base.file_sha(p) for p in wrappers}}
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = None
        if price['shards']:
            job = submit(directory, 'reader-array', [f'--array=0-{len(price["shards"])-1}',
                        f'--job-name=st-overnight-{args.lane}-readers', str(wrappers[0])], env, binding)
        analysis_args = ([f'--dependency=afterok:{job}'] if job else [])
        analysis = submit(directory, 'analysis', [*analysis_args,
                          f'--job-name=st-overnight-{args.lane}-analysis', str(wrappers[1])], env, binding)
        save(directory / 'CHAIN.json', {'schema': 'overnight-measurement-chain-v1', 'binding': binding,
                                       'reader_job': job, 'analysis_job': analysis,
                                       'analysis_output': env['OVERNIGHT_ANALYSIS_OUT']})


if __name__ == '__main__':
    main()
