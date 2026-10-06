"""Attach one complete dense measurement chain to an authorized generation stage."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess

import dispatch_generated_dense_v1 as dispatch
import dispatch_overnight_readers_v1 as shared
import generated_dense_pipeline_v1 as dense

SCRIPTS = shared.REPO / 'scripts/experimental_resume'
PREP = SCRIPTS / 'prepare_generated_dense_v1.sbatch'
DISPATCH = SCRIPTS / 'dispatch_generated_dense_v1.sbatch'


def stage_paths(stage, dense_parent=None):
    dense.require(stage in ('A', 'B', 'C', 'FRESH'), 'unsupported generated dense stage')
    version = 1 if stage in ('A', 'B') else 2
    prefix = 'OVERNIGHT_FRESH_COMPARISON' if stage == 'FRESH' else 'OVERNIGHT_DISCOVERY_' + stage
    manifest_path = shared.DOC / f'{prefix}_MANIFEST_v{version}.json'
    price_path = shared.DOC / f'{prefix}_PRICE_v{version}.json'
    manifest, price = dense.sealed(manifest_path), dense.sealed(price_path)
    dense.require(manifest['schema'] == f'overnight-routing-manifest-v{version}' and
                  price['manifest_sha256'] == manifest['sha256'] and price['shards'] == manifest['shards'] and
                  price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY', 'stage manifest and complete generation price differ')
    tag = manifest['sha256'][:16]
    run_out = shared.RUNS / f'overnight-routing-v{version}-{tag}'
    parent = Path(dense_parent) if dense_parent is not None else shared.RUNS / 'generated-dense'
    dense.require(str(parent.resolve()).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/'),
                  'dense outputs must remain in authorized owned work tree')
    return manifest_path, price_path, manifest, price, run_out, parent


def frame_dependency(stage, manifest, explicit=None):
    if explicit is not None:
        dense.require(bool(re.fullmatch(r'[0-9]+', str(explicit))), 'frame-price job must be a Slurm parent ID')
        return str(explicit), {'source': 'explicit_authorized_frame_price_job', 'job_id': str(explicit)}
    if stage in ('A', 'B'):
        source = shared.DOC / 'OVERNIGHT_AB_RECOVERY_CHAIN_v1.json'
        value = dense.sealed(source)
        dense.require(value['schema'] == 'overnight-ab-recovery-chain-v1', 'A/B recovery chain schema differs')
        rows = [r for r in value['records'] if r['lane'] == stage]
        dense.require(len(rows) == 1, 'A/B frame-price dependency is ambiguous')
        job = rows[0]['frame_price_job']
    else:
        source = shared.DOC / ('overnight-submissions-v2-' + manifest['sha256'][:16]) / 'GENERATION_CHAIN.json'
        value = dense.sealed(source)
        dense.require(value['schema'] == 'overnight-generation-chain-v2' and
                      value['binding']['manifest_sha256'] == manifest['sha256'],
                      'v2 generation chain differs from selected manifest')
        job = value['frame_price_job']
    dense.require(isinstance(job, str) and bool(re.fullmatch(r'[0-9]+', job)), 'invalid frame-price dependency receipt')
    return job, {'path': str(source), 'sha256': value['sha256'], 'job_id': job}


def submit_chain(stage, paths, frame_job, dependency_receipt, *, submit=shared.submit):
    manifest_path, price_path, manifest, price, run_out, parent = paths
    plan = dense.sealed(dispatch.PLAN)
    dense.require(plan['code_files'] == dense.measurement_code(), 'frozen dense measurement source changed')
    directory = shared.DOC / ('generated-dense-chain-' + manifest['sha256'][:16] + '-' + plan['code_digest'][:12])
    directory.mkdir(exist_ok=True)
    args = [str(manifest_path), str(price_path), str(run_out), str(parent)]
    binding = {'schema': 'generated-dense-chain-binding-v1', 'stage': stage,
               'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
               'measurement_plan_sha256': plan['sha256'], 'frame_price_dependency': dependency_receipt,
               'dense_directory': str(dense.output_path(manifest, parent)),
               'chain_driver_sha256': dense.file_sha(__file__),
               'dispatcher_sha256': dense.file_sha(dispatch.__file__),
               'shared_submit_sha256': dense.file_sha(shared.__file__),
               'wrappers': {str(p): dense.file_sha(p) for p in (PREP, DISPATCH)}}
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = 'LIVE_ACCOUNT-' + os.environ.get('SLURM_JOB_ID', str(os.getpid())) + '.json'
        if not (directory / snapshot).exists():
            dispatch.live_account_snapshot(directory, snapshot)
        for wrapper in (PREP, DISPATCH):
            subprocess.run(['bash', '-n', str(wrapper)], check=True)
        env = os.environ.copy()
        prep_job = submit(directory, 'dense-cpu-prepare', [f'--dependency=afterok:{frame_job}',
                          f'--job-name=dense-{stage}-prepare', str(PREP), *args], env, binding)
        dispatch.ensure_verified(directory, 'dense-cpu-prepare', prep_job)
        dispatcher_job = submit(directory, 'dense-gpu-dispatch', [f'--dependency=afterok:{prep_job}',
                                f'--job-name=dense-{stage}-dispatch', str(DISPATCH), *args], env, binding)
        dispatch.ensure_verified(directory, 'dense-gpu-dispatch', dispatcher_job)
        return shared.save(directory / 'CHAIN.json', {'schema': 'generated-dense-parent-chain-v1',
                           'binding': binding, 'frame_price_job': frame_job, 'dense_cpu_prepare_job': prep_job,
                           'dense_gpu_dispatch_job': dispatcher_job, 'dense_directory': binding['dense_directory'],
                           'scope': 'Saved continuation labeling and trajectory measurement only; no native generation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('A', 'B', 'C', 'FRESH'))
    parser.add_argument('--frame-price-job')
    parser.add_argument('--dense-parent', type=Path)
    args = parser.parse_args()
    dense.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
                  getpass.getuser() == 'lmolfett' and not socket.gethostname().startswith('login'),
                  'dense chain attachment requires authorized one-shot viz CPU allocation')
    paths = stage_paths(args.stage, args.dense_parent)
    job, receipt = frame_dependency(args.stage, paths[2], args.frame_price_job)
    result = submit_chain(args.stage, paths, job, receipt)
    print(json.dumps({'stage': args.stage, 'frame_price_job': job,
                      'dense_cpu_prepare_job': result['dense_cpu_prepare_job'],
                      'dense_gpu_dispatch_job': result['dense_gpu_dispatch_job'],
                      'dense_directory': result['dense_directory'], 'sha256': result['sha256']}), flush=True)


if __name__ == '__main__':
    main()
