"""One-shot CPU attachment for frozen routing target engagement; no generation."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import socket
import subprocess

import analyze_routing_first_stage_v1 as analysis
import dispatch_generated_dense_v1 as dispatch
import dispatch_overnight_readers_v1 as shared
import submit_generated_dense_chain_v1 as stages

WRAPPER = Path(__file__).with_name('analyze_routing_first_stage_v1.sbatch')


def submit_stage(stage, paths, frame_job, dependency_receipt, *, submit=shared.submit):
    manifest_path, price_path, manifest, price, generation_out, parent = paths
    plan = analysis.validate_plan()
    directory = shared.DOC / ('routing-first-stage-chain-' + manifest['sha256'][:16] + '-' + plan['sha256'][:12])
    directory.mkdir(exist_ok=True)
    out = analysis.output_path(manifest, parent, plan)
    binding = {'schema': 'routing-first-stage-chain-binding-v1', 'stage': stage,
               'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
               'plan_sha256': plan['sha256'], 'frame_price_dependency': dependency_receipt,
               'analysis_directory': str(out), 'wrapper_sha256': analysis.source.file_sha(WRAPPER),
               'resources': {'partition': 'lrd_all_viz', 'cpus': 2, 'memory_GiB': 16,
                             'wall_seconds': 7200, 'maximum_CPU_core_hours': 4., 'GPU_hours': 0.}}
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = 'LIVE_ACCOUNT-' + os.environ.get('SLURM_JOB_ID', str(os.getpid())) + '.json'
        if not (directory / snapshot).exists():
            dispatch.live_account_snapshot(directory, snapshot)
        subprocess.run(['bash', '-n', str(WRAPPER)], check=True)
        args = [f'--dependency=afterok:{frame_job}', f'--job-name=routing-engagement-{stage}', str(WRAPPER),
                str(manifest_path), str(price_path), str(generation_out), str(parent)]
        job = submit(directory, 'routing-first-stage-analysis', args, os.environ.copy(), binding)
        dispatch.ensure_verified(directory, 'routing-first-stage-analysis', job)
        return shared.save(directory / 'CHAIN.json', {'schema': 'routing-first-stage-chain-v1',
            'binding': binding, 'frame_price_job': frame_job, 'analysis_job': job,
            'analysis_directory': str(out), 'scope': 'Already saved routes and telemetry; CPU only.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('A', 'B', 'C', 'FRESH'), required=True)
    parser.add_argument('--frame-price-job')
    parser.add_argument('--out-parent', type=Path, default=shared.RUNS / 'routing-first-stage')
    args = parser.parse_args()
    analysis.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
                     getpass.getuser() == 'lmolfett' and not socket.gethostname().startswith('login'),
                     'routing first-stage attachment requires authorized one-shot viz allocation')
    paths = stages.stage_paths(args.stage, args.out_parent)
    job, receipt = stages.frame_dependency(args.stage, paths[2], args.frame_price_job)
    result = submit_stage(args.stage, paths, job, receipt)
    print(json.dumps({'stage': args.stage, 'frame_price_job': job, 'analysis_job': result['analysis_job'],
                      'analysis_directory': result['analysis_directory'], 'sha256': result['sha256']}), flush=True)


if __name__ == '__main__':
    main()
