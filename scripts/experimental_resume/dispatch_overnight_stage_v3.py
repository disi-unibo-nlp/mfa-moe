"""One-shot complete-stage generation or reader dispatch for C/fresh v2."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared
import rate_overnight_semantics_v2 as reader


def stage_paths(stage):
    if stage not in ('C', 'FRESH'):
        raise ValueError('unsupported stage')
    prefix = 'OVERNIGHT_DISCOVERY_C' if stage == 'C' else 'OVERNIGHT_FRESH_COMPARISON'
    manifest_path = shared.DOC / (prefix + '_MANIFEST_v2.json')
    manifest = shared.base.sealed(manifest_path)
    tag = manifest['sha256'][:16]
    return prefix, manifest_path, manifest, shared.RUNS / ('overnight-semantic-v2-' + tag), shared.RUNS / ('overnight-readers-v2-' + tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('C', 'FRESH'), required=True)
    parser.add_argument('--phase', choices=('generation', 'readers'), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('dispatch requires authorized one-shot CPU Slurm')
    prefix, manifest_path, manifest, measurement, ratings = stage_paths(args.stage)
    price_path = shared.DOC / (prefix + '_PRICE_v2.json')
    generation_price = shared.base.sealed(price_path)
    envelope = shared.base.sealed(shared.DOC / (prefix + '_ENVELOPE_v2.json'))
    if (generation_price['manifest_sha256'] != manifest['sha256'] or
            generation_price['status'] != 'PASS_COMPLETE_STAGE_GENERATION_ONLY' or
            not manifest['shards'] or generation_price['shards'] != manifest['shards'] or
            generation_price['sha256'] != envelope['generation_price_sha256'] or
            envelope['manifest_sha256'] != manifest['sha256'] or
            envelope['status'] != 'PASS_COMPLETE_GENERATION_AND_SEMANTIC_ENVELOPE' or
            not envelope.get('execution_files') or
            any(shared.base.file_sha(path) != digest for path, digest in envelope['execution_files'].items())):
        raise ValueError('exact generation price/envelope mismatch')
    directory = shared.DOC / ('overnight-submissions-v2-' + manifest['sha256'][:16])
    directory.mkdir(exist_ok=True)
    scripts = shared.REPO / 'scripts/experimental_resume'
    env = os.environ.copy()
    env.update(OVERNIGHT_STAGE=args.stage, MANIFEST=str(manifest_path), PRICE=str(price_path),
               OVERNIGHT_MANIFEST=str(manifest_path), OVERNIGHT_GENERATION_PRICE=str(price_path),
               OVERNIGHT_GENERATION_OUT=str(shared.RUNS / ('overnight-routing-v2-' + manifest['sha256'][:16])),
               OVERNIGHT_MEASUREMENT_OUT=str(measurement),
               OVERNIGHT_READER_GPU_HOUR_CEILING=str(envelope['semantic_measurement']['measurement_gpu_hour_ceiling']),
               OVERNIGHT_BLIND_FRAME=str(measurement / 'BLIND_FRAME.json'),
               OVERNIGHT_READER_PRICE=str(measurement / 'READER_PRICE.json'), OVERNIGHT_READER_OUT=str(ratings),
               OVERNIGHT_ANALYSIS_OUT=str(shared.DOC / (prefix + '_ANALYSIS_v2')))
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'generation':
            wrappers = [scripts / 'overnight_routing_run_v3.sbatch', scripts / 'build_price_overnight_semantics_v3.sbatch',
                        scripts / 'dispatch_overnight_stage_v3.sbatch']
            for path in wrappers:
                subprocess.run(['bash', '-n', str(path)], check=True)
            binding = {'manifest_sha256': manifest['sha256'], 'generation_price_sha256': generation_price['sha256'],
                       'envelope_sha256': envelope['sha256'], 'wrapper_hashes': {str(p): shared.base.file_sha(p) for p in wrappers}}
            seconds = manifest['max_wall_seconds']
            wall = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
            job = shared.submit(directory, 'generation', [f'--array=0-{len(manifest["shards"])-1}',
                                '--time=' + wall, f'--job-name=st-overnight-{args.stage}-v2', str(wrappers[0])], env, binding)
            price_job = shared.submit(directory, 'frame-price', [f'--dependency=afterok:{job}',
                                      f'--job-name=st-overnight-{args.stage}-price-v2', str(wrappers[1])], env, binding)
            env['OVERNIGHT_DISPATCH_PHASE'] = 'readers'
            dispatcher = shared.submit(directory, 'reader-dispatch', [f'--dependency=afterok:{price_job}',
                                       f'--job-name=st-overnight-{args.stage}-dispatch-v2', str(wrappers[2])], env, binding)
            shared.save(directory / 'GENERATION_CHAIN.json', {'schema': 'overnight-generation-chain-v2', 'binding': binding,
                         'generation_job': job, 'frame_price_job': price_job, 'reader_dispatch_job': dispatcher})
        else:
            frame = shared.base.sealed(measurement / 'BLIND_FRAME.json')
            price = shared.base.sealed(measurement / 'READER_PRICE.json')
            reader.validate(frame, price)
            if (frame['generation_manifest_sha256'] != manifest['sha256'] or
                    price['estimated_complete_gpu_hours'] > envelope['semantic_measurement']['measurement_gpu_hour_ceiling']):
                raise ValueError('exact semantic price/envelope mismatch')
            wrappers = [scripts / 'rate_overnight_semantics_v2.sbatch', scripts / 'analyze_overnight_semantics_v2.sbatch']
            for path in wrappers:
                subprocess.run(['bash', '-n', str(path)], check=True)
            binding = {'manifest_sha256': manifest['sha256'], 'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
                       'envelope_sha256': envelope['sha256'], 'wrapper_hashes': {str(p): shared.base.file_sha(p) for p in wrappers}}
            job = None
            if price['shards']:
                job = shared.submit(directory, 'reader-array', [f'--array=0-{len(price["shards"])-1}',
                                    f'--job-name=st-overnight-{args.stage}-read-v2', str(wrappers[0])], env, binding)
            analysis = shared.submit(directory, 'analysis', [*([f'--dependency=afterok:{job}'] if job else []),
                                     f'--job-name=st-overnight-{args.stage}-itt-v2', str(wrappers[1])], env, binding)
            shared.save(directory / 'MEASUREMENT_CHAIN.json', {'schema': 'overnight-measurement-chain-v2', 'binding': binding,
                         'reader_job': job, 'analysis_job': analysis, 'analysis_output': env['OVERNIGHT_ANALYSIS_OUT']})


if __name__ == '__main__':
    main()
