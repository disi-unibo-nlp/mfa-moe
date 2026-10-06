"""One-shot exact-price submission chain for separate reader-cap sensitivity."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared
import mechanism_semantic_cap_recovery_v2 as recovery


def wall(seconds):
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def main():
    recovery.require(os.environ.get('SLURM_JOB_ID') and
                     os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz',
                     'measurement-sensitivity dispatcher requires authorized CPU Slurm')
    plan = shared.base.sealed(recovery.PLAN)
    recovery.validate(plan)
    recovery.require(plan['status'] == 'FROZEN_PRICED_REQUIRES_GPU_QUALIFICATION' and
                     plan['generation_price']['shards'] and
                     plan['qualification_price']['ratings'] == 4 and
                     plan['generation_price']['ratings'] == 180 and
                     plan['qualification_price']['gpus'] == plan['generation_price']['gpus'] == 2 and
                     plan['qualification_price']['max_wall_seconds'] == 3600 and
                     plan['generation_price']['max_wall_seconds'] == 7200 and
                     plan['estimated_complete_gpu_hours'] ==
                     plan['generation_price']['estimated_complete_gpu_hours'] +
                     plan['qualification_price']['estimated_complete_gpu_hours'] and
                     plan['estimated_complete_gpu_hours'] <= plan['complete_gpu_hour_ceiling'],
                     'qualification/full replay exact resource envelope differs')
    scripts = recovery.REPO / 'scripts/experimental_resume'
    gpu = scripts / 'run_mechanism_semantic_cap_recovery_v2.sbatch'
    analyze = scripts / 'analyze_mechanism_semantic_cap_recovery_v2.sbatch'
    for path in (gpu, analyze):
        subprocess.run(['bash', '-n', str(path)], check=True)
        recovery.require(plan['code_files'].get(str(path.resolve())) == shared.base.file_sha(path),
                         'priced wrapper differs')
    binding = {'manifest_sha256': plan['sha256'],
               'qualification_price_sha256': shared.base.digest(plan['qualification_price']),
               'generation_price_sha256': shared.base.digest(plan['generation_price']),
               'complete_gpu_hour_ceiling': plan['complete_gpu_hour_ceiling'],
               'source_hashes': {str(path.resolve()): shared.base.file_sha(path) for path in
                                (Path(__file__), Path(__file__).with_suffix('.sbatch'), gpu, analyze)}}
    directory = recovery.DOC / ('mechanism-cap-recovery-submissions-v2-' + plan['sha256'][:16])
    directory.mkdir(exist_ok=True)
    env = os.environ.copy(); env['CAP_RECOVERY_RESUME_UNCOMMITTED'] = '0'
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env['CAP_RECOVERY_MODE'] = 'qualification'
        pilot = shared.submit(directory, 'qualification',
            ['--time=' + wall(plan['qualification_price']['max_wall_seconds']),
             '--job-name=mechanism-cap-sensitivity-qual-v2', str(gpu)], env, binding)
        env['CAP_RECOVERY_MODE'] = 'generation'
        generation = shared.submit(directory, 'generation',
            ['--dependency=afterok:' + pilot,
             f'--array=0-{len(plan["generation_price"]["shards"])-1}',
             '--time=' + wall(plan['generation_price']['max_wall_seconds']),
             '--job-name=mechanism-cap-sensitivity-replay-v2', str(gpu)], env, binding)
        analysis = shared.submit(directory, 'analysis',
            ['--dependency=afterok:' + generation,
             '--job-name=mechanism-cap-sensitivity-analysis-v2', str(analyze)], env, binding)
        shared.save(directory / 'CHAIN.json', {
            'schema': 'mechanism-semantic-cap-recovery-chain-v2', 'binding': binding,
            'qualification_job': pilot, 'generation_job': generation, 'analysis_job': analysis,
            'original_primary_unchanged': True,
            'analysis_output': str(recovery.DOC / 'MECHANISM_SEMANTIC_CAP_RECOVERY_SENSITIVITY_v2.json')})


if __name__ == '__main__':
    main()
