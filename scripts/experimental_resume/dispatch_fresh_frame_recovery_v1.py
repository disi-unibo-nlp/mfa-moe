"""Dispatch fresh readers only after guarded exact post-generation pricing."""
import fcntl
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared
import fresh_frame_recovery_v1 as recovery


def main():
    recovery.require(os.environ.get('SLURM_JOB_ID') and
                     os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz',
                     'fresh reader dispatch requires one-shot CPU Slurm')
    m, a = shared.base.sealed(recovery.MANIFEST), shared.base.sealed(recovery.AMENDMENT)
    paths = a['paths']
    measurement = Path(paths['measurement'])
    amap, frame, price = (shared.base.sealed(measurement / name) for name in
                          ('ARM_MAP.json', 'BLIND_FRAME.json', 'READER_PRICE.json'))
    fields = recovery.validate_measurement(m, amap, frame, price, a)
    envelope = shared.base.sealed(recovery.ENVELOPE)
    recovery.require(price['estimated_complete_gpu_hours'] <=
                     envelope['semantic_measurement']['measurement_gpu_hour_ceiling'],
                     'recovered exact reader price exceeds frozen complete envelope')
    directory = Path(paths['source'])
    env = os.environ.copy()
    env.update(OVERNIGHT_MANIFEST=str(recovery.MANIFEST), OVERNIGHT_MEASUREMENT_OUT=str(measurement),
               OVERNIGHT_BLIND_FRAME=str(measurement / 'BLIND_FRAME.json'),
               OVERNIGHT_READER_PRICE=str(measurement / 'READER_PRICE.json'),
               OVERNIGHT_READER_OUT=paths['ratings'], OVERNIGHT_ANALYSIS_OUT=paths['analysis'])
    wrappers = [recovery.SCRIPTS / 'rate_overnight_semantics_fresh_adapter_v1.sbatch',
                recovery.SCRIPTS / 'analyze_overnight_semantics_fresh_adapter_v1.sbatch']
    for path in wrappers:
        subprocess.run(['bash', '-n', str(path)], check=True)
    binding = {'manifest_sha256': m['sha256'], 'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
               'envelope_sha256': envelope['sha256'], 'recovery_provenance': fields,
               'wrapper_hashes': {str(p): shared.base.file_sha(p) for p in wrappers}}
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = None
        if price['shards']:
            job = shared.submit(directory, 'reader-array', [f'--array=0-{len(price["shards"])-1}',
                                '--job-name=st-overnight-FRESH-read-recovery-v1', str(wrappers[0])], env, binding)
        analysis = shared.submit(directory, 'analysis', [*([f'--dependency=afterok:{job}'] if job else []),
                                 '--job-name=st-overnight-FRESH-itt-recovery-v1', str(wrappers[1])], env, binding)
        shared.save(directory / 'MEASUREMENT_CHAIN.json', {'schema': 'overnight-measurement-chain-v2',
            'binding': binding, 'reader_job': job, 'analysis_job': analysis,
            'analysis_output': paths['analysis'], 'recovery_amendment_sha256': a['sha256']})


if __name__ == '__main__':
    main()
