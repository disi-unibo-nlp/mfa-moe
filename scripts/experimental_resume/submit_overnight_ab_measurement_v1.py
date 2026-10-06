"""Submit authorized A/B postprocessing chains after exact generation job IDs."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as dispatch


def main():
    for lane, generation_job in [('A', '59344221'), ('B', '59344222')]:
        manifest_path, manifest, measurement, _ = dispatch.paths(lane)
        generation_price = dispatch.DOC / f'OVERNIGHT_DISCOVERY_{lane}_PRICE_v1.json'
        envelope = dispatch.base.sealed(dispatch.DOC / f'OVERNIGHT_DISCOVERY_{lane}_ENVELOPE_v1.json')
        directory = dispatch.DOC / ('overnight-submissions-' + manifest['sha256'][:16])
        directory.mkdir(exist_ok=True)
        scripts = dispatch.REPO / 'scripts/experimental_resume'
        wrappers = [scripts / 'build_price_overnight_semantics_v1.sbatch', scripts / 'dispatch_overnight_readers_v1.sbatch']
        for path in wrappers:
            subprocess.run(['bash', '-n', str(path)], check=True)
        env = os.environ.copy()
        env.update(OVERNIGHT_LANE=lane, OVERNIGHT_MANIFEST=str(manifest_path),
                   OVERNIGHT_GENERATION_PRICE=str(generation_price),
                   OVERNIGHT_GENERATION_OUT=str(dispatch.RUNS / ('overnight-routing-v1-' + manifest['sha256'][:16])),
                   OVERNIGHT_MEASUREMENT_OUT=str(measurement),
                   OVERNIGHT_READER_GPU_HOUR_CEILING=str(envelope['semantic_measurement']['measurement_gpu_hour_ceiling']))
        binding = {'manifest_sha256': manifest['sha256'], 'generation_job': generation_job,
                   'envelope_sha256': envelope['sha256'],
                   'wrapper_hashes': {str(p): dispatch.base.file_sha(p) for p in wrappers}}
        price_job = dispatch.submit(directory, 'frame-price', [f'--dependency=afterok:{generation_job}',
                                    f'--job-name=st-overnight-{lane}-frame-price', str(wrappers[0])], env, binding)
        dispatch_job = dispatch.submit(directory, 'reader-dispatch', [f'--dependency=afterok:{price_job}',
                                       f'--job-name=st-overnight-{lane}-dispatch', str(wrappers[1])], env, binding)
        dispatch.save(directory / 'GENERATION_TO_MEASUREMENT.json',
                      {'schema': 'overnight-generation-measurement-chain-v1', 'lane': lane, 'binding': binding,
                       'generation_job': generation_job, 'frame_price_job': price_job, 'dispatch_job': dispatch_job})


if __name__ == '__main__':
    main()
