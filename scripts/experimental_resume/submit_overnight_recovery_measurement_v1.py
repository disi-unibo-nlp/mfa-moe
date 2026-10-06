"""Record the namespace recovery and reattach immutable A/B measurement chains."""
import os
import subprocess
import dispatch_overnight_readers_v1 as d


def main():
    files = [d.REPO / 'scripts/experimental_resume' / name for name in
             ('overnight_routing_entry_v1.py', 'overnight_routing_run_recovery_v1.sbatch',
              'diagnose_mechanism_validation_v3.py')]
    records = []
    for lane, job, previous in [('A', '59344807', '59344221'), ('B', '59344808', '59344222')]:
        manifest_path, manifest, measurement, _ = d.paths(lane)
        envelope = d.base.sealed(d.DOC / f'OVERNIGHT_DISCOVERY_{lane}_ENVELOPE_v1.json')
        directory = d.DOC / ('overnight-submissions-' + manifest['sha256'][:16])
        verification = subprocess.run(['scontrol', 'show', 'job', job], check=True, capture_output=True, text=True)
        if f'JobId={job}' not in verification.stdout:
            raise ValueError('recovery array not observed')
        d.save(directory / 'generation-recovery-verified.json',
               {'schema': 'overnight-generation-recovery-v1', 'job_id': job, 'previous_job_id': previous,
                'manifest_sha256': manifest['sha256'], 'scontrol': verification.stdout,
                'recovery_code': {str(p): d.base.file_sha(p) for p in files},
                'startup_proof': 'Exact CPU startup ordering passed before relaunch; qualified package pinned before any validation imports.',
                'preservation': 'No completed generation batches existed at recovery decision; previous empty bindings and all Slurm failures preserved. Sampler, policies, prefixes, UIDs and manifest unchanged.'})
        env = os.environ.copy()
        env.update(OVERNIGHT_LANE=lane, OVERNIGHT_MANIFEST=str(manifest_path),
                   OVERNIGHT_GENERATION_PRICE=str(d.DOC / f'OVERNIGHT_DISCOVERY_{lane}_PRICE_v1.json'),
                   OVERNIGHT_GENERATION_OUT=str(d.RUNS / ('overnight-routing-v1-' + manifest['sha256'][:16])),
                   OVERNIGHT_MEASUREMENT_OUT=str(measurement),
                   OVERNIGHT_READER_GPU_HOUR_CEILING=str(envelope['semantic_measurement']['measurement_gpu_hour_ceiling']))
        scripts = d.REPO / 'scripts/experimental_resume'
        wrappers = [scripts / 'build_price_overnight_semantics_v1.sbatch', scripts / 'dispatch_overnight_readers_v1.sbatch']
        binding = {'manifest_sha256': manifest['sha256'], 'generation_job': job,
                   'envelope_sha256': envelope['sha256'],
                   'wrapper_hashes': {str(p): d.base.file_sha(p) for p in wrappers}}
        price = d.submit(directory, 'frame-price-recovery-v1', [f'--dependency=afterok:{job}',
                         f'--job-name=st-overnight-{lane}-price-r1', str(wrappers[0])], env, binding)
        dispatcher = d.submit(directory, 'reader-dispatch-recovery-v1', [f'--dependency=afterok:{price}',
                              f'--job-name=st-overnight-{lane}-dispatch-r1', str(wrappers[1])], env, binding)
        records.append({'lane': lane, 'generation_job': job, 'frame_price_job': price, 'dispatch_job': dispatcher})
    d.save(d.DOC / 'OVERNIGHT_AB_RECOVERY_CHAIN_v1.json', {'schema': 'overnight-ab-recovery-chain-v1', 'records': records})


if __name__ == '__main__':
    main()
