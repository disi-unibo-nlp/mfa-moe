"""Plan or explicitly submit the additive CPU fresh-frame recovery chain."""
import argparse
import fcntl
import json
import os
import pwd
import socket
from pathlib import Path
import subprocess

import dispatch_overnight_readers_v1 as shared
import fresh_frame_recovery_v1 as recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true', help='spend the already authorized recovery resources')
    args = parser.parse_args()
    m, a = shared.base.sealed(recovery.MANIFEST), shared.base.sealed(recovery.AMENDMENT)
    recovery.validate_amendment(a, m)
    recovery.require(pwd.getpwuid(os.geteuid()).pw_name == 'lmolfett' and
                     socket.gethostname().endswith('.leonardo.local'),
                     'recovery submission is restricted to lmolfett')
    paths = a['paths']
    source = Path(paths['source'])
    original = shared.base.sealed(source / 'GENERATION_CHAIN.json')
    stage = shared.base.sealed(Path(paths['generation']) / 'STAGE_COMPLETION.json')
    recovery.require(original['generation_job'] == a['preserved_jobs']['generation'] and
                     original['frame_price_job'] == a['preserved_jobs']['failed_frame_price'] and
                     original['reader_dispatch_job'] == a['preserved_jobs']['cancelled_reader_dispatch'] and
                     stage['status'] == 'COMPLETE_UNGRADED_DISCOVERY_GENERATION' and
                     stage['manifest_sha256'] == m['sha256'] and
                     stage['counts']['assigned'] == m['expected_requests'],
                     'original generation completion or submission differs')
    env = os.environ.copy()
    env.update(OVERNIGHT_MANIFEST=str(recovery.MANIFEST), OVERNIGHT_GENERATION_PRICE=str(recovery.PRICE),
               OVERNIGHT_GENERATION_OUT=paths['generation'], OVERNIGHT_MEASUREMENT_OUT=paths['measurement'],
               OVERNIGHT_FRESH_RECOVERY_AMENDMENT_SHA256=a['sha256'],
               OVERNIGHT_READER_GPU_HOUR_CEILING=str(a['reader_gpu_hour_ceiling']))
    wrappers = [recovery.SCRIPTS / 'build_price_overnight_semantics_fresh_adapter_v1.sbatch',
                recovery.SCRIPTS / 'dispatch_fresh_frame_recovery_v1.sbatch']
    for path in wrappers:
        subprocess.run(['bash', '-n', str(path)], check=True)
    binding = {'manifest_sha256': m['sha256'], 'amendment_sha256': a['sha256'],
               'stage_completion_sha256': stage['sha256'], 'original_generation_chain_sha256': original['sha256'],
               'wrapper_hashes': {str(p): shared.base.file_sha(p) for p in wrappers}}
    print(json.dumps({'schema': 'overnight-fresh-frame-recovery-plan-v1', 'binding': binding,
                      'paths': paths, 'frame_price_arguments': ['--job-name=st-fresh-frame-price-recovery-v1', str(wrappers[0])],
                      'reader_dispatch_dependency': 'afterok:<new-frame-price-job-id>',
                      'accounting_required': original['generation_job'], 'will_submit': args.submit}, indent=2), flush=True)
    if not args.submit:
        return
    completed = subprocess.run(['sacct', '-j', original['generation_job'], '-nP',
                                '--format=JobID,State,ExitCode'], check=True, capture_output=True, text=True)
    records = [line.split('|') for line in completed.stdout.splitlines() if line.strip()]
    tasks = {row[0]: row for row in records if '.' not in row[0] and '_' in row[0]}
    expected = {original['generation_job'] + '_' + str(i) for i in range(len(m['shards']))}
    recovery.require(set(tasks) == expected and all(row[1:3] == ['COMPLETED', '0:0'] for row in tasks.values()),
                     'generation array lacks exact successful accounting')
    directory = Path(paths['recovery_submissions'])
    directory.mkdir(exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        price = shared.submit(directory, 'frame-price', ['--job-name=st-fresh-frame-price-recovery-v1',
                              str(wrappers[0])], env, binding)
        dispatch = shared.submit(directory, 'reader-dispatch', [f'--dependency=afterok:{price}',
                                 '--job-name=st-fresh-reader-dispatch-recovery-v1', str(wrappers[1])], env, binding)
        shared.save(directory / 'RECOVERY_CHAIN.json', {'schema': 'overnight-fresh-frame-recovery-chain-v1',
            'binding': binding, 'amendment_sha256': a['sha256'], 'manifest_sha256': m['sha256'],
            'frame_price_job': price, 'reader_dispatch_job': dispatch,
            'canonical_source': str(source), 'analysis_output': paths['analysis']})


if __name__ == '__main__':
    main()
