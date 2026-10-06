"""Credential-safe, duplicate-resistant receipt for a bounded CPU wake signal."""
import argparse
from pathlib import Path
import re

import native_finalization_correction_v4 as C
C.activate()
import native_finalization_native_dispatch_v6 as D
N = D.N


def main(phase, job):
    D.login_guard()
    root = N.DOC / 'submissions'; attempt_path = root / 'health-start.attempt.json'
    if phase == 'prepare':
        N.require(not attempt_path.exists(), 'start signal attempt already exists; reconcile without duplicate submission')
        chain = N.U.sealed(root / 'GENERATION_CHAIN.json')
        N.require(chain['generation_job'] == job, 'watched job differs from submitted generation')
        manifest = N.U.sealed(N.DOC / 'MANIFEST.json'); N.validate(manifest)
        budget = D.preflight(2. / 60.)
        paths = [Path(__file__), Path(__file__).with_name('native_finalization_start_signal_v1.py'),
            Path(__file__).with_name('native_finalization_start_signal_v1.sbatch')]
        N.save(attempt_path, {'schema': 'native-finalization-health-submission-attempt-v1',
            'watched_job_id': job, 'manifest_sha256': manifest['sha256'],
            'budget_sha256': budget['sha256'], 'dependency': 'after:' + job,
            'cpu_envelope': {'partition': 'lrd_all_viz', 'qos': 'normal', 'cpus': 2, 'memory_GiB': 32, 'wall_seconds': 60},
            'sources': {str(p): N.U.file_sha(p) for p in paths}, 'export': 'NIL',
            'scope': 'User-requested one-shot health notification only; no scientific output or additional GPU allocation.'})
        print('START_SIGNAL_READY', flush=True)
        return
    N.require(re.fullmatch(r'[0-9]+', job) is not None, 'invalid submitted signal job id')
    attempt = N.U.sealed(attempt_path)
    N.require(all(N.U.file_sha(Path(p)) == h for p, h in attempt['sources'].items()), 'signal source changed')
    raw = D.Ops.command(['scontrol', 'show', 'job', '-o', job])
    for fragment in ('JobId=' + job + ' ', 'UserId=lmolfett(', 'Account=iscrc_miosr ',
        'Partition=lrd_all_viz ', 'QOS=normal ', 'NumCPUs=2 ', 'TimeLimit=00:01:00 '):
        N.require(fragment in raw, 'signal submission differs: ' + fragment)
    N.save(root / 'health-start.receipt.json', {'schema': 'overnight-submit-receipt-v1',
        'job_id': job, 'name': 'health-start', 'attempt_sha256': attempt['sha256'],
        'binding': attempt, 'verified_scontrol': raw, 'no_new_GPU_allocation': True})
    print('VERIFIED_START_SIGNAL', job, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('prepare', 'register')); p.add_argument('job_id')
    a = p.parse_args(); main(a.phase, a.job_id)
