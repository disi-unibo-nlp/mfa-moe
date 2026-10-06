"""One-shot, event-driven health signals within an existing Slurm allocation.

Lustre notifications are observed on the writing node, never presumed to
propagate to a login-node watcher. No scheduler polling or new allocation.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import time

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1/native-finalization-v1'
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/native-finalization-v1')
MASK = 0x00000008 | 0x00000080 | 0x00000100 | 0x00000400 | 0x00000800


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('invalid health input seal: ' + str(path))
    return value


def status(phase, job):
    allocation = ROOT / 'allocations' / job
    qualification = allocation / 'QUALIFICATION.json'
    cost = allocation / 'COST.json'
    if phase == 'qualification':
        if qualification.exists():
            proof = sealed(qualification)
            return {'signal': proof['status'], 'qualification_sha256': proof['sha256']}
        if cost.exists():
            return {'signal': 'ENGINEERING_ENDED_WITHOUT_QUALIFICATION', 'cost_sha256': sealed(cost)['sha256']}
        return None
    if phase in ('preparation', 'measurement'):
        path = DOC / ('PREPARED.json' if phase == 'preparation' else 'J1_READY.json')
        if path.exists():
            proof = sealed(path)
            if proof['producer_job_id'] != job:
                raise ValueError('readiness producer differs')
            return {'signal': 'PREPARED' if phase == 'preparation' else 'J1_READY', 'readiness_sha256': proof['sha256']}
        return None
    rows = [sealed(p) for p in (ROOT / 'receipts').glob('*.json')]
    errors = [r['assignment']['uid'] for r in rows if r['status'] == 'GENERATION_ERROR']
    if errors:
        return {'signal': 'GENERATION_ERROR', 'errors': errors, 'receipts': len(rows)}
    if len(rows) == 32 or cost.exists():
        return {'signal': 'GENERATION_COMPLETE' if len(rows) == 32 else 'GENERATION_ENDED_INCOMPLETE',
                'receipts': len(rows), 'cost_present': cost.exists()}
    return None


def emit(body):
    health = DOC / 'health'
    health.mkdir(exist_ok=True)
    body = {'schema': 'native-finalization-health-v1', **body,
            'observed_utc': datetime.now(timezone.utc).isoformat(), 'observer_host': socket.gethostname(),
            'manifest_sha256': sealed(DOC / 'MANIFEST.json')['sha256'],
            'no_scheduler_polling': True, 'no_new_allocation': True}
    value = {**body, 'sha256': digest(body)}
    path = health / (body['job_id'] + '-' + body['phase'] + '-' + value['sha256'][:16] + '.json')
    with path.open('x') as stream:
        json.dump(value, stream, indent=1); stream.write('\n')
    sealed(path)
    print(json.dumps({'health_receipt': str(path), **value}), flush=True)


def watch(phase, job, timeout):
    if os.environ.get('SLURM_JOB_ID') != job or socket.gethostname().startswith('login'):
        raise ValueError('observer must run in the exact existing writer allocation')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    watched = set()
    targets = (DOC,) if phase in ('preparation', 'measurement') else (
        ROOT, ROOT / 'allocations', ROOT / 'allocations' / job, ROOT / 'receipts')
    deadline = time.monotonic() + timeout
    print(json.dumps({'observer': 'ARMED', 'phase': phase, 'job_id': job,
                      'host': socket.gethostname(), 'timeout_seconds': timeout}), flush=True)
    try:
        while True:
            # Add watches before the snapshot to avoid missed local writer events.
            for path in targets:
                if path not in watched and path.is_dir():
                    if libc.inotify_add_watch(fd, os.fsencode(path), MASK) < 0:
                        raise OSError(ctypes.get_errno(), 'inotify_add_watch: ' + str(path))
                    watched.add(path)
            result = status(phase, job)
            if result:
                emit({'job_id': job, 'phase': phase, **result}); return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                emit({'job_id': job, 'phase': phase, 'signal': 'HEALTH_DEADLINE',
                      'timeout_seconds': timeout, 'action': 'Inspect exact job and receipts once; no automatic retry.'})
                return
            ready, _, _ = select.select([fd], [], [], remaining)
            if ready:
                os.read(fd, 65536)
            # The next snapshot is driven by an event or the single deadline.
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preparation', 'qualification', 'generation', 'measurement'), required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    parser.add_argument('--doc', type=Path, default=DOC)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    for path in (args.doc, args.root):
        if not str(path.resolve()).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/'):
            raise ValueError('health paths must stay inside user WORK')
    DOC, ROOT = args.doc, args.root
    watch(args.phase, args.job_id, args.timeout_seconds)
