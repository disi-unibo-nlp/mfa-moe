"""Bounded orchestration snapshot of parallel routing jobs, including failures."""
from __future__ import annotations
from collections import Counter
import datetime
import json
from pathlib import Path
import re
import subprocess

import dispatch_overnight_readers_v1 as shared

ROOT_JOBS = ('59342858', '59342983', '59343015', '59344064', '59344077',
             '59344192', '59344202', '59344223', '59344179', '59344221',
             '59344222', '59344411', '59344413', '59344415', '59344417',
             '59344807', '59344808', '59344841', '59345112', '59345113')


def identifiers(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in ('job_id', 'job', 'parent') or key.endswith('_job')) and re.fullmatch(r'[0-9]+', str(item)):
                yield str(item)
            yield from identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from identifiers(item)


def receipt_paths():
    paths = set()
    for pattern in ('overnight*submissions*', 'generated-dense-chain-*',
                    'mechanism-cap-recovery-submissions*', 'routing-first-stage*submissions*',
                    'utility*submissions*'):
        for directory in shared.DOC.glob(pattern):
            if directory.is_dir():
                paths.update(directory.rglob('*.json'))
    for directory in (shared.RUNS / 'generated-dense').glob('dense-generated-*/dispatch-v1'):
        paths.update(directory.glob('*.json'))
    return sorted(paths)


def main():
    host = subprocess.run(['hostname'], check=True, capture_output=True, text=True).stdout.strip()
    user = subprocess.run(['id', '-un'], check=True, capture_output=True, text=True).stdout.strip()
    if not host.endswith('.leonardo.local') or user != 'lmolfett':
        raise RuntimeError('cluster identity does not match LEONARDO/lmolfett')
    jobs, receipts = set(ROOT_JOBS), {}
    for path in receipt_paths():
        value = shared.base.sealed(path)
        jobs.update(identifiers(value))
        receipts[str(path)] = value['sha256']
    fields = ('JobID', 'State', 'ExitCode', 'ElapsedRaw', 'AllocCPUS', 'AllocTRES', 'Start', 'End')
    query = ['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(jobs)), '-o', ','.join(fields)]
    raw = subprocess.run(query, check=True, capture_output=True, text=True).stdout
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = line.split('|')
        if values and values[-1] == '':
            values.pop()
        if len(values) != len(fields):
            raise ValueError('unexpected accounting row shape')
        row = dict(zip(fields, values))
        tres = dict(v.split('=', 1) for v in row['AllocTRES'].split(',') if '=' in v)
        hours = int(row['ElapsedRaw']) / 3600
        gpu_count = int(tres.get('gres/gpu', '0'))
        row.update(allocated_gpu_hours_so_far=hours * gpu_count,
                   allocated_cpu_only_hours_so_far=hours * int(row['AllocCPUS']) if not gpu_count else 0,
                   billing_unit_hours_so_far=hours * float(tres.get('billing', '0')),
                   final_state=row['State'].split()[0] in ('COMPLETED', 'FAILED', 'CANCELLED',
                                                         'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'))
        rows.append(row)
    if len({r['JobID'] for r in rows}) != len(rows):
        raise ValueError('accounting includes duplicate allocation records')
    now = datetime.datetime.now(datetime.timezone.utc)
    body = {'schema': 'parallel-routing-slurm-snapshot-v1', 'snapshot_UTC': now.isoformat(),
            'host': host, 'user': user, 'requested_job_ids': sorted(jobs),
            'receipt_sha256s': receipts, 'accounting_query': query, 'jobs': rows,
            'state_counts': dict(Counter(r['State'].split()[0] for r in rows)),
            'allocated_gpu_hours_so_far': sum(r['allocated_gpu_hours_so_far'] for r in rows),
            'allocated_cpu_only_hours_so_far': sum(r['allocated_cpu_only_hours_so_far'] for r in rows),
            'billing_unit_hours_so_far': sum(r['billing_unit_hours_so_far'] for r in rows),
            'interpretation': 'Allocation elapsed-time accounting, including preserved failures. Running jobs are provisional. '
                              'CPU-only hours exclude CPU allocations accompanying GPU jobs. Slurm billing units are not '
                              'a substitute for saldo credit balance. Completion status alone does not verify artifacts.'}
    path = shared.DOC / ('OVERNIGHT_SLURM_' + now.strftime('%Y%m%dT%H%M%SZ') + '.json')
    value = shared.save(path, body)
    print(json.dumps({'output': str(path), 'sha256': value['sha256'],
                      'states': value['state_counts'], 'gpu_hours_so_far': value['allocated_gpu_hours_so_far']}))


if __name__ == '__main__':
    main()
