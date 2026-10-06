"""Seal read-only X2 capped-manifest, NLL, and exact Slurm spend closeout."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = ROOT / 'steering-v1'
R = S / 'runs/x2-resume-v1'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
OUT = REPO / 'report/experimental-resume-v1/X2_COMPLETION_AUDIT_v1.json'
IDS = (59093506, 59095712, 59101870)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jobs():
    cmd = ['sacct', '-j', ','.join(map(str, IDS)),
           '--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES', '-P', '-X']
    rows = {}
    for line in subprocess.check_output(cmd, text=True).splitlines()[1:]:
        parts = line.split('|')
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        job_id, state, code, elapsed, tres = parts
        jid = int(job_id)
        if jid not in IDS or 'gres/gpu=2' not in tres or jid in rows:
            raise ValueError('X2 Slurm accounting differs')
        rows[jid] = {'job_id': jid, 'state': state, 'exit_code': code,
                     'elapsed_seconds': int(elapsed), 'gpus': 2,
                     'gpu_hours': int(elapsed)*2/3600}
    if set(rows) != set(IDS):
        raise ValueError('X2 Slurm job missing')
    if (rows[IDS[0]]['state'], rows[IDS[1]]['state'], rows[IDS[2]]['state']) != (
            'COMPLETED', 'FAILED', 'COMPLETED'):
        raise ValueError('X2 Slurm final states changed')
    if (rows[IDS[0]]['exit_code'], rows[IDS[1]]['exit_code'], rows[IDS[2]]['exit_code']) != (
            '0:0', '1:0', '0:0'):
        raise ValueError('X2 Slurm exit codes changed')
    return [rows[jid] for jid in IDS]


def main():
    manifest_path = R / 'x2-resume-v1.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['sha256'] != digest({k: v for k, v in manifest.items() if k != 'sha256'}):
        raise ValueError('X2 manifest seal changed')
    freeze_path = R / 'FROZEN.json'
    freeze = json.loads(freeze_path.read_text())
    if (freeze['manifest_sha256'] != manifest['sha256'] or
        freeze['manifest_file_sha256'] != sha(manifest_path)):
        raise ValueError('X2 output binding changed')
    requests = manifest['requests']
    uids = {row['uid'] for row in requests}
    arms = Counter(row['arm'] for row in requests)
    if (len(requests) != 6630 or len(uids) != 6630 or
        set(row['max_tokens'] for row in requests) != {1024} or
        dict(arms) != {'N': 78, 'E+': 1638, 'E-': 1638, 'M+': 1638, 'M-': 1638}):
        raise ValueError('X2 registered capped workload changed')
    output = Path(freeze['output'])
    status_path = output / 'shard-0.status.json'
    index_path = output / 'shard-0.index.jsonl'
    status = json.loads(status_path.read_text())
    index = [json.loads(line) for line in index_path.open()]
    if (status['manifest_sha256'] != manifest['sha256'] or status['status'] != 'complete' or
        status['n_requests'] != 6630 or status['n_done'] != 6630 or
        status['errored_this_run'] or status['aborted_in_flight'] or
        len(index) != 6630 or {x['uid'] for x in index} != uids):
        raise ValueError('X2 generation is not complete under frozen manifest')
    nll_path = next(R.glob('native-nll/*/*/native-nll.json'))
    nll = json.loads(nll_path.read_text())
    nll_uids = {row['uid'] for row in nll['records']}
    expected_nll = {row['uid'] for row in requests if row['arm'] == 'N' or row['arm'].startswith('E')}
    if (nll['binding']['manifest_sha256'] != manifest['sha256'] or
        len(nll['records']) != 3354 or nll_uids != expected_nll or
        not all(row['status'] == 'MEASURED' and row['n_tokens'] == 256 and
                math.isfinite(row['mean_nll']) for row in nll['records']) or
        not all(nll['validation'][name]['pass'] for name in ('tf1', 'tf8'))):
        raise ValueError('X2 native-NLL coverage, values, or parity changed')
    accounting = jobs()
    spent = sum(row['gpu_hours'] for row in accounting)
    if not math.isclose(spent, 5.127222222222222, rel_tol=0, abs_tol=1e-10):
        raise ValueError('X2 billed GPU-hour total changed')
    body = {'schema': 'legacy-X2-completion-audit-v1', 'status': 'COMPLETE_NO_SUBMISSION',
            'manifest_sha256': manifest['sha256'], 'manifest_file_sha256': sha(manifest_path),
            'freeze_file_sha256': sha(freeze_path),
            'generation_status_sha256': sha(status_path), 'generation_index_sha256': sha(index_path),
            'native_nll_file_sha256': sha(nll_path),
            'generation': {'assigned_requests': 6630, 'completed_requests': 6630,
                           'unique_uids': 6630, 'cap': 1024, 'arms': dict(arms),
                           'status': 'complete', 'errored': 0, 'aborted_in_flight': 0},
            'native_nll': {'assigned_N_E_uids': 3354, 'measured_uids': 3354,
                           'tokens_per_record': 256,
                           'tf1_parity_pass': True, 'tf8_parity_pass': True},
            'slurm': accounting,
            'original_ceiling_GPU_h': 8.0, 'actual_GPU_h': spent,
            'unused_ceiling_GPU_h': 8.0 - spent,
            'action': 'No X2 GPU launch: the registered generation and native-NLL workload is already complete. Preserve failed attempt and saved analysis separately.'}
    body['sha256'] = digest(body)
    if OUT.exists():
        prior = json.loads(OUT.read_text())
        if prior != body:
            raise ValueError('existing X2 closeout version differs')
    else:
        OUT.write_text(json.dumps(body, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'seal': body['sha256'],
                      'requests': 6630, 'native_nll': 3354,
                      'actual_GPU_h': spent, 'unused_GPU_h': body['unused_ceiling_GPU_h']}))


if __name__ == '__main__':
    main()
