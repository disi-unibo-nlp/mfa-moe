"""Bounded native readback of immutable diagnostic artifacts and allocation costs."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1/native-finalization-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('invalid seal: ' + str(path))
    return value


def write(path, body):
    value = {**body, 'sha256': digest(body)}
    with path.open('x') as stream:
        json.dump(value, stream, indent=1, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    if sealed(path) != value:
        raise ValueError('readback failed')
    return value


def main():
    manifest = sealed(DOC / 'MANIFEST.json'); audit = sealed(DOC / 'SAVED_AUDIT.json')
    for path, expected in manifest['binding']['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('sealed source changed: ' + path)
    tests = sealed(DOC / 'TESTS.json'); chain = sealed(DOC / 'submissions/GENERATION_CHAIN.json')
    if chain['manifest_sha256'] != manifest['sha256']:
        raise ValueError('submitted manifest differs')
    jobs = {}
    for path in (DOC / 'submissions').glob('*.json'):
        r = sealed(path)
        if r.get('schema') == 'overnight-submit-receipt-v1':
            jobs[r['job_id']] = path.stem
    records = []
    for job, name in sorted(jobs.items()):
        raw = subprocess.run(['sacct', '-X', '-nP', '-j', job,
            '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'], check=True,
            capture_output=True, text=True).stdout
        rows = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == job]
        if len(rows) != 1:
            raise ValueError('missing exact allocation: ' + job)
        r = rows[0]; tres = dict(s.split('=', 1) for s in r[4].split(',') if '=' in s)
        elapsed = int(r[3])
        records.append({'job_id': job, 'name': name, 'state': r[1], 'exit_code': r[2],
            'elapsed_seconds': elapsed, 'allocation': tres,
            'billing_core_hours': float(tres.get('billing', 0)) * elapsed / 3600,
            'GPU_hours': float(tres.get('gres/gpu', 0)) * elapsed / 3600,
            'provisional': r[1] in ('RUNNING', 'PENDING', 'COMPLETING'), 'sacct': raw})
    table = DOC / 'saved-review.csv'
    if not table.exists():
        with table.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case', 'uid', 'arm', 'trigger_meaning', 'prefix_state',
                'candidate_region', 'candidate_character_offset', 'reasoning_tokens', 'generated_tokens'])
            for r in audit['rows']:
                c = r['first_complete_correct_candidate']
                writer.writerow([r['case'], r['uid'], r['arm'], r['trigger_meaning'],
                    r['preceding_prefix_state'], c['region'], c['character_offset'],
                    r['reasoning_tokens'], r['generated_tokens']])
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    checkpoint = write(DOC / ('CHECKPOINT-' + now + '.json'), {
        'schema': 'native-finalization-checkpoint-v1', 'observed_utc': now,
        'manifest_sha256': manifest['sha256'], 'sources_verified': len(manifest['binding']['sources']),
        'saved_audit_sha256': audit['sha256'], 'tests_sha256': tests['sha256'],
        'generation_chain_sha256': chain['sha256'], 'assignments': len(manifest['rows']),
        'allocations': records, 'observed_billing_core_hours': sum(r['billing_core_hours'] for r in records),
        'observed_GPU_hours': sum(r['GPU_hours'] for r in records),
        'saved_review_csv_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
        'analysis_status': 'PENDING_GENERATION_AND_OFFLINE_GRADING',
        'future_artifacts': ['analysis/assignments.csv', 'analysis/effects.csv',
            'analysis/accuracy-versus-length.png', 'analysis/accuracy-versus-length.pdf', 'analysis/ANALYSIS.json'],
        'next_step': 'After measurement produces J1_READY.json, run the finite native j1 dispatch. No downstream experiment automatically submitted.',
        'claim_limit': 'No accuracy or output-length effect observed yet.'})
    pointer = REPO / 'report/experimental-resume-v1/NATIVE_FINALIZATION_DIAGNOSTIC_v1.json'
    if not pointer.exists():
        write(pointer, {'schema': 'routing-study-native-finalization-addendum-v1',
            'directory': str(DOC), 'manifest_sha256': manifest['sha256'],
            'initial_checkpoint_sha256': checkpoint['sha256'],
            'prior_study_index': str(DOC / 'STUDY_INDEX-59417845.json'),
            'saved_review_table': str(table), 'readout': str(DOC / 'README.md'),
            'status': 'GENERATION_SUBMITTED; refer to subsequent immutable checkpoints and study indices'})
    print(json.dumps({k: checkpoint[k] for k in ('observed_utc', 'sources_verified',
        'observed_billing_core_hours', 'observed_GPU_hours', 'sha256')}))


if __name__ == '__main__':
    main()
