"""Reviewable CPU-prepared qualification recovery proposal; never submits a job."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def run():
    old = json.loads((S/'runs/ordered-qualification-v1/PREPARED.retry.json').read_text())
    fixtures_path = S/'runs/ordered-qualification-v1/CPU_PREFIXES.json'
    fixtures = json.loads(fixtures_path.read_text())
    if fixtures['sha256'] != digest({k: v for k, v in fixtures.items() if k != 'sha256'}):
        raise ValueError('CPU fixture seal differs')
    if fixtures['maximum_requests'] != 17 or fixtures['maximum_decode_tokens'] != 16512:
        raise ValueError('fixed qualification workload differs')
    source = REPO/'scripts/experimental_resume/qualify_ordered.py'
    driver_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = S/'addenda/ordered'/driver_sha[:16]
    directory.mkdir(exist_ok=True)
    driver = directory/source.name
    if driver.exists() and driver.read_bytes() != source.read_bytes():
        raise ValueError('immutable recovery driver differs')
    if not driver.exists():
        driver.write_bytes(source.read_bytes())
        driver.chmod(0o400)
    files = {p: sha for p, sha in old['files'].items() if p != old['driver']}
    files.update({str(driver): driver_sha, str(fixtures_path): hashlib.sha256(fixtures_path.read_bytes()).hexdigest()})
    for path, expected in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen worker or prepared fixture changed: '+path)
    audit = json.loads((REPO/'report/experimental-resume-v1/JOB_AUDIT.json').read_text())
    jobs = [j for j in audit['jobs'] if j['task'].startswith('ordered-qualify')]
    if not all(j['final'] for j in jobs):
        raise ValueError('earlier qualification allocations must be final')
    spent = sum(j['actual_resource_hours'] for j in jobs)
    reserved = (19*60+196)*2/3600
    body = {'schema': 'ordered-qualification-CPU-prepared-recovery-proposal-v1',
        'recorded_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'driver': str(driver), 'overlay': old['overlay'], 'files': files,
        'fixtures': str(fixtures_path), 'fixture_seal': fixtures['sha256'],
        'maximum_requests': 17, 'maximum_decode_tokens': 16512, 'prefill_tokens': fixtures['prefill_tokens'],
        'criteria_change': False, 'worker_change': False, 'selection_change': False,
        'CPU_prepare_job_id': fixtures['job_id'], 'GPU_job_id': None,
        'budget': {'stage_GPU_hour_ceiling_current': .75, 'stage_GPU_hours_already_spent': spent,
            'requested_additional_GPU_hours': .15, 'stage_GPU_hour_ceiling_proposed': .9,
            'GPU_count': 2, 'wall_limit_minutes': 19, 'observed_shutdown_allowance_seconds': 196,
            'maximum_new_allocation_including_shutdown_GPUh': reserved,
            'maximum_stage_total_GPUh': spent+reserved,
            'cold_load_and_fixed_batches_driver_allowance_seconds': 950,
            'deadline_margin_seconds': 30,
            'all_preparation_prefill_decoding_recovery_and_failures_charged': True,
            'no_allocation_transfer': True, 'no_further_retry_authorized': True},
        'status': 'PREPARED_REQUIRES_USER_APPROVAL_FOR_STAGE_INCREASE',
        'does_not_qualify': ['semantic transition detector', 'discovery behavioral eligibility', 'original-prompt controller', 'complete discovery/mechanism/utility prices'],
        'prior_receipt': str(S/'runs/ordered-qualification-v1/PREPARED.retry.json')}
    if spent+reserved > .9:
        raise ValueError('complete fixed allocation exceeds the proposed stage ceiling')
    body['sha256'] = digest(body)
    out = S/'runs/ordered-qualification-v1/PREPARED.cpu-recovery.json'
    if out.exists():
        previous = json.loads(out.read_text())
        if previous['files'] != files:
            raise ValueError('different CPU-prepared recovery proposal exists')
        body = previous
    else:
        out.write_text(json.dumps(body, indent=1)+'\n')
        out.chmod(0o400)
    directory.chmod(0o500)
    repo_out = REPO/'report/experimental-resume-v1/QUALIFICATION_RECOVERY_PROPOSAL.json'
    repo_out.write_text(json.dumps({'receipt': str(out), 'sha256': body['sha256'], 'budget': body['budget'],
        'status': body['status'], 'does_not_qualify': body['does_not_qualify']}, indent=1)+'\n')
    print(json.dumps({'receipt': str(out), 'sha256': body['sha256'], 'budget': body['budget'], 'status': body['status']}))


if __name__ == '__main__':
    run()
