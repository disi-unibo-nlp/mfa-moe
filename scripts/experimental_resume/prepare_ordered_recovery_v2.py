"""Freeze a reviewable one-shot qualification recovery proposal; never submit."""
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
    prior_path = S/'runs/ordered-qualification-v1/PREPARED.cpu-recovery.json'
    prior = json.loads(prior_path.read_text())
    old_result = json.loads((S/'runs/ordered-qualification-v1/results-cpu-recovery/QUALIFICATION.json').read_text())
    if str(old_result['job_id']) != '59103206' or old_result['pass'] or '880s left, 950s needed' not in old_result['error']:
        raise ValueError('the fixed previous failure has different evidence')
    source = REPO/'scripts/experimental_resume/qualify_ordered.py'
    driver_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = S/'addenda/ordered'/driver_sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory/source.name
    if driver.exists() and driver.read_bytes() != source.read_bytes():
        raise ValueError('immutable qualification recovery driver changed')
    if not driver.exists():
        driver.write_bytes(source.read_bytes())
        driver.chmod(0o400)
    files = {path: sha for path, sha in prior['files'].items() if path != prior['driver']}
    files[str(driver)] = driver_sha
    for path, expected in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen worker/fixture changed: ' + path)
    audit = json.loads((REPO/'report/experimental-resume-v1/JOB_AUDIT.json').read_text())
    jobs = [j for j in audit['jobs'] if j['task'].startswith('ordered-qualify')]
    if len(jobs) != 3 or not all(j['final'] for j in jobs):
        raise ValueError('all three qualification attempts must have settled')
    spent = sum(j['actual_resource_hours'] for j in jobs)
    reserved = (20*60+196)*2/3600
    ceiling = 1.10
    if spent+reserved > ceiling:
        raise ValueError('complete fixed recovery price exceeds proposed stage ceiling')
    body = {'schema': 'ordered-qualification-recovery-proposal-v2',
        'recorded_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prior_binding': str(prior_path), 'prior_binding_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        'prior_failed_job': '59103206', 'prior_failure': old_result['error'],
        'driver': str(driver), 'driver_sha256': driver_sha, 'overlay': prior['overlay'],
        'files': files, 'fixtures': prior['fixtures'], 'fixture_seal': prior['fixture_seal'],
        'maximum_requests': 17, 'maximum_decode_tokens': 16512, 'prefill_tokens': 37472,
        'criteria_change': False, 'worker_change': False, 'selection_change': False,
        'code_change': 'move unchanged 950-second allowance before fingerprint/import; record setup and fingerprint timings',
        'budget': {'current_stage_ceiling_GPUh': .90, 'verified_stage_spent_GPUh': spent,
            'requested_additional_GPUh': .20, 'proposed_stage_ceiling_GPUh': ceiling,
            'GPU_count': 2, 'wall_limit_minutes': 20, 'observed_shutdown_allowance_seconds': 196,
            'maximum_new_allocation_including_shutdown_GPUh': reserved,
            'maximum_stage_total_GPUh': spent+reserved,
            'proposed_contingency_GPUh': ceiling-spent-reserved,
            'cold_import_load_and_fixed_batches_allowance_seconds': 950,
            'all_preparation_prefill_decoding_recovery_failure_and_teardown_charged': True,
            'no_allocation_transfer': True, 'no_further_retry_authorized': True},
        'status': 'PREPARED_AWAITING_NEW_USER_AUTHORIZATION',
        'does_not_qualify': ['semantic transition detector', 'behavioral enrollment',
            'original-prompt controller', 'complete discovery/mechanism/utility stage prices']}
    body['sha256'] = digest(body)
    destination = S/'runs/ordered-qualification-v1/PREPARED.cpu-recovery-v2.json'
    if destination.exists() and json.loads(destination.read_text()) != body:
        raise ValueError('different v2 recovery proposal already frozen')
    if not destination.exists():
        destination.write_text(json.dumps(body, indent=1)+'\n')
        destination.chmod(0o400)
    directory.chmod(0o500)
    public = {key: body[key] for key in ('schema','recorded_UTC','prior_binding','prior_failed_job',
        'prior_failure','driver','driver_sha256','fixtures','fixture_seal','maximum_requests',
        'maximum_decode_tokens','prefill_tokens','criteria_change','worker_change',
        'selection_change','code_change','budget','status','does_not_qualify','sha256')}
    public['private_full_binding'] = str(destination)
    report = REPO/'report/experimental-resume-v1/QUALIFICATION_RECOVERY_PROPOSAL_v2.json'
    if report.exists() and json.loads(report.read_text()) != public:
        raise ValueError('different public v2 proposal already exists')
    if not report.exists():
        report.write_text(json.dumps(public, indent=1)+'\n')
    print(json.dumps({'proposal': str(report), 'binding': str(destination),
                      'stage_total_GPUh': spent+reserved, 'additional_GPUh': .20,
                      'status': body['status']}))


if __name__ == '__main__':
    run()
