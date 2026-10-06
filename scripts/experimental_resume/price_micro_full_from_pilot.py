"""Seal a complete 12-family GPU ceiling from an actually completed four-prefix pilot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
PILOT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
FULL = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_MANIFEST_v3.json'
RUN = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_FULL_STAGE_PRICE_v1.json'
JOB = '59180099'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    x = json.loads(path.read_text())
    if x.get('sha256') != digest({k: v for k, v in x.items() if k != 'sha256'}):
        raise ValueError('source JSON seal changed: ' + str(path))
    return x


def elapsed_seconds(text):
    fields = text.split(':')
    if len(fields) != 3:
        raise ValueError('unexpected sacct elapsed time')
    day, hour = fields[0].split('-') if '-' in fields[0] else ('0', fields[0])
    return int(day) * 86400 + int(hour) * 3600 + int(fields[1]) * 60 + int(fields[2])


def main():
    pilot, full = sealed(PILOT), sealed(FULL)
    if (pilot['stage'] != 'pilot' or full['stage'] != 'full' or
        pilot['driver_sha256'] != full['driver_sha256'] or
        full['expected_requests'] != 3 * pilot['expected_requests']):
        raise ValueError('full stage differs beyond enrollment')
    binding, summary = sealed(RUN / 'BINDING.json'), sealed(RUN / 'SUMMARY.json')
    if (binding['manifest_sha256'] != pilot['sha256'] or
        summary['manifest_sha256'] != pilot['sha256'] or
        summary['status'] != 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION' or
        summary['requests'] != pilot['expected_requests']):
        raise ValueError('same-bound pilot is not complete')
    batch = sealed(RUN / 'batch-000.json')
    if batch['manifest_sha256'] != pilot['sha256'] or len(batch['outputs']) != 48:
        raise ValueError('pilot batch receipt incomplete')
    accounting = subprocess.check_output(['sacct', '-j', JOB, '--format=JobID,State,ExitCode,Elapsed,AllocTRES',
                                          '-P', '-n'], text=True, timeout=15).splitlines()
    row = next((line.split('|') for line in accounting if line.split('|')[0] == JOB), None)
    if row is None or row[1:3] != ['COMPLETED', '0:0'] or 'gres/gpu=2' not in row[4]:
        raise ValueError('pilot final job state or allocation differs')
    elapsed = elapsed_seconds(row[3])
    driver_elapsed = float(summary['elapsed_driver_seconds'])
    warm = float(batch['elapsed_seconds'])
    if not 0 < warm < driver_elapsed <= elapsed + 10:
        raise ValueError('pilot timing components differ')
    fixed = driver_elapsed - warm
    outside = max(0., elapsed - driver_elapsed)
    prefill_ratio = full['expected_prefill_tokens'] / pilot['expected_prefill_tokens']
    decode_ratio = full['maximum_decode_tokens'] / pilot['maximum_decode_tokens']
    warm_multiplier = max(prefill_ratio, decode_ratio)
    # One model load; pessimistic 25% fixed, 50% warm/shutdown, plus 120 s.
    projected = fixed * 1.25 + warm * warm_multiplier * 1.5 + outside * 1.5 + 120.
    one_job_minutes = math.ceil(projected / 300) * 5
    body = {'schema': 'routing-micro-screen-full-stage-price-v1',
            'status': 'PRICE_ONLY_PENDING_FIRST_STAGE_AND_ENGINE_GATES',
            'pilot_job_id': JOB, 'pilot_job_state': row[1], 'pilot_exit_code': row[2],
            'pilot_sacct_elapsed_seconds': elapsed, 'pilot_allocated_TRES': row[4],
            'pilot_manifest_sha256': pilot['sha256'], 'full_manifest_sha256': full['sha256'],
            'pilot_generation_binding_sha256': binding['sha256'],
            'pilot_summary_sha256': summary['sha256'], 'pilot_batch_sha256': batch['sha256'],
            'pilot_driver_seconds': driver_elapsed, 'pilot_batch_seconds': warm,
            'pilot_fixed_in_driver_seconds': fixed,
            'pilot_outside_driver_seconds_including_shutdown': outside,
            'full_request_count': full['expected_requests'],
            'full_maximum_prefill_tokens': full['expected_prefill_tokens'],
            'full_maximum_decode_tokens': full['maximum_decode_tokens'],
            'full_maximum_context_tokens': full['maximum_context_tokens'],
            'prefill_ratio_full_over_pilot': prefill_ratio,
            'maximum_decode_ratio_full_over_pilot': decode_ratio,
            'pessimistic_warm_scale': warm_multiplier * 1.5,
            'projected_complete_seconds_including_shutdown_and_margin': projected,
            'proposed_single_job_wall_minutes': one_job_minutes,
            'proposed_two_A100_ceiling_GPU_hours': 2 * one_job_minutes / 60,
            'pricing_rule': '1.25x pilot fixed setup + 1.5x larger token ratio times pilot warm batch + 1.5x outside-driver including shutdown + 120 s, rounded up to 5 min',
            'caveat': 'two-A100 full stage must be repriced if pilot output, code, batching, or resource shape changes; no same-prefix semantic or utility claim'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing stage price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'projected_seconds': projected,
                      'wall_minutes': one_job_minutes,
                      'GPU_hour_ceiling': body['proposed_two_A100_ceiling_GPU_hours']}))


if __name__ == '__main__':
    main()
