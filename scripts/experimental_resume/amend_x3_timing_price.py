"""Seal a 2-hour ceiling amendment for the CPU-prepared X3 cost-only pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1')
SOURCE = REPORT / 'X3_TIMING_PILOT_PRICE_v1.json'
OUT = REPORT / 'X3_TIMING_PILOT_PRICE_v2.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed sealed X3 pilot price or manifest')
    return value


def main():
    source = read_sealed(SOURCE)
    manifest = read_sealed(Path(source['manifest_path']))
    if (source['manifest_sha256'] != manifest['sha256'] or
        source['requests'] != 40 or source['max_decode_tokens'] != 1040440 or
        source['status'] != 'CPU_PREPARED_GPU_HOLD'):
        raise ValueError('X3 pilot workload differs from frozen CPU preparation')
    wall = sum(source['components_wall_seconds'].values())
    if abs(wall - source['projected_wall_seconds']) > 1e-8 or wall >= 7200:
        raise ValueError('pilot stage no longer fits the amended two-hour window')
    body = {'schema': 'x3-32k-timing-pilot-price-v2',
            'status': 'PROPOSED_GPU_HOLD_NEW_STUDY_PRIORITY',
            'source_price_sha256': source['sha256'],
            'manifest_sha256': manifest['sha256'],
            'manifest_path': source['manifest_path'],
            'scope': '40 capped requests on four frozen dev-disc long-prefix families; cost-only measurement; no X3 causal estimates',
            'max_decode_tokens': source['max_decode_tokens'],
            'prefill_tokens_without_cache_reuse': source['prefill_tokens_without_cache_reuse'],
            'components_wall_seconds': source['components_wall_seconds'],
            'projected_wall_seconds': wall,
            'proposed_job_walltime_seconds': 7200,
            'projected_walltime_margin_seconds': 7200 - wall,
            'proposed_GPUs': 2,
            'proposed_stage_ceiling_GPUh': 4.0,
            'post_run_allowed_fields': source['measurement_outputs_allowed'],
            'post_run_embargoed_fields': source['measurement_outputs_embargoed'],
            'prerequisites': ['new-study priority resolved',
                              'frozen capped sampler job plan bound to manifest digest',
                              'Slurm account/partition/QoS and exact two-hour limit verified'],
            'full_X3': 'still HOLD; separately price 770-request generation, frozen GEPA labels, grading, any NLL, loads, retries and shutdown, then amend 12 GPU-hour ceiling if needed'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing v2 price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'ceiling_GPUh': 4.0, 'walltime_margin_seconds': 7200-wall}))


if __name__ == '__main__':
    main()
