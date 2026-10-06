"""CPU preparation and complete measurement reserve for the frozen A/B lanes."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import overnight_routing_runner_v1 as run


def measurement_reserve(requests):
    prior = run.base.sealed(run.DOC / 'MECHANISM_START_READER_PRICE_v2.json')
    load = prior['components_seconds']['two_cold_loads'] / 2
    shutdown = prior['components_seconds']['two_shutdowns'] / 2
    repeat = prior.get('repeat_work_factor', 1.25)
    maximum_prompt = 49152 - 1024
    per_row = repeat * 2 * (maximum_prompt / prior['bounded_prefill_tps'] +
                             1024 / prior['bounded_decode_tps'])
    usable = 7200 - load - shutdown - 900
    rows_per_shard = 8 * math.floor(usable / (8 * per_row))
    if rows_per_shard < 8:
        raise ValueError('cannot fit a complete two-reader batch')
    jobs = math.ceil(requests / rows_per_shard)
    loads = jobs + max(1, math.ceil(jobs * (repeat - 1)))
    gpu_hours = 2 * (requests * per_row + loads * (load + shutdown)) / 3600
    return {'rows': requests, 'ratings': 2 * requests, 'max_prompt_tokens_each': maximum_prompt,
            'rating_output_cap': 1024, 'jobs_upper_bound': jobs,
            'max_wall_seconds': 7200, 'prior_price_sha256': prior['sha256'],
            'measurement_gpu_hour_ceiling': math.ceil(gpu_hours * 100) / 100,
            'scope': 'Upper bound using full49152-token reader context; exact generated-frame price required before rating. Two readers,25percent repeat work,recovery loads and shutdown included.'}


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('preparation requires a CPU Slurm step')
    records = []
    for lane, wall, rows in [('A', 8100, 2), ('B', 7200, 2)]:
        design = run.DOC / f'OVERNIGHT_DISCOVERY_{lane}_DESIGN_v1.json'
        manifest_path = run.DOC / f'OVERNIGHT_DISCOVERY_{lane}_MANIFEST_v1.json'
        price_path = run.DOC / f'OVERNIGHT_DISCOVERY_{lane}_PRICE_v1.json'
        manifest, price = run.prepare(design, manifest_path, price_path, rows, wall)
        subprocess.run([sys.executable, '-B', str(Path(run.__file__).resolve()),
                        '--manifest', str(manifest_path), '--cpu-preflight'], check=True)
        reserve = measurement_reserve(manifest['expected_requests'])
        body = {'schema': 'overnight-complete-resource-envelope-v1', 'lane': lane,
                'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
                'generation_gpu_hours': price['estimated_complete_gpu_hours'],
                'semantic_measurement': reserve,
                'total_generation_plus_semantic_gpu_hours': price['estimated_complete_gpu_hours'] + reserve['measurement_gpu_hour_ceiling'],
                'authorization': 'RESOURCE_AUTHORIZATION_2026-10-02_v2.json; user approved parallel implementation2026-10-04',
                'dense_labels': 'Separately priced secondary stage; no unpriced GPU labeling submission.',
                'status': 'PASS_COMPLETE_GENERATION_AND_SEMANTIC_ENVELOPE'}
        envelope = run.write_once(run.DOC / f'OVERNIGHT_DISCOVERY_{lane}_ENVELOPE_v1.json', body)
        records.append({'lane': lane, 'manifest_sha256': manifest['sha256'], 'envelope_sha256': envelope['sha256'],
                        'shards': len(manifest['shards']), 'requests': manifest['expected_requests'],
                        'complete_gpu_hours': body['total_generation_plus_semantic_gpu_hours']})
    print(json.dumps({'status': 'PASS_AB_CPU_PREFLIGHT', 'records': records}), flush=True)


if __name__ == '__main__':
    main()
