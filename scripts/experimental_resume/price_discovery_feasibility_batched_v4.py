"""Price the complete 21-family homogeneous sentinel discovery feasibility stage."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_discovery_feasibility_v2 as discovery

REPORT = discovery.REPORT
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
NEIGHBOR_RUN = ROOT / 'runs/routing-control-v1/micro-neighbor-qual-e0c498b65dce4dc7'
OUT = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_BATCHED_PRICE_v4.json'


def main():
    amendment = base.sealed(discovery.AMENDMENT)
    neighbor = base.sealed(NEIGHBOR_RUN / 'SUMMARY.json')
    neighbor_audit = base.sealed(REPORT / 'CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json')
    if (neighbor_audit['summary_sha256'] != neighbor['sha256'] or
            not neighbor_audit['accepted_for_limited_batched_behavioral_discovery']):
        raise ValueError('accepted neighbor qualification differs from measured generation')
    batches = [base.sealed(NEIGHBOR_RUN / f'batch-{i:03d}.json') for i in range(10)]
    if (neighbor['requests'] != 80 or neighbor['batches'] != 10 or
            any(len(x['outputs']) != 8 or any(y['error'] for y in x['outputs'])
                for x in batches)):
        raise ValueError('measured eight-request neighbor source incomplete')
    counts = discovery.workload_counts({'stage': 'discovery_feasibility21',
                                        'rows': amendment['rows']})
    if (counts['expected_requests'] != 480 or counts['expected_batches'] != 60 or
            counts['maximum_decode_tokens'] != 122880):
        raise ValueError('complete padded feasibility workload changed')
    batch_seconds = [x['elapsed_seconds'] for x in batches]
    batch_sum = sum(batch_seconds)
    observed_cold_and_setup = neighbor['elapsed_driver_seconds'] - batch_sum
    shutdown = 228.
    repeat = 1.25
    stress_decode = 100.
    stress_prefill = 1000.
    estimate = observed_cold_and_setup + shutdown + repeat * (
        counts['expected_prefill_tokens'] / stress_prefill +
        counts['maximum_decode_tokens'] / stress_decode)
    body = {
        'schema': 'routing-discovery-feasibility-homogeneous-batched-price-v4',
        'status': 'PRICE_ONLY_PENDING_LAYER24_QUALIFICATION',
        'amendment_sha256': amendment['sha256'],
        'new_driver_sha256': base.file_sha(discovery.__file__),
        'neighbor_generation_summary_sha256': neighbor['sha256'],
        'accepted_neighbor_audit_sha256': neighbor_audit['sha256'],
        'neighbor_batch_sha256s': [x['sha256'] for x in batches],
        'source_neighbor_job': '59203153',
        'source_neighbor_job_elapsed_seconds': 776,
        'source_neighbor_driver_elapsed_seconds': neighbor['elapsed_driver_seconds'],
        'source_neighbor_batch_seconds_range': [min(batch_seconds), max(batch_seconds)],
        'source_neighbor_cold_and_setup_seconds': observed_cold_and_setup,
        'source_neighbor_total_decode_tokens': 20480,
        'stage': 'discovery_feasibility21_local_256_token_endpoint',
        'distinct_analysis_families': 21,
        'candidate_analysis_families': 12,
        'approach_analysis_families': 9,
        'technical_padding_requests': 60,
        'analysis_request_slots': 420,
        **counts,
        'stress_aggregate_decode_tokens_per_second': stress_decode,
        'stress_prefill_tokens_per_second': stress_prefill,
        'repeat_factor': repeat,
        'shutdown_seconds': shutdown,
        'estimated_complete_wall_seconds': estimate,
        'requested_wall_seconds': 7200,
        'gpus': 2,
        'gpu_hour_ceiling': 4,
        'interpretation': 'All 480 assigned requests, model load, prefill, decode, telemetry, artifacts and shutdown are priced. Technical filler repeats support full four-family batches and are excluded from ITT semantic estimands. This is a smaller exploratory 256-token feasibility screen, not the registered 48-family discovery completion or 1,024-token mechanism validation.',
    }
    value = {**body, 'sha256': base.digest(body)}
    if OUT.exists():
        if base.sealed(OUT) != value:
            raise ValueError('existing complete-stage price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'requests': counts['expected_requests'],
                      'prefill': counts['expected_prefill_tokens'],
                      'decode': counts['maximum_decode_tokens'],
                      'estimated_wall_seconds': estimate,
                      'ceiling_gpu_hours': 4}))


if __name__ == '__main__':
    main()
