"""Price the complete two-reader discovery transition audit before GPU launch."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FIXTURE = R / 'steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_AUDIT_CANDIDATES.json'
PARITY = R / 'steering-v1/runs/routing-control-v1/dense-judge-parity/results-8e939bee-2341bac1/PARITY.json'
OUT = REPO / 'report/experimental-resume-v1/TRANSITION_RATING_COMPLETE_PRICE.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('transition rating input seal differs: ' + str(path))
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('semantic rating prompt pricing requires CPU Slurm')
    import sys
    sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
    import rate_transition_candidates as rating
    from transformers import AutoTokenizer

    fixture = sealed(FIXTURE)
    parity = sealed(PARITY)
    if fixture['schema'] != 'transition-audit-candidates-unlabeled-v1' or fixture['families'] != 48:
        raise ValueError('unexpected discovery frame')
    if parity['job_id'] != '59112590' or parity['coverage'] < .99:
        raise ValueError('direct judge throughput source not qualified')
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = []
    for row in fixture['records']:
        tokens = tokenizer.apply_chat_template(rating.messages(row), tokenize=True,
                                               add_generation_prompt=True,
                                               enable_thinking=True, reasoning_effort='low')
        lengths.append(len(tokens))
    if not lengths or max(lengths) + rating.MAX_TOKENS > 49152:
        raise ValueError('rating prompt exceeds configured judge context')
    ratings = 2 * len(lengths)
    prefill = 2 * sum(lengths)
    decode = ratings * rating.MAX_TOKENS
    observed_decode_tps = parity['total_generated_tokens'] / parity['generation_seconds']
    pessimistic_decode_tps = observed_decode_tps * .65
    pessimistic_prefill_tps = 19656.064021098042 * .5
    repeat_work = 1.25
    decode_seconds = decode / pessimistic_decode_tps * repeat_work
    prefill_seconds = prefill / pessimistic_prefill_tps * repeat_work
    load_seconds = 2 * parity['load_seconds'] * 1.25
    shutdown_seconds = 2 * 196
    complete_GPU_h = 2 * (decode_seconds + prefill_seconds + load_seconds + shutdown_seconds) / 3600
    # A three-hour first slice and 75-minute same-manifest recovery retain
    # a bounded complete-stage capacity, including both cold loads.
    reservation_ceiling = 2 * (3 + 1.25)
    status = 'PASS_COMPLETE_STAGE' if complete_GPU_h <= reservation_ceiling else 'HOLD_STAGE_PRICE_EXCEEDS_CAPACITY'
    body = {'schema': 'transition-two-reader-complete-price-v1',
            'status': status, 'job_id': os.environ['SLURM_JOB_ID'],
            'fixture_sha256': fixture['sha256'],
            'rating_driver_sha256': hashlib.sha256(Path(rating.__file__).read_bytes()).hexdigest(),
            'rubric_sha256': hashlib.sha256(rating.RUBRIC.read_bytes()).hexdigest(),
            'parity_sha256': parity['sha256'], 'rows': len(lengths), 'ratings': ratings,
            'prompt_tokens_exact_twice': prefill,
            'prompt_tokens_min': min(lengths), 'prompt_tokens_max': max(lengths),
            'max_decode_tokens': decode,
            'observed_parity_decode_tokens_per_s': observed_decode_tps,
            'pessimistic_decode_tokens_per_s': pessimistic_decode_tps,
            'pessimistic_prefill_tokens_per_s': pessimistic_prefill_tps,
            'repeat_work_factor': repeat_work,
            'components_seconds': {'decode': decode_seconds, 'prefill': prefill_seconds,
                                   'two_cold_loads': load_seconds, 'two_shutdowns': shutdown_seconds},
            'complete_stage_projected_GPU_h': complete_GPU_h,
            'allocation_plan': {'first_slice_GPU_h_max': 6.,
                                'one_recovery_GPU_h_max': 2.5,
                                'combined_GPU_h_ceiling': reservation_ceiling},
            'previous_discovery_GPU_h_line': 2.,
            'additional_capacity_under_current_user_authorization_GPU_h': reservation_ceiling - 2.,
            'interpretation': 'Two same-model independent arm-blind LLM draws, discovery-only; no human truth or causal effect'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing transition rating price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'status': status, 'rows': len(lengths),
                      'complete_projected_GPU_h': complete_GPU_h,
                      'reservation_ceiling_GPU_h': reservation_ceiling}), flush=True)


if __name__ == '__main__':
    main()
