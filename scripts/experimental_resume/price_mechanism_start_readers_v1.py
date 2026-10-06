"""Exact-token, complete-stage price for the frozen mechanism Qwen3.8 start audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from prepare_mechanism_start_frame_v1 import FRAME, REPO, SELECTION, digest, file_sha, sealed, write_once
import rate_transition_v22_fullprefix_starts_v2 as prior_reader
from price_transition_ratings_v3 import count_prompt_tokens

OUT = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v1.json'
PRIOR_PRICE = REPO / 'report/experimental-resume-v1/TRANSITION_V22_FULL_PREFIX_START_RATING_PRICE_v2.json'
PRIOR_RUN = (Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24') /
             'steering-v1/runs/routing-control-v1/dense-discovery/ratings-v22-fullprefix-v2-6c10499b-1ef8863f')
MAX_TOKENS = 1024
MAX_MODEL_LEN = 49152
RATING_DRIVER = REPO / 'scripts/experimental_resume/rate_mechanism_start_readers_v1.py'


def price(frame, selection, prior_price, prior_summary, prior_binding, lengths):
    if (frame['schema'] != 'mechanism-start-frame-v1' or
            frame['selection_sha256'] != selection['sha256'] or
            frame['families_in_frozen_pool'] != 128 or
            frame['rows'] != len(selection['records']) or
            prior_price['schema'] != 'transition-v22-fullprefix-start-rating-price-v2' or
            prior_summary['binding_sha256'] != prior_binding['sha256'] or
            prior_binding['frame_sha256'] != prior_price['frame_sha256'] or
            len(lengths) != frame['rows'] or not lengths or
            max(lengths) + MAX_TOKENS > MAX_MODEL_LEN):
        raise ValueError('frame, selection, prior timing or context mismatch')
    n = len(lengths)
    prefill = 2 * sum(lengths)
    decode = 2 * n * MAX_TOKENS
    prior_timings = [t for b in prior_summary['timings'] for t in b['reader_timings']]
    prior_generated = sum(t['generated_tokens'] for t in prior_timings)
    prior_seconds = sum(t['wall_seconds'] for t in prior_timings)
    if prior_generated != prior_summary['counts']['generated_tokens'] or prior_seconds <= 0:
        raise ValueError('prior full-stage measured throughput differs')
    observed_tps = prior_generated / prior_seconds
    bounded_decode_tps = min(prior_price['bounded_decode_tokens_per_s'], .8 * observed_tps)
    bounded_prefill_tps = prior_price['bounded_prefill_tokens_per_s']
    repeat = 1.25
    # Two cold loads and shutdowns include one possible same-manifest recovery.
    loads = prior_price['components_seconds']['two_cold_loads']
    shutdowns = prior_price['components_seconds']['two_shutdowns']
    decode_seconds = decode / bounded_decode_tps * repeat
    prefill_seconds = prefill / bounded_prefill_tps * repeat
    wall = loads + shutdowns + decode_seconds + prefill_seconds
    gpu_h = 2 * wall / 3600
    return {
        'schema': 'mechanism-start-reader-price-v1',
        'status': 'PASS_COMPLETE_20_GPUH' if gpu_h <= 20 else 'HOLD_PRICE_EXCEEDS_20_GPUH',
        'frame_sha256': frame['sha256'],
        'selection_sha256': selection['sha256'],
        'rating_driver_sha256': file_sha(RATING_DRIVER),
        'message_source_sha256': file_sha(prior_reader.__file__),
        'rubric_sha256': file_sha(prior_reader.RUBRIC),
        'pricing_driver_sha256': file_sha(__file__),
        'prior_price_sha256': prior_price['sha256'],
        'prior_binding_sha256': prior_binding['sha256'],
        'prior_summary_sha256': prior_summary['sha256'],
        'rows': n, 'ratings': 2 * n,
        'prompt_tokens_exact_twice': prefill,
        'prompt_tokens_min': min(lengths), 'prompt_tokens_max': max(lengths),
        'max_decode_tokens': decode, 'max_model_len': MAX_MODEL_LEN,
        'prior_observed_generated_tokens': prior_generated,
        'prior_observed_generation_seconds': prior_seconds,
        'prior_observed_effective_tps': observed_tps,
        'bounded_decode_tps': bounded_decode_tps,
        'bounded_prefill_tps': bounded_prefill_tps,
        'components_seconds': {'two_cold_loads': loads, 'decode': decode_seconds,
                               'prefill': prefill_seconds, 'two_shutdowns': shutdowns},
        'complete_stage_projected_wall_seconds': wall,
        'complete_stage_projected_GPU_h': gpu_h,
        'allocation_plan': {'first_job': '2 GPUs x 10 hours = 20 GPU-h ceiling',
                            'complete_stage_GPU_h_ceiling': 20.0,
                            'recovery': 'same-manifest resume only after fresh complete-stage reprice'},
        'interpretation': 'Exact Qwen3.8 chat-template prompt IDs for all assigned readers; 1024-token worst-case outputs, conservative prior throughput, two cold loads and shutdowns. This is a resource ceiling, not an effect estimate.'
    }


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('exact token pricing requires CPU Slurm step')
    from transformers import AutoTokenizer
    frame, selection = sealed(FRAME), sealed(SELECTION)
    prior_price = sealed(PRIOR_PRICE)
    prior_binding = sealed(PRIOR_RUN / 'BINDING.json')
    prior_summary = sealed(PRIOR_RUN / 'SUMMARY.json')
    if [r['uid'] for r in frame['records']] != [r['uid'] for r in selection['records']]:
        raise ValueError('frame has changed deterministic start order')
    tokenizer = AutoTokenizer.from_pretrained(prior_reader.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        prior_reader.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort='low')) for row in frame['records']]
    value = write_once(OUT, price(frame, selection, prior_price, prior_summary,
                                  prior_binding, lengths))
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'status': value['status'], 'rows': value['rows'],
                      'projected_GPU_h': value['complete_stage_projected_GPU_h']}), flush=True)


if __name__ == '__main__':
    main()
