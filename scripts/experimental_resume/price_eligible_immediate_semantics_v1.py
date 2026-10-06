"""Price the complete two-reader 1,024-token audit from its sealed blind frame.

Run on CPU Slurm after generation and frame construction.  The price is bound to
the exact frame bytes, rubric, rating driver, model snapshot, and scout timing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from price_transition_ratings_v3 import count_prompt_tokens
import rate_eligible_immediate_semantics_v1 as rating


SCOUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
             'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
             'dense-discovery/ratings-v22-qwen-scout-d8f50b7e-06f8fabd/SUMMARY.json')
WALL_SECONDS = 4 * 3600
GPU_COUNT = 2


def estimate(lengths, scout):
    if not lengths or any(type(n) is not int or n < 16 for n in lengths):
        raise ValueError('no valid gradeable prompt lengths')
    scout_seconds = sum(reader['wall_seconds'] for timing in scout['timings']
                        for reader in timing['reader_timings'])
    if scout_seconds <= 0 or scout['counts']['generated_tokens'] <= 0:
        raise ValueError('invalid reader timing reference')
    observed_tps = scout['counts']['generated_tokens'] / scout_seconds
    decode_tps = min(43.71, .8 * observed_tps)
    prompt_tokens = len(rating.READERS) * sum(lengths)
    max_decode = len(rating.READERS) * len(lengths) * rating.MAX_TOKENS
    cold_load = 1.25 * sum(timing['load_seconds'] for timing in scout['timings'])
    components = {'decode': 1.25 * max_decode / decode_tps,
                  'prefill': 1.25 * prompt_tokens / 4914.,
                  'cold_load': cold_load, 'shutdown': 196.}
    return {'observed_scout_effective_generated_tokens_per_s': observed_tps,
            'priced_decode_tokens_per_s': decode_tps,
            'priced_prefill_tokens_per_s': 4914., 'repeat_work_factor': 1.25,
            'prompt_tokens_exact_twice': prompt_tokens,
            'max_decode_tokens': max_decode,
            'max_model_context_tokens': max(lengths) + rating.MAX_TOKENS,
            'prompt_tokens_per_request_min': min(lengths),
            'prompt_tokens_per_request_p90': sorted(lengths)[math.ceil(.9 * len(lengths)) - 1],
            'prompt_tokens_per_request_max': max(lengths),
            'components_seconds': components,
            'projected_complete_seconds': sum(components.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('full-frame tokenizer pricing requires a CPU Slurm step')
    if not args.frame.resolve().is_relative_to(rating.STAGE) or not args.out.resolve().is_relative_to(rating.REPO):
        raise ValueError('rating price paths must remain in owned staging/repository')
    frame, scout = rating.sealed(args.frame), rating.sealed(SCOUT)
    rows = frame.get('records', [])
    if (frame.get('schema') != 'eligible-immediate-blind-frame-v1' or
            not 1 <= len(rows) <= 156 or
            frame.get('rubric_sha256') != rating.file_sha(rating.RUBRIC) or
            frame.get('continuation_max_tokens') != 256 or
            set(frame.get('reader_input_allowlist', [])) != rating.ALLOWLIST or
            any(set(row) != {'blind_id', 'reader_input'} or
                set(row['reader_input']) != rating.ALLOWLIST for row in rows)):
        raise ValueError('blind rating frame does not match the frozen immediate rubric')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        rating.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort='low')) for row in rows]
    estimate_data = estimate(lengths, scout)
    if estimate_data['max_model_context_tokens'] > 49152:
        raise ValueError('reader prompt exceeds frozen model context')
    body = {'schema': 'eligible-immediate-blind-rating-price-v1',
            'status': ('PASS_COMPLETE_STAGE' if
                       estimate_data['projected_complete_seconds'] <= .9 * WALL_SECONDS
                       else 'HOLD_PRICE_TOO_CLOSE_TO_WALLTIME'),
            'pricing_job_id': os.environ['SLURM_JOB_ID'],
            'frame_sha256': frame['sha256'],
            'frame_file_sha256': rating.file_sha(args.frame),
            'driver_sha256': rating.file_sha(rating.__file__),
            'price_driver_sha256': rating.file_sha(__file__),
            'rubric_sha256': rating.file_sha(rating.RUBRIC),
            'scout_sha256': scout['sha256'],
            'model_snapshot': str(rating.MODEL),
            'assigned_requests': 156,
            'gradeable_requests': len(rows),
            'ratings': len(rows) * len(rating.READERS),
            'gpus': GPU_COUNT,
            'requested_wall_seconds': WALL_SECONDS,
            'ceiling_gpu_hours': GPU_COUNT * WALL_SECONDS / 3600,
            **estimate_data}
    price = rating.write_once(args.out, body)
    if body['status'] == 'PASS_COMPLETE_STAGE':
        rating.validate(frame, price, args.frame)
    print(json.dumps({'status': body['status'], 'price': str(args.out),
                      'sha256': price['sha256'],
                      'projected_seconds': estimate_data['projected_complete_seconds'],
                      'gradeable': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
