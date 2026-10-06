"""Exact-token complete-stage price for the 48-row arm-blind pilot audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket

from price_transition_ratings_v3 import count_prompt_tokens
import rate_micro_blind_semantics_v2_2 as rating

OUT = rating.PRICE
SCOUT = rating.ROOT / ('steering-v1/runs/routing-control-v1/dense-discovery/'
                       'ratings-v22-qwen-scout-d8f50b7e-06f8fabd/SUMMARY.json')


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('tokenization and full-stage price require CPU Slurm')
    from transformers import AutoTokenizer

    frame, scout = rating.sealed(rating.FRAME), rating.sealed(SCOUT)
    if frame['schema'] != 'routing-micro-screen-blind-frame-v1' or len(frame['records']) != 48:
        raise ValueError('pilot blind frame changed')
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    start_lengths, target_lengths = [], []
    for _, row in rating.start_groups(frame['records']):
        ids = tokenizer.apply_chat_template(rating.messages(row, 'start'), tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=True, reasoning_effort='low')
        start_lengths.append(count_prompt_tokens(ids))
    for row in frame['records']:
        ids = tokenizer.apply_chat_template(rating.messages(row, 'target'), tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=True, reasoning_effort='low')
        target_lengths.append(count_prompt_tokens(ids))
    lengths = start_lengths + target_lengths
    ratings = 2 * len(lengths)
    prompt_tokens = 2 * sum(lengths)
    max_decode = ratings * rating.MAX_TOKENS
    max_context = max(lengths) + rating.MAX_TOKENS
    if max_context > 49152:
        raise ValueError('judge prompt exceeds model context')
    # Conservative Qwen3.8 scout: 59.82 observed generated tokens/s including
    # prefill; use the slower independent short-context stress estimate.
    scout_time = sum(reader['wall_seconds'] for timing in scout['timings']
                     for reader in timing['reader_timings'])
    observed_scout_effective = scout['counts']['generated_tokens'] / scout_time
    decode_tps = min(43.71, .8 * observed_scout_effective)
    prefill_tps = 4914.0
    repeat = 1.25
    cold_load = 1.25 * sum(timing['load_seconds'] for timing in scout['timings'])
    shutdown = 196.0
    components = {'decode': repeat * max_decode / decode_tps,
                  'prefill': repeat * prompt_tokens / prefill_tps,
                  'cold_load': cold_load, 'shutdown': shutdown}
    predicted_s = sum(components.values())
    walltime_s = 120 * 60
    gpu_ceiling = 2 * walltime_s / 3600
    status = ('PASS_COMPLETE_STAGE' if predicted_s <= .90 * walltime_s
              else 'HOLD_PRICE_TOO_CLOSE_TO_WALLTIME')
    body = {'schema': 'micro-blind-semantic-rating-price-v2.2',
            'status': status, 'job_id': os.environ['SLURM_JOB_ID'],
            'frame_sha256': frame['sha256'],
            'driver_sha256': rating.file_sha(rating.__file__),
            'price_driver_sha256': rating.file_sha(__file__),
            'rubric_sha256': rating.file_sha(rating.RUBRIC),
            'qwen_scout_sha256': scout['sha256'],
            'model_snapshot': str(rating.MODEL),
            'rows': len(frame['records']), 'unique_native_prefixes': len(start_lengths),
            'ratings': ratings, 'start_ratings': 2 * len(start_lengths),
            'target_ratings': 2 * len(target_lengths),
            'prompt_tokens_exact_twice': prompt_tokens,
            'prompt_tokens_per_request_min': min(lengths),
            'prompt_tokens_per_request_p90': sorted(lengths)[math.ceil(.9 * len(lengths)) - 1],
            'prompt_tokens_per_request_max': max(lengths),
            'max_decode_tokens': max_decode, 'max_model_context_tokens': max_context,
            'observed_scout_effective_generated_tokens_per_s': observed_scout_effective,
            'priced_decode_tokens_per_s': decode_tps,
            'priced_prefill_tokens_per_s': prefill_tps,
            'repeat_work_factor': repeat,
            'components_seconds': components,
            'projected_complete_seconds': predicted_s,
            'projected_complete_GPU_h': 2 * predicted_s / 3600,
            'proposed_walltime_seconds': walltime_s,
            'stage_GPU_h_ceiling': gpu_ceiling,
            'interpretation': 'Two same-model arm-blind LLM audits of frozen pilot continuations; no human truth.'}
    value = {**body, 'sha256': rating.digest(body)}
    if OUT.exists():
        if rating.sealed(OUT) != value:
            raise ValueError('existing sealed rating price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'sha256': value['sha256'],
                      'status': status, 'projected_seconds': predicted_s,
                      'ceiling_GPU_h': gpu_ceiling}), flush=True)


if __name__ == '__main__':
    main()
