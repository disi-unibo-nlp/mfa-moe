"""Exact complete-stage Qwen3.8 two-reader price for the sealed blind frame.

Tokenization runs only in a CPU Slurm step. Shards contain whole eight-row
blocks and both readers. The caller supplies a reviewed GPU-hour ceiling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import run_boundary_micro_screen as base
import prepare_mechanism_validation_v1 as generation_prep
import rate_overnight_semantics_v2 as rating
from price_transition_ratings_v3 import count_prompt_tokens

PRIOR_PRICE = rating.REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v2.json'
PRIOR_BINDING = generation_prep.RATINGS / 'BINDING.json'
PRIOR_SUMMARY = generation_prep.RATINGS / 'SUMMARY.json'


def price_stage(frame, lengths, prior_price, prior_binding, prior_summary,
                wall_seconds, gpu_hour_ceiling):
    rating.require(frame['schema'] == 'overnight-semantic-blind-frame-v2' and
                   frame['continuation_max_tokens'] in (256, 1024) and
                   prior_price['schema'] == 'mechanism-start-reader-price-v2' and
                   prior_price['status'] == 'PASS_COMPLETE_20_GPUH' and
                   prior_binding['schema'] == 'mechanism-start-reader-binding-v2' and
                   prior_binding['price_sha256'] == prior_price['sha256'] and
                   prior_binding['model_snapshot'] == str(rating.MODEL) and
                   prior_binding['readers'] == 2 and
                   prior_summary['schema'] == 'mechanism-start-reader-summary-v2' and
                   prior_summary['binding_sha256'] == prior_binding['sha256'] and
                   prior_summary['counts']['rows'] == 454 and
                   prior_summary['counts']['ratings'] == 908 and
                   len(lengths) == len(frame['records']) and
                   max(lengths, default=0) + rating.MAX_TOKENS <= rating.MAX_MODEL_LEN and
                   type(wall_seconds) is int and wall_seconds >= 3600 and
                   gpu_hour_ceiling > 0, 'reader price inputs, context or resource ceiling differ')
    decode_tps = prior_price['bounded_decode_tps']
    prefill_tps = prior_price['bounded_prefill_tps']
    repeat = prior_price.get('repeat_work_factor', 1.25)
    load = prior_price['components_seconds']['two_cold_loads'] / 2
    shutdown = prior_price['components_seconds']['two_shutdowns'] / 2
    reserve = 900
    usable = wall_seconds - load - shutdown - reserve
    rating.require(usable > 0, 'one reader job cannot fit load and shutdown')
    shards = []
    start, accumulated = 0, 0.0
    for offset in range(0, len(lengths), rating.BATCH):
        block = lengths[offset:offset + rating.BATCH]
        seconds = repeat * (2 * sum(block) / prefill_tps +
                            2 * len(block) * rating.MAX_TOKENS / decode_tps)
        rating.require(seconds <= usable, 'one complete two-reader batch exceeds walltime')
        if accumulated and accumulated + seconds > usable:
            shards.append({'start': start, 'end': offset,
                           'estimated_work_seconds': accumulated})
            start, accumulated = offset, 0.0
        accumulated += seconds
    if lengths:
        shards.append({'start': start, 'end': len(lengths),
                       'estimated_work_seconds': accumulated})
    jobs = len(shards)
    recovery_loads = max(1, math.ceil(jobs * (repeat - 1))) if jobs else 0
    work = sum(x['estimated_work_seconds'] for x in shards)
    total_seconds = work + (jobs + recovery_loads) * (load + shutdown)
    gpu_hours = 2 * total_seconds / 3600
    return {
        'schema': 'overnight-semantic-reader-price-v2',
        'status': 'PASS_COMPLETE_STAGE' if gpu_hours <= gpu_hour_ceiling else
                  'HOLD_EXCEEDS_GPU_HOUR_CEILING',
        'frame_sha256': frame['sha256'],
        'rating_driver_sha256': base.file_sha(rating.__file__),
        'pricing_driver_sha256': base.file_sha(__file__),
        'rubric_sha256': base.file_sha(rating.RUBRIC),
        'model_snapshot': str(rating.MODEL),
        'prior_price_sha256': prior_price['sha256'],
        'prior_binding_sha256': prior_binding['sha256'],
        'prior_summary_sha256': prior_summary['sha256'],
        'rows': len(lengths), 'ratings': 2 * len(lengths),
        'batch_size': rating.BATCH, 'readers': 2,
        'prompt_token_lengths': lengths,
        'prompt_tokens_exact_twice': 2 * sum(lengths),
        'prompt_tokens_min': min(lengths, default=0), 'prompt_tokens_max': max(lengths, default=0),
        'max_decode_tokens': 2 * len(lengths) * rating.MAX_TOKENS,
        'max_model_context_tokens': max(lengths, default=0) + rating.MAX_TOKENS,
        'bounded_decode_tokens_per_second': decode_tps,
        'bounded_prefill_tokens_per_second': prefill_tps,
        'repeat_work_factor': repeat,
        'cold_load_seconds': load, 'shutdown_seconds': shutdown,
        'shards': shards, 'planned_jobs': jobs,
        'recovery_load_reserve': recovery_loads,
        'max_wall_seconds_per_job': wall_seconds,
        'preemption_reserve_seconds_per_job': reserve,
        'estimated_complete_wall_seconds': total_seconds,
        'estimated_complete_gpu_hours': gpu_hours,
        'gpu_hour_ceiling': gpu_hour_ceiling,
        'gpus': 2,
        'interpretation': ('Exact Qwen3.8 chat-template prompts for two arm-blind ratings '
                           'of every gradeable continuation within the frozen horizon; maximum output '
                           'caps, measured conservative throughput, complete shards, '
                           'cold loads and a recovery reserve. Price is not an effect estimate.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-wall-seconds', type=int, required=True)
    parser.add_argument('--gpu-hour-ceiling', type=float, required=True)
    args = parser.parse_args()
    rating.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
                   os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz',
                   'exact token pricing requires lrd_all_viz CPU Slurm step')
    from transformers import AutoTokenizer
    frame = base.sealed(args.frame)
    prior_price = base.sealed(PRIOR_PRICE)
    prior_binding = base.sealed(PRIOR_BINDING)
    prior_summary = base.sealed(PRIOR_SUMMARY)
    tokenizer = AutoTokenizer.from_pretrained(rating.MODEL, local_files_only=True)
    lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
        rating.messages(row), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort='low')) for row in frame['records']]
    body = price_stage(frame, lengths, prior_price, prior_binding, prior_summary,
                       args.max_wall_seconds, args.gpu_hour_ceiling)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    value = rating.save(args.out, body, existing_ok=True)
    print(json.dumps({'price_sha256': value['sha256'], 'status': value['status'],
                      'rows': value['rows'], 'ratings': value['ratings'],
                      'shards': len(value['shards']),
                      'estimated_gpu_hours': value['estimated_complete_gpu_hours']}), flush=True)


if __name__ == '__main__':
    main()
