"""Bound the 30-minute checkpointed arm-blind rating slices."""
from __future__ import annotations

import json
from pathlib import Path

import rate_micro_blind_semantics_v2_2 as rating

PRICE = rating.sealed(rating.PRICE)
OUT = rating.REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_DEBUG_PRICE_v2.2.json'


def main():
    if PRICE['status'] != 'PASS_COMPLETE_STAGE' or PRICE['ratings'] != 104:
        raise ValueError('complete-stage semantic price differs')
    # Two 12-continuation target batches give the largest two-batch decode.
    # The first slice's two 4-prefix start batches are less expensive.
    max_decode = 2 * rating.BATCH * rating.MAX_TOKENS
    max_prefill = 2 * rating.BATCH * PRICE['prompt_tokens_per_request_max']
    components = {'decode': 1.25 * max_decode / PRICE['priced_decode_tokens_per_s'],
                  'prefill': 1.25 * max_prefill / PRICE['priced_prefill_tokens_per_s'],
                  'cold_load': PRICE['components_seconds']['cold_load'],
                  'shutdown': PRICE['components_seconds']['shutdown']}
    projection = sum(components.values())
    body = {'schema': 'micro-blind-semantic-debug-slice-price-v2.2',
            'complete_price_sha256': PRICE['sha256'],
            'frame_sha256': PRICE['frame_sha256'],
            'rating_driver_sha256': PRICE['driver_sha256'],
            'price_driver_sha256': rating.file_sha(__file__),
            'ratings_total': 104, 'batches_total': 10,
            'max_ratings_in_two_target_batches': 24,
            'max_decode_tokens_two_target_batches': max_decode,
            'max_prompt_tokens_two_target_batches': max_prefill,
            'two_batch_conservative_components_seconds': components,
            'two_batch_projected_seconds': projection,
            'debug_walltime_seconds': 1800,
            'debug_GPU_h_ceiling_per_slice': 1.0,
            'five_slice_GPU_h_ceiling': 5.0,
            'deadline_guard': 'stop before next batch if <440s remain before Slurm end minus120s',
            'status': 'PASS_TWO_COMPLETE_TARGET_BATCHES_PER_SLICE' if projection < 1800
                      else 'HOLD_SLICE_PRICE',
            'interpretation': 'Bounded same-manifest checkpoint/resume; actual slices may cover more batches when observed judge outputs are shorter.'}
    value = {**body, 'sha256': rating.digest(body)}
    if OUT.exists():
        if rating.sealed(OUT) != value:
            raise ValueError('debug slice price changed')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'status': value['status'], 'two_batch_projected_seconds': projection}))


if __name__ == '__main__':
    main()
