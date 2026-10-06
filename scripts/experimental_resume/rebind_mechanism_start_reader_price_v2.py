"""Rebind the exact v1 prompt price to the v2 append-only recovery driver."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_mechanism_start_frame_v1 import FRAME, REPO, SELECTION, file_sha, sealed, write_once
import rate_transition_v22_fullprefix_starts_v2 as prior_reader

SOURCE = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v1.json'
OUT = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v2.json'
DRIVER = REPO / 'scripts/experimental_resume/rate_mechanism_start_readers_v2.py'


def main():
    source, frame, selection = sealed(SOURCE), sealed(FRAME), sealed(SELECTION)
    if (source['schema'] != 'mechanism-start-reader-price-v1' or
            source['status'] != 'PASS_COMPLETE_20_GPUH' or
            source['frame_sha256'] != frame['sha256'] or
            source['selection_sha256'] != selection['sha256'] or
            source['rows'] != frame['rows'] or source['ratings'] != 2 * frame['rows'] or
            source['message_source_sha256'] != file_sha(prior_reader.__file__) or
            source['rubric_sha256'] != file_sha(prior_reader.RUBRIC) or
            source['max_decode_tokens'] != 2 * frame['rows'] * 1024 or
            source['complete_stage_projected_GPU_h'] > 20):
        raise ValueError('exact v1 stage price is absent or changed')
    body = {**{k: v for k, v in source.items() if k != 'sha256'},
            'schema': 'mechanism-start-reader-price-v2',
            'source_exact_price_sha256': source['sha256'],
            'rating_driver_sha256': file_sha(DRIVER),
            'pricing_driver_sha256': file_sha(__file__),
            'recovery_amendment': 'Append-only attempted-batch ledger and per-UID receipts; same seeds and one committed result per UID/reader. Uncommitted calls may be recomputed and are counted, not silently treated as absent. Any recovery requires fresh remaining-cost assessment.',
            'allocation_plan': {**source['allocation_plan'],
                                'recovery': 'same-manifest uncommitted-attempt resume only after fresh remaining-cost assessment; no automatic extra allocation'}}
    value = write_once(OUT, body)
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'source_exact_price_sha256': source['sha256'],
                      'projected_GPU_h': value['complete_stage_projected_GPU_h']}))


if __name__ == '__main__':
    main()
