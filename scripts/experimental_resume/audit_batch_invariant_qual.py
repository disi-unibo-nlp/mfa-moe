"""Audit within-regime native repeats and first-stage routing after BI qualification.

The established MARLIN run is included as a descriptive comparator only. The
different engine implementations do not form a randomized efficacy contrast.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

from analyze_micro_first_stage import analyze
from audit_native_only_replay_v3 import load_run, pair_report, summary
from build_micro_blind_frame import digest, sealed, write_once

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
BI_MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_BATCH_INVARIANT_QUAL_MANIFEST_v1.json'
BASE_MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
BI_RUN = ROOT / 'runs/routing-control-v1/batch-invariant-qual-db182a1debc7465c'
BASE_RUN = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_BATCH_INVARIANT_QUAL_AUDIT_v1.json'


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('routed-array analysis belongs on CPU Slurm')
    bi = load_run(BI_MANIFEST, BI_RUN, 48)
    base = load_run(BASE_MANIFEST, BASE_RUN, 48)
    bi_pairs = pair_report(bi[3], bi[4], ('native', 'native_duplicate'))
    base_pairs = pair_report(base[3], base[4], ('native', 'native_duplicate'))
    if [(x['prefix_uid'], x['seed']) for x in bi_pairs] != [
        (x['prefix_uid'], x['seed']) for x in base_pairs]:
        raise ValueError('source and distinct-engine native pair enrollment differs')
    first_stage = analyze(bi[0], BI_RUN, 48)
    if len(first_stage['records']) != 48:
        raise ValueError('batch-invariant first stage omitted assigned requests')
    body = {'schema': 'routing-batch-invariant-qualification-audit-v1',
            'manifest_sha256': bi[0]['sha256'],
            'source_pilot_manifest_sha256': base[0]['sha256'],
            'batch_invariant_summary_sha256': bi[2]['sha256'],
            'source_pilot_summary_sha256': base[2]['sha256'],
            'batch_invariant_array_sha256': bi[3]['array_sha256'],
            'source_pilot_array_sha256': base[3]['array_sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'batch_invariant_native_pairs': bi_pairs,
            'established_engine_native_pairs': base_pairs,
            'batch_invariant_native_summary': summary(bi_pairs),
            'established_engine_native_summary': summary(base_pairs),
            'first_stage': first_stage,
            'interpretation': 'within-regime native-repeat/isolation engineering audit; source engine and batch-invariant engine are distinct and cannot be pooled for semantic effect inference'}
    value = write_once(OUT, body)
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'batch_invariant_native': body['batch_invariant_native_summary'],
                      'established_native': body['established_engine_native_summary'],
                      'first_stage_arms': first_stage['arm_summaries']}))


if __name__ == '__main__':
    main()
