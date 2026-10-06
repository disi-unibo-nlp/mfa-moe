"""Audit serial/eager within-profile native repeats and 256-token action dose."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from analyze_micro_first_stage import analyze
from audit_native_only_replay_v3 import first_difference, pair_report, summary
from build_micro_blind_frame import file_sha, sealed, write_once

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICRO_SERIAL_QUAL_MANIFEST_v1.json'
SOURCE_MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
RUN = ROOT / 'runs/routing-control-v1/micro-serial-qual-68670e1dcdb8f3a2'
SOURCE_RUN = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICRO_SERIAL_QUAL_AUDIT_v1.json'
TARGET = (9, 189)


def serial_native_pairs(manifest):
    by_pair = defaultdict(dict)
    receipts = []
    for index in range(8):
        batch = sealed(RUN / f'batch-{index:03d}.json')
        array = RUN / f'batch-{index:03d}.npz'
        if (batch['manifest_sha256'] != manifest['sha256'] or
                file_sha(array) != batch['array_sha256'] or len(batch['outputs']) != 6):
            raise ValueError('serial checkpoint receipt differs')
        receipts.append(batch['sha256'])
        with np.load(array, allow_pickle=False) as arrays:
            for row in batch['outputs']:
                if row['role'] == 'native':
                    if row['error'] or not row['routed_present'] or row['uid'] not in arrays:
                        raise ValueError('native serial request has missing result')
                    routed = np.asarray(arrays[row['uid']])
                    if routed.shape != (len(row['tokens']), 40, 8):
                        raise ValueError('serial tokens and routes differ')
                    by_pair[(row['prefix_uid'], row['seed'])][row['arm']] = (row, routed)
    if len(by_pair) != 8 or any(set(p) != {'native', 'native_duplicate'} for p in by_pair.values()):
        raise ValueError('serial native pairs incomplete')
    pairs = []
    for (prefix_uid, seed), arms in sorted(by_pair.items()):
        a, ra = arms['native']
        b, rb = arms['native_duplicate']
        if a['prompt_sha256'] != b['prompt_sha256']:
            raise ValueError('native duplicate prompts differ')
        first_token = first_difference(a['tokens'], b['tokens'])
        shared = min(len(ra), len(rb))
        set_a, set_b = np.sort(ra[:shared], axis=2), np.sort(rb[:shared], axis=2)
        first_set = next((i for i in range(shared) if not np.array_equal(set_a[i], set_b[i])),
                         shared if len(ra) != len(rb) else None)
        hit_a = np.isin(ra[:shared, 28], TARGET).any(axis=1)
        hit_b = np.isin(rb[:shared, 28], TARGET).any(axis=1)
        before = first_token if first_token is not None else shared
        pairs.append({'prefix_uid': prefix_uid, 'seed': seed,
                      'tokens_equal': first_token is None,
                      'first_token_difference_index': first_token,
                      'first_unordered_topk_set_difference_index': first_set,
                      'first_token_unordered_topk_equal': bool(np.array_equal(set_a[0], set_b[0])),
                      'first_token_target_mask_equal': bool(hit_a[0] == hit_b[0]),
                      'unordered_topk_equal_before_token_difference': bool(np.array_equal(set_a[:before], set_b[:before])),
                      'target_mask_equal_before_token_difference': bool(np.array_equal(hit_a[:before], hit_b[:before])),
                      'first_token_target_selected': [bool(hit_a[0]), bool(hit_b[0])],
                      'first_emitted_tokens': [a['tokens'][0], b['tokens'][0]],
                      'target_occupancy_fraction': [float(hit_a.mean()), float(hit_b.mean())],
                      'finish_equal': a['finish'] == b['finish']})
    return pairs, receipts


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('routed-array audit belongs on CPU Slurm')
    manifest, source_manifest = sealed(MANIFEST), sealed(SOURCE_MANIFEST)
    if (manifest['source_pilot_sha256'] != source_manifest['sha256'] or
            manifest['engine_overrides'] != {
                'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}):
        raise ValueError('serial execution profile differs')
    serial_first = analyze(manifest, RUN, 6)
    if serial_first['assigned'] != 48:
        raise ValueError('serial qualification lost assignments')
    native_pairs, receipts = serial_native_pairs(manifest)
    source = sealed(SOURCE_RUN / 'batch-000.json')
    source_array = SOURCE_RUN / 'batch-000.npz'
    if (source['manifest_sha256'] != source_manifest['sha256'] or
            file_sha(source_array) != source['array_sha256']):
        raise ValueError('established pilot source differs')
    established_pairs = pair_report(source, source_array, ('native', 'native_duplicate'))
    if [(r['prefix_uid'], r['seed']) for r in native_pairs] != [
        (r['prefix_uid'], r['seed']) for r in established_pairs]:
        raise ValueError('serial and established pair enrollment differs')
    body = {'schema': 'routing-boundary-serial-eager-qualification-audit-v1',
            'manifest_sha256': manifest['sha256'],
            'source_pilot_manifest_sha256': source_manifest['sha256'],
            'serial_generation_summary_sha256': sealed(RUN / 'SUMMARY.json')['sha256'],
            'serial_batch_sha256s': receipts,
            'source_pilot_batch_sha256': source['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'serial_native_pairs': native_pairs,
            'established_native_pairs': established_pairs,
            'serial_native_summary': summary(native_pairs),
            'established_native_summary': summary(established_pairs),
            'serial_first_stage': serial_first,
            'qualification_limits': ['max_num_seqs=1 prevents simultaneous edited neighbors and artificial preemption',
                                     'only 256-token branch-zero pulses are exercised',
                                     'semantically invalid old pilot starts remain engineering fixtures'],
            'interpretation': 'new within-profile native-repeat and pulse-engine evidence; no semantic effect, cross-profile equivalence, or later two-action preemption claim'}
    value = write_once(OUT, body)
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'serial_native': body['serial_native_summary'],
                      'established_native': body['established_native_summary'],
                      'arm_first_stage': serial_first['arm_summaries']}))


if __name__ == '__main__':
    main()
