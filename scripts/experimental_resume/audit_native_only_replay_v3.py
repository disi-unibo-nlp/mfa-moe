"""Compare native duplicate variability with and without edited neighbors."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from build_micro_blind_frame import digest, file_sha, sealed

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPLAY_MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_NATIVE_ONLY_REPLAY_MANIFEST_v3.json'
PILOT_MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
REPLAY = ROOT / 'runs/routing-control-v1/native-only-replay-4b4eda23ab7747ec'
PILOT = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_NATIVE_ONLY_REPLAY_AUDIT_v3.json'
TARGET = (9, 189)


def first_difference(a, b):
    return next((i for i, (left, right) in enumerate(zip(a, b)) if left != right),
                min(len(a), len(b)) if len(a) != len(b) else None)


def load_run(manifest_path, run_path, expected):
    manifest = sealed(manifest_path)
    binding = sealed(run_path / 'BINDING.json')
    summary = sealed(run_path / 'SUMMARY.json')
    batch = sealed(run_path / 'batch-000.json')
    array_path = run_path / 'batch-000.npz'
    if (manifest['expected_requests'] != expected or
            binding['manifest_sha256'] != manifest['sha256'] or
            summary['manifest_sha256'] != manifest['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            summary['requests'] != expected or
            batch['manifest_sha256'] != manifest['sha256'] or
            batch['binding_sha256'] != binding['sha256'] or
            len(batch['outputs']) != expected or
            file_sha(array_path) != batch['array_sha256']):
        raise ValueError('completed generation receipts differ')
    return manifest, binding, summary, batch, array_path


def pair_report(batch, array_path, arm_names):
    pairs = defaultdict(dict)
    with np.load(array_path, allow_pickle=False) as arrays:
        for row in batch['outputs']:
            if row['arm'] not in arm_names:
                continue
            if row['error'] or not row['routed_present'] or row['uid'] not in arrays:
                raise ValueError('native replay has missing routed result')
            route = np.asarray(arrays[row['uid']])
            if route.shape != (len(row['tokens']), 40, 8):
                raise ValueError('routed array does not match tokens')
            if any(any(v.get('expert_identity_mismatches') or v.get('weight_mismatches')
                       for v in rank.values()) for rank in row['inactive_native_checks'].values()):
                raise ValueError('inactive-native worker check failed')
            pairs[(row['prefix_uid'], row['seed'])][row['arm']] = (row, route)
    if len(pairs) != 8 or any(set(arms) != set(arm_names) for arms in pairs.values()):
        raise ValueError('native pair enrollment differs')
    records = []
    for (prefix_uid, seed), arms in sorted(pairs.items()):
        a, ra = arms[arm_names[0]]
        b, rb = arms[arm_names[1]]
        if a['prompt_sha256'] != b['prompt_sha256']:
            raise ValueError('duplicate-native prompts differ')
        first_token_difference = first_difference(a['tokens'], b['tokens'])
        shared = min(len(ra), len(rb))
        sets_a = np.sort(ra[:shared], axis=2)
        sets_b = np.sort(rb[:shared], axis=2)
        first_set_difference = next((i for i in range(shared) if not np.array_equal(
            sets_a[i], sets_b[i])), shared if len(ra) != len(rb) else None)
        target_a = np.isin(ra[:shared, 28], TARGET).any(axis=1)
        target_b = np.isin(rb[:shared, 28], TARGET).any(axis=1)
        before = first_token_difference if first_token_difference is not None else shared
        records.append({'prefix_uid': prefix_uid, 'seed': seed,
                        'tokens_equal': first_token_difference is None,
                        'first_token_difference_index': first_token_difference,
                        'first_unordered_topk_set_difference_index': first_set_difference,
                        'first_token_unordered_topk_equal':
                        bool(np.array_equal(sets_a[0], sets_b[0])),
                        'first_token_target_mask_equal': bool(target_a[0] == target_b[0]),
                        'unordered_topk_equal_before_token_difference':
                        bool(np.array_equal(sets_a[:before], sets_b[:before])),
                        'target_mask_equal_before_token_difference':
                        bool(np.array_equal(target_a[:before], target_b[:before])),
                        'first_token_target_selected': [bool(target_a[0]), bool(target_b[0])],
                        'first_emitted_tokens': [a['tokens'][0], b['tokens'][0]],
                        'target_occupancy_fraction': [float(target_a.mean()),
                                                      float(target_b.mean())],
                        'finish_equal': a['finish'] == b['finish']})
    return records


def summary(rows):
    return {'pairs': len(rows), 'tokens_equal': sum(x['tokens_equal'] for x in rows),
            'first_token_unordered_topk_equal': sum(x['first_token_unordered_topk_equal'] for x in rows),
            'first_token_target_mask_equal': sum(x['first_token_target_mask_equal'] for x in rows),
            'unordered_topk_equal_before_token_difference': sum(
                x['unordered_topk_equal_before_token_difference'] for x in rows),
            'target_mask_equal_before_token_difference': sum(
                x['target_mask_equal_before_token_difference'] for x in rows),
            'first_token_difference_indices': [x['first_token_difference_index'] for x in rows],
            'first_unordered_topk_set_difference_indices': [
                x['first_unordered_topk_set_difference_index'] for x in rows]}


def main():
    replay = load_run(REPLAY_MANIFEST, REPLAY, 16)
    pilot = load_run(PILOT_MANIFEST, PILOT, 48)
    replay_rows = pair_report(replay[3], replay[4], ('native_A', 'native_B'))
    pilot_rows = pair_report(pilot[3], pilot[4], ('native', 'native_duplicate'))
    if [(x['prefix_uid'], x['seed']) for x in replay_rows] != [
        (x['prefix_uid'], x['seed']) for x in pilot_rows]:
        raise ValueError('native-only replay and mixed pilot pair identities differ')
    body = {'schema': 'routing-native-only-replay-comparison-v3',
            'replay_manifest_sha256': replay[0]['sha256'],
            'replay_summary_sha256': replay[2]['sha256'],
            'replay_array_sha256': replay[3]['array_sha256'],
            'mixed_pilot_manifest_sha256': pilot[0]['sha256'],
            'mixed_pilot_summary_sha256': pilot[2]['sha256'],
            'mixed_pilot_array_sha256': pilot[3]['array_sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_only_pairs': replay_rows, 'mixed_pilot_pairs': pilot_rows,
            'native_only_summary': summary(replay_rows),
            'mixed_pilot_summary': summary(pilot_rows),
            'interpretation': 'small diagnostic comparison of same-prefix duplicates; does not by itself identify source of numerical variability'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing native replay audit differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'native_only': body['native_only_summary'],
                      'mixed_pilot': body['mixed_pilot_summary']}))


if __name__ == '__main__':
    main()
