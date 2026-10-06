"""Bounded read-only diagnosis of the eight duplicate-native pilot pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from build_micro_blind_frame import digest, file_sha, sealed

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
RUN = ROOT / 'runs/routing-control-v1/micro-screen-qual4-ac4c9651e71fe067'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_NATIVE_ISOLATION_AUDIT_v2.json'


def first_difference(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def main():
    manifest = sealed(MANIFEST)
    binding = sealed(RUN / 'BINDING.json')
    batch = sealed(RUN / 'batch-000.json')
    array_path = RUN / 'batch-000.npz'
    if (binding['manifest_sha256'] != manifest['sha256'] or
        batch['manifest_sha256'] != manifest['sha256'] or
        batch['binding_sha256'] != binding['sha256'] or
        file_sha(array_path) != batch['array_sha256']):
        raise ValueError('pilot artifacts differ')
    groups = {}
    for row in batch['outputs']:
        if row['role'] != 'native':
            continue
        groups.setdefault((row['prefix_uid'], row['seed']), {})[row['arm']] = row
    if len(groups) != 8 or any(set(group) != {'native', 'native_duplicate'} for group in groups.values()):
        raise ValueError('duplicate-native enrollment differs')
    records = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for (prefix_uid, seed), group in sorted(groups.items()):
            a, b = group['native'], group['native_duplicate']
            ra, rb = arrays[a['uid']], arrays[b['uid']]
            token_first = first_difference(a['tokens'], b['tokens'])
            shared = min(ra.shape[0], rb.shape[0])
            routing_differ = np.any(ra[:shared] != rb[:shared], axis=(1, 2))
            first_route = int(np.flatnonzero(routing_differ)[0]) if np.any(routing_differ) else (
                shared if ra.shape[0] != rb.shape[0] else None)
            sets_a, sets_b = np.sort(ra[:shared], axis=2), np.sort(rb[:shared], axis=2)
            set_differ = np.any(sets_a != sets_b, axis=(1, 2))
            first_set = int(np.flatnonzero(set_differ)[0]) if np.any(set_differ) else (
                shared if ra.shape[0] != rb.shape[0] else None)
            layer28_equal = np.all(sets_a[:, 28] == sets_b[:, 28], axis=1)
            target_a = np.isin(ra[:shared, 28], [9, 189]).any(axis=1)
            target_b = np.isin(rb[:shared, 28], [9, 189]).any(axis=1)
            before_token = token_first if token_first is not None else shared
            records.append({'prefix_uid': prefix_uid, 'seed': seed,
                            'prompt_sha_equal': a['prompt_sha256'] == b['prompt_sha256'],
                            'assigned_prompt_lengths': [a['prompt_len'], b['prompt_len']],
                            'emitted_token_lengths': [len(a['tokens']), len(b['tokens'])],
                            'first_emitted_token_difference_index': token_first,
                            'first_routed_expert_difference_index': first_route,
                            'first_unordered_topk_set_difference_index': first_set,
                            'routed_token_difference_count_in_shared_length': int(routing_differ.sum()),
                            'unordered_topk_set_difference_count_in_shared_length': int(set_differ.sum()),
                            'unordered_topk_set_equal_before_first_token_difference':
                            bool(np.array_equal(sets_a[:before_token], sets_b[:before_token])),
                            'layer28_unordered_set_equal_token_count': int(layer28_equal.sum()),
                            'layer28_target_occupancy_equal_token_count': int((target_a == target_b).sum()),
                            'layer28_target_occupancy_equal_before_first_token_difference':
                            bool(np.array_equal(target_a[:before_token], target_b[:before_token])),
                            'routed_shared_token_length': shared,
                            'routing_equal_before_first_token_difference':
                            bool(np.array_equal(ra[:token_first], rb[:token_first]))
                            if token_first is not None else bool(np.array_equal(ra, rb)),
                            'finish_equal': a['finish'] == b['finish'],
                            'both_unerrored': not a['error'] and not b['error'],
                            'inactive_native_checks_a': a.get('inactive_native_checks'),
                            'inactive_native_checks_b': b.get('inactive_native_checks')})
    body = {'schema': 'routing-micro-screen-native-isolation-audit-v2',
            'manifest_sha256': manifest['sha256'],
            'generation_binding_sha256': binding['sha256'],
            'batch_receipt_sha256': batch['sha256'],
            'routed_array_sha256': batch['array_sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pairs': records,
            'tokens_equal_pairs': sum(r['first_emitted_token_difference_index'] is None for r in records),
            'routed_equal_pairs': sum(r['first_routed_expert_difference_index'] is None for r in records),
            'unordered_topk_set_equal_pairs': sum(r['first_unordered_topk_set_difference_index'] is None for r in records),
            'unordered_topk_set_equal_before_token_divergence_pairs':
            sum(r['unordered_topk_set_equal_before_first_token_difference'] for r in records),
            'target_occupancy_equal_before_token_divergence_pairs':
            sum(r['layer28_target_occupancy_equal_before_first_token_difference'] for r in records),
            'prompt_equal_pairs': sum(r['prompt_sha_equal'] for r in records),
            'interpretation': 'diagnostic of same-seed native duplicate reproducibility; edited-neighbor isolation cannot be qualified from this failure'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing native audit differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'tokens_equal_pairs': body['tokens_equal_pairs'],
                      'routed_equal_pairs': body['routed_equal_pairs'],
                      'unordered_topk_set_equal_before_token_divergence_pairs': body['unordered_topk_set_equal_before_token_divergence_pairs'],
                      'target_occupancy_equal_before_token_divergence_pairs': body['target_occupancy_equal_before_token_divergence_pairs'],
                      'first_token_differences': [r['first_emitted_token_difference_index'] for r in records],
                      'first_unordered_set_differences': [r['first_unordered_topk_set_difference_index'] for r in records]}))


if __name__ == '__main__':
    main()
