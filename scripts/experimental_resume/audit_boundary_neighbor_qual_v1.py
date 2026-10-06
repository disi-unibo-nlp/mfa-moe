"""Audit sealed native-sentinel neighbor qualification against prospective gates."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from build_micro_blind_frame import file_sha, sealed, write_once
from audit_native_only_replay_v3 import first_difference

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_BATCHED_NEIGHBOR_QUAL_MANIFEST_v1.json'
GATES = REPO / 'report/experimental-resume-v1/CAUSAL_BATCHED_NEIGHBOR_QUAL_GATES_v1.json'
RUN = ROOT / 'runs/routing-control-v1/micro-neighbor-qual-e0c498b65dce4dc7'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v1.json'
TARGET = (9, 189)


def first_target(route):
    return bool(np.isin(route[0, 28], TARGET).any())


def compare(a, b):
    ra, rb = a['route'], b['route']
    first_text = first_difference(a['tokens'], b['tokens'])
    shared = min(len(ra), len(rb))
    before = min(first_text if first_text is not None else shared, shared)
    masks_a = np.isin(ra[:before, 28], TARGET).any(axis=1)
    masks_b = np.isin(rb[:before, 28], TARGET).any(axis=1)
    return {'first_emitted_token_equal': a['tokens'][0] == b['tokens'][0],
            'first_token_target_mask_equal': first_target(ra) == first_target(rb),
            'first_token_unordered_topk_equal': bool(np.array_equal(
                np.sort(ra[0], axis=1), np.sort(rb[0], axis=1))),
            'first_text_divergence_index': first_text,
            'same_text_prefix_tokens': before,
            'target_mask_equal_on_same_text_prefix': bool(np.array_equal(masks_a, masks_b))}


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('routed-array audit belongs on CPU Slurm')
    manifest, gates = sealed(MANIFEST), sealed(GATES)
    if gates['manifest_sha256'] != manifest['sha256']:
        raise ValueError('prospective gates not bound to manifest')
    binding, summary = sealed(RUN / 'BINDING.json'), sealed(RUN / 'SUMMARY.json')
    if (binding['manifest_sha256'] != manifest['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            summary['manifest_sha256'] != manifest['sha256']):
        raise ValueError('generation binding/summary differs')
    data = defaultdict(dict)
    batch_hashes = []
    hard = {'assigned': 0, 'errored': 0, 'missing_routed_or_telemetry': 0,
            'inactive_expert_identity_mismatches': 0,
            'inactive_weight_mismatches': 0, 'sentinel_active_rows': 0}
    for index in range(10):
        batch = sealed(RUN / f'batch-{index:03d}.json')
        array = RUN / f'batch-{index:03d}.npz'
        if (batch['manifest_sha256'] != manifest['sha256'] or
                batch['binding_sha256'] != binding['sha256'] or
                file_sha(array) != batch['array_sha256'] or
                len(batch['outputs']) != 8):
            raise ValueError('checkpoint batch differs: ' + str(index))
        batch_hashes.append(batch['sha256'])
        with np.load(array, allow_pickle=False) as arrays:
            for row in batch['outputs']:
                hard['assigned'] += 1
                if row['error']:
                    hard['errored'] += 1
                if not row['routed_present'] or row['missing_telemetry_ranks'] or row['uid'] not in arrays:
                    hard['missing_routed_or_telemetry'] += 1
                    continue
                routed = np.asarray(arrays[row['uid']])
                if routed.shape != (len(row['tokens']), 40, 8) or len(routed) == 0:
                    hard['missing_routed_or_telemetry'] += 1
                    continue
                for rank in row['inactive_native_checks'].values():
                    for check in rank.values():
                        hard['inactive_expert_identity_mismatches'] += check['expert_identity_mismatches']
                        hard['inactive_weight_mismatches'] += check['weight_mismatches']
                if row['slot'] == 'sentinel':
                    for dose in row['action_dose'].values():
                        hard['sentinel_active_rows'] += dose.get('rows') or 0
                key = (row['family'], row['seed'])
                subkey = (row['condition'], row['slot'])
                if subkey in data[key]:
                    raise ValueError('duplicate native-sentinel assignment')
                data[key][subkey] = {'tokens': row['tokens'], 'route': routed,
                                     'prompt_sha256': row['prompt_sha256'],
                                     'finish': row['finish']}
    conditions = [x['name'] for x in manifest['conditions']]
    if len(data) != 8 or any(set(row) != {(condition, slot) for condition in conditions
                                         for slot in ('active', 'sentinel')}
                             for row in data.values()):
        raise ValueError('incomplete matched family/seed/condition layout')
    comparisons = defaultdict(list)
    for (family, seed), rows in sorted(data.items()):
        baseline = rows[('zero', 'sentinel')]
        for condition in conditions:
            sentinel = rows[(condition, 'sentinel')]
            active = rows[(condition, 'active')]
            if baseline['prompt_sha256'] != sentinel['prompt_sha256'] or baseline['prompt_sha256'] != active['prompt_sha256']:
                raise ValueError('matched family/seed prompt differs')
            comparison = compare(baseline, sentinel if condition != 'zero' else active)
            comparisons[condition].append({'family': family, 'seed': seed, **comparison})
    aggregates = {}
    for condition, rows in comparisons.items():
        aggregates[condition] = {
            'paired_family_seed': len(rows),
            'first_emitted_token_discordant': sum(not x['first_emitted_token_equal'] for x in rows),
            'first_token_target_mask_discordant': sum(not x['first_token_target_mask_equal'] for x in rows),
            'first_token_unordered_topk_discordant': sum(not x['first_token_unordered_topk_equal'] for x in rows),
            'same_text_prefix_target_mask_discordant': sum(
                not x['target_mask_equal_on_same_text_prefix'] for x in rows),
        }
    hard_gate_pass = (
        hard['assigned'] == gates['hard_engine_gates']['completed_assigned_requests'] and
        summary['requests'] == hard['assigned'] and summary['batches'] == 10 and
        all(value == 0 for key, value in hard.items() if key != 'assigned'))
    limit_mask = gates['behavioral_sensitivity_gates']['maximum_target_mask_discordant_pairs_per_edited_condition']
    limit_token = gates['behavioral_sensitivity_gates']['maximum_first_emitted_token_discordant_pairs_per_edited_condition']
    behavior_gate_pass = all(
        aggregates[name]['first_token_target_mask_discordant'] <= limit_mask and
        aggregates[name]['first_emitted_token_discordant'] <= limit_token
        for name in conditions if name != 'zero')
    body = {'schema': 'routing-boundary-neighbor-qualification-audit-v1',
            'manifest_sha256': manifest['sha256'], 'gates_sha256': gates['sha256'],
            'binding_sha256': binding['sha256'], 'summary_sha256': summary['sha256'],
            'batch_sha256s': batch_hashes,
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'hard_checks': hard, 'comparisons': dict(comparisons),
            'aggregates': aggregates,
            'hard_gate_pass': hard_gate_pass,
            'behavioral_sensitivity_gate_pass': behavior_gate_pass,
            'accepted_for_limited_batched_behavioral_discovery': hard_gate_pass and behavior_gate_pass,
            'interpretation': 'Only four semantically invalid pilot families; observed non-rejection cannot establish engine equivalence or a population-tight neighbor bound. Randomization and family-clustered inference for later discovery remain separate.'}
    value = write_once(OUT, body)
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'hard_gate_pass': hard_gate_pass,
                      'behavioral_sensitivity_gate_pass': behavior_gate_pass,
                      'aggregates': aggregates}))


if __name__ == '__main__':
    main()
