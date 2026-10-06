"""Audit assigned micro-screen first-stage routing from sealed batch/NPZ outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

from build_micro_blind_frame import THINK_END_ID, assigned, digest, file_sha, sealed, write_once

TARGET = (9, 189)


def rates(routed, tokens, own):
    if routed.shape != (len(tokens), 40, 8):
        raise ValueError('routed array does not match emitted token count')
    stop = tokens.index(THINK_END_ID) if THINK_END_ID in tokens else len(tokens)
    section = routed[:stop, 28, :]
    target_hits = np.isin(section, TARGET).any(axis=1)
    own_hits = np.isin(section, own).any(axis=1)
    return {'reasoning_tokens': int(stop),
            'target_selected_tokens': int(target_hits.sum()),
            'own_set_selected_tokens': int(own_hits.sum()),
            'target_selected_fraction': float(target_hits.mean()) if stop else None,
            'own_set_selected_fraction': float(own_hits.mean()) if stop else None,
            'target_selected_per_256_assigned': float(target_hits.sum() / 256),
            'own_set_selected_per_256_assigned': float(own_hits.sum() / 256),
            'target_expert_token_counts': {str(e): int((section == e).any(axis=1).sum()) for e in TARGET},
            'closed_reasoning': THINK_END_ID in tokens}


def analyze(manifest, run_out, batch_size):
    binding = sealed(run_out / 'BINDING.json')
    if binding['manifest_sha256'] != manifest['sha256']:
        raise ValueError('generation output belongs to another manifest')
    summary = sealed(run_out / 'SUMMARY.json')
    if (summary['manifest_sha256'] != manifest['sha256'] or
        summary['binding_sha256'] != binding['sha256'] or
        summary['requests'] != manifest['expected_requests'] or
        summary['status'] != 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION'):
        raise ValueError('generation not complete and same-bound')
    plan = list(assigned(manifest))
    if len(plan) != manifest['expected_requests']:
        raise ValueError('assigned request count differs')
    action_by_name = {x['name']: x for x in manifest['actions']}
    data = {}
    receipts = []
    for i in range(math.ceil(len(plan) / batch_size)):
        expected = plan[i * batch_size:(i + 1) * batch_size]
        a = sealed(run_out / f'batch-{i:03d}-assignment.json')
        b = sealed(run_out / f'batch-{i:03d}.json')
        npz = run_out / f'batch-{i:03d}.npz'
        if (a['manifest_sha256'] != manifest['sha256'] or
            b['manifest_sha256'] != manifest['sha256'] or
            a['binding_sha256'] != binding['sha256'] or
            b['binding_sha256'] != binding['sha256'] or
            [x['uid'] for x in a['requests']] != [x['uid'] for x in expected] or
            [x['uid'] for x in b['outputs']] != [x['uid'] for x in expected] or
            file_sha(npz) != b['array_sha256']):
            raise ValueError('assignment, result or routed array seal differs')
        receipts.append({'batch': i, 'assignment_sha256': a['sha256'],
                         'receipt_sha256': b['sha256'], 'routed_array_sha256': b['array_sha256']})
        with np.load(npz, allow_pickle=False) as arrays:
            for metadata, output in zip(expected, b['outputs'], strict=True):
                if output['uid'] != metadata['uid']:
                    raise ValueError('request alignment differs')
                if output.get('routed_present'):
                    if metadata['uid'] not in arrays:
                        raise ValueError('completed request lacks routed array')
                    routed = np.asarray(arrays[metadata['uid']])
                else:
                    routed = None
                data[metadata['uid']] = (metadata, output, routed)
    if set(data) != {x['uid'] for x in plan}:
        raise ValueError('all assigned requests must have a result or explicit failure')

    records = []
    by_arm = defaultdict(list)
    by_pair = defaultdict(dict)
    for item in plan:
        meta, output, routed = data[item['uid']]
        policy = output['policy']
        own = TARGET if meta['role'] != 'random' else tuple(action_by_name[policy]['experts'][0][1])
        if meta['role'] == 'random' and policy not in action_by_name:
            raise ValueError('random policy differs from frozen set')
        usable = not output.get('error') and routed is not None
        observed = rates(routed, output['tokens'], own) if usable else {
            'reasoning_tokens': 0, 'target_selected_tokens': 0, 'own_set_selected_tokens': 0,
            'target_selected_fraction': None, 'own_set_selected_fraction': None,
            'target_selected_per_256_assigned': 0.0, 'own_set_selected_per_256_assigned': 0.0,
            'target_expert_token_counts': {str(e): 0 for e in TARGET}, 'closed_reasoning': False}
        record = {**meta, 'policy': policy, 'error': output.get('error'),
                  'finish': output.get('finish'), 'stop_reason': output.get('stop_reason'),
                  'emitted_tokens': len(output['tokens']), 'routed_present': routed is not None,
                  'action_dose': output.get('action_dose'),
                  'inactive_native_checks': output.get('inactive_native_checks'),
                  **observed}
        records.append(record)
        by_arm[meta['arm']].append(record)
        by_pair[(meta['prefix_uid'], meta['seed'])][meta['arm']] = (output, routed)
    duplicates = []
    native_arms = [a['name'] for a in manifest['arms'] if a['role'] == 'native']
    if len(native_arms) != 2:
        raise ValueError('expected exactly two duplicate native arms')
    for (prefix_uid, seed), pair in sorted(by_pair.items()):
        a, b = (pair[name] for name in native_arms)
        duplicates.append({'prefix_uid': prefix_uid, 'seed': seed,
                           'tokens_equal': a[0]['tokens'] == b[0]['tokens'],
                           'finish_equal': a[0]['finish'] == b[0]['finish'],
                           'routed_equal': a[1] is not None and b[1] is not None and np.array_equal(a[1], b[1]),
                           'both_unerrored': not a[0].get('error') and not b[0].get('error')})
    summaries = {}
    for name, subset in by_arm.items():
        valid = [x for x in subset if x['target_selected_fraction'] is not None]
        summaries[name] = {'assigned': len(subset), 'valid_routed_nonzero_reasoning': len(valid),
                           'generation_errors': sum(bool(x['error']) for x in subset),
                           'mean_target_fraction_on_valid':
                           sum(x['target_selected_fraction'] for x in valid) / len(valid) if valid else None,
                           'mean_own_fraction_on_valid':
                           sum(x['own_set_selected_fraction'] for x in valid) / len(valid) if valid else None,
                           'mean_target_selected_per_256_assigned':
                           sum(x['target_selected_per_256_assigned'] for x in subset) / len(subset)}
    return {'schema': 'routing-micro-screen-first-stage-v1',
            'manifest_sha256': manifest['sha256'], 'generation_binding_sha256': binding['sha256'],
            'generation_summary_sha256': summary['sha256'], 'batch_receipts': receipts,
            'assigned': len(plan), 'arm_summaries': summaries,
            'duplicate_native': duplicates,
            'duplicate_native_exact_all': all(d['tokens_equal'] and d['finish_equal'] and d['routed_equal']
                                              and d['both_unerrored'] for d in duplicates),
            'records': records,
            'interpretation': 'exploratory first-stage only; semantic ratings separate; all assigned requests retained'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('first-stage routed-array audit requires CPU Slurm')
    manifest = sealed(args.manifest)
    result = analyze(manifest, args.run_out, args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    value = write_once(args.out / 'FIRST_STAGE.json', result)
    print(json.dumps({'path': str(args.out / 'FIRST_STAGE.json'), 'sha256': value['sha256'],
                      'assigned': result['assigned'], 'duplicate_native_exact_all': result['duplicate_native_exact_all'],
                      'arm_summaries': result['arm_summaries']}))


if __name__ == '__main__':
    main()
