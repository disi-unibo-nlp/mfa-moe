"""Audit new layer24 action hook and realized top-k dose against frozen gates."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

from build_micro_blind_frame import file_sha, sealed, write_once

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPORT = REPO / 'report/experimental-resume-v1'
MANIFEST = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_MANIFEST_v1.json'
GATES = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_GATES_v1.json'
RUN = ROOT / 'runs/routing-control-v1/layer24-qual-87bda38632b02054'
OUT = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_AUDIT_v1.json'


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('routed-array qualification audit requires CPU Slurm')
    manifest, gates = sealed(MANIFEST), sealed(GATES)
    if gates['manifest_sha256'] != manifest['sha256']:
        raise ValueError('layer24 gates are not bound to manifest')
    binding, summary = sealed(RUN / 'BINDING.json'), sealed(RUN / 'SUMMARY.json')
    if (binding['manifest_sha256'] != manifest['sha256'] or
            summary['manifest_sha256'] != manifest['sha256'] or
            summary['binding_sha256'] != binding['sha256']):
        raise ValueError('generation binding or summary differs')
    action_by_name = {a['name']: a for a in manifest['actions']}
    rows = defaultdict(dict)
    hard = {'assigned': 0, 'errors': 0, 'missing_routed_or_telemetry': 0,
            'inactive_expert_identity_mismatches': 0,
            'inactive_weight_mismatches': 0,
            'invalid_edited_dose_or_segment': 0}
    receipts = []
    dose_rows = []
    for index in range(8):
        batch = sealed(RUN / f'batch-{index:03d}.json')
        array_path = RUN / f'batch-{index:03d}.npz'
        if (batch['manifest_sha256'] != manifest['sha256'] or
                batch['binding_sha256'] != binding['sha256'] or
                len(batch['outputs']) != 8 or
                file_sha(array_path) != batch['array_sha256']):
            raise ValueError('layer24 checkpoint receipt differs')
        receipts.append(batch['sha256'])
        with np.load(array_path, allow_pickle=False) as arrays:
            for row in batch['outputs']:
                hard['assigned'] += 1
                if row['error']:
                    hard['errors'] += 1
                if (not row['routed_present'] or row['missing_telemetry_ranks'] or
                        row['uid'] not in arrays):
                    hard['missing_routed_or_telemetry'] += 1
                    continue
                route = np.asarray(arrays[row['uid']])
                if route.shape != (len(row['tokens']), 40, 8) or len(route) == 0:
                    hard['missing_routed_or_telemetry'] += 1
                    continue
                for rank in row['inactive_native_checks'].values():
                    for check in rank.values():
                        hard['inactive_expert_identity_mismatches'] += check['expert_identity_mismatches']
                        hard['inactive_weight_mismatches'] += check['weight_mismatches']
                key = (row['family'], row['seed'], row['condition'])
                if row['slot'] in rows[key]:
                    raise ValueError('duplicate action/sentinel row')
                rows[key][row['slot']] = (row, route)
                if row['role'] == 'native':
                    if row['action_dose']:
                        hard['invalid_edited_dose_or_segment'] += 1
                    continue
                expected = row['policy']
                if (expected not in action_by_name or
                        action_by_name[expected]['experts'][0][0] != 24 or
                        set(row['action_dose']) != {'0', '1'}):
                    hard['invalid_edited_dose_or_segment'] += 1
                    continue
                rank_values = []
                for rank in ('0', '1'):
                    info = row['action_dose'][rank]
                    dose = (info.get('dose') or {}).get(expected, {}).get('24')
                    segments = (info.get('segments') or {}).get(expected, [])
                    active_rows = (info.get('rows') or {}).get(expected)
                    if (dose is None or not isinstance(segments, list) or
                            not segments or segments[0][0] != 0 or
                            active_rows is None or active_rows <= 0 or
                            dose['active_rows'] <= 0 or
                            not math.isfinite(dose['weight_l1']) or
                            dose['weight_l1'] < 0):
                        hard['invalid_edited_dose_or_segment'] += 1
                        continue
                    rank_values.append(dose)
                if len(rank_values) == 2:
                    dose_rows.append({'family': row['family'], 'seed': row['seed'],
                                      'condition': row['condition'], 'policy': expected,
                                      'rank0': rank_values[0], 'rank1': rank_values[1]})
    if (len(rows) != 32 or any(set(v) != {'active', 'sentinel'} for v in rows.values())):
        raise ValueError('64-request paired qualification layout incomplete')
    comparisons = defaultdict(list)
    for (family, seed, condition), pair in sorted(rows.items()):
        active, active_route = pair['active']
        sentinel, sentinel_route = pair['sentinel']
        if active['prompt_sha256'] != sentinel['prompt_sha256']:
            raise ValueError('action and sentinel prefix differ')
        active_ids = (159,) if condition.startswith('target') else (
            tuple(action_by_name[active['policy']]['experts'][0][1])
            if condition.startswith('random') else (159,))
        mask_a = np.isin(active_route[:, 24], active_ids).any(axis=1)
        mask_b = np.isin(sentinel_route[:, 24], active_ids).any(axis=1)
        comparisons[condition].append({
            'family': family, 'seed': seed, 'policy': active['policy'],
            'first_token_active_selected': bool(mask_a[0]),
            'first_token_sentinel_selected': bool(mask_b[0]),
            'active_selected_fraction': float(mask_a.mean()),
            'sentinel_selected_fraction': float(mask_b.mean()),
            'first_emitted_token_equal': active['tokens'][0] == sentinel['tokens'][0],
            'finish_equal': active['finish'] == sentinel['finish']})
    target1 = [r['rank0'] for r in dose_rows if r['condition'] == 'target1']
    random1 = [r['rank0'] for r in dose_rows if r['condition'] == 'random1']
    dose_summary = {
        'target1_rows': len(target1), 'random1_rows': len(random1),
        'target1_membership_changes': sum(r['membership_changes'] for r in target1),
        'target1_actual_minus_native_target_hits': sum(
            r['actual_target_hits'] - r['native_target_hits'] for r in target1),
        'random1_membership_changes': sum(r['membership_changes'] for r in random1),
        'target1_weight_l1': sum(r['weight_l1'] for r in target1),
        'random1_weight_l1': sum(r['weight_l1'] for r in random1),
    }
    first_gate = gates['first_stage_gate']
    hard_pass = (hard['assigned'] == 64 and summary['requests'] == 64 and
                 summary['batches'] == 8 and
                 all(v == 0 for k, v in hard.items() if k != 'assigned') and
                 len(dose_rows) == 24)
    dose_pass = (
        len(target1) == 8 and len(random1) == 8 and
        dose_summary['target1_membership_changes'] >=
        first_gate['target_bias1_total_membership_changes_minimum'] and
        dose_summary['target1_actual_minus_native_target_hits'] >=
        first_gate['target_bias1_total_actual_minus_native_target_hits_minimum'] and
        dose_summary['random1_membership_changes'] >=
        first_gate['random_bias1_total_membership_changes_minimum'])
    body = {'schema': 'routing-discovery-layer24-hook-dose-audit-v1',
            'manifest_sha256': manifest['sha256'], 'gates_sha256': gates['sha256'],
            'binding_sha256': binding['sha256'], 'summary_sha256': summary['sha256'],
            'batch_sha256s': receipts,
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'hard_checks': hard, 'dose_rows': dose_rows,
            'dose_summary': dose_summary, 'paired_first_stage': dict(comparisons),
            'mechanical_pass': hard_pass, 'routing_first_stage_pass': dose_pass,
            'pass': hard_pass and dose_pass,
            'interpretation': 'Four semantically invalid prefixes qualify mechanics and local routing dose only; no semantic effect or engine equivalence. Discovery feasibility remains separately gated.'}
    value = write_once(OUT, body)
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'pass': body['pass'], 'dose_summary': dose_summary,
                      'hard_checks': hard}))


if __name__ == '__main__':
    main()
