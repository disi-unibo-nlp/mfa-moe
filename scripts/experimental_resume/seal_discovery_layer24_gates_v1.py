"""Freeze layer24 mechanical and routing-dose gates before GPU qualification."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
MANIFEST = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_MANIFEST_v1.json'
NATIVE = REPORT / 'CAUSAL_NATIVE_ONLY_REPLAY_AUDIT_v3.json'
OUT = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_GATES_v1.json'


def main():
    manifest, native = base.sealed(MANIFEST), base.sealed(NATIVE)
    body = {
        'schema': 'routing-discovery-layer24-hook-dose-gates-v1',
        'manifest_sha256': manifest['sha256'],
        'native_replay_audit_sha256': native['sha256'],
        'population': 'four old semantically invalid candidate prefixes times two seeds; engineering only',
        'hard_gates': {
            'complete_requests': 64,
            'complete_batches': 8,
            'missing_routed_arrays_or_rank_telemetry': 0,
            'inactive_expert_identity_mismatches': 0,
            'inactive_weight_mismatches': 0,
            'native_sentinel_active_rows': 0,
            'policy_table_hooked_layers': [24, 28],
            'target_and_random_policy_executed_at_layer24': True,
            'all_edited_requests_have_positive_active_rows_and_finite_weight_l1': True,
            'all_edited_requests_have_exact_branch_relative_segment_start': 0,
        },
        'first_stage_gate': {
            'target_bias1_total_membership_changes_minimum': 1,
            'target_bias1_total_actual_minus_native_target_hits_minimum': 1,
            'random_bias1_total_membership_changes_minimum': 1,
            'meaning': 'Confirm new layer24 action reaches the actual top-k router at least once; not semantic control or a population effect.',
        },
        'descriptive_comparisons': ['first-generated-token expert159 selected mask by native/target/random condition',
                                    'paired native-sentinel target occupancy at layer24 through common generated-text prefix',
                                    'dose active_rows, target_mass, weight_l1, native/actual target hits and membership changes per rank',
                                    'random expert realized dose and native numerical variation'],
        'prior_native_layer24_expert159_repeat': {
            'pairs': 8, 'first_token_masks_equal': 8,
            'mean_native_fraction': 0.080078125,
            'mean_duplicate_fraction': 0.064453125,
            'mean_absolute_pair_difference': 0.0205078125,
            'maximum_absolute_pair_difference': 0.13671875,
        },
        'consequence': 'A first-stage or hard-gate failure holds the approach action and its 21-family feasibility generation pending prospectively amended expert/timing qualification. Passing is local engineering only; no bitwise engine equivalence or semantic benefit follows.',
    }
    value = {**body, 'sha256': base.digest(body)}
    if OUT.exists():
        if base.sealed(OUT) != value:
            raise ValueError('existing layer24 gates differ')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256']}))


if __name__ == '__main__':
    main()
