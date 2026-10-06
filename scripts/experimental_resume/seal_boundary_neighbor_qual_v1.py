"""Freeze matched sentinel engineering workload and outcome-independent gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_boundary_neighbor_qual_v1 as neighbor

REPO = neighbor.REPO
REPORT = REPO / 'report/experimental-resume-v1'
MANIFEST = REPORT / 'CAUSAL_BATCHED_NEIGHBOR_QUAL_MANIFEST_v1.json'
PRICE = REPORT / 'CAUSAL_BATCHED_NEIGHBOR_QUAL_PRICE_v1.json'
GATES = REPORT / 'CAUSAL_BATCHED_NEIGHBOR_QUAL_GATES_v1.json'
NATIVE_REPLAY = REPORT / 'CAUSAL_NATIVE_ONLY_REPLAY_AUDIT_v3.json'


def write_once(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('sealed artifact differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1) + '\n')
    return value


def main():
    pilot = base.sealed(neighbor.PILOT)
    replay = base.sealed(NATIVE_REPLAY)
    entry = str(Path(neighbor.__file__).resolve())
    codes = dict(pilot['code_files'])
    codes[entry] = base.file_sha(entry)
    rows = pilot['rows']
    requests = len(rows) * len(pilot['seeds']) * len(neighbor.CONDITIONS) * 2
    prefill = sum((len(r['prompt_ids']) + len(r['prefix_ids'])) * 20 for r in rows)
    manifest_body = {
        'schema': 'routing-boundary-neighbor-qualification-v1',
        'source_pilot_sha256': pilot['sha256'],
        'entry_driver_sha256': codes[entry],
        'driver_sha256': pilot['driver_sha256'],
        'code_files': codes,
        'profile': {'max_num_seqs': 8, 'enforce_eager': False,
                    'VLLM_BATCH_INVARIANT': 0, 'batch_size': 8},
        'conditions': neighbor.CONDITIONS,
        'condition_order_by_seed': {'0': [x['name'] for x in neighbor.CONDITIONS],
                                    '1': [x['name'] for x in reversed(neighbor.CONDITIONS)]},
        'source_native_replay_audit_sha256': replay['sha256'],
        'base_tree_sha256': pilot['base_tree_sha256'],
        'family_freeze_sha256': pilot['family_freeze_sha256'],
        'prefix_scout_sha256': pilot['prefix_scout_sha256'],
        'prepared_sha256': pilot['prepared_sha256'],
        'qualified_worker_sha256': pilot['qualified_worker_sha256'],
        'rows': pilot['rows'], 'actions': pilot['actions'],
        'arms': [{'name': 'active', 'policy': 'condition', 'role': 'conditional'},
                 {'name': 'sentinel', 'policy': 'zero', 'role': 'native'}],
        'seeds': pilot['seeds'], 'max_tokens': pilot['max_tokens'],
        'stage': pilot['stage'],
        'random_set_by_family_seed': pilot['random_set_by_family_seed'],
        'expected_requests': requests,
        'expected_prefill_tokens': prefill,
        'maximum_decode_tokens': requests * pilot['max_tokens'],
        'maximum_context_tokens': pilot['maximum_context_tokens'],
        'scope': 'Engineering only on four semantically ineligible prior pilot starts; no causal semantic claim',
    }
    manifest = write_once(MANIFEST, manifest_body)
    gate_body = {
        'schema': 'routing-boundary-neighbor-qualification-gates-v1',
        'manifest_sha256': manifest['sha256'],
        'native_only_replay_audit_sha256': replay['sha256'],
        'registered_population': 'four frozen semantically ineligible pilot families, two seeds',
        'prior_native_replay_baseline': {
            'pairs': replay['native_only_summary']['pairs'],
            'first_token_target_mask_equal_pairs': replay['native_only_summary']['first_token_target_mask_equal'],
            'first_token_unordered_topk_equal_pairs': replay['native_only_summary']['first_token_unordered_topk_equal'],
            'tokens_equal_pairs': replay['native_only_summary']['tokens_equal'],
        },
        'hard_engine_gates': {
            'completed_assigned_requests': requests,
            'completed_checkpoint_batches': 10,
            'missing_routed_arrays_or_telemetry': 0,
            'inactive_native_expert_identity_mismatches_per_token': 0,
            'inactive_native_weight_mismatches_per_token': 0,
            'native_sentinel_cpu_active_rows': 0,
            'policy_table_hash_matches_original_pilot': True,
        },
        'behavioral_sensitivity_gates': {
            'comparison': 'At first generated token, compare the same family and seed zero-policy sentinel in each edited-condition batch against the zero-condition sentinel; also compare first emitted token. Four edited conditions each have eight paired sentinels.',
            'maximum_target_mask_discordant_pairs_per_edited_condition': 2,
            'maximum_first_emitted_token_discordant_pairs_per_edited_condition': 2,
            'baseline_zero_condition_active_vs_sentinel_reported_separately': True,
            'unordered_topk_set_discordance': 'report per condition and compare with the frozen native-only replay 8-pair baseline; no equality gate because native-only pairs already diverged at token zero',
            'all_256_token_trajectories': 'report overlap only until first text divergence; report family-clustered descriptive intervals, not a bitwise identity gate',
        },
        'consequence': 'Any hard-gate failure or >2/8 behavioral discordance in any condition holds this batched profile; serial/eager is fallback. Passing is a limited four-family engineering screen, not engine equivalence or tight population bound.',
        'provenance': 'All gates fixed before native-sentinel qualification outputs; tolerance 2/8 is an engineering sensitivity threshold informed by 0/8 target-mask and first-token-token discordance in native-only replay, not chosen from edited-neighbor results.',
    }
    gates = write_once(GATES, gate_body)
    # The slower historical serial 256-token batches give the fallback price;
    # this faster profile is conservatively planned at 50 aggregate tok/s.
    load, shutdown = 685.0234088897705, 196.
    decode_rate, prefill_rate, repeat = 50., 1000., 1.25
    estimate = load + shutdown + repeat * (prefill / prefill_rate +
                                            manifest['maximum_decode_tokens'] / decode_rate)
    price_body = {
        'schema': 'routing-boundary-neighbor-qualification-price-v1',
        'manifest_sha256': manifest['sha256'], 'gates_sha256': gates['sha256'],
        'source_serial_qualification_job': '59200002',
        'source_batched_pilot_job': '59180099',
        'expected_requests': requests, 'checkpoint_batches': 10,
        'expected_prefill_tokens': prefill,
        'maximum_decode_tokens': manifest['maximum_decode_tokens'],
        'stress_aggregate_decode_tokens_per_second': decode_rate,
        'stress_prefill_tokens_per_second': prefill_rate,
        'repeat_factor': repeat, 'cold_load_seconds': load,
        'shutdown_seconds': shutdown,
        'estimated_complete_wall_seconds': estimate,
        'requested_wall_seconds': 3600, 'gpus': 2,
        'gpu_hour_ceiling': 2.,
        'pricing_status': 'Prospective conservative engineering price; no causal enrollment or semantic evaluation',
    }
    price = write_once(PRICE, price_body)
    print(json.dumps({'manifest': str(MANIFEST), 'manifest_sha256': manifest['sha256'],
                      'gates': str(GATES), 'gates_sha256': gates['sha256'],
                      'price': str(PRICE), 'price_sha256': price['sha256'],
                      'estimate_seconds': estimate}))


if __name__ == '__main__':
    main()
