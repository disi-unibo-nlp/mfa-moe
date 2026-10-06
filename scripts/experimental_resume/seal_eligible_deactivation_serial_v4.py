"""Freeze counterbalanced deactivation assignments and serial GPU price."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_eligible_deactivation_serial_v4 as screen

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
PILOT = DOC / 'CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
SERIAL_PRICE = DOC / 'CAUSAL_MICRO_SERIAL_QUAL_PRICE_v1.json'
OUT = screen.MANIFEST
PRICE_OUT = DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_PRICE_v4.json'
INVALID_V1 = DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v1.json'
FAILED_V2 = DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v2.json'
POSITIVE = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json'
OLD_V3 = DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v3.json'
ARMS = [
    {'name': 'native', 'role': 'native'},
    {'name': 'native_duplicate', 'role': 'native'},
    {'name': 'target_bias_minus1', 'role': 'target'},
    {'name': 'target_force_off', 'role': 'target'},
    {'name': 'random_bias_minus1', 'role': 'random'},
    {'name': 'random_force_off', 'role': 'random'},
]


def write_once(path, payload):
    value = {**payload, 'sha256': base.digest(payload)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('sealed output already differs: ' + str(path))
    else:
        path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    return value


def main():
    pool = base.sealed(screen.POOL)
    dictionary = base.sealed(screen.DICTIONARY)
    pilot = base.sealed(PILOT)
    family = base.sealed(base.FAMILY_FREEZE)
    reference = base.sealed(SERIAL_PRICE)
    positive = base.sealed(POSITIVE)
    old_v3 = base.sealed(OLD_V3)
    invalid_v1 = base.sealed(INVALID_V1)
    failed_v2 = base.sealed(FAILED_V2)
    if pool.get('schema') != 'joint-qwen-native-exact-pool-v1':
        raise ValueError('unexpected exact high-agreement pool schema')
    rows = screen.expected_rows(pool)
    if set(pool.get('selected_global_uids', [])) != {r['uid'] for r in rows}:
        raise ValueError('global family-disjoint selection disagrees with sealed pool')
    if Counter(r['transition'] for r in rows) != {'candidate_to_verify': 8, 'approach_to_commit': 5}:
        raise ValueError('high-agreement transition enrollment differs')
    actions = []
    for transition, stem in (('candidate_to_verify', 'verify'), ('approach_to_commit', 'commit')):
        template = next(t for t in dictionary['target_templates'] if t['transition'] == transition)
        controls = dictionary['matched_random_control_sets'][transition]
        for kind, magnitude, suffix in (('bias', 1.0, 'bias_minus1'), ('force', 0.0, 'force_off')):
            actions.append({'name': f'{stem}_target_{suffix}', 'transition': transition,
                            'experts': template['experts'], 'kind': kind,
                            'sign': -1, 'magnitude': magnitude})
            for index, control in enumerate(controls):
                actions.append({'name': f'{stem}_random{index}_{suffix}',
                                'transition': transition, 'experts': control['experts'],
                                'kind': kind, 'sign': -1, 'magnitude': magnitude})
    if positive['rows'] != rows:
        raise ValueError('positive and deactivation diagnostic do not share exact assigned starts')
    schedule, arm_orders = screen.expected_control_schedules(rows, ARMS, positive)
    reqs = len(rows) * 2 * len(ARMS)
    prefill = sum(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 2 * len(ARMS)
    decode = reqs * 256
    maximum_context = max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256 for r in rows)
    code_files = dict(pilot['code_files'])
    code_files[str(Path(screen.__file__))] = base.file_sha(screen.__file__)
    code_files[str(Path(base.__file__))] = base.file_sha(base.__file__)
    body = {
        'schema': 'routing-eligible-deactivation-serial-v4',
        'scope': 'counterbalanced exploratory necessity diagnostic in 13 discovery families; finite negative bias and hard target exclusion, each with exposure-matched random control',
        'execution_mode': 'base-hook-always-256-no-ordered-metadata',
        'supersedes_invalid_queued_manifest_sha256': invalid_v1['sha256'],
        'supersedes_failed_cpu_preflight_manifest_sha256': failed_v2['sha256'],
        'supersedes_uncounterbalanced_v3_manifest_sha256': old_v3['sha256'],
        'base_hook_dose_rule': 'Use per-layer base counters and emitted routed arrays to measure executed expert removal; ordered_action_dose is absent when ordered metadata is omitted.',
        'source_pool_sha256': pool['sha256'],
        'action_dictionary_sha256': dictionary['sha256'],
        'positive_immediate_manifest_sha256': positive['sha256'],
        'family_freeze_sha256': family['sha256'],
        'base_tree_sha256': base.REQUIRED_BASE_TREE,
        'qualified_worker_sha256': pilot['qualified_worker_sha256'],
        'entry_driver_sha256': base.file_sha(screen.__file__),
        'driver_sha256': base.file_sha(base.__file__),
        'code_files': code_files,
        'engine_profile': screen.PROFILE,
        'rows': rows, 'actions': actions, 'arms': ARMS,
        'random_set_by_family_seed': schedule,
        'arm_order_by_family_seed': arm_orders,
        'seeds': [0, 1], 'max_tokens': 256,
        'expected_requests': reqs,
        'expected_prefill_tokens': prefill,
        'maximum_decode_tokens': decode,
        'maximum_context_tokens': maximum_context,
        'assigned_outcome_rule': 'All assigned requests retained, including native duplicates, caps, natural stops, failures and nonfire; independent arm-blind semantic ratings required.',
        'interpretation': 'Exploratory necessity diagnostic, distinct from registered positive action dictionary. Native top-k remains eight and the shared expert is unchanged. Negative finite bias may fail to remove a strongly selected expert; assess actual executed dose. Immediate 256-token outcome only.'
    }
    manifest = write_once(OUT, body)
    screen.validate(manifest, base.__file__)
    projected = (reference['cold_load_seconds'] +
                 reference['repeat_factor'] *
                 (prefill / reference['serial_prefill_stress_tokens_per_second'] +
                  decode / reference['serial_decode_stress_tokens_per_second']) +
                 reference['shutdown_seconds'])
    wall = 10800
    if projected > wall - 900:
        raise ValueError('complete serial screen exceeds 3h ceiling; reprice before submission')
    price = write_once(PRICE_OUT, {
        'schema': 'routing-eligible-deactivation-serial-price-v4',
        'manifest_sha256': manifest['sha256'],
        'measured_engineering_reference_job': '59200002',
        'prior_prospective_price_sha256': reference['sha256'],
        'expected_requests': reqs, 'checkpoint_batches': reqs // 6,
        'arm_order_design': 'same mapped six-arm slot balance as positive v3; exact schedule sealed in manifest',
        'random_control_design': 'exact positive v3 sets, balanced separately per transition and seed',
        'expected_prefill_tokens': prefill, 'maximum_decode_tokens': decode,
        'maximum_context_tokens': maximum_context,
        'cold_load_seconds': reference['cold_load_seconds'],
        'serial_prefill_stress_tokens_per_second': reference['serial_prefill_stress_tokens_per_second'],
        'serial_decode_stress_tokens_per_second': reference['serial_decode_stress_tokens_per_second'],
        'repeat_factor': reference['repeat_factor'],
        'shutdown_seconds': reference['shutdown_seconds'],
        'estimated_complete_wall_seconds': projected,
        'requested_wall_seconds': wall, 'gpus': 2, 'gpu_hour_ceiling': 6.0,
        'all_loads_prefill_decode_retries_shutdown_charged': True,
        'gate': 'exact pool/worker hashes, CPU negative-kernel compilation and no-ordered-metadata request preflight, base-hook closure/timing checks, serial 59200002 audit, Slurm acceptance; downstream blinded rating separately priced',
        'interpretation': 'Separate exploratory negative-bias/force-off diagnostic on 13 globally distinct discovery families; matched random and duplicate-native arms are included. This is not registered 48-family discovery or independent validation.'
    })
    print(json.dumps({'manifest': str(OUT), 'sha256': manifest['sha256'],
                      'price': str(PRICE_OUT), 'price_sha256': price['sha256'],
                      'families': len(rows), 'transition_counts': dict(Counter(r['transition'] for r in rows)),
                      'requests': reqs, 'prefill': prefill, 'decode': decode,
                      'projected_wall_seconds': projected}))


if __name__ == '__main__':
    main()
