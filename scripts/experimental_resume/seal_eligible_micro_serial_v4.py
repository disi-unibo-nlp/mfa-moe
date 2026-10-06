"""Freeze counterbalanced eligible discovery assignments and serial GPU price."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import run_boundary_micro_screen as base
import run_eligible_micro_serial_v3 as screen

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
PILOT = DOC / 'CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
SERIAL_PRICE = DOC / 'CAUSAL_MICRO_SERIAL_QUAL_PRICE_v1.json'
OUT = screen.MANIFEST
PRICE_OUT = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v3.json'
ARMS = [
    {'name': 'native', 'role': 'native'},
    {'name': 'native_duplicate', 'role': 'native'},
    {'name': 'target_bias0.5', 'role': 'target'},
    {'name': 'target_bias1', 'role': 'target'},
    {'name': 'random_bias0.5', 'role': 'random'},
    {'name': 'random_bias1', 'role': 'random'},
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
        for bias, suffix in ((0.5, '0.5'), (1.0, '1')):
            actions.append({'name': f'{stem}_target_bias{suffix}', 'transition': transition,
                            'experts': template['experts'], 'bias': bias})
            for index, control in enumerate(controls):
                actions.append({'name': f'{stem}_random{index}_bias{suffix}',
                                'transition': transition, 'experts': control['experts'], 'bias': bias})
    schedule, arm_orders = screen.expected_control_schedules(rows, ARMS)
    reqs = len(rows) * 2 * len(ARMS)
    prefill = sum(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 2 * len(ARMS)
    decode = reqs * 256
    maximum_context = max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256 for r in rows)
    code_files = dict(pilot['code_files'])
    code_files[str(Path(screen.__file__))] = base.file_sha(screen.__file__)
    code_files[str(Path(base.__file__))] = base.file_sha(base.__file__)
    body = {
        'schema': 'routing-eligible-micro-serial-v3',
        'scope': 'exploratory same-prefix causal screen in 13 discovery families; six-arm serial positions counterbalanced; semantic outcome rating follows generation',
        'source_pool_sha256': pool['sha256'],
        'action_dictionary_sha256': dictionary['sha256'],
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
        'interpretation': 'Local discovery action screen only. Its starts are LLM-audited; no human truth, mechanism validation, accuracy or token utility is established by this manifest.'
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
        'schema': 'routing-eligible-micro-serial-price-v3',
        'manifest_sha256': manifest['sha256'],
        'measured_engineering_reference_job': '59200002',
        'prior_prospective_price_sha256': reference['sha256'],
        'expected_requests': reqs, 'checkpoint_batches': reqs // 6,
        'arm_order_design': 'deterministic six-arm cyclic counterbalance per transition and globally; exact schedule is sealed in the manifest',
        'random_control_design': 'four sets balanced separately per transition and seed; exact schedule is sealed in the manifest',
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
        'gate': 'exact pool/worker hashes, CPU request preflight, serial 59200002 audit, Slurm acceptance; downstream blinded rating separately priced',
        'interpretation': 'Discovery maximum support is 13 globally distinct families, below the registered 48-family maximum; this is a labeled small causal screen, not validation.'
    })
    print(json.dumps({'manifest': str(OUT), 'sha256': manifest['sha256'],
                      'price': str(PRICE_OUT), 'price_sha256': price['sha256'],
                      'families': len(rows), 'transition_counts': dict(Counter(r['transition'] for r in rows)),
                      'requests': reqs, 'prefill': prefill, 'decode': decode,
                      'projected_wall_seconds': projected}))


if __name__ == '__main__':
    main()
