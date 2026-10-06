"""Seal a price-only +2/+4 rescue design; current ordered worker rejects it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SOURCE = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_STRONG_BIAS_PREPARED_v0.1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    data = json.loads(path.read_text())
    if digest({k: v for k, v in data.items() if k != 'sha256'}) != data['sha256']:
        raise ValueError('source seal differs')
    return data


def main():
    source = sealed(SOURCE)
    if (source['stage'] != 'pilot' or source['expected_requests'] != 48 or
        len(source['rows']) != 4 or len(source['arms']) != 6 or
        source['sha256'] != 'ac4c9651e71fe067726b467c40d7dd5a2bf8c22538c85fc88381f8615b13fcaf'):
        raise ValueError('qualified four-family source changed')
    actions = []
    for name, ids in [('target', [189, 9]),
                      ('random0', [139, 120]), ('random1', [255, 133]),
                      ('random2', [43, 5]), ('random3', [196, 24])]:
        for dose in (2., 4.):
            actions.append({'name': f'{name}_bias{dose:g}',
                            'transition': 'candidate_to_verify',
                            'experts': [[28, ids]], 'bias': dose})
    arms = [{'name': 'native', 'policy': 'zero', 'role': 'native'},
            {'name': 'native_duplicate', 'policy': 'zero', 'role': 'native'},
            {'name': 'target_bias2', 'policy': 'target_bias2', 'role': 'target'},
            {'name': 'target_bias4', 'policy': 'target_bias4', 'role': 'target'},
            {'name': 'random_bias2', 'policy': 'random_selector_bias2', 'role': 'random'},
            {'name': 'random_bias4', 'policy': 'random_selector_bias4', 'role': 'random'}]
    body = {'schema': 'routing-boundary-strong-bias-prepared-v1',
            'status': 'HOLD_ORDERED_WORKER_AMENDMENT_ENGINE_QUALIFICATION_AND_MEASURED_PRICE',
            'conditional_trigger': 'pilot +0.5/+1 has verified executed dose but no target expert top-k entry',
            'source_pilot_manifest_sha256': source['sha256'],
            'source_driver_sha256': source['driver_sha256'],
            'source_qualified_worker_sha256': source['qualified_worker_sha256'],
            'source_family_freeze_sha256': source['family_freeze_sha256'],
            'source_prefix_scout_sha256': source['prefix_scout_sha256'],
            'source_prepared_action_sha256': source['prepared_sha256'],
            'rows': source['rows'], 'random_set_by_family_seed': source['random_set_by_family_seed'],
            'actions': actions, 'arms': arms, 'seeds': source['seeds'],
            'max_tokens': source['max_tokens'],
            'expected_requests': source['expected_requests'],
            'expected_prefill_tokens': source['expected_prefill_tokens'],
            'maximum_decode_tokens': source['maximum_decode_tokens'],
            'maximum_context_tokens': source['maximum_context_tokens'],
            'proposed_new_engineering_qualification_ceiling_GPU_hours': 0.5,
            'proposed_new_generation_ceiling_GPU_hours': 1.0,
            'complete_stage_price_status': 'PENDING_MEASURED_PILOT_RUNTIME_AND_NEW_WORKER_QUALIFICATION',
            'not_executable_reason': 'qualified ordered worker_adapter accepts positive bias only at 0.5 or 1.0; Action.validate and current runner also reject 2.0/4.0',
            'interpretation': 'discovery-only rescue proposal, not a causal result or approved job'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing price-only proposal differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'requests': value['expected_requests'],
                      'prefill': value['expected_prefill_tokens'],
                      'decode': value['maximum_decode_tokens']}))


if __name__ == '__main__':
    main()
