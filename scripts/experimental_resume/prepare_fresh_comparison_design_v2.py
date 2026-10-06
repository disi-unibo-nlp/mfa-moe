"""Bind the approved fresh comparison to explicit actions before eligibility outcomes."""
from pathlib import Path
import json

from prepare_overnight_designs_v1 import DOC, TRANSITIONS, expert_sets, save
import run_boundary_micro_screen as base


def main():
    protocol = base.sealed(DOC / 'OVERNIGHT_FRESH_COMPARISON_PROTOCOL_v1.json')
    dictionary = base.sealed(DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json')
    if protocol['dictionary_sha256'] != dictionary['sha256']:
        raise ValueError('protocol expert dictionary changed')
    actions, arms_by = [], {}
    for transition in TRANSITIONS:
        variants = [('bias', None), ('force', None), ('reweight', None)]
        if transition == TRANSITIONS[0]:
            variants += [('expert189', 189), ('expert9', 9)]
        for name, singleton in variants:
            target, peers = expert_sets(dictionary, transition, singleton)
            kind = name if name in ('force', 'reweight') else 'bias'
            for suffix, experts in [('', target), *[(f'_random{i}', e) for i, e in enumerate(peers)]]:
                actions.append({'name': transition + '_' + name + suffix, 'transition': transition,
                                'experts': experts, 'kind': kind, 'sign': 1,
                                'magnitude': 0. if kind == 'force' else 1.})
        arms = []
        for name in protocol['arms_by_transition'][transition]:
            if name.startswith('native'):
                arms.append({'name': name, 'role': 'native', 'policies': {}, 'slots': []})
                continue
            random = name.startswith('random_')
            variant = name.removeprefix('random_')
            slots = [0, 512] if variant == 'repeat' else [0]
            policy = transition + '_' + ('bias' if variant == 'repeat' else variant)
            if random:
                policy += '_random{random_set}'
            arms.append({'name': name, 'role': 'random' if random else 'target',
                         'policies': {transition: [policy] * len(slots)}, 'slots': slots})
        arms_by[transition] = arms
    body = {'schema': 'overnight-routing-design-v2', 'protocol_sha256': protocol['sha256'],
            'dictionary_sha256': dictionary['sha256'], 'horizon': 1024, 'transitions': TRANSITIONS,
            'source_enrollment_path': str(DOC / 'MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_v1.json'),
            'source_frame_path': '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-mechanism-extension-220/MECHANISM_EXTENSION_START_FRAME_v1.json',
            'actions': actions, 'arms_by_transition': arms_by,
            'planned_contrasts': protocol['primary_contrasts'],
            'analysis_scopes': ['all', *TRANSITIONS], 'analysis_seed': protocol['analysis_seed'],
            'bootstrap_replicates': protocol['bootstrap_replicates'],
            'claim_limit': protocol['scope']}
    path = DOC / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'
    value = save(path, body)
    print(json.dumps({'design': str(path), 'sha256': value['sha256'],
                      'arms': {k: len(v) for k, v in arms_by.items()}, 'contrasts': len(body['planned_contrasts'])}))


if __name__ == '__main__':
    main()
