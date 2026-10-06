"""Freeze correlation-guided comparisons without reading intervention outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import run_boundary_micro_screen as base

DOC = Path(__file__).resolve().parents[2] / 'report/experimental-resume-v1'
DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
TRANSITIONS = ['candidate_to_verify', 'approach_to_commit']


def save(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('refusing to overwrite a frozen design: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=1, ensure_ascii=False)
            stream.write('\n')
    return value


def expert_sets(dictionary, transition, singleton=None):
    target = next(x['experts'] for x in dictionary['target_templates'] if x['transition'] == transition)
    random = dictionary['matched_random_control_sets'][transition]
    if singleton is None:
        return target, [r['experts'] for r in random]
    if transition != TRANSITIONS[0] or singleton not in (9, 189):
        raise ValueError('unsupported singleton')
    peers = []
    for record in random:
        peer = next(x['peer_expert'] for x in record['native_exposure_by_target_and_peer']
                    if x['target_expert'] == singleton)
        peers.append([[28, [peer]]])
    return [[28, [singleton]]], peers


def design(dictionary, lane):
    transitions = TRANSITIONS[:1] if lane == 'A' else TRANSITIONS
    mode = 'native_operator' if lane == 'C' else 'ordered_positive'
    actions = []
    arms = [{'name': name, 'role': 'native', 'policies': {}, 'slots': []}
            for name in ('native', 'native_duplicate')]
    candidates = ({'pair': None, 'expert189': 189, 'expert9': 9} if lane == 'A' else
                  {'once': None, 'repeat': None} if lane == 'B' else
                  {'bias': None, 'force': None, 'reweight': None})
    for name, singleton in candidates.items():
        count = 2 if name == 'repeat' else 1
        target_policies, random_policies = {}, {}
        for transition in transitions:
            target, peers = expert_sets(dictionary, transition, singleton)
            policy_name = transition + '_' + name
            for suffix, experts in [('', target), *[(f'_random{i}', e) for i, e in enumerate(peers)]]:
                item = {'name': policy_name + suffix, 'transition': transition, 'experts': experts}
                item.update({'kind': name, 'sign': 1, 'magnitude': 0. if name == 'force' else 1.}
                            if lane == 'C' else {'bias': 1.})
                actions.append(item)
            target_policies[transition] = [policy_name] * count
            random_policies[transition] = [policy_name + '_random{random_set}'] * count
        slots = [] if lane == 'C' else [0, 512][:count]
        arms.extend([{'name': name, 'role': 'target', 'policies': target_policies, 'slots': slots},
                     {'name': 'random_' + name, 'role': 'random', 'policies': random_policies, 'slots': slots}])
    contrasts = [[name, control] for name in candidates for control in ('native', 'random_' + name)]
    contrasts.extend([['pair', 'expert189'], ['pair', 'expert9']] if lane == 'A' else
                     [['repeat', 'once']] if lane == 'B' else
                     [['force', 'bias'], ['reweight', 'bias']])
    return {'schema': 'overnight-routing-design-v1', 'lane': lane, 'mode': mode,
            'horizon': 256 if lane == 'C' else 1024, 'transitions': transitions,
            'actions': actions, 'arms': arms, 'planned_contrasts': contrasts,
            'analysis_scopes': ['all', *transitions], 'analysis_seed': 20261004,
            'dictionary_sha256': dictionary['sha256'],
            'bootstrap_replicates': 50000,
            'claim_limit': 'Exploratory discovery prefixes previously examined; no independent confirmation or utility claim.',
            'selection_information': 'Frozen native discovery correlations and existing discovery audits only; no newly completing mechanism outcomes.'}


def fresh_spec(dictionary):
    contrasts = []
    arms = {}
    for transition in TRANSITIONS:
        names = ['native', 'native_duplicate', 'bias', 'random_bias', 'repeat', 'random_repeat',
                 'force', 'random_force', 'reweight', 'random_reweight']
        pairs = [[name, control] for name in ('bias', 'force', 'reweight')
                 for control in ('native', 'random_' + name)]
        pairs += [['force', 'bias'], ['reweight', 'bias'], ['repeat', 'native'],
                  ['repeat', 'random_repeat'], ['repeat', 'bias']]
        if transition == TRANSITIONS[0]:
            names += ['expert189', 'random_expert189', 'expert9', 'random_expert9']
            pairs += [[name, control] for name in ('expert189', 'expert9')
                      for control in ('native', 'random_' + name)]
            pairs += [['bias', 'expert189'], ['bias', 'expert9']]
        arms[transition] = names
        contrasts += [{'transition': transition, 'a': a, 'b': b} for a, b in pairs]
    assert len(contrasts) == 28
    return {'schema': 'overnight-fresh-comparison-protocol-v1', 'dictionary_sha256': dictionary['sha256'],
            'source': 'MECHANISM_EXTENSION_220_FAMILY_FREEZE_v1.json',
            'enrollment': 'First primary two-reader accepted start per family and transition in frozen candidate order; no outcome-dependent replacement.',
            'strict_veto': 'Sensitivity on enrolled starts only; never changes primary enrollment.',
            'arms_by_transition': arms, 'primary_contrasts': contrasts, 'horizon': 1024,
            'seeds': [0, 1], 'pulse_width': 256, 'pulse_slots': {'default': [0], 'repeat': [0, 512]},
            'operators': {'bias': {'sign': 1, 'magnitude': 1.}, 'force': {'sign': 1, 'magnitude': 0.},
                          'reweight': {'sign': 1, 'magnitude': 1.}},
            'bootstrap_replicates': 50000, 'analysis_seed': 20261004,
            'primary': 'Both arm-blind semantic readers accept substantive target transition after trigger within1024tokens.',
            'weighting': 'Equal families within transition, averaging two assigned seeds.',
            'simultaneous_family': 'All28 primary contrasts,95percent Bonferroni family-clustered percentile intervals.',
            'missing': 'All assigned cells retained; two-reader-valid success primary; either-reader and missing bounds reported.',
            'scope': 'All three approaches compared prospectively; no winner selection from13-family screens. No accuracy or token-utility claim.'}


def main():
    dictionary = base.sealed(DICTIONARY)
    outputs = {}
    for lane in ('A', 'B', 'C'):
        path = DOC / f'OVERNIGHT_DISCOVERY_{lane}_DESIGN_v1.json'
        outputs[lane] = save(path, design(dictionary, lane))['sha256']
    outputs['fresh_protocol'] = save(DOC / 'OVERNIGHT_FRESH_COMPARISON_PROTOCOL_v1.json', fresh_spec(dictionary))['sha256']
    print(json.dumps(outputs))


if __name__ == '__main__':
    main()
