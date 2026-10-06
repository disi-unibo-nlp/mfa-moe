"""Freeze two discovery-only expert actions and exposure-matched controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
BASE = ROOT / 'runs/routing-control-v1'
CANDIDATE = BASE / 'dense-verify-source-v2/RESULT.json'
APPROACH = BASE / 'approach-boundary-supported-v2/RESULT.json'
EXPOSURE = BASE / 'dense-discovery/route-profiles-84a85c92-40c3d3b0/NATIVE_ROUTE_SUMMARY.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
RANDOM = {'candidate_to_verify': [[139, 120], [255, 133], [43, 5], [196, 24]]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('source JSON seal differs: ' + str(path))
    return value


def one_action(transition, path, expected, layer, exposure, random_sets):
    source = sealed(path)
    selected = source['proposal']['experts']
    if [(x['layer'], x['expert']) for x in selected] != [(layer, e) for e in expected]:
        raise ValueError('native contrast selected experts changed')
    target = {'transition': transition, 'experts': [[layer, sorted(expected)]],
              'biases': [0.5, 1.0], 'source_sha256': source['sha256'],
              'source_population': source['population'],
              'source_support': source['support'],
              'source_fold_effects': source['four_fold_family_crossfit']}
    peers = []
    for index, ids in enumerate(random_sets):
        if len(set(ids)) != len(expected) or set(ids) & set(expected):
            raise ValueError('invalid frozen matched-random set')
        if any(abs(exposure[layer][peer_id] - exposure[layer][target_id]) >
               .1 * exposure[layer][target_id]
               for target_id, peer_id in zip(expected, ids, strict=True)):
            raise ValueError('random set exceeds ±10% native exposure match')
        peers.append({'set_index': index, 'experts': [[layer, sorted(ids)]],
                      'native_exposure_by_target_and_peer': [
                          {'target_expert': target_id, 'target_exposure': exposure[layer][target_id],
                           'peer_expert': peer_id, 'peer_exposure': exposure[layer][peer_id],
                           'relative_exposure_gap': abs(exposure[layer][peer_id] - exposure[layer][target_id]) /
                           exposure[layer][target_id]}
                          for target_id, peer_id in zip(expected, ids, strict=True)]})
    return target, peers


def main():
    exposure_source = sealed(EXPOSURE)
    matrix = exposure_source['native_expert_selection_exposure']
    if len(matrix) != 40 or any(len(row) != 256 for row in matrix):
        raise ValueError('native exposure shape differs')
    candidate, candidate_random = one_action('candidate_to_verify', CANDIDATE,
                                             [189, 9], 28, matrix,
                                             RANDOM['candidate_to_verify'])
    approach_source = sealed(APPROACH)
    if approach_source['hold_for_causal_test'] or not approach_source['random_exposure_supported']:
        raise ValueError('supported approach source did not pass prospective discovery gate')
    approach_expert = approach_source['proposal']['experts'][0]
    approach_layer, approach_id = approach_expert['layer'], approach_expert['expert']
    approach_peers = approach_expert['first_four_matched_random_experts']
    if len(approach_peers) != 4:
        raise ValueError('four exposure-matched approach controls unavailable')
    approach, approach_random = one_action('approach_to_commit', APPROACH,
                                           [approach_id], approach_layer, matrix,
                                           [[peer] for peer in approach_peers])
    if candidate['source_support']['within_family_matched_positive_families'] != 48:
        raise ValueError('candidate native support differs')
    if approach['source_support']['matched_positive_families'] != 20:
        raise ValueError('approach native support differs')
    targets = [candidate, approach]
    body = {'schema': 'routing-discovery-action-dictionary-v1',
            'version': 1, 'max_candidate_action_definitions': 6,
            'actual_candidate_action_definitions': 4,
            'native_top_k': 8, 'shared_expert_preserved': True,
            'discovery_family_pool_size': 48,
            'native_exposure_source_sha256': exposure_source['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'target_templates': targets,
            'matched_random_control_sets': {
                'candidate_to_verify': candidate_random,
                'approach_to_commit': approach_random},
            'eligibility_gate': 'independent full-prefix pretreatment start audit and globally family-disjoint enrollment; no intervention outcomes used for selecting these expert IDs',
            'approach_weakness': 'Explore→Plan/Implement class proxy, not identified approach→semantic commitment; a single supported expert replaces the infeasible two-expert pair; held-out fold signs are reported, not a confirmatory efficacy claim',
            'interpretation': 'observational routing contrasts select sparse proposals, not causal expert importance or optimal trajectories; evaluate biases +0.5 and +1.0 against native duplicates and frozen random sets'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('frozen discovery action dictionary differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'target_templates': len(targets),
                      'target_dose_definitions': value['actual_candidate_action_definitions'],
                      'approach_max_random_exposure_gap': max(
                          x['relative_exposure_gap'] for p in approach_random
                          for x in p['native_exposure_by_target_and_peer'])}))


if __name__ == '__main__':
    main()
