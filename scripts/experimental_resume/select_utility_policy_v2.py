"""Freeze utility action choices from the complete fresh-family v2 analysis.

Selection uses semantic effect estimates against native routing, never utility
outcomes, random-control winners, accuracy, token cost or side-screen outcomes.
The result binds local actions; a separately frozen controller and exact GPU
qualification are required before original-prompt utility generation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import utility_scout_v1 as utility
import rate_overnight_semantics_v2 as storage

REPO = utility.REPO
FRESH_DESIGN = REPO / 'report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'
MEASUREMENT_FREEZE = REPO / 'report/experimental-resume-v1/OVERNIGHT_MEASUREMENT_PIPELINE_v2.json'
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit')
require = storage.require


def check_seal(value):
    require(value.get('sha256') == utility.digest({k: v for k, v in value.items() if k != 'sha256'}),
            'input JSON seal differs')


def native_evidence(effect):
    interval = effect['simultaneous_ci95']
    require(isinstance(interval, list) and len(interval) == 2 and
            all(type(v) in (int, float) and math.isfinite(v) for v in interval) and
            interval[0] <= interval[1], 'native comparison lacks finite simultaneous interval')
    estimate = effect['estimate']
    return {'estimate_vs_native': estimate, 'simultaneous_ci95': interval,
            'native_competitive': interval[0] <= 0,
            'native_comparison': ('POSITIVE_INTERVAL' if interval[0] > 0 else
                                  'NEGATIVE_INTERVAL' if interval[1] < 0 else
                                  'INTERVAL_INCLUDES_NO_EFFECT'),
            'positive_point_estimate': estimate > 0,
            'families': effect['families'], 'assigned_starts': effect['assigned_starts'],
            'precision_status': effect['precision_status']}


def select(manifest, analysis, design, measurement_freeze):
    for value in (manifest, analysis, design, measurement_freeze):
        check_seal(value)
    require(manifest['schema'] == 'overnight-routing-manifest-v2' and
            manifest['horizon'] == 1024 and manifest['design_sha256'] == design['sha256'] and
            design['schema'] == 'overnight-routing-design-v2' and
            set(manifest['transitions']) == set(TRANSITIONS) and
            len(manifest['planned_contrasts_scoped']) == 28 and
            all(p['scope'] in TRANSITIONS for p in manifest['planned_contrasts_scoped']),
            'selection requires the fresh 28-contrast common-1024 design')
    require(analysis['schema'] == 'overnight-semantic-itt-v2' and
            analysis['manifest_sha256'] == manifest['sha256'] and analysis['horizon'] == 1024 and
            analysis['assigned'] == manifest['expected_requests'] and
            analysis['starts'] == len(manifest['rows']) and
            analysis['source_enrollment_path'] == manifest['source_enrollment_path'] and
            measurement_freeze['schema'] == 'overnight-measurement-pipeline-freeze-v2' and
            analysis['analysis_driver_sha256'] == measurement_freeze['code_files'][
                str(REPO / 'scripts/experimental_resume/analyze_overnight_semantics_v2.py')],
            'analysis is not complete or is bound to different source/code')
    primary = analysis['primary']
    require(primary['endpoints'] == ['both_positive'] and primary['multiplicity'] == 28 and
            primary['replicates'] == 50000 and primary['seed'] == 20261004 and
            len(primary['contrasts']) == 28, 'frozen primary semantic family differs')
    by_comparison = {(c['scope'], c['arm'], c['reference']): c for c in primary['contrasts']}
    frozen_keys = {(p['scope'], p['left'], p['right']) for p in manifest['planned_contrasts_scoped']}
    require(len(by_comparison) == 28 and set(by_comparison) == frozen_keys and
            all(c['endpoint'] == 'both_positive' for c in primary['contrasts']),
            'primary contrasts omit or add assigned comparisons')
    actions = {a['name']: a for a in manifest['actions']}
    selections, rankings, omitted = {}, {}, {}
    for transition in TRANSITIONS:
        native_comparisons = [by_comparison.get((transition, arm['name'], 'native'))
                              for arm in manifest['arms_by_transition'][transition] if arm['role'] == 'target']
        require(native_comparisons and all(c is not None for c in native_comparisons),
                'target lacks frozen native comparison')
        if all(c['estimate'] is None and c.get('precision_status') == 'INSUFFICIENT_FAMILIES'
               for c in native_comparisons):
            omitted[transition] = {'reason': 'INSUFFICIENT_FAMILIES',
                                   'frozen_native_contrasts': native_comparisons}
            rankings[transition] = []
            continue
        ranked = []
        for position, arm in enumerate(manifest['arms_by_transition'][transition]):
            if arm['role'] != 'target':
                continue
            names = arm['policies'][transition]
            require(names and all(name in actions and '{' not in name and
                    actions[name]['transition'] == transition for name in names),
                    'selected action is missing, random, or from another transition')
            effect = by_comparison.get((transition, arm['name'], 'native'))
            require(effect is not None and type(effect['estimate']) in (int, float) and
                    math.isfinite(effect['estimate']), 'target lacks estimable frozen native contrast')
            experts = sorted({(layer, expert) for name in names
                              for layer, ids in actions[name]['experts'] for expert in ids})
            require(arm['slots'] in ([0], [0, 512]) and len(names) == len(arm['slots']),
                    'target pulse schedule is not one of the frozen local templates')
            related = [c for c in primary['contrasts'] if c['scope'] == transition and
                       (c['arm'] == arm['name'] or c['reference'] == arm['name'])]
            ranked.append({'arm': arm['name'], 'role': 'target', 'manifest_order': position,
                'selection_label': 'best estimated tested intervention when ranked first; utility benefit untested',
                'transition': transition, 'slots': arm['slots'], 'pulse_width': manifest['pulse_width'],
                'action_names': names, 'actions': [actions[n] for n in names],
                'unique_expert_identities': [list(x) for x in experts],
                'pulse_count': len(arm['slots']), 'expert_count': len(experts),
                'native_evidence': native_evidence(effect), 'related_frozen_contrasts': related})
        require(ranked, 'no frozen target arm for transition')
        ranked.sort(key=lambda row: (-row['native_evidence']['estimate_vs_native'],
                    row['pulse_count'], row['expert_count'], row['manifest_order']))
        selections[transition] = ranked[0]
        rankings[transition] = ranked
    return utility.seal({'schema': 'routing-frozen-utility-policy-v2',
        'status': ('FROZEN_LOCAL_ACTIONS_CONTROLLER_AND_ENGINE_REQUIRED' if selections else
                   'HOLD_NO_SUPPORTED_LOCAL_ACTION'),
        'selection_rule': 'Maximum semantic effect estimate vs native among frozen target arms per transition; exact ties fewer pulses, fewer unique layer/expert identities, fixed manifest order.',
        'selection_driver_sha256': utility.file_sha(Path(__file__)),
        'mechanism_manifest_sha256': manifest['sha256'], 'mechanism_analysis_sha256': analysis['sha256'],
        'fresh_design_sha256': design['sha256'], 'measurement_pipeline_sha256': measurement_freeze['sha256'],
        'source_enrollment_path': manifest['source_enrollment_path'],
        'source_enrollment_sha256': manifest['source_enrollment_sha256'],
        'source_frame_sha256': manifest['source_frame_sha256'],
        'selections': selections, 'all_target_rankings': rankings, 'omitted_transitions': omitted,
        'selection_horizon': 1024, 'utility_maximum_output_tokens': 16384,
        'utility_outcomes_used': False, 'random_arms_eligible_for_selection': False,
        'claim_limit': 'Selection makes no winner, replication or utility-benefit claim. Intervals describe the frozen preselection contrasts; the largest point estimate may be noise. Native can remain competitive even when an action is selected. Online timing and reader costs require separate qualification and utility evaluation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'analysis', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = select(utility.sealed(args.manifest), utility.sealed(args.analysis),
                    utility.sealed(FRESH_DESIGN), utility.sealed(MEASUREMENT_FREEZE))
    body = {k: v for k, v in result.items() if k != 'sha256'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    storage.save(args.out, body, existing_ok=True)
    print(result['status'], result['sha256'], flush=True)


if __name__ == '__main__':
    main()
