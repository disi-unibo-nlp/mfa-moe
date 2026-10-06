"""Selection is complete-analysis bound, deterministic and target-only."""
import copy
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import select_utility_policy_v2 as selector


def sealed(body):
    return selector.utility.seal({k: v for k, v in body.items() if k != 'sha256'})


def fixtures():
    design = selector.utility.sealed(selector.FRESH_DESIGN)
    freeze = selector.utility.sealed(selector.MEASUREMENT_FREEZE)
    planned = [{'scope': c['transition'], 'left': c['a'], 'right': c['b']}
               for c in design['planned_contrasts']]
    manifest = sealed({'schema': 'overnight-routing-manifest-v2', 'design_sha256': design['sha256'],
        'horizon': 1024, 'transitions': design['transitions'], 'planned_contrasts_scoped': planned,
        'expected_requests': 480, 'rows': [{}] * 20, 'source_enrollment_path': '/fresh/enrollment',
        'source_enrollment_sha256': 'enrollment-sha', 'source_frame_sha256': 'frame-sha',
        'actions': design['actions'], 'arms_by_transition': design['arms_by_transition'], 'pulse_width': 256})
    contrasts = [{'scope': c['scope'], 'arm': c['left'], 'reference': c['right'],
        'endpoint': 'both_positive', 'estimate': .1, 'simultaneous_ci95': [-.1, .3],
        'families': 10, 'assigned_starts': 10, 'precision_status': 'FAMILY_BOOTSTRAP_APPROXIMATION'}
        for c in planned]
    analysis = sealed({'schema': 'overnight-semantic-itt-v2', 'manifest_sha256': manifest['sha256'],
        'horizon': 1024, 'assigned': 480, 'starts': 20, 'source_enrollment_path': '/fresh/enrollment',
        'analysis_driver_sha256': freeze['code_files'][str(REPO /
            'scripts/experimental_resume/analyze_overnight_semantics_v2.py')],
        'primary': {'endpoints': ['both_positive'], 'multiplicity': 28,
                    'replicates': 50000, 'seed': 20261004, 'contrasts': contrasts}})
    return manifest, analysis, design, freeze


class SelectionTests(unittest.TestCase):
    def test_exact_ties_use_pulses_experts_manifest_order(self):
        result = selector.select(*fixtures())
        self.assertEqual(result['selections']['candidate_to_verify']['arm'], 'expert189')
        self.assertEqual(result['selections']['approach_to_commit']['arm'], 'bias')
        self.assertTrue(result['selections']['candidate_to_verify']['native_evidence']['native_competitive'])
        self.assertFalse(result['utility_outcomes_used'])
        self.assertFalse(result['random_arms_eligible_for_selection'])

    def test_largest_estimate_wins_despite_cost_and_wide_interval(self):
        args = list(fixtures())
        for row in args[1]['primary']['contrasts']:
            if row['arm'] == 'repeat' and row['reference'] == 'native':
                row['estimate'] = .11
        args[1] = sealed(args[1])
        result = selector.select(*args)
        self.assertTrue(all(row['arm'] == 'repeat' for row in result['selections'].values()))
        self.assertTrue(all(row['native_evidence']['native_competitive'] for row in result['selections'].values()))

    def test_incomplete_or_foreign_analysis_cannot_choose(self):
        args = list(fixtures())
        args[1]['assigned'] -= 1
        args[1] = sealed(args[1])
        with self.assertRaisesRegex(ValueError, 'not complete'):
            selector.select(*args)
        args = list(fixtures())
        args[1]['primary']['contrasts'][0]['estimate'] = None
        args[1] = sealed(args[1])
        with self.assertRaisesRegex(ValueError, 'estimable'):
            selector.select(*args)
        args = list(fixtures())
        args[1]['primary']['contrasts'][0]['arm'] = 'random_bias'
        args[1] = sealed(args[1])
        with self.assertRaisesRegex(ValueError, 'omit or add'):
            selector.select(*args)


if __name__ == '__main__':
    unittest.main()
