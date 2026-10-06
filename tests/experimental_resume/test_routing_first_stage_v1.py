"""Meaningful bounded checks for target engagement, missingness and pairing."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import analyze_routing_first_stage_v1 as analysis
import submit_routing_first_stage_v1 as attach


def spec(slots=(0,)):
    return {'pulses': [{'slot': start, 'end': start + 256, 'action': 'selected',
                        'targets': [[28, 189], [28, 9]]} for start in slots],
            'intended_opportunities': 512 * len(slots)}


def row(uid='u', *, tokens=None, error=None, finish='length'):
    return {'uid': uid, 'tokens': list(tokens if tokens is not None else [1] * 1024),
            'error': error, 'finish': finish}


def routes(n=1024, targets=(189, 9)):
    value = np.tile(np.arange(8), (n, 40, 1))
    for index, target in enumerate(targets):
        value[:, 28, index] = target
    return value


class EngagementTests(unittest.TestCase):
    def test_closure_row_included_but_no_postclosure_routes(self):
        tokens = [1, 1, analysis.source.THINK_END, 1, 1]
        result = analysis.measure(row(tokens=tokens, finish='stop'), routes(5), spec())
        self.assertEqual(result['observed_executed_opportunities'], 6)
        self.assertEqual(result['observed_selected_expert_hits'], 6)
        self.assertEqual(result['endpoint'], 6 / 512)
        self.assertEqual(result['conditional_inclusion'], 1.)

    def test_reference_measured_on_left_targets_not_own_random_experts(self):
        target = analysis.measure(row(), routes(), spec())
        reference = analysis.measure(row(), routes(targets=(120, 121)), spec())
        self.assertEqual(target['endpoint'], 1.)
        self.assertEqual(reference['endpoint'], 0.)

    def test_repeat_uses_both_slots_with_unexecuted_pulse_zero(self):
        tokens = [1] * 300
        result = analysis.measure(row(tokens=tokens, finish='stop'), routes(300), spec((0, 512)))
        self.assertEqual(result['observed_executed_opportunities'], 512)
        self.assertEqual(result['endpoint'], .5)
        self.assertEqual(result['conditional_inclusion'], 1.)

    def test_failure_and_missing_array_bounds_are_not_imputed_points(self):
        failed = analysis.measure(row(tokens=[1] * 10, error='timeout'), routes(10), spec())
        missing = analysis.measure(row(tokens=[1] * 10, finish='stop'), None, spec())
        empty = analysis.measure(row(tokens=[], finish='stop'), routes(0), spec())
        empty_without_array = analysis.measure(row(tokens=[], finish='stop'), None, spec())
        self.assertIsNone(failed['endpoint']); self.assertEqual(failed['endpoint_bounds'], [0., 1.])
        self.assertIsNone(missing['endpoint']); self.assertEqual(missing['endpoint_bounds'], [0., 20 / 512])
        self.assertEqual(empty['endpoint'], 0.); self.assertIsNone(empty['conditional_inclusion'])
        self.assertTrue(empty['no_pulse_exposure'])
        self.assertTrue(empty_without_array['point_identified'])
        self.assertEqual(empty_without_array['endpoint'], 0.)

    def test_duplicate_routed_experts_rejected(self):
        invalid = routes(1)
        invalid[0, 0, 1] = invalid[0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'top-eight'):
            analysis.measure(row(tokens=[1]), invalid, spec())

    def test_family_weighting_seed_pairing_and_missing_envelope(self):
        pair = {'scope': 'all', 'left': 'bias', 'right': 'native', 'supported': True, 'records': []}
        values = {}
        for family, prefixes, effect in [('f0', ['p0', 'p1'], 1.), ('f1', ['p2'], 0.)]:
            for prefix in prefixes:
                for seed in (0, 1):
                    left, right = f'{prefix}-{seed}-left', f'{prefix}-{seed}-right'
                    pair['records'].append({'prefix_uid': prefix, 'family': family, 'seed': seed,
                        'transition': 'candidate_to_verify', 'left_uid': left, 'right_uid': right,
                        'target_spec_sha256': 's'})
                    values[left] = {'s': {'endpoint_bounds': [effect, effect], 'point_identified': True}}
                    values[right] = {'s': {'endpoint_bounds': [0., 0.], 'point_identified': True}}
        result, assigned = analysis.infer([pair], values, n_boot=200)
        self.assertEqual(len(assigned), 6)
        self.assertEqual(result['contrasts'][0]['estimate'], .5)
        self.assertEqual(result['multiplicity'], 1)
        values['p2-0-left']['s'] = {'endpoint_bounds': [0., 1.], 'point_identified': False}
        result, _ = analysis.infer([pair], values, n_boot=200)
        self.assertIsNone(result['contrasts'][0]['estimate'])
        self.assertEqual(result['contrasts'][0]['identification_bounds'], [.5, .75])
        self.assertIsNotNone(result['contrasts'][0]['simultaneous_uncertainty_envelope95'])
        pair['records'].pop()
        with self.assertRaisesRegex(ValueError, 'seeds incomplete'):
            analysis.infer([pair], values, n_boot=200)

    def test_scope_alias_deduplicates_only_single_transition(self):
        manifest = {'schema': 'overnight-routing-manifest-v1', 'rows': [{'transition': 'verify'}],
                    'analysis_scopes': ['all', 'verify'], 'planned_contrasts': [['bias', 'native']]}
        self.assertEqual(len(analysis.frozen_contrasts(manifest)), 1)
        manifest['rows'].append({'transition': 'commit'})
        manifest['analysis_scopes'].append('commit')
        self.assertEqual(len(analysis.frozen_contrasts(manifest)), 3)

    def test_telemetry_ranks_never_summed_as_independent_observations(self):
        payload = {'uid': 'u', 'transition': 'verify', 'arm': 'bias', 'role': 'target',
                   'action_dose': {str(rank): {'dose': {'action': {'28': {'active_rows': 2, 'weight_l1': .5}}}}
                                   for rank in (0, 1)}}
        result = analysis.telemetry_summary([payload])
        self.assertEqual(len(result['records']), 2)
        self.assertEqual({r['rank'] for r in result['records']}, {'0', '1'})
        self.assertEqual([r['mean_sparse_weight_l1_per_active_row'] for r in result['records']], [.25, .25])

    def test_changed_array_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as directory:
            path = Path(directory) / 'routes.npz'
            np.savez(path, u=routes(1))
            arrays = {'u': {'path': str(path), 'available': True, 'sha256': '0' * 64}}
            with self.assertRaisesRegex(ValueError, 'changed route archive'):
                analysis.measurements([row(tokens=[1])], arrays, {'u': {'s': spec()}})

    def test_standalone_plot_and_csv_export(self):
        import os
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as directory:
            root = Path(directory)
            inference = {'contrasts': [{'scope': 'all', 'left': 'bias', 'right': 'native',
                'status': 'POINT_IDENTIFIED', 'estimate': .1, 'simultaneous_ci95': [-.1, .3]}]}
            with mock.patch.dict(os.environ, {'MPLCONFIGDIR': str(root / 'matplotlib')}):
                files = analysis.artifacts(root, inference, [{'arm': 'native', 'assigned': 4}])
            self.assertEqual(set(files), {'CONTRASTS.csv', 'ARMS.csv', 'TARGET_ENGAGEMENT.png', 'TARGET_ENGAGEMENT.pdf'})
            self.assertTrue(all((root / path).stat().st_size > 0 for path in files))

    def test_attachment_uses_exact_frame_dependency_and_cpu_only_wrapper(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as directory:
            root = Path(directory)
            manifest = {'sha256': 'a' * 64}
            plan = {'sha256': 'p' * 64}
            paths = (root / 'manifest.json', root / 'price.json', manifest,
                     {'sha256': 'b' * 64}, root / 'generation', root / 'analysis')
            calls = []
            def submit(directory, name, args, env, binding):
                calls.append((name, args, binding)); return '12345678'
            with mock.patch.object(analysis, 'validate_plan', return_value=plan), \
                 mock.patch.object(attach.shared, 'DOC', root), \
                 mock.patch.object(attach.dispatch, 'live_account_snapshot'), \
                 mock.patch.object(attach.dispatch, 'ensure_verified'), \
                 mock.patch.object(attach.subprocess, 'run'):
                result = attach.submit_stage('A', paths, '12345670', {'job_id': '12345670'}, submit=submit)
            self.assertEqual(len(calls), 1)
            self.assertIn('--dependency=afterok:12345670', calls[0][1])
            self.assertEqual(calls[0][2]['resources']['GPU_hours'], 0.)
            self.assertEqual(result['analysis_job'], '12345678')


if __name__ == '__main__':
    unittest.main()
