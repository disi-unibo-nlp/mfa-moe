"""No-inference/no-submit tests for the independent operator panel."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import operator_panel_v1 as P
import operator_panel_outcomes_v1 as O
from operator_panel_backend_v1 import PanelController, paired_reader_seed
import utility_controller_interface_v1 as B
import dispatch_overnight_readers_v1 as shared


def plan():
    return P.build_plan(P.U.sealed(P.DOC / 'UTILITY_SCOUT_PLAN_v1.json'),
                        P.U.sealed(P.DOC / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json'))


def rows(n=12):
    p = plan()
    return [{**a, 'operational_correct': a['arm'] == 'bias',
             'reasoning_tokens': (1 + p['family_order'].index(a['family'])) * (10 if a['arm'] == 'native' else 9),
             'tokens': (1 + p['family_order'].index(a['family'])) * 12}
            for a in P.assignments(p, p['family_order'][:n])]


class PanelTests(unittest.TestCase):
    def test_factorial_order_seeds_and_policies(self):
        p = plan(); P.validate_plan(p)
        self.assertEqual(len(p['assignments']), 768)
        positions = {(arm, i): 0 for arm in P.ARMS for i in range(4)}
        for family in p['family_order']:
            block = [a for a in p['assignments'] if a['family'] == family]
            self.assertEqual({(r['arm'], r['seed']) for r in block}, {(a, s) for a in P.ARMS for s in (0, 1)})
            for seed in (0, 1):
                paired = [a for a in block if a['seed'] == seed]
                self.assertEqual(len({a['screening_uid'] for a in paired}), 1)
                self.assertEqual(len({a['prompt_token_ids_sha256'] for a in paired}), 1)
                for a in paired: positions[a['arm'], a['execution_position']] += 1
        self.assertEqual(set(positions.values()), {48})
        self.assertEqual(P.assignments(p, p['family_order'][:2]), p['assignments'][:16])
        for operator, selected in p['policies'].items():
            for transition, policy in selected.items():
                self.assertEqual(policy['actions'][0]['kind'], operator)
                self.assertEqual(policy['slots'], [0]); self.assertEqual(policy['pulse_width'], 256)
        with self.assertRaises(ValueError): P.assignments(p, p['family_order'][1:13])
        broken = copy.deepcopy(p); broken['assignments'][0]['seed'] = 1
        with self.assertRaises(ValueError): P.validate_plan(broken)

    def test_screening_common_seed_operator_binding_and_native(self):
        p = plan(); controls = []
        for arm in P.ARMS:
            a = next(r for r in p['assignments'] if r['family'] == p['family_order'][0] and r['seed'] == 0 and r['arm'] == arm)
            def screen(request):
                return [B.SideResult(request.request_id, i, 'stop', '{"start":true}', 100, 5, .1) for i in (0, 1)]
            ctl = PanelController(a, 'Find x', p['policies'], lambda ids: ''.join(map(chr, ids)), screen)
            for token in map(ord, 'The answer is 3. '):
                ctl.observe_cumulative((*ctl.state.completion_token_ids, token))
            self.assertEqual(ctl.state.uid, a['uid'])
            if arm == 'native':
                self.assertEqual(ctl.queries, []); self.assertIsNone(ctl.episode)
            else:
                self.assertEqual(ctl.episode['selected_arm'], arm)
                self.assertEqual(ctl.episode['action_names'], p['policies'][arm]['candidate_to_verify']['action_names'])
                controls.append(ctl)
        self.assertEqual(len({c.queries[0]['request']['request_id'] for c in controls}), 1)
        self.assertEqual(len({c.physical_screening_ids[0]['physical_request_id'] for c in controls}), 3)
        seed = lambda key, reader: P.U.digest([key, reader])
        self.assertEqual(len({paired_reader_seed(c.physical_screening_ids[0]['physical_request_id'], 0, seed)
                              for c in controls}), 1)
        for ctl in controls:
            before = len(ctl.queries)
            for token in map(ord, 'The answer is 4. '):
                ctl.observe_cumulative((*ctl.state.completion_token_ids, token))
            self.assertEqual(len(ctl.queries), before)

    def test_reasoning_marker_caps_missing_and_errors(self):
        self.assertEqual(O.lengths([7, 8, P.THINK_END_ID, 9], 'stop')['reasoning_tokens'], 2)
        self.assertEqual(O.lengths([P.THINK_END_ID, 9], 'stop')['answer_tokens'], 1)
        self.assertEqual(O.lengths([7] * 10, 'length', cap=10)['reasoning_tokens'], 10)
        self.assertIsNone(O.lengths([7], 'stop')['reasoning_tokens'])
        self.assertIsNone(O.lengths([7], 'error')['reasoning_tokens'])
        self.assertEqual(O.lengths([P.THINK_END_ID] * 2, 'stop')['boundary_status'], 'MULTIPLE_MARKERS_FIRST_CLOSURE')
        self.assertEqual(O.lengths([7], 'stop', prompt_reasoning_open=False)['boundary_status'], 'PREFIX_ALREADY_CLOSED')

    def test_pair_inference_nine_effects_equal_family_and_missing_bounds(self):
        p = plan(); data = rows(); families = p['family_order'][:12]
        result = O.inference(data, families, replicates=500)
        self.assertEqual(len(result['simultaneous_95_intervals']), 9)
        self.assertEqual(result['point_estimates'][0], 1.)
        self.assertEqual(result['point_estimates'][1], -6.5)
        self.assertTrue(result['small_sample']); self.assertTrue(result['degenerate_intervals'])
        data[0]['operational_correct'] = None
        result = O.inference(data, families, replicates=500)
        self.assertEqual(result['status'], 'INCOMPLETE_ENDPOINTS')
        self.assertIsNone(result['simultaneous_95_intervals'])
        self.assertLess(result['identification_bounds'][0][0], result['identification_bounds'][0][1])
        with self.assertRaises(ValueError): O.paired_values(data + data[:1], families)

    def test_unfinished_grades_remain_unknown(self):
        score = SimpleNamespace(adjudicate_uid=lambda strict, finish, verdict: (None, verdict, verdict == 'EQUIVALENT'))
        self.assertEqual(O.adjudicated(False, 'stop', None, score), (None, 'J1_PENDING'))
        self.assertEqual(O.adjudicated(True, 'stop', None, score)[0], True)
        self.assertEqual(O.adjudicated(False, 'length', None, score)[0], False)
        self.assertEqual(O.adjudicated(False, 'stop', 'UNCERTAIN', score)[0], False)

    def test_measured_pricing_uses_runtime_and_preserves_grading_recovery(self):
        p = plan(); assigned = P.assignments(p, p['family_order'][:2]); receipts = []
        for a in assigned:
            side = 1. if a['arm'] != 'native' else 0.
            receipts.append({'status': 'COMMITTED_GENERATION', 'assignment': a,
                'result': {'generation_wall_seconds': 11. + side, 'controller': {
                    'state': {'completion_token_ids': [7] * 1000}, 'side_elapsed_seconds': side,
                    'queries': [{}] if side else []}}})
        allocations = [{'elapsed_seconds': 1100, 'billing_core_hours': 32 * 1100 / 3600}] * 2
        profile = P.measured_profile(receipts, [{'load_wall_seconds': 700}] * 2,
            [{'allocated_driver_wall_seconds': 1000}] * 2, allocations)
        self.assertEqual(profile['status'], 'MEASURED_RUNTIME_ONLY')
        grading = {n: {'requested_allocation_GPU_hour_ceiling': n * 4.5} for n in P.COHORTS}
        price = P.choose_cohort(p, profile, grading)
        self.assertIsNotNone(price['selected'])
        self.assertLessEqual(price['selected']['additional_total_billing_core_hours'], 6000)
        self.assertGreater(price['selected']['recovery_shards'], 0)
        self.assertGreater(price['selected']['grading_reserve_billing_core_hours'], 0)
        receipts[0]['operational_correct'] = False
        self.assertEqual(profile, P.measured_profile(receipts, [{'load_wall_seconds': 700}] * 2,
            [{'allocated_driver_wall_seconds': 1000}] * 2, allocations))
        hold = P.choose_cohort(p, {**profile, 'generator_seconds_per_token': 100}, grading)
        self.assertEqual(hold['status'], 'HOLD_NO_COMPLETE_COHORT_FITS')
        for r in receipts: r['result']['controller']['queries'] = []
        self.assertEqual(P.measured_profile(receipts, [], [], allocations)['status'], 'HOLD_MEASURED_PROFILE')

    def test_no_duplicate_submission_after_receipt_or_unresolved_attempt(self):
        with tempfile.TemporaryDirectory(prefix='.panel-test-', dir=REPO) as temp:
            directory = Path(temp)
            shared.save(directory / 'pilot.json', {'binding': {'panel': 'x'}, 'arguments': ['wrapper'], 'job_id': '7'})
            with patch.object(shared.subprocess, 'run') as command:
                self.assertEqual(shared.submit(directory, 'pilot', ['wrapper'], {}, {'panel': 'x'}), '7')
                command.assert_not_called()
            shared.save(directory / 'pending.attempt.json', {'identity': 'attempt'})
            with self.assertRaises(RuntimeError): shared.submit(directory, 'pending', ['wrapper'], {}, {})


if __name__ == '__main__':
    unittest.main()
