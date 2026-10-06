"""CPU acceptance tests for the serial/eager 1,024-token engine gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/qualify_mechanism_engine_1024_v1.py'
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location('qualify_mechanism_engine_1024_v1', SOURCE)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class EngineQualificationTests(unittest.TestCase):
    def test_sealed_manifest_and_complete_price(self):
        manifest = Q.base.sealed(Q.MANIFEST)
        Q.validate_manifest(manifest)
        price = Q.base.sealed(Q.REPO / 'report/experimental-resume-v1/MECHANISM_ENGINE_1024_QUAL_PRICE_v1.json')
        self.assertEqual((manifest['expected_requests'], manifest['maximum_decode_tokens']),
                         (12, 11 * 1024 + 128))
        self.assertEqual(price['qualification_manifest_sha256'], manifest['sha256'])
        self.assertEqual(price['status'], 'PASS_COMPLETE_STAGE')
        self.assertLessEqual(price['modeled_complete_seconds'], price['requested_wall_seconds'])
        self.assertEqual(manifest['bias'], 1.0)

    def test_native_neighbor_edit_is_detected(self):
        meta = {'uid': 'n', 'role': 'native', 'transition': 'candidate_to_verify',
                'action_names': [], 'closed_prefix': False, 'cap': 1024}
        outcome = SimpleNamespace(error=None, routed=__import__('numpy').zeros((4, 40, 8)),
                                  tokens=[1, 2, 3, 4])
        record = {'inactive_native_checks': {'24': {'expert_identity_mismatches': 0,
                                                    'weight_mismatches': 0}},
                  'cpu_active_rows': 1}
        check, _ = Q.check_case(meta, outcome, {0: {'n': record}, 1: {'n': record}},
                               SimpleNamespace(THINK_END_ID=999), [])
        self.assertFalse(check['pass'])
        self.assertIn('native request edited rank 0', check['reasons'])

    def test_two_pulse_order_and_recovery(self):
        names = ['verify', 'commit']
        meta = {'uid': 'o', 'role': 'preempted_ab', 'transition': 'ordered',
                'action_names': names, 'closed_prefix': False, 'cap': 1024}
        outcome = SimpleNamespace(error=None, routed=__import__('numpy').zeros((800, 40, 8)),
                                  tokens=[1] * 800)
        record = {'inactive_native_checks': {'24': {'expert_identity_mismatches': 0,
                                                    'weight_mismatches': 0}},
                  'ordered_action_rows': {'verify': 256, 'commit': 256},
                  'ordered_segments': {'verify': [[0, 256]], 'commit': [[512, 768]]},
                  'ordered_action_dose': {'verify': {'24': {'active_rows': 256}},
                                          'commit': {'28': {'active_rows': 256}}},
                  'preemptions': 1, 'recompute_rows': 32}
        check, _ = Q.check_case(meta, outcome, {0: {'o': record}, 1: {'o': record}},
                               SimpleNamespace(THINK_END_ID=999),
                               [{'reset_ok': True}])
        self.assertTrue(check['pass'], check['reasons'])
        self.assertEqual(check['expected_segments']['commit'], [[512, 768]])
        no_recovery, _ = Q.check_case(meta, outcome, {0: {'o': record}, 1: {'o': record}},
                                      SimpleNamespace(THINK_END_ID=999), [])
        self.assertFalse(no_recovery['pass'])

    def test_closed_prefix_suppresses_both_actions(self):
        meta = {'uid': 'c', 'role': 'closed', 'transition': 'closed',
                'action_names': ['verify', 'commit'], 'closed_prefix': True, 'cap': 128}
        outcome = SimpleNamespace(error=None, routed=__import__('numpy').zeros((16, 40, 8)),
                                  tokens=[1] * 16)
        record = {'inactive_native_checks': {'24': {'expert_identity_mismatches': 0,
                                                    'weight_mismatches': 0}},
                  'ordered_action_rows': {'verify': 0, 'commit': 0},
                  'ordered_segments': {'verify': [], 'commit': []},
                  'ordered_action_dose': {'verify': {'28': {'active_rows': 0}},
                                          'commit': {'24': {'active_rows': 0}}}}
        check, _ = Q.check_case(meta, outcome, {0: {'c': record}, 1: {'c': record}},
                               SimpleNamespace(THINK_END_ID=999), [])
        self.assertTrue(check['pass'], check['reasons'])


if __name__ == '__main__':
    unittest.main()
