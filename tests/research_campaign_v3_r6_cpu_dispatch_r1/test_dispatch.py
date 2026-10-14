import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r6.contracts import publish, seal
from moe_exp.research_campaign_v3_r6.scheduler import serial_capacity
from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1 import policy
from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import amend_card, retirable


def fixture(stage='coordination', partition='lrd_all_serial', gpus=0):
    return seal({'stage': stage, 'partition': partition, 'gpus': gpus,
        'cpus': 4 if not gpus else 16, 'memory_gib': 24 if not gpus else 120,
        'allocation_cpus': 4 if not gpus else 16,
        'memory_mib': (24 if not gpus else 120) * 1024,
        'billing_units_upper': 4 if not gpus else 16, 'reservation_hours': 12,
        'model': 'glm', 'shard_index': 12, 'minutes': 180,
        'campaign_sha256': 'original', 'ops_binding_sha256': 'operations',
        'claim_cell': {'sources': ['original-pair-bank'], 'families': 129},
        'comment': 'original-comment'})


class DispatchContracts(unittest.TestCase):
    def test_seven_serial_owners_prevent_protected_cpu_wave(self):
        snapshot = {'jobs': [{'partition': 'lrd_all_serial', 'state': 'PENDING'} for _ in range(7)]}
        self.assertFalse(serial_capacity(snapshot, [fixture()])['submitted'])
        corrected = amend_card(fixture(), 'dispatch')
        self.assertTrue(all(serial_capacity(snapshot, [corrected]).values()))

    def test_cpu_work_preserves_scientific_contract_and_cost(self):
        original = fixture(); preserved = copy.deepcopy(original)
        amended = amend_card(original, 'dispatch')
        self.assertEqual(original, preserved)
        self.assertEqual(amended['partition'], 'boost_usr_prod')
        self.assertEqual(amended['gpus'], 0)
        for key in ('cpus', 'memory_gib', 'minutes', 'billing_units_upper', 'reservation_hours',
                    'claim_cell', 'campaign_sha256', 'ops_binding_sha256', 'model', 'shard_index'):
            self.assertEqual(amended[key], original[key])

    def test_all_protected_serial_owners_keep_partition(self):
        for stage in ('controller', 'snapshot', 'validate'):
            with self.subTest(stage=stage):
                self.assertEqual(amend_card(fixture(stage=stage), 'dispatch')['partition'], 'lrd_all_serial')

    def test_existing_reporting_card_remains_byte_identical(self):
        original = fixture(stage='snapshot'); original['ops_report_only'] = True
        self.assertIs(amend_card(original, 'dispatch'), original)

    def test_gpu_price_and_original_assignments_keep_resources(self):
        original = fixture(stage='reader_price', partition='boost_usr_prod', gpus=2)
        amended = amend_card(original, 'dispatch')
        for key in ('partition', 'gpus', 'cpus', 'memory_gib', 'minutes', 'model', 'shard_index', 'claim_cell'):
            self.assertEqual(amended[key], original[key])

    def test_only_pending_zero_runtime_scientific_owners_retire(self):
        value = fixture(stage='controller')
        pending = {'state': 'PENDING', 'elapsed_seconds': 0, 'start': 'None'}
        self.assertTrue(retirable(value, pending))
        forbidden = [dict(pending, state='RUNNING'), dict(pending, elapsed_seconds=1),
                     dict(pending, start='2026-10-14T12:00:00')]
        for accounting in forbidden:
            self.assertFalse(retirable(value, accounting))
        self.assertFalse(retirable(dict(value, ops_report_only=True), pending))
        self.assertFalse(retirable(fixture(), pending))

    def test_unpriced_resource_expansion_rejected(self):
        for change in ({'cpus': 8}, {'memory_gib': 48}, {'gpus': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                amend_card(dict(fixture(), **change), 'dispatch')

    def test_stale_dispatch_or_full_graph_proof_rejected(self):
        good = {'passed': True, 'binding_sha256': 'dispatch',
                'base_validation_sha256': 'validated', 'actual_graph_sha256': 'graph'}
        for change in ({'passed': False}, {'binding_sha256': 'other'},
                       {'base_validation_sha256': 'old'}, {'actual_graph_sha256': 'old'}):
            proof = dict(good, **change)
            def read(path):
                return {'sha256': 'graph'} if path.name == 'IMPORT_GRAPH_VALIDATION.json' else proof
            with self.subTest(change=change), patch.object(policy, 'check_binding', return_value={'sha256': 'dispatch'}), \
                    patch.object(policy.base, 'require_readiness', return_value={'sha256': 'validated'}), \
                    patch.object(policy, 'load', side_effect=read), self.assertRaises(ValueError):
                policy.require_readiness({'root': '/unused'})

    def test_transfer_verifies_three_successors_before_any_retirement(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary).resolve(); events = []; verified = set()
            pending = {'state': 'PENDING', 'elapsed_seconds': 0, 'start': 'None', 'terminal': False}
            owners = [{'card': fixture(stage='controller'), 'accepted': {'job_id': str(i)},
                       'accounting': dict(pending)} for i in (101, 102, 103)]
            publish(root / 'LAUNCH.json', {'actual_launch': True})
            publish(root / 'handoffs/0000.json', {'generation': 0, 'wait_ids': ['99'],
                'primary': {'job_id': '101'}, 'guards': [{'job_id': '102'}, {'job_id': '103'}]})

            class FakeScheduler:
                states = {str(i): dict(pending) for i in (101, 102, 103)}
                def snapshot(self):
                    return {'jobs': [{'partition': 'lrd_all_serial'} for _ in range(7)]}
                def observe(self, jid):
                    return self.states[jid]
                def command(self, args):
                    if args[:2] == ['scontrol', 'hold']:
                        events.append(('hold', args[2])); return ''
                    if args[:3] == ['scontrol', 'show', 'job']:
                        return 'Reason=JobHeldUser'
                    if args[0] == 'scancel':
                        self.assert_protected(); events.append(('cancel', args[1]))
                        self.states[args[1]] = dict(pending, state='CANCELLED', terminal=True)
                        return ''
                    raise AssertionError(args)
                def assert_protected(self):
                    if verified != {'201', '202', '203'}:
                        raise AssertionError('Retirement preceded verification of all three owners')
                def verify(self, jid, value):
                    verified.add(jid); events.append(('verify', jid))
                    return {'fields': {'Reason': 'Dependency'}}

            def protect(campaign, scheduler, *, generation, wait_ids=(), release=True):
                receipts = []
                for jid in ('201', '202', '203'):
                    value = fixture(stage='controller')
                    publish(root / 'cards' / (value['sha256'] + '.json'), value)
                    receipts.append({'job_id': jid, 'card_sha256': value['sha256']})
                events.append(('release' if release else 'accept', tuple(r['job_id'] for r in receipts)))
                return {'primary': receipts[0], 'guards': receipts[1:], 'generation': generation}

            with patch.object(policy, 'require_readiness', return_value={}), \
                    patch.object(policy, 'check_binding', return_value={'sha256': 'dispatch'}), \
                    patch('moe_exp.research_campaign_v3_r6.controller.jobs', return_value=owners), \
                    patch('moe_exp.research_campaign_v3_r6.controller._check', return_value={'admitted': True}), \
                    patch('moe_exp.research_campaign_v3_r6.scheduler.protected_controller', side_effect=protect):
                scheduler = FakeScheduler()
                result = policy.transfer({'root': str(root)}, scheduler)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual([event[1] for event in events if event[0] == 'cancel'], ['101', '102', '103'])
                before = list(events)
                self.assertEqual(policy.transfer({'root': str(root)}, scheduler), result)
                self.assertEqual(events, before)


if __name__ == '__main__':
    unittest.main()
