import os
import unittest
from unittest.mock import Mock, patch

from moe_exp.research_campaign_v3_r6_cpu_allocation_r1 import guard_for, verification_target


class CpuAllocationContracts(unittest.TestCase):
    def setUp(self):
        self.card = {'gpus': 0, 'partition': 'boost_usr_prod', 'cpus': 4,
                     'memory_gib': 24, 'stage': 'analysis_pilot'}
        self.env = {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0', 'USER': 'lmolfett',
                    'SLURM_JOB_ACCOUNT': 'iscrc_miosr', 'SLURM_JOB_PARTITION': 'boost_usr_prod'}
        self.native = Mock()
        self.verify = Mock(return_value={'fields': {'ReqTRES': 'cpu=4,mem=24G,node=1,billing=4'},
                                        'resources': {'billing_units': 4}})

    def run_guard(self, guard):
        with patch.dict(os.environ, self.env), patch('socket.getfqdn', return_value='lrdn0001.leonardo.local'), patch('socket.gethostname', return_value='lrdn0001'):
            guard()

    def test_real_cpu_booster_proof_accepts_without_environment_spoofing(self):
        guard = guard_for(self.card, self.native, self.verify)
        self.run_guard(guard)
        self.native.assert_not_called()
        self.verify.assert_called_once_with('123', self.card)

    def test_gpu_and_serial_guards_remain_native(self):
        guard_for({**self.card, 'gpus': 2}, self.native, self.verify)(gpu=True)
        guard_for({**self.card, 'partition': 'lrd_all_serial'}, self.native, self.verify)()
        self.assertEqual(self.native.call_count, 2)
        self.verify.assert_not_called()

    def test_gpu_request_or_expanded_billing_rejected(self):
        for tres, billing in (('cpu=4,gres/gpu=1', 4), ('cpu=8,mem=24G', 8)):
            self.verify.return_value = {'fields': {'ReqTRES': tres}, 'resources': {'billing_units': billing}}
            with self.assertRaises(ValueError):
                self.run_guard(guard_for(self.card, self.native, self.verify))

    def test_wrong_account_or_no_step_rejected(self):
        for field, value in (('SLURM_JOB_ACCOUNT', 'foreign'), ('SLURM_STEP_ID', None)):
            env = dict(self.env)
            if value is None:
                env.pop(field)
            else:
                env[field] = value
            with patch.dict(os.environ, env, clear=True), patch('socket.getfqdn', return_value='lrdn0001.leonardo.local'), patch('socket.gethostname', return_value='lrdn0001'), self.assertRaises(ValueError):
                guard_for(self.card, self.native, self.verify)()

    def test_cpu_card_cannot_start_gpu_work(self):
        with self.assertRaises(ValueError):
            guard_for(self.card, self.native, self.verify)(gpu=True)

    def test_array_verifies_physical_submission_and_rejects_wrong_child(self):
        physical = {**self.card, 'array_cards': [self.card], 'comment': 'physical-comment'}
        with patch.dict(os.environ, {'SLURM_ARRAY_JOB_ID': '200', 'SLURM_ARRAY_TASK_ID': '0'}):
            self.assertEqual(verification_target('201', self.card, physical), ('200_0', physical))
            with self.assertRaises(ValueError):
                verification_target('201', {**self.card, 'cpus': 8}, physical)


if __name__ == '__main__':
    unittest.main()
