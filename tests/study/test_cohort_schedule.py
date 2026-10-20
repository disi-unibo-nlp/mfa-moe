"""Full-limit reservation, ambiguous sbatch reconciliation and cutoff contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from moe_exp.study.common import WORK, publish, verified
from moe_exp.study.reservations import CohortSchedule


class CohortScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.value = {'sha256': 'fixed-cohort', 'paths': {'control': str(self.root)},
                      'science_cutoff': '2026-10-23T05:30:00+02:00',
                      'report_deadline': '2026-10-23T11:30:00+02:00'}
        self.read = patch('moe_exp.study.cohort.read', return_value=self.value)
        self.read.start()
        self.addCleanup(self.read.stop)

    def test_reserved_full_limit_exists_before_ambiguous_submission_and_is_not_resubmitted(self):
        calls = []
        def run(arguments):
            calls.append(arguments)
            if arguments[0] == 'sbatch':
                reservation = verified(self.root/'schedule'/'validate-all'/'RESERVED.json')
                self.assertEqual(reservation['full_limit_hours'], 4)
                raise subprocess.CalledProcessError(1, arguments, stderr='connection lost after possible acceptance')
            return ''
        schedule = CohortSchedule(self.root/'COHORT.json', run=run)
        with patch.object(schedule, '_preflight', return_value={'checked': True}), \
                patch.object(schedule, '_dependencies', return_value=[]), \
                patch('moe_exp.study.ownership.check_time', return_value=1):
            with self.assertRaises(subprocess.CalledProcessError):
                schedule.submit('validate')
            with self.assertRaisesRegex(ValueError, 'ambiguous sbatch intent retained'):
                schedule.submit('validate')
        self.assertEqual(sum(arguments[0] == 'sbatch' for arguments in calls), 1)
        self.assertEqual(schedule.status()['totals']['cpu_core_hours'], 4)

    def test_final_analysis_four_core_hours_are_protected(self):
        schedule = CohortSchedule(self.root/'COHORT.json')
        with patch.object(schedule, '_dependencies', return_value=[]), \
                patch.object(schedule, 'status', return_value={'totals': {'gpu_hours': 0, 'cpu_core_hours': 12}}), \
                patch('moe_exp.study.ownership.check_time', return_value=1):
            with self.assertRaisesRegex(ValueError, 'protected final-analysis reserve'):
                schedule.submit('prepare')

    def test_expired_full_allocation_rejects_before_submission(self):
        schedule = CohortSchedule(self.root/'COHORT.json', run=lambda args: self.fail('submission after cutoff'))
        with patch('moe_exp.study.ownership.check_time', side_effect=ValueError('deadline expired')):
            with self.assertRaisesRegex(ValueError, 'deadline expired'):
                schedule.submit('validate')
        self.assertFalse((self.root/'schedule'/'validate-all'/'INTENT.json').exists())

    def test_terminal_actuals_replace_full_limit_once(self):
        schedule = CohortSchedule(self.root/'COHORT.json')
        card = schedule.card('generate', 'qwen36')
        folder = self.root/'schedule'/'generate-qwen36'
        publish(folder/'RESERVED.json', {'card': card})
        self.assertEqual(schedule.status()['totals']['gpu_hours'], 12)
        publish(folder/'SBATCH.json', {'job': '123'})
        with patch('moe_exp.study.ownership.accounting', return_value={
            'job': '123', 'state': 'FAILED', 'elapsed_seconds': 1800, 'exit_code': '1:0',
        }):
            self.assertEqual(schedule.status()['totals']['gpu_hours'], 1)

    def test_contingent_engineering_uses_verified_cpu_acceptance(self):
        schedule = CohortSchedule(self.root/'COHORT.json')
        predecessor = {'stage': 'validate', 'model': None, 'job': '123',
                       'usage': {'state': 'PENDING', 'exit_code': '0:0'}}
        publish(self.root/'schedule'/'validate-all'/'ACCEPTED.json', {
            'job': '123', 'card': {'cohort_sha256': self.value['sha256']}})
        with patch.object(schedule, 'status', return_value={'records': [predecessor]}):
            self.assertEqual(schedule._dependencies(schedule.card('engineering', 'gpt')), ['123'])


if __name__ == '__main__':
    unittest.main()
