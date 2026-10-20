"""Full-limit reservation, ambiguous sbatch reconciliation and cutoff contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from moe_exp.study.common import WORK, publish, verified
from moe_exp.study.reservations import CohortSchedule, freeze_predecessor


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

    def predecessor_fixture(self):
        """Build one accepted failed CPU job in an immutable predecessor journal."""
        source = self.root/'SOURCE.json'
        publish(source, {'release': 'failed'})
        self.value.update({'records': [{'uid': 'original'}], 'source': {'path': str(source)}})
        self.value['paths']['bulk'] = str(self.root/'old-bulk')
        path = self.root/'COHORT.json'
        publish(path, self.value)
        schedule = CohortSchedule(path)
        card = schedule.card('validate', None)
        folder = self.root/'schedule'/'validate-all'
        publish(folder/'RESERVED.json', {'card': card})
        publish(folder/'SBATCH.json', {'job': '7'})
        publish(folder/'ACCEPTED.json', {'job': '7', 'card': card})
        return path

    def test_terminal_predecessor_is_charged_once_and_cannot_satisfy_new_prerequisites(self):
        path = self.predecessor_fixture()
        usage = {'job': '7', 'state': 'FAILED', 'elapsed_seconds': 176, 'exit_code': '1:0'}
        with patch('moe_exp.study.ownership.accounting', return_value=usage):
            previous = freeze_predecessor(path, self.value['records'], self.root/'new-bulk')
            self.value['predecessor'] = previous
            self.value['paths'] = {'control': str(self.root/'replacement')}
            schedule = CohortSchedule(self.root/'replacement'/'COHORT.json')
            state = schedule.status()
            self.assertAlmostEqual(state['totals']['cpu_core_hours'], 176*4/3600)
            self.assertEqual(state['records'], [])
            self.assertEqual(len(state['predecessor_records']), 1)
            with self.assertRaisesRegex(ValueError, 'prerequisite not accepted'):
                schedule._dependencies(schedule.card('engineering', 'gpt'))
            card = schedule.card('validate', None)
            publish(schedule.root/'schedule'/'validate-all'/'RESERVED.json', {'card': card})
            self.assertAlmostEqual(schedule.status()['totals']['cpu_core_hours'], 4+176*4/3600)

    def test_live_or_unresolved_predecessor_rejects_replacement(self):
        path = self.predecessor_fixture()
        with patch('moe_exp.study.ownership.accounting', return_value={'state': 'RUNNING'}):
            with self.assertRaisesRegex(ValueError, 'still live'):
                freeze_predecessor(path, self.value['records'], self.root/'new-bulk')
        (self.root/'schedule'/'validate-all'/'ACCEPTED.json').unlink()
        with self.assertRaisesRegex(ValueError, 'unresolved reservation'):
            freeze_predecessor(path, self.value['records'], self.root/'new-bulk')

    def test_failed_engineering_startup_reduces_each_corrected_full_wall_limit(self):
        schedule = CohortSchedule(self.root/'COHORT.json')
        prior = [{'stage': 'engineering', 'model': model, 'charged': seconds*2/3600}
                 for model, seconds in (('gpt', 312), ('qwen36', 311))]
        with patch.object(schedule, '_predecessor_usage', return_value=prior):
            cards = [schedule.card('engineering', model) for model in ('gpt', 'qwen36')]
        self.assertEqual([card['minutes'] for card in cards], [114, 114])
        engineering_hours = sum(row['charged'] for row in prior)+sum(card['minutes']/60*card['gpus'] for card in cards)
        self.assertLessEqual(engineering_hours, 8)
        self.assertLessEqual(engineering_hours+12+2+24+18, 64)


if __name__ == '__main__':
    unittest.main()
