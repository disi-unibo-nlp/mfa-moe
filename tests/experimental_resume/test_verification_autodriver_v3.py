from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import verification_study_v3_autodriver_r1 as driver
from moe_exp.verification_study.contracts import save, load, file_sha

STAGING = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')


class DriverTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory(prefix='cv3-autodriver-test-', dir=STAGING)
        self.root = Path(self.temp.name)
        self.path = self.root / 'CAMPAIGN.json'
        source = self.root / 'source.py'; source.write_text('source fixture')
        self.campaign = save(self.path, {'schema': 'test-campaign', 'code_files': {str(source): file_sha(source)},
            'result_root': str(self.root), 'models': {k: k for k in driver.MODEL_KEYS}})

    def tearDown(self):
        self.temp.cleanup()

    def card(self, mode='bootstrap', **kwargs):
        return driver.controller_card(self.path, mode, files={'test-source': 'sealed'}, **kwargs)

    def test_successor_uses_any_completion_not_all(self):
        card = self.card('successor', parent='30', wait_ids=['20', '10', '20'])
        self.assertEqual(card['dependency'], 'afterany:10?afterany:20')
        self.assertEqual(card['wait_job_ids'], ['10', '20'])
        self.assertNotIn('afterok', card['dependency'])

    def test_rescue_only_fires_on_failure(self):
        card = self.card('rescue', parent='30', consecutive_rescues=1)
        self.assertEqual(card['dependency'], 'afternotok:30')
        self.assertEqual(card['resources']['gpus'], 0)
        self.assertEqual(card['resources']['billing_core_hours_max'], 0)

    def test_no_empty_or_invalid_successor(self):
        with self.assertRaises(ValueError):
            self.card('successor', parent='30')
        with self.assertRaises(ValueError):
            self.card('successor', parent='30', wait_ids=['10;sbatch'])

    def test_consecutive_rescue_is_bounded(self):
        with self.assertRaises(ValueError):
            self.card('rescue', parent='30', consecutive_rescues=3)
        self.assertEqual(self.card('successor', parent='30', wait_ids=['10'])['consecutive_rescues'], 0)

    def test_test_rejection_creates_no_intent_or_persisted_preflight(self):
        card = self.card()
        with patch.object(driver, 'check_files'), patch.object(driver.dispatch, 'serial_slots', return_value={'free': 3}), \
             patch.object(driver.native, 'preflight', return_value={'schema': 'test-preflight'}), \
             patch.object(driver, 'command', return_value='DependencyParameters = kill_invalid_depend\nMinJobAge = 30'), \
             patch.object(driver.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'sbatch')):
            with self.assertRaises(subprocess.CalledProcessError):
                driver.submit_controller(card)
        slot = self.root / 'automation/submissions' / card['sha256']
        self.assertFalse((slot / 'INTENT.json').exists())
        self.assertFalse((slot / 'PREFLIGHT.json').exists())

    def test_bootstrap_reserves_room_for_rescue_and_handoff(self):
        card = self.card()
        with patch.object(driver, 'check_files'), patch.object(driver.dispatch, 'serial_slots', return_value={'free': 2}), \
             patch.object(driver.subprocess, 'run') as submit:
            with self.assertRaises(ValueError):
                driver.submit_controller(card)
        submit.assert_not_called()
        self.assertFalse((self.root / 'automation/submissions' / card['sha256'] / 'INTENT.json').exists())

    def test_submit_intent_is_reconciled_without_new_job(self):
        card = self.card()
        slot = self.root / 'automation/submissions' / card['sha256']
        save(slot / 'INTENT.json', {'schema': 'intent', 'controller_card_sha256': card['sha256']})
        with patch.object(driver, 'check_files'), patch.object(driver, 'comment_matches', return_value={
            'job_id': '10', 'partition': 'lrd_all_serial', 'state': 'COMPLETED'}), \
             patch.object(driver.subprocess, 'run') as run:
            result = driver.submit_controller(card)
        run.assert_not_called()
        self.assertEqual(result['job_id'], '10')
        self.assertTrue(result['reconciled'])

    def test_aged_completed_intent_found_in_history(self):
        with patch.object(driver, 'command', side_effect=['', '10|cv3auto:sha|iscrc_miosr|lrd_all_serial|COMPLETED|0:0|']):
            result = driver.comment_matches('cv3auto:sha')
        self.assertEqual(result['job_id'], '10')
        self.assertEqual(result['state'], 'COMPLETED')

    def test_ambiguous_or_absent_intent_cannot_be_retried(self):
        with patch.object(driver, 'command', side_effect=['', '']):
            with self.assertRaises(ValueError):
                driver.comment_matches('cv3auto:sha')
        with patch.object(driver, 'command', side_effect=[
            '10|cv3auto:sha|iscrc_miosr|lrd_all_serial\n11|cv3auto:sha|iscrc_miosr|lrd_all_serial', '']):
            with self.assertRaises(ValueError):
                driver.comment_matches('cv3auto:sha')

    def test_queue_presence_overrides_stale_terminal_accounting(self):
        with patch.object(driver, 'command', side_effect=[
            '10|COMPLETED|0:0|10|cpu=4|', '10|RUNNING|boost_usr_prod|cv3:sha|None']):
            result = driver.observations(['10'])
        self.assertTrue(result['10']['live'])
        self.assertEqual(result['10']['state'], 'RUNNING')

    def test_known_terminal_event_is_reused(self):
        save(self.root / 'health/events/one.json', {'schema': 'verification-health-event-v3',
            'campaign_sha256': self.campaign['sha256'], 'predecessor_job_id': '10',
            'job_card_sha256': 'card-sha', 'state': 'STEP_COMPLETE'})
        rows = [{'job_id': '10', 'card': {'sha256': 'card-sha'}}]
        with patch.object(driver.health, 'check') as check:
            result = driver.observe_terminal(self.path, self.root, self.campaign, rows,
                                            {'10': {'state': 'COMPLETED'}}, self.card())
        check.assert_not_called()
        self.assertEqual(result[0]['state'], 'STEP_COMPLETE')


    def test_accepted_successor_is_reused_before_handoff_was_saved(self):
        card = self.card('successor', parent='30', wait_ids=['10'])
        slot = self.root / 'automation/submissions' / card['sha256']
        save(slot / 'CARD.json', card)
        save(slot / 'SUBMISSION.json', {'schema': 'verification-autodriver-submission-v3',
            'controller_card_sha256': card['sha256'], 'job_id': '31'})
        self.assertFalse((self.root / 'automation/runs/30/HANDOFF.json').exists())
        with patch.object(driver, 'submit_controller') as submit:
            result = driver.accepted_successor(self.root, '30', '32')
        submit.assert_not_called()
        self.assertEqual(result['job_id'], '31')

    def test_last_completion_is_consumed_before_chain_stops(self):
        import verification_study_v3_autostages_r1 as stages
        completed = {'10': {'state': 'COMPLETED', 'exit_code': '0:0'}}
        with patch.object(stages, 'next_actions', side_effect=[{'status': 'WAITING', 'actions': []}, {'status': 'COMPLETE', 'actions': []}]) as advance, \
             patch.object(driver, 'scientific_submissions', return_value=[{'job_id': '10'}]), \
             patch.object(driver, 'observations', return_value=completed), \
             patch.object(driver, 'observe_terminal', return_value=[{'job_id': '10', 'state': 'STEP_COMPLETE'}]) as observe:
            result, states, events = driver.advance_frontier(self.path, self.root, self.campaign,
                self.card(), {'10': {'state': 'RUNNING', 'live': True}})
        self.assertEqual(advance.call_count, 2)
        observe.assert_called_once()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(events[0]['job_id'], '10')
        self.assertFalse(result.get('ready_cpu_actions'))

    def test_finite_race_pass_budget_records_exact_unconsumed_work(self):
        import verification_study_v3_autostages_r1 as stages
        with patch.object(stages, 'next_actions', return_value={'status': 'WAITING', 'actions': []}), \
             patch.object(driver, 'scientific_submissions', return_value=[{'job_id': '10'}]), \
             patch.object(driver, 'observations', return_value={'10': {'state': 'COMPLETED', 'exit_code': '0:0'}}), \
             patch.object(driver, 'observe_terminal', return_value=[]):
            result, states, events = driver.advance_frontier(self.path, self.root, self.campaign,
                self.card(), {'10': {'state': 'RUNNING', 'live': True}}, passes=1)
        self.assertTrue(result['ready_cpu_actions'])
        self.assertEqual(result['unconsumed_terminal_job_ids'], ['10'])


if __name__ == '__main__':
    unittest.main()
