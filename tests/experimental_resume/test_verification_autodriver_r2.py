"""No Slurm allocations: interrupted held/prearmed controller transactions."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import verification_study_v3_autodriver_r2 as driver
from moe_exp.verification_study.contracts import file_sha, load, save

STAGING = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')


class FakeSlurm:
    def __init__(self):
        self.jobs = {}; self.operations = []; self.calls = []

    def text(self, job):
        value = self.jobs[job]
        reason = 'JobHeldUser' if value['held'] else 'Dependency' if value['dependency'] else 'None'
        dependency = value['dependency'] or '(null)'
        if value['dependency']:
            parts = dependency.split(':'); dependency = parts[0] + ':' + ':'.join(p + '(unfulfilled)' for p in parts[1:])
        return (f"JobId={job} Account=iscrc_miosr Partition=lrd_all_serial Comment={value['comment']} "
                f"JobState={value['state']} Reason={reason} Priority={0 if value['held'] else 1} Dependency={dependency}")

    def run(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if argv[:3] == ['scontrol', 'show', 'config']:
            output = 'DependencyParameters = kill_invalid_depend\nMinJobAge = 30'
        elif argv[:3] == ['scontrol', 'show', 'job']:
            output = self.text(argv[3])
        elif argv[:2] == ['scontrol', 'release']:
            self.jobs[argv[2]]['held'] = False; self.operations.append(('release', argv[2])); output = ''
        elif Path(argv[0]).name == 'sbatch':
            if '--test-only' in argv:
                output = 'estimated start';
            else:
                assert '--hold' in argv
                job = str(100 + len(self.jobs))
                comment = next(a.split('=', 1)[1] for a in argv if a.startswith('--comment='))
                dependency = next((a.split('=', 1)[1] for a in argv if a.startswith('--dependency=')), None)
                self.jobs[job] = {'comment': comment, 'dependency': dependency, 'held': True, 'state': 'PENDING'}
                self.operations.append(('accept', job)); output = job
        elif argv[0] == 'squeue':
            output = '\n'.join(f"{j}|{v['comment']}|iscrc_miosr|lrd_all_serial" for j, v in self.jobs.items())
        elif argv[0] == 'sacct':
            output = '\n'.join(f"{j}|{v['comment']}|iscrc_miosr|lrd_all_serial|{v['state']}|0:0|" for j, v in self.jobs.items())
        else:
            raise AssertionError('unexpected mocked command: ' + str(argv))
        return subprocess.CompletedProcess(argv, 0, output, '')


class ControllerR2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cv3-driver-r2-test-', dir=STAGING)
        self.root = Path(self.temp.name); self.path = self.root / 'CAMPAIGN.json'
        self.source = self.root / 'source.py'; self.source.write_text('sealed source fixture')
        self.files = {str(self.source): file_sha(self.source)}
        self.campaign = save(self.path, {'schema': 'test-campaign', 'code_files': self.files,
            'result_root': str(self.root), 'models': {key: key for key in driver.MODEL_KEYS}})
        self.slurm = FakeSlurm()
        from verification_study_v3_transport_r2 import pin_runtime_environment
        self.transport = types.ModuleType('verification_study_v3_transport_r2')
        self.transport.pin_runtime_environment = pin_runtime_environment
        self.transport.clean_submission_environment = lambda mapping=None: {
            key: value for key, value in (os.environ if mapping is None else mapping).items()
            if not key.startswith(('SLURM_', 'SRUN_', 'SBATCH_'))}
        self.modules = patch.dict(sys.modules, {'verification_study_v3_transport_r2': self.transport})
        self.modules.start()
        self.subprocess = patch.object(driver.subprocess, 'run', side_effect=self.slurm.run); self.subprocess.start()
        self.preflight = patch.object(driver.native, 'preflight', return_value={'schema': 'test-preflight'}); self.preflight.start()
        self.slots = patch.object(driver.dispatch, 'serial_slots', side_effect=lambda: {'free': 10 - len(self.slurm.jobs)}); self.slots.start()

    def tearDown(self):
        self.slots.stop(); self.preflight.stop(); self.subprocess.stop(); self.modules.stop(); self.temp.cleanup()

    def card(self, mode='bootstrap', **kwargs):
        return driver.controller_card(self.path, mode, files=self.files, **kwargs)

    def test_complete_chain_is_held_then_released_deepest_first(self):
        card = self.card(); receipt = driver.submit_controller(card)
        self.assertEqual(receipt['job_id'], '100')
        self.assertEqual(self.slurm.operations, [('accept', '100'), ('accept', '101'), ('accept', '102'),
                                               ('release', '102'), ('release', '101'), ('release', '100')])
        proof = load(driver.slot_for(card) / 'PREARM.json')
        self.assertEqual(proof['children'][0]['job_id'], '101')
        self.assertEqual(self.slurm.jobs['101']['dependency'], 'afternotok:100')
        self.assertEqual(self.slurm.jobs['102']['dependency'], 'afternotok:101:100')
        second = load(Path(proof['children'][0]['prearm_path']))
        self.assertEqual(second['failure_ancestor_job_ids'], ['100'])
        self.assertEqual(len(second['children']), 1)

    def test_successful_ancestor_blocks_cancelled_rescue_descendant(self):
        driver.submit_controller(self.card())
        self.slurm.jobs['100']['state'] = 'COMPLETED'; self.slurm.jobs['101']['state'] = 'CANCELLED'
        ids = self.slurm.jobs['102']['dependency'].split(':')[1:]
        # afternotok is conjunctive: cancellation of inactive rescue1 cannot
        # override successful primary100 and activate rescue2.
        self.assertFalse(all(self.slurm.jobs[j]['state'] != 'COMPLETED' for j in ids))
        self.slurm.jobs['100']['state'] = 'FAILED'; self.slurm.jobs['101']['state'] = 'FAILED'
        self.assertTrue(all(self.slurm.jobs[j]['state'] != 'COMPLETED' for j in ids))

    def test_deferred_bootstrap_has_full_protection_but_primary_stays_held(self):
        card = self.card(); receipt = driver.submit_controller(card, defer_release=True)
        self.assertTrue(self.slurm.jobs['100']['held'])
        self.assertFalse(self.slurm.jobs['101']['held']); self.assertFalse(self.slurm.jobs['102']['held'])
        self.assertFalse((driver.slot_for(card) / 'RELEASE.json').exists())
        self.assertTrue((driver.slot_for(card) / 'PREARM.json').exists())
        driver.release_controller(card)
        self.assertFalse(self.slurm.jobs['100']['held']); self.assertEqual(len(self.slurm.jobs), 3)
        self.assertEqual(driver.accepted_receipt(card)['sha256'], receipt['sha256'])

    def test_readback_accepts_expanded_conjunctive_dependencies_only(self):
        self.assertEqual(driver.dependency_ids('Dependency=afternotok:101(unfulfilled):100(fulfilled)', 'afternotok'), ['101', '100'])
        self.assertEqual(driver.dependency_ids('Dependency=afternotok:101(unfulfilled),afternotok:100(fulfilled)', 'afternotok'), ['101', '100'])
        for dependency in ('afternotok:101?afternotok:100', 'afternotok:101,afterany:100', 'afternotok:101:101'):
            with self.assertRaises(ValueError):
                driver.dependency_ids('Dependency=' + dependency, 'afternotok')

    def test_second_rescue_reuses_roots_accepted_handoff(self):
        next_card = self.card('successor', parent='50', wait_ids=['40'])
        receipt = driver.submit_controller(next_card, defer_release=True)
        operations = len(self.slurm.operations)
        result = driver.accepted_successor(self.root, ['51', '50'], current='52')
        self.assertEqual(result['job_id'], receipt['job_id'])
        self.assertEqual(len(self.slurm.jobs), 3)
        self.assertEqual(self.slurm.operations[operations:], [('release', '100')])

    def test_foreign_or_competing_ancestor_handoffs_are_not_allocated(self):
        card = self.card('successor', parent='50', wait_ids=['40'])
        slot = driver.slot_for(card); save(slot / 'CARD.json', {**card, 'operational_version': 'autodriver-r1'})
        saved = load(slot / 'CARD.json')
        save(slot / 'SUBMISSION.json', {'schema': 'verification-autodriver-submission-v3',
            'controller_card_sha256': saved['sha256'], 'job_id': '60'})
        with self.assertRaises(ValueError):
            driver.accepted_successor(self.root, ['51', '50'])
        self.assertFalse(self.slurm.jobs)

    def test_second_rescue_has_no_third_descendant(self):
        card = self.card('rescue', parent='50', consecutive_rescues=2, failure_ancestor_job_ids=['50', '49'])
        driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 1)
        self.assertEqual(load(driver.slot_for(card) / 'PREARM.json')['children'], [])
        with self.assertRaises(ValueError):
            self.card('rescue', parent='50', consecutive_rescues=3, failure_ancestor_job_ids=['50', '49', '48'])
        with self.assertRaises(ValueError):
            self.card('rescue', parent='50', consecutive_rescues=2)

    def test_crash_after_primary_acceptance_reconciles_same_job(self):
        card = self.card(); original = driver.save; failed = []
        def crash(path, value, **kwargs):
            if Path(path).name == 'SUBMISSION.json' and value.get('job_id') == '100' and not failed:
                failed.append(True); raise RuntimeError('crash before primary receipt')
            return original(path, value, **kwargs)
        with patch.object(driver, 'save', side_effect=crash):
            with self.assertRaises(RuntimeError):
                driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 1); self.assertTrue(self.slurm.jobs['100']['held'])
        receipt = driver.submit_controller(card)
        self.assertEqual(receipt['job_id'], '100'); self.assertTrue(receipt['reconciled'])
        self.assertEqual(len(self.slurm.jobs), 3)

    def test_existing_primary_receipt_resumes_missing_prearm(self):
        card = self.card()
        with patch.object(driver, 'ensure_prearmed', side_effect=RuntimeError('crash after primary receipt')):
            with self.assertRaises(RuntimeError):
                driver.submit_controller(card)
        self.assertTrue((driver.slot_for(card) / 'SUBMISSION.json').exists())
        self.assertEqual(driver.missing_chain_slots(card), 2)
        receipt = driver.submit_controller(card)
        self.assertEqual(receipt['job_id'], '100'); self.assertEqual(len(self.slurm.jobs), 3)

    def test_crash_after_rescue_acceptance_does_not_duplicate_it(self):
        card = self.card(); original = driver.save; failed = []
        def crash(path, value, **kwargs):
            if Path(path).name == 'SUBMISSION.json' and value.get('job_id') == '101' and not failed:
                failed.append(True); raise RuntimeError('crash before rescue receipt')
            return original(path, value, **kwargs)
        with patch.object(driver, 'save', side_effect=crash):
            with self.assertRaises(RuntimeError):
                driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 2)
        driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 3)
        self.assertEqual(sum(op == ('accept', '101') for op in self.slurm.operations), 1)

    def test_crash_after_release_reconciles_without_second_release(self):
        card = self.card(); original = driver.save; failed = []
        def crash(path, value, **kwargs):
            if Path(path) == driver.slot_for(card) / 'RELEASE.json' and not failed:
                failed.append(True); raise RuntimeError('crash after scontrol release')
            return original(path, value, **kwargs)
        with patch.object(driver, 'save', side_effect=crash):
            with self.assertRaises(RuntimeError):
                driver.submit_controller(card)
        self.assertFalse(self.slurm.jobs['100']['held'])
        self.assertTrue((driver.slot_for(card) / 'RELEASE_INTENT.json').exists())
        driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 3)
        self.assertEqual(self.slurm.operations.count(('release', '100')), 1)

    def test_completed_transaction_does_not_allocate_or_release_again(self):
        card = self.card(); first = driver.submit_controller(card); operations = list(self.slurm.operations)
        second = driver.submit_controller(card)
        self.assertEqual(first, second); self.assertEqual(self.slurm.operations, operations)

    def test_release_resume_after_fast_completion_and_rescue_cancellation(self):
        card = self.card(); original = driver.save
        def crash(path, value, **kwargs):
            if Path(path) == driver.slot_for(card) / 'RELEASE.json':
                raise RuntimeError('lost receipt after release')
            return original(path, value, **kwargs)
        with patch.object(driver, 'save', side_effect=crash):
            with self.assertRaises(RuntimeError):
                driver.submit_controller(card)
        self.slurm.jobs['100']['state'] = 'COMPLETED'
        self.slurm.jobs['101']['state'] = 'CANCELLED'; self.slurm.jobs['102']['state'] = 'CANCELLED'
        driver.submit_controller(card)
        self.assertEqual(len(self.slurm.jobs), 3)
        self.assertEqual(self.slurm.operations.count(('release', '100')), 1)

    def test_slot_limit_keeps_existing_protection_and_creates_no_intent(self):
        card = self.card()
        with patch.object(driver.dispatch, 'serial_slots', return_value={'free': 2}):
            with self.assertRaises(driver.SerialSlotLimit):
                driver.submit_controller(card)
        self.assertEqual(self.slurm.jobs, {})
        self.assertFalse((driver.slot_for(card) / 'INTENT.json').exists())

    def test_ambiguous_intent_is_never_retried(self):
        card = self.card(); slot = driver.slot_for(card)
        save(slot / 'INTENT.json', {'schema': 'intent', 'controller_card_sha256': card['sha256']})
        with patch.object(driver, 'comment_matches', side_effect=ValueError('ambiguous intent')):
            with self.assertRaises(ValueError):
                driver.submit_controller(card)
        self.assertFalse(self.slurm.jobs)

    def test_changed_sealed_source_rejects_before_allocation(self):
        card = self.card(); self.source.write_text('changed source')
        with self.assertRaises(ValueError):
            driver.submit_controller(card)
        self.assertFalse(self.slurm.jobs)

    def test_scoped_clean_environment_preserves_parent_and_other_commands(self):
        environment = {'PATH': '/usr/bin', 'SLURM_JOB_ID': 'parent', 'SLURM_CPU_BIND': 'mask_cpu:bad',
            'SLURM_CPU_BIND_LIST': 'bad', 'SRUN_CPUS_PER_TASK': '4', 'SBATCH_EXPORT': 'ALL', 'KEEP': 'value'}
        with patch.dict(os.environ, environment, clear=True):
            with driver.submission_transport():
                driver.subprocess.run(['sbatch', '--test-only'], check=True)
                driver.subprocess.run(['scontrol', 'show', 'config'], check=True)
            self.assertEqual(os.environ['SLURM_JOB_ID'], 'parent')
            self.assertEqual(os.environ['SLURM_CPU_BIND'], 'mask_cpu:bad')
        self.assertEqual(self.slurm.calls[-2][1]['env'], {'PATH': '/usr/bin', 'KEEP': 'value'})
        self.assertNotIn('env', self.slurm.calls[-1][1])
        explicit = {'SLURM_CPU_BIND': 'bad', 'KEEP': 'explicit'}
        with driver.submission_transport(), driver.submission_transport():
            driver.subprocess.run(['/usr/bin/sbatch', '--test-only'], env=explicit)
        self.assertEqual(self.slurm.calls[-1][1]['env'], {'KEEP': 'explicit'})
        self.assertIn('SLURM_CPU_BIND', explicit)

    def test_transport_scientific_intent_uses_transport_comment(self):
        card = save(self.root / 'submissions/transport/JOB_CARD.json', {'schema': 'verification-job-card-v3',
            'campaign_sha256': self.campaign['sha256'], 'code_files': self.files, 'transport_executor': {'sha256': 'transport'},
            'resources': {'partition': 'boost_usr_prod'}})
        save(self.root / 'submissions/transport/INTENT.json', {'schema': 'intent', 'job_card_sha256': card['sha256'],
            'preflight_sha256': 'preflight'})
        with patch.object(driver, 'comment_matches', return_value={'job_id': '40', 'partition': 'boost_usr_prod'}) as match:
            result = driver.reconcile_scientific_intents(self.root, self.campaign)
        match.assert_called_once_with('cv3t:' + card['sha256']); self.assertEqual(result, ['40'])

    def test_runtime_paths_are_reasserted_after_step_prolog(self):
        with patch.dict(os.environ, {'TMPDIR': '/tmp', 'SLURM_JOB_ID': '50', 'HOME': '/private/home'}):
            directory = driver.pin_runtime_environment(self.root, '50')
            self.assertEqual(Path(os.environ['TMPDIR']), directory / 'tmp')
            self.assertEqual(tempfile.gettempdir(), str(directory / 'tmp'))
            self.assertEqual(os.environ['SLURM_JOB_ID'], '50'); self.assertEqual(os.environ['HOME'], '/private/home')
        tempfile.tempdir = None

    def test_launch_shell_overrides_binding_and_signals_early_failure(self):
        shell = driver.SHELL.read_text()
        self.assertIn('--cpu-bind=none --mem-bind=none', shell)
        self.assertIn('srun --ntasks=1 --cpus-per-task=4', shell)
        self.assertIn('SLURM_CPU_BIND@', shell); self.assertIn('SHELL_FAILURE.json', shell)
        self.assertLess(shell.index("trap 'cv3_early_failure"), shell.index('srun --ntasks'))


if __name__ == '__main__':
    unittest.main()
