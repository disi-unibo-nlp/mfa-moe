"""No-submit regressions for exact dependencies and native live accounting."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import dispatch_utility_production_v2 as entry
import attach_utility_price_pilot_v5 as pilot
import submit_utility_completion_chain_v2 as root


class RecoveryTests(unittest.TestCase):
    def tearDown(self):
        entry.INSTALLED = None
        pilot.INSTALLED = None

    def test_native_accounting_path_and_operational_identity_reach_every_cpu_and_gpu_child(self):
        recovery = {'sha256': 'recovery'}
        original_cpu, original_gpu = entry.original.cpu_submit, entry.original.gpu_submit
        calls = []
        try:
            with patch.object(entry.original.shared, 'submit', side_effect=lambda *args: calls.append(args) or '701'), \
                 patch.object(entry.original, 'gpu_submit', return_value='702') as gpu, \
                 patch.dict(os.environ, {'PATH': '/usr/bin'}):
                installed = entry.install(recovery)
                self.assertEqual(os.environ['PATH'], '/cineca/bin:/usr/bin')
                self.assertEqual(installed.cpu_submit(Path('.'), 'price', dict(os.environ), {'science': 'frozen'},
                    ['--dependency=afterok:700']), '701')
                self.assertEqual(installed.gpu_submit(Path('.'), 'initial', {}, {}, dict(os.environ),
                    {'science': 'frozen'}, [0, 1]), '702')
                self.assertEqual(calls[0][2][0], '--dependency=afterok:700')
                self.assertEqual(calls[0][2][-1], str(entry.WRAPPER))
                self.assertEqual(calls[0][4], {'science': 'frozen', 'operational_recovery_sha256': 'recovery'})
                self.assertEqual(gpu.call_args.args[-1], [0, 1])
                self.assertEqual(gpu.call_args.args[-2], calls[0][4])
                self.assertIn('/cineca/bin', calls[0][3]['PATH'])
        finally:
            entry.original.cpu_submit, entry.original.gpu_submit = original_cpu, original_gpu

    def test_unavailable_native_accounting_fails_before_any_child(self):
        with patch.object(entry, 'SALDO', Path('/definitely/missing/saldo')), \
             patch.object(entry.original.shared, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'native saldo unavailable'):
                entry.install({'sha256': 'recovery'})
            submit.assert_not_called()

    def test_restart_preserves_three_exact_dependency_edges_and_original_grading_wrapper(self):
        with tempfile.TemporaryDirectory(prefix='.utility-recovery-test-', dir=REPO) as temporary:
            directory = Path(temporary)
            plan = {'reader_dispatch_job': '700', 'measurement_source_dir': str(directory / 'source'),
                'proposal_path': '/frozen/proposal', 'fresh_generation_dispatch_job': '600',
                'attachment_dir': str(directory / 'pilot'), 'sha256': 'pilot-recovery',
                'proposal_sha256': 'proposal'}
            recovery = {'sha256': 'operation', 'config_path': '/frozen/config',
                'production_chain_dir': str(directory / 'production'), 'grading_plan_sha256': 'j1', 'scope': 'frozen'}
            calls = []
            def dependency(identifier, *args, **kwargs):
                return ['--dependency=afterok:' + identifier], {'active_predecessor': identifier}
            def submit(*args):
                calls.append(args)
                return str(700 + len(calls))
            with patch.object(root.dispatcher, 'dependency', side_effect=dependency), \
                 patch.object(root.shared, 'submit', side_effect=submit), \
                 patch.dict(os.environ, {'PATH': '/usr/bin'}):
                chain = root.dispatch(directory, {'sha256': 'config'}, directory / 'plan.json', plan,
                    directory / 'recovery.json', recovery)
            self.assertEqual([call[2][0] for call in calls],
                ['--dependency=afterok:700', '--dependency=afterok:701', '--dependency=afterok:702'])
            self.assertEqual(calls[0][2][-1], str(root.P.SCRIPTS / 'attach_utility_price_pilot_v5.sbatch'))
            self.assertEqual(calls[1][2][-1], str(root.P.SCRIPTS / 'dispatch_utility_production_v2.sbatch'))
            self.assertIn(str(root.P.SCRIPTS / 'utility_j1_attach_v2.sbatch'), calls[2][2])
            self.assertEqual([chain[key] for key in ('pilot_generation_attachment_job',
                'production_follow_job', 'grading_follow_job')], ['701', '702', '703'])
            self.assertTrue(all(call[3]['UTILITY_PILOT_ATTACHMENT'] == plan['attachment_dir'] for call in calls))
            self.assertTrue(all(call[3]['PATH'].startswith('/cineca/bin:') for call in calls))
            self.assertTrue(all(call[4]['operational_recovery_sha256'] == 'operation' for call in calls))

    def test_original_live_commitments_gate_includes_fixed_pilot_and_all_reserved_work(self):
        def live(command, **kwargs):
            if command[0] == '/cineca/bin/saldo':
                return SimpleNamespace(stdout='IscrC_MIOSR 20260504 20270204 68000 21350 21350 31.4 7391 1251\n')
            if command[0] == 'squeue':
                return SimpleNamespace(stdout='700_[0-3%2]|PENDING|2:00:00|2:00:00|16|gres/gpu:2|1\n')
            if command[0] == 'sacctmgr':
                return SimpleNamespace(stdout='lmolfett|iscrc_miosr||normal|\n')
            return SimpleNamespace(stdout='PartitionName=' + command[-1])
        with tempfile.TemporaryDirectory(prefix='.utility-recovery-test-', dir=REPO) as temporary, \
             patch.object(root.subprocess, 'run', side_effect=live):
            value = root.live_preflight(Path(temporary), {'available_generation_billing_core_hours': 16000,
                'offline_grading_reserve_gpu_hours': 216})
            self.assertEqual(value['reserved_billing_core_hours'], 17984)
            self.assertEqual(value['budget']['active_remaining_commitment_billing_core_hours'], 128)
            self.assertEqual(value['budget']['uncommitted_reported_billing_core_hours'], 46522)
            with self.assertRaisesRegex(ValueError, 'live account cannot fit'):
                root.live_preflight(Path(temporary), {'available_generation_billing_core_hours': 45000,
                    'offline_grading_reserve_gpu_hours': 216})

    def test_pilot_generation_uses_restored_reader_and_preserves_original_generation_binding(self):
        with tempfile.TemporaryDirectory(prefix='.utility-recovery-test-', dir=REPO) as temporary:
            directory = Path(temporary)
            source, own = directory / 'source', directory / 'own'
            source.mkdir(); own.mkdir()
            pilot.original.shared.save(source / 'GENERATION_CHAIN.json', {
                'schema': 'overnight-generation-chain-v2', 'binding': {'manifest_sha256': 'manifest'},
                'reader_dispatch_job': 'failed-old-reader'})
            plan = {'reader_dispatch_job': '700', 'fresh_generation_dispatch_job': '600',
                'reader_recovery_chain_sha256': 'repair', 'sha256': 'pilot-recovery'}
            with patch.object(pilot.original, 'paths', return_value=(None, {'sha256': 'manifest'}, source, own)), \
                 patch.object(pilot.original, 'validate_sources'), patch.object(pilot.original, 'active_preflight'), \
                 patch.object(pilot.original, 'dependency_arguments', return_value=['--dependency=afterok:700']) as deps, \
                 patch.object(pilot.original, 'submit', return_value='701') as submit:
                pilot.generation(plan, {'sha256': 'proposal'}, directory / 'proposal.json')
            self.assertEqual(deps.call_args.args[0], [('reader_dispatch', '700')])
            self.assertIn('--dependency=afterok:700', submit.call_args.args[2])
            result = pilot.original.utility.sealed(own / 'GENERATION_ATTACHMENT.json')
            self.assertEqual(result['reader_dispatch_job'], '700')
            self.assertEqual(result['fresh_generation_dispatch_job'], '600')
            self.assertEqual(result['next_attachment_job'], '701')

    def test_v5_successors_change_only_wrapper_paths_and_record_operational_amendment(self):
        old_paths, old_submit = pilot.original.paths, pilot.original.submit
        old_run, old_inputs = pilot.original.subprocess.run, pilot.original.pilot_inputs
        with tempfile.TemporaryDirectory(prefix='.utility-recovery-test-', dir=REPO) as temporary:
            directory = Path(temporary)
            plan = {'sha256': 'pilot-recovery', 'manifest_sha256': 'manifest',
                'measurement_source_dir': str(directory / 'source'), 'attachment_dir': str(directory / 'own')}
            try:
                with patch.object(pilot.original, 'paths', return_value=(None, {'sha256': 'manifest'},
                    directory / 'source', directory / 'old-own')), \
                     patch.object(pilot.original, 'submit', return_value='701') as submit, \
                     patch.dict(os.environ, {'UTILITY_PILOT_OPERATIONAL_RECOVERY': '/exact/plan.json'}):
                    module = pilot.install(plan)
                    module.submit(directory, 'runtime-pilot',
                        [str(pilot.SCRIPTS / 'run_utility_price_pilot_v4.sbatch')], {'frozen': 'environment'},
                        {'pilot_manifest_sha256': 'frozen-pilot'})
                    self.assertEqual(submit.call_args.args[2], [str(pilot.GPU_WRAPPER)])
                    self.assertEqual(submit.call_args.args[3]['UTILITY_PILOT_OPERATIONAL_RECOVERY'], '/exact/plan.json')
                    self.assertEqual(submit.call_args.args[4], {'pilot_manifest_sha256': 'frozen-pilot',
                        'pilot_operational_recovery_sha256': 'pilot-recovery'})
                    self.assertEqual(module.paths({})[-1], directory / 'own')
            finally:
                pilot.original.paths, pilot.original.submit = old_paths, old_submit
                pilot.original.subprocess.run, pilot.original.pilot_inputs = old_run, old_inputs

    def test_guarded_selector_replaces_only_the_frozen_selector_program_and_checks_before_pilot(self):
        old_paths, old_submit = pilot.original.paths, pilot.original.submit
        old_run, old_inputs = pilot.original.subprocess.run, pilot.original.pilot_inputs
        try:
            with patch.object(pilot.original.subprocess, 'run', return_value='result') as run, \
                 patch.object(pilot.original, 'pilot_inputs') as inputs, \
                 patch.object(pilot, 'validate_selected_policy', side_effect=ValueError('unbound recovered policy')) as guard:
                installed = pilot.install({'sha256': 'pilot-recovery'})
                command = ['qualified-python', '-B', str(pilot.SCRIPTS / 'select_utility_policy_v2.py'),
                    '--manifest', '/exact/manifest', '--analysis', '/exact/analysis', '--out', '/exact/policy']
                installed.subprocess.run(command, check=True)
                self.assertEqual(run.call_args.args[0], [*command[:2], str(pilot.GUARDED_SELECTOR), *command[3:]])
                installed.subprocess.run(['saldo', '-b', 'lmolfett'], check=True)
                self.assertEqual(run.call_args.args[0], ['saldo', '-b', 'lmolfett'])
                with self.assertRaisesRegex(ValueError, 'unbound recovered policy'):
                    installed.pilot_inputs({}, Path('/exact/pilot'), {'sha256': 'manifest'})
                guard.assert_called_once()
                inputs.assert_not_called()
        finally:
            pilot.original.paths, pilot.original.submit = old_paths, old_submit
            pilot.original.subprocess.run, pilot.original.pilot_inputs = old_run, old_inputs

    def test_policy_guard_requires_exact_recovered_analysis_provenance_and_guarded_entry(self):
        provenance = {'amendment_sha256': 'exact', 'scientific_functions_unchanged': True}
        fake = SimpleNamespace(validate_analysis=lambda *args: provenance)
        with tempfile.TemporaryDirectory(prefix='.utility-recovery-test-', dir=REPO) as temporary:
            directory = Path(temporary)
            own = directory / 'pilot'; own.mkdir()
            analysis = directory / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2'
            analysis.mkdir()
            pilot.original.shared.save(analysis / 'ANALYSIS.json', {'assigned': 3480})
            amendment = directory / 'AMENDMENT.json'
            pilot.original.shared.save(amendment, {'manifest_sha256': 'manifest'})
            plan = {'reader_amendment_path': str(amendment), 'guarded_selector_sha256': 'guarded'}
            policy = own / 'SELECTED_POLICY.json'
            pilot.original.shared.save(policy, {'recovery_provenance': provenance,
                'selection_operational_entry_sha256': 'foreign'})
            with patch.dict(sys.modules, {'fresh_frame_recovery_v1': fake}), \
                 patch.object(pilot.original, 'DOC', directory):
                with self.assertRaisesRegex(ValueError, 'lacks guarded'):
                    pilot.validate_selected_policy(plan, {'sha256': 'manifest'}, own)
                policy.unlink()
                pilot.original.shared.save(policy, {'recovery_provenance': provenance,
                    'selection_operational_entry_sha256': 'guarded'})
                pilot.validate_selected_policy(plan, {'sha256': 'manifest'}, own)


if __name__ == '__main__':
    unittest.main()
