"""Prevent source, sampling and manifest changes from adopting saved rescue work."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moe_exp.study import rescue_source
from moe_exp.study.common import FAST, WORK, publish, verified


class RescueExecutionTest(unittest.TestCase):
    def setUp(self):
        """Build isolated source, binding and store fixtures in user WORK staging."""
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = {
            'config': {'bulk_root': str(self.root/'bulk')},
            'models': {'gpt': {'sampler': {'temperature': .8}, 'card': {'max_tokens': 512}}},
        }
        self.original_binding = self.root/'original-binding.json'
        publish(self.original_binding, self.study)
        self.binding_patch = patch.object(rescue_source, 'STUDY_BINDING', self.original_binding)
        self.binding_patch.start()
        self.addCleanup(self.binding_patch.stop)
        self.manifest = publish(self.root/'manifest.json', {
            'model': 'gpt', 'records': [{'uid': 'a'*64, 'horizon': 512}],
        })
        self.operator = self.root/'operator.py'
        self.operator.write_text('original operator\n')
        self.runtime = {'packages': {'vllm': 'qualified'}, 'backend_source_files': {'selector.py': 'sealed'}}
        self.paths_patch = patch.object(rescue_source, 'paths', return_value={str(self.operator): self.operator})
        self.runtime_patch = patch.object(rescue_source, 'runtime_binding', return_value=self.runtime)
        self.paths_patch.start()
        self.runtime_patch.start()
        self.addCleanup(self.paths_patch.stop)
        self.addCleanup(self.runtime_patch.stop)
        self.source_path = self.root/'source.json'
        self.source = rescue_source.build(self.source_path)
        self.units = self.root/'bulk'/'driver'/'gpt'/'rescue'/'units'
        self.receipt = self.units.parent/'EXECUTION_IDENTITY.json'

    def test_explicit_source_receipt_preserves_historical_default(self):
        historical = self.root/'historical.json'
        with patch.object(rescue_source, 'RECEIPT', historical):
            self.assertEqual(rescue_source.verify(self.source_path), self.source)
        self.assertFalse(historical.exists())

    def test_source_and_installed_runtime_changes_reject_verification(self):
        self.operator.write_text('different operator\n')
        with self.assertRaisesRegex(ValueError, 'current files'):
            rescue_source.verify(self.source_path)
        self.operator.write_text('original operator\n')
        with patch.object(rescue_source, 'runtime_binding', return_value={'packages': {'vllm': 'changed'}}):
            with self.assertRaisesRegex(ValueError, 'installed runtime'):
                rescue_source.verify(self.source_path)

    def test_explicit_fresh_binding_is_selected_without_changing_original(self):
        from moe_exp.study import driver

        original = self.original_binding.read_bytes()
        fresh = {**self.study, 'config': {'bulk_root': str(self.root/'fresh-bulk')}}
        fresh_path = self.root/'fresh-binding.json'
        publish(fresh_path, fresh)
        source_path = self.root/'fresh-source.json'
        source = rescue_source.build(source_path, study_binding_path=fresh_path)
        self.assertEqual(source['study_binding_path'], str(fresh_path))
        self.assertEqual(rescue_source.verify(source_path), source)
        manifests = self.root/'manifests'
        publish(manifests/'gpt'/'RESCUE.json', {key: value for key, value in self.manifest.items() if key != 'sha256'})
        with patch.object(driver, 'binding') as original_loader, \
                patch.object(driver, 'DRIVER', manifests), \
                patch.object(driver, 'shard_rows', return_value=[]), \
                patch.object(driver, 'code_hashes', return_value={}), \
                patch.object(driver, 'hooked_llm') as engine:
            execution = rescue_source.bind_execution(fresh, 'gpt', self.manifest, source)
        self.assertEqual(execution['model'], 'gpt')
        original_loader.assert_not_called()
        engine.assert_not_called()
        self.assertTrue((self.root/'fresh-bulk'/'driver'/'gpt'/'rescue'/'EXECUTION_IDENTITY.json').is_file())
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.original_binding.read_bytes(), original)

    def test_fresh_binding_cannot_change_the_scientific_profile(self):
        changed = copy.deepcopy(self.study)
        changed['models']['gpt']['sampler']['temperature'] = .9
        path = self.root/'changed-binding.json'
        publish(path, changed)
        with self.assertRaisesRegex(ValueError, 'may change only'):
            rescue_source.load_study_binding(path)

    def test_fresh_store_binds_once_and_identical_resume_keeps_receipt(self):
        first = rescue_source.bind_execution(self.study, 'gpt', self.manifest, self.source)
        original = self.receipt.read_bytes()
        self.assertFalse(self.units.exists())
        folder = self.units/('a'*64)
        folder.mkdir(parents=True)
        (folder/'PARTIAL.json').write_text('{"tokens": [1]}\n')
        second = rescue_source.bind_execution(self.study, 'gpt', self.manifest, self.source)
        self.assertEqual(first, second)
        self.assertEqual(self.receipt.read_bytes(), original)

    def test_unbound_completed_and_partial_stores_are_preserved(self):
        folder = self.units/('a'*64)
        folder.mkdir(parents=True)
        for name in ('RESULT.json', 'PARTIAL.json'):
            with self.subTest(name=name):
                path = folder/name
                path.write_text('legacy state\n')
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'unbound prior outputs or partials'):
                    rescue_source.bind_execution(self.study, 'gpt', self.manifest, self.source)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(self.receipt.exists())
                path.unlink()

    def test_manifest_sampling_cap_and_source_changes_reject_store_reuse(self):
        rescue_source.bind_execution(self.study, 'gpt', self.manifest, self.source)
        original = self.receipt.read_bytes()
        changed_sampler = copy.deepcopy(self.study)
        changed_sampler['models']['gpt']['sampler']['temperature'] = .9
        changed_cap = copy.deepcopy(self.study)
        changed_cap['models']['gpt']['card']['max_tokens'] = 1024
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest['records'][0]['horizon'] = 1024
        changed_source = {**self.source, 'sha256': 'different-source-receipt'}
        cases = (
            (changed_sampler, self.manifest, self.source),
            (changed_cap, self.manifest, self.source),
            (self.study, changed_manifest, self.source),
            (self.study, self.manifest, changed_source),
        )
        for study, manifest, source in cases:
            with self.subTest(study=study, manifest=manifest, source=source):
                with self.assertRaisesRegex(ValueError, 'execution identity differs'):
                    rescue_source.bind_execution(study, 'gpt', manifest, source)
                self.assertEqual(self.receipt.read_bytes(), original)

    def test_fast_store_rejects_before_any_write(self):
        study = {**self.study, 'config': {'bulk_root': str(FAST/'uncreated-execution-guard-test')}}
        with self.assertRaisesRegex(ValueError, 'fresh HOME/WORK binding'):
            rescue_source.bind_execution(study, 'gpt', self.manifest, self.source)
        self.assertFalse(self.receipt.exists())

    def test_driver_rejects_unbound_store_before_recovery_or_engine(self):
        from moe_exp.study import driver

        folder = self.units/('a'*64)
        folder.mkdir(parents=True)
        (folder/'RESULT.json').write_text('legacy state\n')
        manifests = self.root/'manifests'
        publish(manifests/'gpt'/'RESCUE.json', {key: value for key, value in self.manifest.items() if key != 'sha256'})
        with patch.object(driver, 'binding', return_value=self.study), \
                patch.object(driver, 'DRIVER', manifests), \
                patch.object(rescue_source, 'verify', return_value=self.source), \
                patch.object(driver, 'recover_outputs') as recovery, \
                patch.object(driver, 'hooked_llm') as engine:
            with self.assertRaisesRegex(ValueError, 'requires --cohort'):
                driver.run_model('gpt', (0, 1), 0, design='rescue', source_receipt=self.source_path)
        recovery.assert_not_called()
        engine.assert_not_called()
        self.assertEqual((folder/'RESULT.json').read_text(), 'legacy state\n')
        self.assertFalse(self.receipt.exists())

    def test_completion_requires_complete_results_for_every_assigned_state(self):
        """Keep uncommitted, unknown and failed assignments visible in completion counts."""
        from moe_exp.study import driver

        rows = [{'uid': f'{index:064x}'} for index in range(7)]
        store = driver.Store(self.units)
        statuses = ('complete', 'unknown', 'execution_failed', 'resource_limited')
        for row, status in zip(rows, statuses):
            publish(store.folder(row['uid'])/'RESULT.json', {'status': status})
        publish(store.folder(rows[4]['uid'])/'OUTPUT.json', {'token_ids': [1]})
        driver.checkpoint(store.folder(rows[5]['uid'])/'PARTIAL.json', {'tokens': [1]})
        result = driver.rescue_completion(store, rows)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['rows'], 7)
        self.assertEqual(result['state_counts'], {
            'complete': 1,
            'unknown': 1,
            'execution_failed': 1,
            'resource_limited': 1,
            'output': 1,
            'partial': 1,
            'untouched': 1,
        })
        self.assertEqual(driver.rescue_completion(store, rows[:1])['status'], 'complete')

    def test_no_pending_unknown_is_persisted_as_incomplete_without_engine(self):
        """An empty runnable queue cannot convert an unknown result into success."""
        from moe_exp.study import driver

        study = rescue_source.load_study_binding()
        rescue_source.bind_execution(study, 'gpt', self.manifest, self.source)
        row = self.manifest['records'][0]
        result_path = self.units/row['uid']/'RESULT.json'
        publish(result_path, {'uid': row['uid'], 'status': 'unknown'})
        original = result_path.read_bytes()
        manifests = self.root/'manifests'
        publish(manifests/'gpt'/'RESCUE.json', {key: value for key, value in self.manifest.items() if key != 'sha256'})
        with patch.object(driver, 'DRIVER', manifests), \
                patch.object(driver, 'shard_rows', return_value=[row]), \
                patch.object(driver, 'code_hashes', return_value={}), \
                patch.object(driver, 'hooked_llm') as engine, \
                patch.dict(os.environ, {'SLURM_JOB_ID': 'completion-test'}):
            result = driver.execute_rows(study, 'gpt', [row], driver.Store(self.units), 0,
                                         design='rescue', max_inflight=8, execution={'cohort': 'frozen'}, pending_uids=set())
        engine.assert_not_called()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['state_counts'], {'unknown': 1})
        receipt = verified(self.root/'bulk'/'driver'/'gpt'/'jobs'/'rescue-completion-test.json')
        self.assertEqual(receipt['status'], 'incomplete')
        self.assertEqual(receipt['state_counts'], {'unknown': 1})
        self.assertEqual(result_path.read_bytes(), original)

    def test_cli_rescue_incomplete_exits_nonzero_and_other_designs_keep_checkpoint_behavior(self):
        """Only rescue requires scientific completion for a successful CLI exit."""
        from moe_exp.study import driver

        allocation = {'SLURM_JOB_ID': 'cli-fixture', 'SLURM_JOB_GPUS': '0,1'}
        for design in ('rescue', 'main', 'strong'):
            with self.subTest(design=design), patch.dict(os.environ, allocation), \
                    patch.object(driver, 'deadline_from_slurm', return_value=123.), \
                    patch.object(driver, 'run_model', return_value={'status': 'incomplete'}) as run:
                arguments = ['run', '--model', 'gpt', '--design', design]
                if design == 'rescue':
                    arguments += ['--source-receipt', str(self.source_path)]
                    with self.assertRaises(SystemExit) as error:
                        driver.main(arguments)
                    self.assertEqual(error.exception.code, 2)
                else:
                    driver.main(arguments)
                self.assertEqual(run.call_args.args[:3], ('gpt', (0, 1), 123.))
        with patch.dict(os.environ, allocation), \
                patch.object(driver, 'deadline_from_slurm', return_value=123.), \
                patch.object(driver, 'run_model', return_value={'status': 'complete'}):
            driver.main(['run', '--model', 'gpt', '--design', 'rescue', '--source-receipt', str(self.source_path)])


if __name__ == '__main__':
    unittest.main()
