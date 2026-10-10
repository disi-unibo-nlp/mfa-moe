"""No allocations: amendment binding, immutable CPU decisions and r5 rescues."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import verification_study_v3_autodriver_r5 as driver
import verification_study_v3_autostages_r5 as auto
import verification_study_v3_semantic_format_r1 as fmt
from moe_exp.verification_study import semantics, stages, execution, closeout
from moe_exp.verification_study.contracts import digest, file_sha, load, save, seal

STAGING = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')
CP = Path(__file__).resolve().parents[2] / 'report/verification-study-v3/CAMPAIGN.json'
spec = importlib.util.spec_from_file_location('r5_inherited_lifecycle_contracts',
    Path(__file__).with_name('test_verification_autodriver_r4.py'))
inherited = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inherited)
inherited.driver = driver


class R5LifecycleTests(inherited.ControllerR4Tests):
    def setUp(self):
        super().setUp()
        binding = save(self.root / 'SEMANTIC_FORMAT_BINDING-r1.json', {'schema': 'fixture'})
        self.files[str(fmt.SCRIPT)] = file_sha(fmt.SCRIPT)
        self.files[str(self.root / 'SEMANTIC_FORMAT_BINDING-r1.json')] = file_sha(self.root / 'SEMANTIC_FORMAT_BINDING-r1.json')
        self.campaign = save(self.path, {**self.campaign, 'code_files': self.files}, replace=True)
        self.binding_patch = patch.object(fmt, 'load_binding', return_value=binding)
        self.binding_patch.start()

    def tearDown(self):
        self.binding_patch.stop()
        super().tearDown()


class R5BindingTests(unittest.TestCase):
    def test_actual_card_closes_exact_format_annex_and_helper(self):
        card = driver.controller_card(CP, 'bootstrap')
        binding = fmt.load_binding(CP)
        self.assertEqual(card['semantic_format_binding_sha256'], binding['sha256'])
        self.assertEqual(card['operational_version'], 'autodriver-r5')
        self.assertEqual(card['program_files'][str(fmt.SCRIPT)], file_sha(fmt.SCRIPT))
        self.assertEqual(card['program_files'][str(CP.parent / 'SEMANTIC_FORMAT_BINDING-r1.json')],
                         file_sha(CP.parent / 'SEMANTIC_FORMAT_BINDING-r1.json'))
        driver.validate_card(card)

    def test_changed_annex_identity_rejects_before_scheduler_action(self):
        card = driver.controller_card(CP, 'bootstrap')
        card = seal({**card, 'semantic_format_binding_sha256': '0' * 64})
        with patch.object(driver, 'command') as command, self.assertRaisesRegex(ValueError, 'amendment differs'):
            driver.validate_card(card)
        command.assert_not_called()

    def test_missing_annex_closure_rejects(self):
        card = driver.controller_card(CP, 'bootstrap')
        files = dict(card['program_files']); files.pop(str(CP.parent / 'SEMANTIC_FORMAT_BINDING-r1.json'))
        card = seal({**card, 'program_files': files})
        with self.assertRaisesRegex(ValueError, 'omits semantic amendment'):
            driver.validate_card(card)


class R5SemanticCpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cv3-format-stage-test-', dir=STAGING)
        self.root = Path(self.tmp.name)
        self.cp = self.root / 'CAMPAIGN.json'
        self.campaign = save(self.cp, {'schema': 'fixture', 'code_files': {'source': 'sha'}})
        self.binding = save(self.root / 'SEMANTIC_FORMAT_BINDING-r1.json', {'schema': 'fixture'})
        self.frame_path = self.root / 'AUDIT_FRAME.json'
        self.record = {'uid': 'u', 'family': 'f', 'split': 'apparent', 'stratum': 'unfiltered',
                       'candidate': {}, 'proposed': {'eligible': True, 'already_checked': False}}
        self.frame = save(self.frame_path, {'schema': 'verification-semantic-frame-v3',
            'model_sha256': 'm', 'records': [self.record]})
        self.ledger_path = self.root / 'LEDGER.json'
        self.ledger = save(self.ledger_path, {'schema': 'fixture', 'raw_source_sha256': 'unchanged'})
        self.metadata = {'semantic_format_binding_sha256': self.binding['sha256'],
                         'semantic_decision_ledger_sha256': self.ledger['sha256'],
                         'semantic_decision_ledger_path': str(self.ledger_path)}
        self.run = object.__new__(auto.Advance)
        self.run.path = self.cp; self.run.root = self.root; self.run.campaign = self.campaign
        self.run.home = self.root / 'autostages-r5'; self.run.actions = []
        self.run.semantic_binding = self.binding
        self.output = self.root / 'RECOGNIZER.json'
        self.cpu_patch = patch.object(closeout, 'require_cpu'); self.cpu_patch.start()
        self.inputs_patch = patch.object(self.run, 'semantic_inputs',
            side_effect=lambda *args: (dict(self.metadata), [self.ledger_path]))
        self.inputs_patch.start()

    def tearDown(self):
        self.inputs_patch.stop(); self.cpu_patch.stop(); self.tmp.cleanup()

    def producer(self):
        return stages.save(self.output, {'schema': 'fixture-output', 'status': 'PASS'})

    def execute(self, producer=None):
        return self.run.cpu('gpt', 'recognition', producer or self.producer,
                            [self.output], [self.frame_path])

    def test_metadata_is_part_of_first_sealed_output(self):
        result = self.execute()
        value = load(self.output)
        for key, expected in self.metadata.items():
            self.assertEqual(value[key], expected)
            self.assertEqual(result[key], expected)

    def test_unchanged_resume_does_not_recompute_or_rewrite(self):
        first = self.execute(); before = file_sha(self.output)
        function = Mock(side_effect=AssertionError('must reuse bound output'))
        second = self.execute(function)
        function.assert_not_called()
        self.assertEqual(first, second); self.assertEqual(before, file_sha(self.output))

    def test_existing_unbound_old_output_is_not_adopted(self):
        save(self.output, {'schema': 'fixture-output', 'status': 'PASS'})
        before = file_sha(self.output); producer = Mock()
        with self.assertRaisesRegex(ValueError, 'lacks matching amendment'):
            self.execute(producer)
        producer.assert_not_called(); self.assertEqual(before, file_sha(self.output))

    def test_changed_raw_decision_ledger_cannot_reuse_old_selection(self):
        self.execute(); before = file_sha(self.output)
        replacement = save(self.root / 'NEW_LEDGER.json', {'schema': 'fixture', 'raw_source_sha256': 'changed'})
        self.metadata['semantic_decision_ledger_sha256'] = replacement['sha256']
        with self.assertRaisesRegex(ValueError, 'lacks matching amendment'):
            self.execute()
        self.assertEqual(before, file_sha(self.output))

    def test_new_unbound_producer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'omitted mandatory'):
            self.execute(lambda: save(self.output, {'schema': 'fixture-output'}))

    def test_context_restores_parser_and_all_save_aliases_after_failure(self):
        old_parser = semantics.parse_rating
        writers = (stages.save, execution.save, closeout.save)
        def fail():
            self.assertIs(semantics.parse_rating, fmt.parse_rating)
            raise RuntimeError('synthetic producer interruption')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.execute(fail)
        self.assertIs(semantics.parse_rating, old_parser)
        self.assertEqual((stages.save, execution.save, closeout.save), writers)

    def test_writer_does_not_annotate_unrelated_output(self):
        other = self.root / 'OTHER.json'
        with auto.semantic_output_writer([self.output], self.metadata):
            stages.save(other, {'schema': 'fixture'})
        self.assertNotIn('semantic_format_binding_sha256', load(other))

    def test_real_recognizer_uses_adapter_without_waiving_support(self):
        contract = save(self.root / 'READER.json', {'schema': 'fixture', 'checkpoint': 'checkpoint'})
        directory = self.root / 'ratings'; directory.mkdir()
        text = '{"candidate_valid":true,"constraint_bound":true,"already_checked":false,"eligible":true}'
        for draw, response in ((0, text), (1, '```json\n' + text + '\n```')):
            save(directory / f'u.reader-{draw}.json', {'schema': 'verification-reader-receipt-v3',
                'uid': 'u', 'reader': draw, 'frame_sha256': self.frame['sha256'],
                'role': 'eligibility', 'input_sha256': digest(self.record), 'text': response,
                'finish_reason': 'stop', 'checkpoint': 'checkpoint',
                'reader_contract_sha256': contract['sha256'], 'runtime_sha256': 'runtime'})
        campaign = save(self.root / 'ACTUAL.json', {'schema': 'fixture', 'runtime_sha256': 'runtime',
                                                 'code_files': {'source': 'sha'}})
        with patch.object(stages, 'check_files'):
            self.execute(lambda: stages.recognition(self.root / 'ACTUAL.json', 'gpt', self.frame_path,
                                                    directory, self.output))
        value = load(self.output)
        self.assertEqual(value['decisions']['eligible']['apparent_family_precision'], 1.)
        self.assertEqual(value['status'], 'FAIL')
        self.assertIn('eligible:SUPPORT_OR_PRECISION', value['failure_reasons'])
        self.assertEqual(value['semantic_format_binding_sha256'], self.binding['sha256'])

    def test_request_preserves_explicit_annex_in_immutable_intent(self):
        path, value = self.run.request('gpt', 'rate', ['--role', 'eligibility'],
                                      label='fixture', population='discovery', count=1)
        self.assertEqual(value['semantic_format_binding_sha256'], self.binding['sha256'])
        self.assertEqual(value['semantic_format_binding_path'], str(self.cp.parent / 'SEMANTIC_FORMAT_BINDING-r1.json'))
        self.assertEqual(value, load(path))


if __name__ == '__main__':
    unittest.main()
