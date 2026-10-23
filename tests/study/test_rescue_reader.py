"""Reader regression checks for saved-driver paths, context identity and provenance."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from moe_exp.io import WORK, digest, file_sha, publish
from moe_exp.study import rescue_reader
from moe_exp.study.rescue_reader import RescueContexts, frozen_campaign, generation_matches, validate_inventory


class RescueReaderTest(unittest.TestCase):
    def setUp(self):
        """Build a saved-driver fixture with valid assignment/output/result joins."""
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = {
            'config': {'bulk_root': str(self.root/'bulk')},
            'source_campaign': str(self.root/'parent'),
            'models': {'gpt': {'profile_sha256': 'frozen-profile'}},
        }
        self.program = self.root/'program.json'
        self.program.write_text('{"prompt": "frozen program"}')
        campaign = publish(self.root/'parent'/'CAMPAIGN.json', {
            'annotator': {
                'config': {'program_sha256': file_sha(self.program)},
                'revision': 'frozen-reader',
                'program': str(self.program),
            },
        })
        self.study['source_campaign_sha256'] = file_sha(self.root/'parent'/'CAMPAIGN.json')
        self.campaign_receipt_sha256 = campaign['sha256']
        self.row = {
            'uid': 'a'*64,
            'family': 'held-out-family',
            'arm': 'target_activate',
            'horizon': 512,
            'seed': 17,
            'profile_sha256': 'frozen-profile',
        }
        self.raw = {
            'uid': self.row['uid'],
            'assignment_sha256': digest(self.row),
            'token_ids': [11, 12, 13],
            'total_allowance': 512,
            'segments': [{'job': 'qualified', 'history': 0, 'max_tokens': 512, 'seed': 17}],
            'finish_reason': 'stop',
            'profile_sha256': 'frozen-profile',
        }
        self.folder = self.root/'bulk'/'driver'/'gpt'/'rescue'/'units'/self.row['uid']
        raw = publish(self.folder/'OUTPUT.json', self.raw)
        publish(self.folder/'RESULT.json', {
            'uid': self.row['uid'],
            'assignment_sha256': digest(self.row),
            'raw_sha256': raw['sha256'],
            'status': 'complete',
        })
        self.rebuilt = self.replayed_sentence()

    @staticmethod
    def replayed_sentence():
        """Return a replay fixture with empty neighboring context preserved."""
        return {
            'sentences': [{
                'first_complete': True,
                'complete': True,
                'straddles_prefix': False,
                'sentence_index': 2,
                'token_start': 1,
                'token_end': 3,
                'inputs': {
                    'problem_statement': 'Frozen problem',
                    'previous_sentence': '',
                    'sentence': 'Check the calculation.',
                    'next_sentence': '',
                },
            }],
            'finish_reason': 'stop',
            'generated_tokens': 3,
            'reasoning_end': 3,
        }

    def builder(self):
        with patch('moe_exp.study.runtime.closure', return_value=[]):
            return RescueContexts(self.study, 'gpt', object(), {'jobs': ['qualified']})

    def test_driver_store_and_exact_reader_inputs(self):
        builder = self.builder()
        with patch('moe_exp.study.outcomes.record', return_value=self.rebuilt):
            first = builder.context(self.row)
            repeated = builder.context(self.row)
        self.assertEqual(first['status'], 'reader_task')
        self.assertEqual(first, repeated)
        self.assertEqual(len(builder.tasks), 1)
        task = builder.tasks[first['annotation_uid']]
        self.assertEqual(task['inputs'], self.rebuilt['sentences'][0]['inputs'])
        self.assertEqual(task['inputs']['previous_sentence'], '')
        self.assertEqual(task['inputs']['next_sentence'], '')

    def test_no_complete_sentence_and_extraction_failure_are_inventory_rows(self):
        builder = self.builder()
        rebuilt = {**self.rebuilt, 'sentences': []}
        with patch('moe_exp.study.outcomes.record', return_value=rebuilt):
            self.assertEqual(builder.context(self.row)['status'], 'no_complete_sentence')
        with patch('moe_exp.study.outcomes.record', side_effect=ValueError('bad replay')):
            row = builder.context(self.row)
        self.assertEqual(row['status'], 'extraction_failed')
        self.assertEqual(row['error'], 'ValueError: bad replay')
        self.assertFalse(builder.tasks)

    def test_straddling_sentence_cannot_be_first_complete_task(self):
        rebuilt = copy.deepcopy(self.rebuilt)
        rebuilt['sentences'][0]['straddles_prefix'] = True
        with patch('moe_exp.study.outcomes.record', return_value=rebuilt):
            with self.assertRaisesRegex(ValueError, 'straddles'):
                self.builder().context(self.row)

    def test_missing_generation_is_not_an_empty_success(self):
        missing = {**self.row, 'uid': 'b'*64}
        self.assertEqual(self.builder().context(missing)['status'], 'generation_unavailable')

    def test_uncapped_and_cross_revision_outputs_fail_provenance(self):
        self.assertTrue(generation_matches(self.row, self.raw, {'qualified'}))
        cases = []
        uncapped = copy.deepcopy(self.raw)
        uncapped['total_allowance'] = 130611
        cases.append(uncapped)
        old = copy.deepcopy(self.raw)
        old['segments'][0]['job'] = 'failed-old-operator'
        cases.append(old)
        mixed = copy.deepcopy(self.raw)
        mixed['segments'].insert(0, {'job': 'failed-old-operator', 'history': 0, 'generated': 1})
        mixed['segments'][-1]['history'] = 1
        cases.append(mixed)
        resumed = copy.deepcopy(self.raw)
        resumed['segments'][0]['history'] = 2
        cases.append(resumed)
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertFalse(generation_matches(self.row, raw, {'qualified'}))

    def test_empty_earlier_admission_does_not_change_the_generated_draw(self):
        raw = copy.deepcopy(self.raw)
        raw['segments'].insert(0, {'job': 'failed-old-operator', 'history': 0, 'generated': 0})
        self.assertTrue(generation_matches(self.row, raw, {'qualified'}))

    def test_actual_empty_admission_without_redundant_generated_field(self):
        raw = copy.deepcopy(self.raw)
        raw['segments'].insert(0, {'job': 'failed-old-operator', 'history': 0})
        self.assertTrue(generation_matches(self.row, raw, {'qualified'}))

    def test_source_mismatch_is_rejected_before_text_extraction(self):
        with patch('moe_exp.study.runtime.closure', return_value=[]):
            builder = RescueContexts(self.study, 'gpt', object(), {'jobs': ['another-release']})
        with patch('moe_exp.study.outcomes.record') as extraction:
            row = builder.context(self.row)
        self.assertEqual(row['status'], 'generation_source_mismatch')
        extraction.assert_not_called()

    def test_frozen_program_bytes_and_parent_identity_are_verified(self):
        self.assertNotEqual(self.campaign_receipt_sha256, self.study['source_campaign_sha256'])
        frozen_campaign(self.study)
        with self.assertRaisesRegex(ValueError, 'parent campaign differs'):
            frozen_campaign({**self.study, 'source_campaign_sha256': self.campaign_receipt_sha256})
        with self.assertRaisesRegex(ValueError, 'parent campaign differs'):
            frozen_campaign({**self.study, 'source_campaign_sha256': 'another-campaign'})
        self.program.write_text('{"prompt": "changed program"}')
        with self.assertRaisesRegex(ValueError, 'program bytes changed'):
            frozen_campaign(self.study)

    def test_recorded_generation_profile_must_match_study_profile(self):
        builder = self.builder()
        builder.study = copy.deepcopy(self.study)
        builder.study['models']['gpt']['profile_sha256'] = 'different-runtime-profile'
        with patch('moe_exp.study.outcomes.record') as extraction:
            with self.assertRaisesRegex(ValueError, 'generation profile differs'):
                builder.context(self.row)
        extraction.assert_not_called()

    def test_inventory_must_cover_every_assignment_and_exact_task_input(self):
        builder = self.builder()
        with patch('moe_exp.study.outcomes.record', return_value=self.rebuilt):
            inventory = builder.context(self.row)
        receipt = {
            'contexts': 1,
            'tasks': 1,
            'inventory': [inventory],
            'records': list(builder.tasks.values()),
        }
        manifest = {'records': [self.row]}
        validate_inventory(receipt, manifest, builder.annotator, 'gpt')
        missing = {**receipt, 'inventory': []}
        with self.assertRaisesRegex(ValueError, 'exact assignment roster'):
            validate_inventory(missing, manifest, builder.annotator, 'gpt')
        changed = copy.deepcopy(receipt)
        changed['records'][0]['inputs']['next_sentence'] = '<END OF RESPONSE>'
        with self.assertRaisesRegex(ValueError, 'frozen context identity'):
            validate_inventory(changed, manifest, builder.annotator, 'gpt')

    def test_incomplete_task_receipt_rejects_before_reader_construction(self):
        builder = self.builder()
        driver = self.root/'manifests'
        manifest = publish(driver/'gpt'/'RESCUE.json', {'records': [self.row]})
        release = {'jobs': ['qualified']}
        publish(driver/'gpt'/rescue_reader.TASK_FILE, {
            'model': 'gpt',
            'manifest_sha256': manifest['sha256'],
            'profile_sha256': 'frozen-profile',
            'annotator_sha256': digest(builder.annotator),
            'generation_release': release,
            'context_sources': {'reader.py': 'frozen'},
            'context_data': {'source_campaign_file_sha256': self.study['source_campaign_sha256']},
            'contexts': 1,
            'tasks': 0,
            'inventory': [{
                'uid': self.row['uid'],
                'family': self.row['family'],
                'arm': self.row['arm'],
                'status': 'generation_unavailable',
            }],
            'records': [],
        })
        with patch.object(rescue_reader, 'DRIVER', driver), \
                patch.object(rescue_reader, 'binding', return_value=self.study), \
                patch.object(rescue_reader, 'generation_release', return_value=release), \
                patch.object(rescue_reader, 'context_sources', return_value={'reader.py': 'frozen'}), \
                patch.object(rescue_reader, 'context_data', return_value={
                    'source_campaign_file_sha256': self.study['source_campaign_sha256'],
                }), patch('moe_exp.study.reader.Reader') as reader:
            with self.assertRaisesRegex(ValueError, 'requires --cohort'):
                rescue_reader.run('gpt', 0)
        reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
