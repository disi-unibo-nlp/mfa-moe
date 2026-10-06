"""Recovery-only metadata and tamper checks; no model or GPU execution."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/experimental_resume'), str(REPO / 'src')]
import generated_dense_fresh_metadata_recovery_v1 as recovery


def sealed(body):
    body = {k: v for k, v in body.items() if k != 'sha256'}
    return {**body, 'sha256': recovery.digest(body)}


class DenseFreshRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'dense-recovery-fixture'
        self.directory.mkdir()
        self.tokenizer = self.directory / 'tokenizer'
        self.tokenizer.mkdir()
        (self.tokenizer / 'tokenizer_config.json').write_text('{"test":true}\n')
        self.start = {'uid': 'start', 'family': 'family', 'transition': 'candidate_to_verify',
                      'prefix_ids': [7], 'prompt_ids': [4, 7],
                      'prefix_ids_sha256': recovery.digest([7]),
                      'prompt_ids_sha256': recovery.digest([4, 7])}
        self.assigned = {'uid': 'generated', 'prefix_uid': 'start', 'family': 'family',
                         'transition': 'candidate_to_verify', 'arm': 'bias', 'seed': 0}
        self.manifest = sealed({'schema': 'overnight-routing-manifest-v2', 'rows': [self.start],
                                'expected_requests': 1, 'horizon': 4,
                                'source_frame_path': '/original/frozen/source.json',
                                'source_frame_sha256': 'original-source-seal'})
        # The raw fresh source lacks these top-level identities. The checked
        # normalizer derives them from the selection, with its own seal.
        self.raw_original = {'uid': 'start', 'transition': 'candidate_to_verify',
             'reader_input': {'problem': 'Original problem', 'emitted_prefix': 'Trigger.',
                              'triggering_sentence': 'Trigger.'},
             'analysis_meta': {'prefix_ids_sha256': self.start['prefix_ids_sha256'],
                               'prompt_ids_sha256': self.start['prompt_ids_sha256']}}
        self.derived = sealed({'schema': 'fresh-start-context-normalization-v1',
             'source_frame_sha256': self.manifest['source_frame_sha256'],
             'records': [{**self.raw_original, 'family': 'family', 'prefix_tokens': 1}]})
        self.provenance = {'amendment_sha256': 'recovery-amendment',
                           'source_frame_sha256': self.manifest['source_frame_sha256'],
                           'normalized_context_sha256': self.derived['sha256']}
        self.codes = {'/frozen/native.py': 'native-code', '/new/adapter.py': 'adapter-code'}
        self.layout = {'text': 'Answer.', 'offsets': [[0, 7]], 'token_owner': [0],
             'reasoning_end': 1, 'closed_reasoning': False,
             'sentences': [{'sentence_index': 0, 'segment': 0, 'char_start': 0, 'char_end': 7,
                            'text': 'Answer.', 'token_indices': [0], 'token_start': 0,
                            'token_end': 1, 'complete': True}]}

    def build(self):
        outputs = [{**self.assigned, 'tokens': [2], 'finish': 'length', 'error': False}]
        source_outputs = (sealed({'assigned': 1}), [self.assigned], outputs,
                          {'generated': {'path': '/raw/routes.npz', 'sha256': 'routes',
                                         'key': 'generated', 'available': True}}, [])
        with patch.object(recovery, 'checked_recovery', return_value=(
                  {'generation_out': str(self.directory)}, self.derived,
                  {'start': self.raw_original['reader_input']}, self.provenance)), \
             patch.object(recovery, 'output_path', return_value=self.directory), \
             patch.object(recovery, 'measurement_code', return_value=self.codes), \
             patch.object(recovery.native, 'source_outputs', return_value=source_outputs), \
             patch.object(recovery.native, 'sentence_layout', return_value=deepcopy(self.layout)), \
             patch.object(recovery.native, 'TOKENIZER', self.tokenizer):
            arm_map, frame = recovery.build_frame(self.manifest, {}, self.directory,
                                                   self.directory, None)
        price = sealed({'frame_sha256': frame['sha256'], 'recovery_provenance': self.provenance})
        return arm_map, frame, price

    def validate(self, arm_map, frame, price, derived=None):
        runner = SimpleNamespace(request_metadata=lambda *args: [(None, self.assigned)])
        with patch.object(recovery, 'measurement_code', return_value=self.codes), \
             patch.object(recovery, 'output_path', return_value=self.directory), \
             patch.object(recovery.native, 'TOKENIZER', self.tokenizer), \
             patch.dict(sys.modules, {'overnight_routing_runner_v2': runner}):
            recovery.validate_recovered_inputs(self.directory, self.manifest, frame, price,
                arm_map, self.derived if derived is None else derived, self.provenance)

    def test_missing_top_level_source_metadata_uses_separately_sealed_normalization(self):
        self.assertNotIn('family', self.raw_original)
        self.assertNotIn('prefix_tokens', self.raw_original)
        arm_map, frame, price = self.build()
        self.validate(arm_map, frame, price)
        normalized = recovery.sealed(self.directory / 'NORMALIZED_SOURCE_CONTEXT.json')
        self.assertEqual(normalized, self.derived)
        self.assertEqual(frame['source_frame_sha256'], 'original-source-seal')
        self.assertNotEqual(frame['source_frame_sha256'], frame['normalized_context_sha256'])
        self.assertEqual(frame['assigned'], 1)
        self.assertEqual(arm_map['records'][0]['emitted_token_ids_sha256'], recovery.digest([2]))
        self.assertEqual(set(frame['records'][0]['inputs']),
                         {'problem_statement', 'previous_sentence', 'sentence', 'next_sentence'})
        self.assertNotIn('bias', str(frame['records']))

    def test_normalized_source_cannot_be_advertised_as_original_source(self):
        arm_map, frame, price = self.build()
        for value in (arm_map, frame):
            value['source_frame_sha256'] = self.derived['sha256']
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.validate(sealed(arm_map), sealed(frame), price)

    def test_resealed_changed_normalization_is_rejected(self):
        arm_map, frame, price = self.build()
        derived = deepcopy(self.derived)
        derived['records'][0]['family'] = 'different-family'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.validate(arm_map, frame, price, sealed(derived))

    def test_resealed_changed_assignment_uid_is_rejected(self):
        arm_map, frame, price = self.build()
        arm_map['records'][0]['uid'] = 'unassigned'
        with self.assertRaisesRegex(ValueError, 'assignment'):
            self.validate(sealed(arm_map), frame, price)

    def test_resealed_changed_blind_problem_text_is_rejected(self):
        arm_map, frame, price = self.build()
        frame['records'][0]['inputs']['problem_statement'] = 'Different problem'
        with self.assertRaisesRegex(ValueError, 'text/context'):
            self.validate(arm_map, sealed(frame), price)

    def test_unsealed_metadata_and_missing_adapter_provenance_are_rejected(self):
        arm_map, frame, price = self.build()
        frame['horizon'] = 999
        with self.assertRaisesRegex(ValueError, 'seal'):
            self.validate(arm_map, frame, price)
        frame = sealed(frame)
        price['recovery_provenance'] = {}
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.validate(arm_map, frame, sealed(price))

    def test_resealed_changed_horizon_or_continuation_tokens_are_rejected(self):
        arm_map, frame, price = self.build()
        frame['horizon'] = 999
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.validate(arm_map, sealed(frame), price)
        frame['horizon'] = self.manifest['horizon']
        arm_map['records'][0]['emitted_token_ids'] = [99]
        with self.assertRaisesRegex(ValueError, 'token/horizon'):
            self.validate(sealed(arm_map), sealed(frame), price)


if __name__ == '__main__':
    unittest.main()
