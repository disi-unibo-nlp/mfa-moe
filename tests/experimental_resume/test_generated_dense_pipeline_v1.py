"""Exact dense joins, gap handling, blind inputs and IDs-only route semantics."""
from collections import UserDict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/experimental_resume'), str(REPO / 'src')]
import generated_dense_pipeline_v1 as dense
from moe_exp.routing_control.analysis import class_summary


class ExactTokenizer:
    def __init__(self):
        from tokenizers import Tokenizer, models, decoders
        self.backend_tokenizer = Tokenizer(models.BPE(
            vocab={'A': 0, '.': 1, 'ĠB': 2, 'Ċ': 3, 'â': 4, 'Ĥ': 5, '¬': 6,
                   'A.ĠB': 7, 'ĠC': 8}, merges=[]))
        self.backend_tokenizer.decoder = decoders.ByteLevel()

    def decode(self, ids, **kwargs):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()


class PromptTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return UserDict(input_ids=list(range(32)), attention_mask=[1] * 32)


class GeneratedDenseTests(unittest.TestCase):
    def test_exact_unicode_offset_ownership_and_reasoning_closure(self):
        tokenizer = ExactTokenizer()
        layout = dense.sentence_layout(tokenizer, [0], [0, 1, 2, 3, 4, 5, 6, dense.THINK_END, 0])
        self.assertEqual(layout['reasoning_end'], 7)
        self.assertEqual(layout['text'], 'A. B\n€')
        self.assertEqual(layout['offsets'][4:7], [[5, 6]] * 3)
        self.assertTrue(layout['closed_reasoning'])
        self.assertEqual([s['text'] for s in layout['sentences']], ['A.', 'B', '€'])
        self.assertEqual(layout['token_owner'][4:7], [2, 2, 2])

    def test_boundary_merged_token_is_never_duplicated_or_retokenized(self):
        layout = dense.sentence_layout(ExactTokenizer(), [0], [7])
        self.assertEqual(layout['text'], 'A. B')
        self.assertEqual(len(layout['offsets']), 1)
        self.assertEqual(layout['sentences'][0]['token_indices'], [0])
        self.assertEqual(layout['sentences'][1]['token_indices'], [])
        self.assertFalse(layout['sentences'][-1]['complete'])

    def test_blind_classifier_allowlist_excludes_arm_seed_dose_and_outcomes(self):
        layout = dense.sentence_layout(ExactTokenizer(), [0], [0, 1, 2, dense.THINK_END])
        rows = dense.make_blind_inputs(layout, 'original problem', 'trigger sentence', 'private-arm-id', b'x' * 32)
        self.assertEqual(set(rows[0]), {'blind_id', 'inputs'})
        self.assertEqual(set(rows[0]['inputs']), {'problem_statement', 'previous_sentence', 'sentence', 'next_sentence'})
        self.assertEqual(rows[0]['inputs']['previous_sentence'], 'trigger sentence')
        self.assertEqual(rows[0]['inputs']['sentence'], 'A.')
        self.assertNotIn('private-arm-id', str(rows))
        self.assertEqual(len({r['blind_id'] for r in rows}), 2)

    def test_unparsed_middle_label_breaks_transition_and_loop_adjacency(self):
        tokens = [0, 1, 2, 1, 8, 1, dense.THINK_END]
        record = dense.sentence_layout(ExactTokenizer(), [0], tokens)
        blind = dense.make_blind_inputs(record, 'problem', 'trigger', 'uid', b'x' * 32)
        record.update(emitted_token_ids=tokens, emitted_token_ids_sha256=dense.digest(tokens))
        labels = {row['blind_id']: {'label': label, 'finish_reason': 'stop'}
                  for row, label in zip(blind, ['Read', None, 'Verify'])}
        valid, all_rows = dense.joined_sentences(record, labels)
        self.assertEqual([r['sentence_index'] for r in valid], [0, 2])
        self.assertEqual(sum(map(sum, class_summary(valid)['transition_counts'])), 0)
        self.assertEqual(all_rows[1]['measurement_status'], 'unparsed_or_capped_label')
        record['token_owner'][0] = 99
        with self.assertRaisesRegex(ValueError, 'ownership'):
            dense.joined_sentences(record, labels)

    def test_id_frequency_profiles_never_claim_gate_weights(self):
        routes = np.tile(np.arange(8, dtype=np.int16), (192, 40, 1))
        routes[64:] += 8
        value = dense.id_routing_profiles(routes)
        self.assertEqual(value['complete_windows'], 3)
        self.assertAlmostEqual(value['selection_frequency_velocity'][0], 1 / 64)
        self.assertEqual(value['selection_frequency_velocity'][1], 0)
        self.assertAlmostEqual(value['selection_frequency_acceleration'][0], -1 / 4096)
        self.assertIsNone(value['gate_distribution_kinematics'])
        self.assertIn('UNAVAILABLE', value['gate_distribution_status'])
        self.assertAlmostEqual(value['mean_adjacent_expert_set_turnover'], 1 / 191)
        with self.assertRaisesRegex(ValueError, 'unique'):
            dense.id_routing_profiles(np.zeros((5, 40, 8), dtype=np.int16))

    def test_complete_price_counts_every_exact_prompt_and_full_decode_cap(self):
        frame = {'sha256': 'source-frame', 'records': [
            {'blind_id': str(i), 'inputs': {'problem_statement': 'p', 'previous_sentence': 'b',
                                         'sentence': 's', 'next_sentence': 'n'}} for i in range(64)]}
        price = dense.price_frame(frame, PromptTokenizer(), 7200)
        self.assertEqual(price['sentences'], 64)
        self.assertEqual(price['prompt_tokens_exact'], 64 * 32)
        self.assertEqual(price['maximum_decode_tokens'], 64 * 1024)
        self.assertEqual(price['shards'][0]['start'], 0)
        self.assertEqual(price['shards'][-1]['stop'], 64)
        self.assertTrue(all(s['complete_wall_seconds'] <= 7200 for s in price['shards']))
        self.assertGreater(price['complete_stage_GPU_h'], 0)

    def test_saved_labels_bind_exact_prompt_ids_and_attempt_receipt(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            out = Path(temporary)
            binding = {'sha256': 'binding'}
            prompt = {'blind_id': 'blind', 'prompt_tokens': 32, 'prompt_ids_sha256': dense.digest(list(range(32)))}
            attempt = dense.save(out / 'attempts/0000000.json', {'binding_sha256': 'binding',
                'start': 0, 'stop': 1, 'blind_ids': ['blind']})
            body = {'schema': 'generated-dense-label-batch-v1', 'binding_sha256': 'binding',
                    'attempt_sha256': attempt['sha256'], 'start': 0, 'stop': 1, 'records': [
                        {**prompt, 'label': 'Read', 'raw_completion': 'Read', 'finish_reason': 'stop', 'generated_tokens': 1}]}
            dense.save(out / 'batches/0000000.json', body)
            self.assertIsNotNone(dense.committed_labels(out, 0, 1, binding, [prompt]))
            with self.assertRaisesRegex(ValueError, 'token identity'):
                dense.committed_labels(out, 0, 1, binding, [{**prompt, 'prompt_ids_sha256': 'changed'}])

    def test_generation_join_rejects_changed_route_archive(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            root = Path(temporary); shard = root / 'shard-000'; shard.mkdir()
            runner_source = root / 'runner_source.py'; runner_source.write_text('# frozen fake fixture\n')
            manifest = {'schema': 'overnight-routing-manifest-v1', 'sha256': 'manifest', 'rows': [{}],
                        'arms': [], 'expected_requests': 1, 'shards': [{'start_row': 0, 'end_row': 1}],
                        'code_files': {str(runner_source.resolve()): dense.file_sha(runner_source)}}
            meta = {'uid': 'u', 'prefix_uid': 'p', 'family': 'f', 'arm': 'native', 'seed': 0}
            runner = SimpleNamespace(__file__=str(runner_source), base=SimpleNamespace(__file__='source'), validate=lambda *a: None,
                                     request_metadata=lambda *a: [(None, meta)])
            binding = dense.save(shard / 'BINDING.json', {'manifest_sha256': 'manifest'})
            summary = dense.save(shard / 'SUMMARY.json', {'manifest_sha256': 'manifest',
                                'binding_sha256': binding['sha256'], 'batches': 1})
            completion = dense.save(shard / 'OVERNIGHT_COMPLETION.json', {'manifest_sha256': 'manifest',
                                    'source_summary_sha256': summary['sha256']})
            dense.save(root / 'STAGE_COMPLETION.json', {'schema': 'overnight-routing-stage-completion-v1',
                'manifest_sha256': 'manifest', 'counts': {'assigned': 1}, 'shard_completion_sha256s': [completion['sha256']]})
            np.savez_compressed(shard / 'batch-000.npz', u=np.tile(np.arange(8), (3, 40, 1)))
            dense.save(shard / 'batch-000-assignment.json', {'manifest_sha256': 'manifest',
                        'binding_sha256': binding['sha256'], 'requests': [meta]})
            dense.save(shard / 'batch-000.json', {'manifest_sha256': 'manifest',
                       'binding_sha256': binding['sha256'], 'array_sha256': dense.file_sha(shard / 'batch-000.npz'),
                       'outputs': [{**meta, 'tokens': [0, 1, 2], 'routed_present': True}]})
            price = {'manifest_sha256': 'manifest', 'shards': manifest['shards'], 'status': 'PASS_COMPLETE_STAGE_GENERATION_ONLY'}
            with patch.object(dense.importlib, 'import_module', return_value=runner):
                stage, plan, found, arrays, receipts = dense.source_outputs(manifest, price, root)
                self.assertEqual(len(found), 1)
                np.savez_compressed(shard / 'batch-000.npz', u=np.zeros((3, 40, 8), dtype=np.int64))
                with self.assertRaisesRegex(ValueError, 'batch/array'):
                    dense.source_outputs(manifest, price, root)


if __name__ == '__main__':
    unittest.main()
