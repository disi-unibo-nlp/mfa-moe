"""CPU-only contract and recovery checks for the separate extension readers."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import mechanism_extension_reader_contract_v1 as contract  # noqa: E402
import rate_mechanism_extension_start_readers_v1 as reader  # noqa: E402
import seal_mechanism_extension_reader_v1 as sealer  # noqa: E402


class ExtensionReaderTests(unittest.TestCase):
    def test_real_source_and_qualification_plan_bind_all_ratings(self):
        family, selection, frame, price = contract.source_inputs()
        manifest = sealer.prepare_body(family, selection, frame, price)
        self.assertEqual(len(family['families']), 220)
        self.assertEqual(frame['rows'], 758)
        self.assertEqual(price['ratings'], 4 * frame['rows'])
        self.assertEqual(price['max_decode_tokens'], 4 * frame['rows'] * 1024)
        self.assertEqual(manifest['reader_channels'], [
            {'kind': kind, 'reader': reader_id}
            for kind in contract.KINDS for reader_id in contract.READERS])
        self.assertEqual(set(manifest['sample_transitions']), set(
            ('candidate_to_verify', 'approach_to_commit')))
        self.assertLess(manifest['qualification_wall_seconds_ceiling'], 3600)
        self.assertEqual(manifest['output_root'], str(contract.OUTPUT_ROOT))

    def test_primary_and_specific_prior_completion_sensitivity_denominators(self):
        rows = [
            {'uid': 'a', 'primary_readers': [
                {'rating': {'start': True}, 'finish_reason': 'stop', 'generated_tokens': 1}] * 2,
             'veto_readers': [
                {'already_completed': False, 'finish_reason': 'stop', 'generated_tokens': 1}] * 2},
            {'uid': 'b', 'primary_readers': [
                {'rating': {'start': True}, 'finish_reason': 'stop', 'generated_tokens': 1}] * 2,
             'veto_readers': [
                {'already_completed': True, 'finish_reason': 'stop', 'generated_tokens': 1}] * 2},
        ]
        counts = reader.summarize_records(rows, {'a': 'family-a', 'b': 'family-b'})
        self.assertEqual(counts['rows'], 2)
        self.assertEqual(counts['ratings'], 8)
        self.assertEqual(counts['primary_accepted_starts'], 2)
        self.assertEqual(counts['strict_sensitivity_starts'], 1)
        self.assertEqual(counts['primary_accepted_families'], 2)
        self.assertEqual(counts['strict_sensitivity_families'], 1)

    def test_crashed_channel_attempt_is_preserved_before_committed_retry(self):
        block = [{'uid': 'uid-a', 'transition': 'candidate_to_verify'},
                 {'uid': 'uid-b', 'transition': 'approach_to_commit'}]
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as folder:
            out = Path(folder)
            for name in ('attempts', 'assignments', 'batches'):
                (out / name).mkdir()
            binding_sha = 'synthetic-binding'
            for kind in contract.KINDS:
                for reader_id in contract.READERS:
                    attempts = []
                    if kind == 'primary' and reader_id == 0:
                        attempts.append(self._attempt(out, block, binding_sha,
                                                      kind, reader_id, 0, []))
                    index = len(attempts)
                    committed = self._attempt(out, block, binding_sha,
                                              kind, reader_id, index, attempts)
                    for row in block:
                        contract.save(out / 'assignments' /
                                      f"{row['uid']}-{kind}-reader{reader_id}-attempt{index:03d}.json",
                                      {'schema': 'mechanism-extension-rating-assignment-v1',
                                       'binding_sha256': binding_sha, 'uid': row['uid'],
                                       'kind': kind, 'reader': reader_id, 'start': 0,
                                       'attempt_index': index,
                                       'attempt_sha256': committed['sha256'],
                                       'state': 'attempted_before_model_chat',
                                       'job_id': 'test'})
                    results = [({'rating': {'start': True}}
                                if kind == 'primary' else
                                {'already_completed': False}) |
                               {'finish_reason': 'stop', 'generated_tokens': 1,
                                'raw_completion': '{}'} for _ in block]
                    contract.save(out / 'batches' /
                                  f'000000-{kind}-reader{reader_id}.json',
                                  {'schema': 'mechanism-extension-reader-channel-batch-v1',
                                   'binding_sha256': binding_sha, 'start': 0,
                                   'kind': kind, 'reader': reader_id,
                                   'attempt_index': index,
                                   'attempt_sha256': committed['sha256'],
                                   'timing': {'kind': kind, 'reader': reader_id},
                                   'records': [{'uid': row['uid'], 'result': results[i]}
                                               for i, row in enumerate(block)]})
            self.assertEqual(len(reader.attempt_ledger(
                out, 0, 'primary', 0, block, binding_sha)), 2)
            combined = reader.combined_batch(out, 0, block, binding_sha)
            self.assertEqual(len(combined['records']), 2)
            self.assertEqual(len(list((out / 'attempts').glob('*.json'))), 5)
            self.assertEqual(reader.combined_batch(out, 0, block, binding_sha), combined)
            receipt = out / 'assignments' / 'uid-a-primary-reader0-attempt001.json'
            changed = json.loads(receipt.read_text())
            changed['uid'] = 'tampered'
            receipt.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'changed source seal'):
                reader.committed_channel(out, 0, 'primary', 0, block, binding_sha)

    def _attempt(self, out, block, binding_sha, kind, reader_id, index, earlier):
        return contract.save(
            out / 'attempts' / f'000000-{kind}-reader{reader_id}-attempt{index:03d}.json',
            {'schema': 'mechanism-extension-reader-attempt-v1',
             'binding_sha256': binding_sha, 'start': 0,
             'kind': kind, 'reader': reader_id, 'attempt_index': index,
             'recovery_of_uncommitted_attempts': [a['sha256'] for a in earlier],
             'uids': [r['uid'] for r in block],
             'seeds': [contract.rating_seed(r['uid'], kind, reader_id)
                       for r in block], 'job_id': 'test',
             'state': 'started_before_model_chat'})

    def test_price_rebind_rejects_nonpassing_qualification(self):
        family, selection, frame, source = contract.source_inputs()
        body = sealer.prepare_body(family, selection, frame, source)
        manifest = {**body, 'sha256': contract.digest(body)}
        result = {'schema': 'mechanism-extension-reader-qualification-result-v1',
                  'status': 'FAIL', 'pass': False,
                  'qualification_manifest_sha256': manifest['sha256'],
                  'records': []}
        with self.assertRaisesRegex(ValueError, 'qualification'):
            sealer.rebind_body(family, selection, frame, source, manifest, result)

    def test_price_rebind_accepts_exact_live_pass_shape(self):
        family, selection, frame, source = contract.source_inputs()
        body = sealer.prepare_body(family, selection, frame, source)
        manifest = {**body, 'sha256': contract.digest(body)}
        records = [
            {'kind': kind, 'reader': reader_id, 'uid': uid,
             'seed': contract.rating_seed(uid, kind, reader_id),
             'generated_tokens': 0}
            for kind in contract.KINDS for reader_id in contract.READERS
            for uid in manifest['sample_uids']]
        result = {
            'schema': 'mechanism-extension-reader-qualification-result-v1',
            'status': 'PASS', 'pass': True,
            'qualification_manifest_sha256': manifest['sha256'],
            'source_exact_price_sha256': source['sha256'],
            'rating_driver_sha256': contract.file_sha(contract.DRIVER),
            'contract_sha256': contract.file_sha(contract.__file__),
            'model_profile': contract.PROFILE,
            'sample_uids': manifest['sample_uids'],
            'reader_channels': manifest['reader_channels'],
            'ratings': manifest['ratings'], 'generated_tokens': 0,
            'prompt_ids_exact_pass': True,
            'model_output_count_pass': True,
            'cap_and_finish_pass': True,
            'records': records, 'sha256': 'synthetic-pass'}
        rebound = sealer.rebind_body(family, selection, frame,
                                     source, manifest, result)
        self.assertEqual(rebound['status'], 'PASS_COMPLETE_STAGE')
        self.assertEqual(rebound['source_exact_price_sha256'], source['sha256'])
        self.assertEqual(rebound['qualification_result_sha256'], 'synthetic-pass')
        self.assertEqual(rebound['ratings'], 3032)


if __name__ == '__main__':
    unittest.main()
