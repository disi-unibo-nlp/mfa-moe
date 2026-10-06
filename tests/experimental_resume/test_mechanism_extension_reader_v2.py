"""Recovery regression: Mapping IDs, unchanged prompts, separate attempts and pricing."""
from collections import UserDict
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import mechanism_extension_reader_contract_v1 as prior
import mechanism_extension_reader_contract_v2 as contract
import seal_mechanism_extension_reader_v2 as sealer


class TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class ExtensionRecoveryTests(unittest.TestCase):
    def test_mapping_and_single_prompt_containers_preserve_exact_ids(self):
        ids = list(range(32))
        for value in (ids, tuple(ids), [ids], UserDict(input_ids=ids),
                      UserDict(input_ids=[ids], attention_mask=[[1] * 32]),
                      TensorLike([ids]), {'input_ids': TensorLike([ids])}):
            with self.subTest(container=type(value).__name__):
                self.assertEqual(contract.exact_token_ids(value), ids)

    def test_ambiguous_or_malformed_prompt_ids_fail_closed(self):
        ids = list(range(32))
        for value in (None, {}, UserDict(attention_mask=ids), [ids, ids],
                      [[ids]], list(range(15)), ids + [True], ids + [-1],
                      ids + [1.0], ids + ['4'], '01234567890123456'):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'exact token IDs'):
                    contract.exact_token_ids(value)

    def test_recovery_changes_container_handling_without_source_or_seed_changes(self):
        family, selection, frame, source = contract.source_inputs()
        manifest = sealer.prepare_body(family, selection, frame, source)
        self.assertEqual(frame['rows'], 758)
        self.assertEqual(manifest['ratings'], 12)
        self.assertNotEqual(contract.OUTPUT_ROOT, prior.OUTPUT_ROOT)
        self.assertNotEqual(contract.QUAL_MANIFEST, prior.QUAL_MANIFEST)
        self.assertEqual(manifest['source_exact_price_sha256'], source['sha256'])
        self.assertEqual(manifest['recovery_provenance']['prior_job_id'], '59273283')
        for kind in contract.KINDS:
            row = frame['records'][0]
            self.assertEqual(contract.messages(row, kind), prior.messages(row, kind))
            for reader_id in contract.READERS:
                self.assertEqual(contract.rating_seed(row['uid'], kind, reader_id),
                                 prior.rating_seed(row['uid'], kind, reader_id))

    def test_parallel_price_preserves_ratings_and_adds_load_cost(self):
        family, selection, frame, source = contract.source_inputs()
        previous = sealer.sealed(sealer.extension.PRIOR_PRICE)
        primary, veto = [1000] * frame['rows'], [1500] * frame['rows']
        old = sealer.extension.price_reader(frame, selection, family, previous,
                                            primary, veto, 86400)
        new = sealer.extension.price_reader(frame, selection, family, previous,
                                            primary, veto, contract.PARALLEL_WALL_SECONDS)
        self.assertEqual(new['ratings'], old['ratings'])
        self.assertEqual(new['prompt_tokens_all_ratings_exact'], old['prompt_tokens_all_ratings_exact'])
        self.assertGreater(len(new['shards']), len(old['shards']))
        self.assertGreater(new['complete_stage_projected_GPU_h'], old['complete_stage_projected_GPU_h'])
        self.assertEqual(new['shards'][0]['start_row'], 0)
        self.assertEqual(new['shards'][-1]['end_row'], frame['rows'])
        overhead = (previous['components_seconds']['two_cold_loads'] +
                    previous['components_seconds']['two_shutdowns']) / 2 + 900
        for shard in new['shards']:
            self.assertLessEqual(shard['estimated_work_seconds'] + overhead, 7200)
            self.assertEqual(shard['start_row'] % 16, 0)


if __name__ == '__main__':
    unittest.main()
