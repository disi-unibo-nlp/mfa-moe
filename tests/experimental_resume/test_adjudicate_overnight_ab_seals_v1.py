"""Preserve raw evidence and old passing seals; adapt only the bad bound."""
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
import adjudicate_overnight_ab_seals_v1 as recovery


class AdjudicationTests(unittest.TestCase):
    def fixture(self, root, old_error=recovery.ORIGINAL_DIMENSION_FAILURE):
        shard = root / 'shard-000'; shard.mkdir()
        binding = dense.save(shard / 'BINDING.json', {'manifest_sha256': 'manifest'})
        summary = dense.save(shard / 'SUMMARY.json', {'manifest_sha256': 'manifest',
                            'binding_sha256': binding['sha256'], 'batches': 1})
        dense.save(shard / 'batch-000-assignment.json', {'manifest_sha256': 'manifest',
                   'binding_sha256': binding['sha256'], 'requests': [{'uid': 'u'}]})
        dense.save(shard / 'batch-000.json', {'manifest_sha256': 'manifest',
                   'binding_sha256': binding['sha256'], 'outputs': [{'uid': 'u'}], 'array_sha256': 'array'})
        core = {'schema': 'overnight-routing-completion-v1', 'manifest_sha256': 'manifest',
                'source_summary_sha256': summary['sha256'], 'shard': 0, 'counts': {'assigned': 1},
                'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION'}
        def original(result, manifest):
            if old_error:
                raise ValueError(old_error)
            return {'status': 'PASS'}
        common = SimpleNamespace(base=SimpleNamespace(sealed=dense.sealed, digest=dense.digest),
                                 require=dense.require, audit_output_dose=original, write_once=dense.save)
        def seal(out, manifest, index):
            common.audit_output_dose({'uid': 'u'}, manifest)
            return common.write_once(shard / 'OVERNIGHT_COMPLETION.json', core)
        common.seal_shard = seal
        corrected = SimpleNamespace(audit_output_dose=lambda *a: {'status': 'pulse_closure_TP_dose_pass'})
        return common, corrected, core

    def test_new_seal_links_independent_adjudication_and_repeat_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temporary:
            root = Path(temporary)
            common, corrected, core = self.fixture(root)
            result, proof = recovery.seal_one(common, corrected, root, {'sha256': 'manifest'}, 0, {'plan': 'p'})
            self.assertEqual(proof['original_dimension_rejections'], 1)
            self.assertEqual(result['dose_adjudication']['sha256'], proof['sha256'])
            self.assertTrue(all(result[k] == v for k, v in core.items()))
            again, _ = recovery.seal_one(common, corrected, root, {'sha256': 'manifest'}, 0, {'plan': 'p'})
            self.assertEqual(result, again)

    def test_original_passing_seal_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temporary:
            root = Path(temporary)
            common, corrected, core = self.fixture(root, old_error=None)
            path = root / 'shard-000/OVERNIGHT_COMPLETION.json'
            original = dense.save(path, core); before = path.read_bytes()
            result, proof = recovery.seal_one(common, corrected, root, {'sha256': 'manifest'}, 0, {'plan': 'p'})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result, original)
            self.assertEqual(proof['original_dimension_rejections'], 0)

    def test_other_old_or_new_failures_cannot_be_adjudicated(self):
        for old_error, corrected_error in [('inactive routing mismatch', None),
                                            (recovery.ORIGINAL_DIMENSION_FAILURE, 'TP rank mismatch')]:
            with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temporary:
                root = Path(temporary)
                common, corrected, _ = self.fixture(root, old_error)
                if corrected_error:
                    def reject(*a): raise ValueError(corrected_error)
                    corrected.audit_output_dose = reject
                with self.assertRaises(ValueError):
                    recovery.seal_one(common, corrected, root, {'sha256': 'manifest'}, 0, {'plan': 'p'})
                self.assertFalse((root / 'shard-000/OVERNIGHT_COMPLETION.json').exists())

    def test_changed_existing_completion_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temporary:
            root = Path(temporary)
            common, _, core = self.fixture(root)
            path = root / 'completion.json'
            dense.save(path, {**core, 'counts': {'assigned': 0}})
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'completion core'):
                recovery.preserved_or_new(common, path, core, {'sha256': 'proof'})
            self.assertEqual(before, path.read_bytes())

    def test_frozen_dense_first_stage_join_accepts_explicit_adjudication_without_old_checker(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests') as temporary:
            root = Path(temporary); shard = root / 'shard-000'; shard.mkdir()
            runner_source = root / 'runner.py'; runner_source.write_text('# synthetic frozen runner\n')
            manifest = {'schema': 'overnight-routing-manifest-v1', 'sha256': 'manifest', 'rows': [{}],
                        'arms': [], 'expected_requests': 1, 'shards': [{}],
                        'code_files': {str(runner_source): dense.file_sha(runner_source)}}
            meta = {'uid': 'u', 'prefix_uid': 'p', 'family': 'f', 'arm': 'native', 'seed': 0}
            def forbidden(*a): raise AssertionError('old checker or validation was called')
            runner = SimpleNamespace(__file__=str(runner_source), validate=forbidden,
                                     request_metadata=lambda *a: [(None, meta)])
            binding = dense.save(shard / 'BINDING.json', {'manifest_sha256': 'manifest'})
            summary = dense.save(shard / 'SUMMARY.json', {'manifest_sha256': 'manifest',
                'binding_sha256': binding['sha256'], 'batches': 1})
            provenance = {'status': 'PASS_CORRECTED_CPU_DOSE_AUDIT', 'sha256': 'independent-proof'}
            completion = dense.save(shard / 'OVERNIGHT_COMPLETION.json', {'manifest_sha256': 'manifest',
                'source_summary_sha256': summary['sha256'], 'dose_adjudication': provenance})
            stage = dense.save(root / 'STAGE_COMPLETION.json', {'schema': 'overnight-routing-stage-completion-v1',
                'manifest_sha256': 'manifest', 'counts': {'assigned': 1},
                'shard_completion_sha256s': [completion['sha256']], 'dose_adjudication': provenance})
            np.savez_compressed(shard / 'batch-000.npz', u=np.tile(np.arange(8), (3, 40, 1)))
            dense.save(shard / 'batch-000-assignment.json', {'manifest_sha256': 'manifest',
                'binding_sha256': binding['sha256'], 'requests': [meta]})
            dense.save(shard / 'batch-000.json', {'manifest_sha256': 'manifest',
                'binding_sha256': binding['sha256'], 'array_sha256': dense.file_sha(shard / 'batch-000.npz'),
                'outputs': [{**meta, 'tokens': [0, 1, 2], 'routed_present': True}]})
            price = {'manifest_sha256': 'manifest', 'shards': manifest['shards'], 'status': 'PASS_COMPLETE_STAGE_GENERATION_ONLY'}
            with patch.object(dense.importlib, 'import_module', return_value=runner), \
                 patch.object(dense.subprocess, 'run', side_effect=forbidden):
                self.assertEqual(dense.seal_generation_if_needed(manifest, root / 'm.json', root), stage)
                accepted, _, found, _, _ = dense.source_outputs(manifest, price, root)
                self.assertEqual(accepted, stage)
                self.assertEqual(len(found), 1)


if __name__ == '__main__':
    unittest.main()
