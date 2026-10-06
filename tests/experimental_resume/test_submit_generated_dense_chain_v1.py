"""Bind stage paths and dependencies before attaching a dense measurement chain."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/experimental_resume'), str(REPO / 'src')]
import submit_generated_dense_chain_v1 as chain


class DenseChainTests(unittest.TestCase):
    def test_explicit_dependency_must_be_exact_parent_job_id(self):
        job, receipt = chain.frame_dependency('A', {}, '59344822')
        self.assertEqual(job, '59344822')
        self.assertEqual(receipt['job_id'], job)
        for invalid in ('123_2', '123:456', '1; sbatch', ''):
            with self.assertRaises(ValueError):
                chain.frame_dependency('A', {}, invalid)

    def test_v2_dependency_binds_generation_manifest(self):
        manifest = {'sha256': 'f' * 64}
        value = {'schema': 'overnight-generation-chain-v2', 'binding': {'manifest_sha256': manifest['sha256']},
                 'frame_price_job': '123', 'sha256': 'chain'}
        with patch.object(chain.dense, 'sealed', return_value=value):
            job, receipt = chain.frame_dependency('FRESH', manifest)
        self.assertEqual(job, '123')
        self.assertIn('GENERATION_CHAIN.json', receipt['path'])
        value['binding']['manifest_sha256'] = 'different'
        with patch.object(chain.dense, 'sealed', return_value=value):
            with self.assertRaises(ValueError):
                chain.frame_dependency('C', manifest)

    def test_cpu_prepare_follows_frame_price_then_dispatch_follows_prepare(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            folder = Path(temporary)
            paths = chain.stage_paths('A', folder / 'outputs')
            calls = []
            def submit(directory, name, args, env, binding):
                calls.append((name, args)); return str(4000 + len(calls))
            with patch.object(chain.shared, 'DOC', folder), patch.object(chain.dispatch, 'live_account_snapshot'), \
                    patch.object(chain.dispatch, 'ensure_verified') as verify:
                result = chain.submit_chain('A', paths, '59344822', {'job_id': '59344822'}, submit=submit)
        self.assertEqual([name for name, _ in calls], ['dense-cpu-prepare', 'dense-gpu-dispatch'])
        self.assertIn('--dependency=afterok:59344822', calls[0][1])
        self.assertIn('--dependency=afterok:4001', calls[1][1])
        self.assertEqual(result['dense_gpu_dispatch_job'], '4002')
        self.assertEqual(verify.call_count, 2)


if __name__ == '__main__':
    unittest.main()
