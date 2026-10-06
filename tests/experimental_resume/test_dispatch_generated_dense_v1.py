"""One-shot dispatch must cover the whole exact price and preserve dependencies."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/experimental_resume'), str(REPO / 'src')]
import dispatch_generated_dense_v1 as dispatch
from test_generated_dense_pipeline_v1 import PromptTokenizer


class DenseDispatchTests(unittest.TestCase):
    def test_live_snapshot_uses_requested_filename(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            folder = Path(temporary)
            with patch.object(dispatch.subprocess, 'run', return_value=SimpleNamespace(stdout='iscrc_miosr|normal')):
                dispatch.live_account_snapshot(folder, 'LIVE_ACCOUNT-123.json')
            self.assertTrue((folder / 'LIVE_ACCOUNT-123.json').exists())
            self.assertFalse((folder / 'association').exists())
            value = dispatch.dense.sealed(folder / 'LIVE_ACCOUNT-123.json')
            self.assertEqual(set(value['records']), {'gpu_partition', 'cpu_partition', 'association'})

    def fixture(self, count=64):
        frame = {'sha256': 'frame', 'sentences': count, 'generation_manifest_sha256': 'f' * 64,
                 'records': [{'blind_id': str(i), 'inputs': {'problem_statement': 'p', 'previous_sentence': 'b',
                                                           'sentence': 's', 'next_sentence': 'n'}} for i in range(count)]}
        return frame, dispatch.dense.price_frame(frame, PromptTokenizer(), 7200)

    def test_complete_price_recomputed_and_dropped_or_underpriced_work_rejected(self):
        frame, price = self.fixture()
        dispatch.validate_complete_price(frame, price)
        for changed in ('complete_stage_GPU_h', 'prompt_tokens_exact', 'maximum_decode_tokens'):
            bad = copy.deepcopy(price); bad[changed] *= .5
            with self.assertRaises(ValueError):
                dispatch.validate_complete_price(frame, bad)
        bad = copy.deepcopy(price); bad['shards'][0]['stop'] -= 32
        with self.assertRaises(ValueError):
            dispatch.validate_complete_price(frame, bad)

    def test_gpu_array_then_exact_afterok_analysis(self):
        frame, price = self.fixture()
        calls = []
        def submit(directory, name, args, env, binding):
            calls.append((name, args)); return str(1000 + len(calls))
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            with patch.object(dispatch, 'live_account_snapshot'), patch.object(dispatch, 'ensure_verified') as verify:
                result = dispatch.dispatch(Path(temporary), frame, price, {'source': 'test'}, submit=submit)
        self.assertEqual([name for name, _ in calls], ['label-array', 'trajectory-analysis'])
        self.assertIn(f'--array=0-{len(price["shards"])-1}', calls[0][1])
        self.assertIn('--dependency=afterok:1001', calls[1][1])
        self.assertEqual(result['analysis_job'], '1002')
        self.assertEqual(verify.call_count, 2)

    def test_zero_sentences_skip_gpu_and_keep_cpu_analysis(self):
        frame, price = self.fixture(0)
        dispatch.validate_complete_price(frame, price)
        calls = []
        def submit(directory, name, args, env, binding):
            calls.append((name, args)); return '2001'
        with tempfile.TemporaryDirectory(dir=REPO / 'tests/experimental_resume') as temporary:
            with patch.object(dispatch, 'live_account_snapshot'), patch.object(dispatch, 'ensure_verified'):
                result = dispatch.dispatch(Path(temporary), frame, price, {'source': 'test'}, submit=submit)
        self.assertEqual([name for name, _ in calls], ['trajectory-analysis'])
        self.assertFalse(any(arg.startswith('--dependency') for arg in calls[0][1]))
        self.assertIsNone(result['label_array_job'])
        self.assertEqual(result['complete_stage_GPU_h'], 0)


if __name__ == '__main__':
    unittest.main()
