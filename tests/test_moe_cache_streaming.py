import unittest
from moe_exp.moe_cache_streaming.cache import simulate
from moe_exp.moe_cache_streaming.run import select_rows


class CacheTest(unittest.TestCase):
    def test_cold_lru_and_recency(self):
        r = simulate([[0], [1], [0], [2], [1]], 2)
        self.assertEqual((r['hits'], r['total_loads']), (1, 4))

    def test_pins_have_initialization_cost(self):
        r = simulate([[0], [1], [2], [0]], 2, [0])
        self.assertEqual((r['hits'], r['demand_loads'], r['total_loads']), (2, 2, 3))

    def test_simultaneous_request_cannot_evict_its_own_hits(self):
        self.assertEqual(simulate([[0, 1], [2, 0]], 2)['total_loads'], 3)
        self.assertEqual(simulate([[1, 0], [0, 2]], 2)['total_loads'], 3)

    def test_full_cache_loads_each_expert_once(self):
        for pins in ([], [0], [0, 1]):
            self.assertEqual(simulate([[0, 1], [1, 2], [0, 2]], 3, pins)['total_loads'], 3)

    def test_invalid_budget_and_duplicates(self):
        for sequence, capacity, pins in [([[1, 2]], 2, [0]), ([[1, 1]], 2, []), ([[1]], 0, [])]:
            with self.assertRaises(ValueError):
                simulate(sequence, capacity, pins)

    def test_selection_is_order_independent_and_problem_disjoint(self):
        rows = [dict(id=str(i), dataset=str(i % 2), source_problem_id=str(i // 2)) for i in range(12)]
        a = select_rows(rows, 6)
        self.assertEqual(a, select_rows(rows[::-1], 6))
        self.assertEqual(sum(r['dataset'] == '0' for r in a), 3)
        self.assertEqual(len({(r['dataset'], r['source_problem_id']) for r in a}), 6)

class AnalysisTest(unittest.TestCase):
    def test_paired_artifacts_and_tamper_detection(self):
        import gzip
        import json
        import tempfile
        from pathlib import Path
        from argparse import Namespace
        from moe_exp.moe_cache_streaming.run import analyze, digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for condition in ['baseline', 'guided']:
                directory = root / condition
                directory.mkdir()
                seq = [[0], [1], [2], [0]] if condition == 'baseline' else [[0], [0], [0], [0]]
                with gzip.open(directory / 'trace.gz', 'wt') as f:
                    json.dump({'0': seq}, f)
                record = dict(id='a', input={'id': 'a'}, prompt_token_ids=[10], sampling_args={},
                              routing='trace.gz', routing_sha256=digest(directory / 'trace.gz'),
                              generated_token_count=5, is_correct=True)
                (directory / 'generations.jsonl').write_text(json.dumps(record)+'\n')
                manifest = dict(status='complete', condition=condition, selected_ids=['a'],
                    generations_sha256=digest(directory / 'generations.jsonl'),
                    engine={}, sampling={}, policy_sha256='same', prompts_sha256='same',
                    versions={}, template_date='2026-09-22', strength=1,
                    scoring_contract='test', routing={'layers':['0']},
                    policy={'top_k':1,'num_experts':3,'layers':{'0':{'scores':[1,0,0],
                    'experts':[{'expert':i,'frequency_mass':3-i} for i in range(3)]}}})
                (directory / 'manifest.json').write_text(json.dumps(manifest))
            args = Namespace(baseline=root/'baseline', guided=root/'guided', budgets=[2,3], output=root/'analysis')
            analyze(args)
            results = json.loads((args.output/'summary.json').read_text())['summaries']
            first = {r['condition']:r for r in results if r['capacity_per_layer']==2}
            self.assertEqual(first['baseline_lru']['total_loads'], 4)
            self.assertEqual(first['guided_target_pins']['total_loads'], 1)
            with (root/'guided'/'trace.gz').open('ab') as f:
                f.write(b'bad')
            args.output = root/'tampered'
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                analyze(args)


class RecorderTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is available in the guiding Docker image')
        from moe_exp.moe_cache_streaming.routing import DecodeRecorder
        self.torch = torch
        self.recorder = DecodeRecorder(3, 1)

    def test_chunked_prefill_followed_by_ordered_decode(self):
        r = self.recorder
        r.begin(5)
        for size in (2, 2, 1):
            r(None, None, self.torch.ones(size, 3))
        self.assertEqual(r.rows, [])
        self.assertEqual(r.prefill_remaining, 0)
        r(None, None, self.torch.tensor([[0., 4., 1.]]))
        r(None, None, self.torch.tensor([[5., 0., 1.]]))
        self.assertEqual(r.rows, [[1], [0]])
        from moe_exp.moe_cache_streaming.routing import CacheWorkerExtension
        from types import SimpleNamespace
        worker = SimpleNamespace(cache_recorders={'0': r})
        self.assertEqual(CacheWorkerExtension.cache_finish(worker, 3), {'0': [[1], [0]]})

    def test_async_scheduler_rejected_before_hooks(self):
        from types import SimpleNamespace
        from moe_exp.moe_cache_streaming.routing import CacheWorkerExtension
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=True, max_num_seqs=1))
        worker = SimpleNamespace(vllm_config=config)
        with self.assertRaisesRegex(ValueError, 'synchronous scheduling'):
            CacheWorkerExtension.cache_configure(worker, {}, 1, 'baseline')

    def test_extra_decode_step_is_not_silently_trimmed(self):
        from types import SimpleNamespace
        from moe_exp.moe_cache_streaming.routing import CacheWorkerExtension
        r = self.recorder
        r.begin(1)
        for _ in range(4):
            r(None, None, self.torch.ones(1, 3))
        with self.assertRaisesRegex(RuntimeError, '3 decode steps, 3 returned tokens'):
            CacheWorkerExtension.cache_finish(SimpleNamespace(cache_recorders={'0': r}), 3)

    def test_prefill_overrun_and_batched_decode_rejected(self):
        r = self.recorder
        r.begin(1)
        with self.assertRaises(RuntimeError):
            r(None, None, self.torch.ones(2, 3))
        r(None, None, self.torch.ones(1, 3))
        with self.assertRaises(RuntimeError):
            r(None, None, self.torch.ones(2, 3))


if __name__ == '__main__':
    unittest.main()
