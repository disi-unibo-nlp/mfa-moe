"""The sliding-window routing reducer, on synthetic expert choices (numpy only)."""
import unittest

import numpy as np

from capture.dynamics.routing import RoutingReducer, probabilities, reduce_arrays

EXPERTS, TOP_K = 16, 4


def expert_ids(tokens, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(EXPERTS)[:TOP_K] for _ in range(tokens)])


def windows_of(ids, chunk):
    reducer = RoutingReducer(EXPERTS, TOP_K, layer=3, model='m')
    rows = []
    for start in range(0, len(ids), chunk):
        rows += reducer.push(ids[start:start + chunk])
    return rows


class ReducerTests(unittest.TestCase):
    def test_windows_do_not_depend_on_how_the_tokens_arrive(self):
        ids = expert_ids(320)
        whole, pieces = windows_of(ids, 320), windows_of(ids, 40)
        self.assertEqual(len(whole), len(pieces))
        for left, right in zip(whole, pieces):
            self.assertEqual(left, right)

    def test_windows_end_every_stride_for_each_width(self):
        rows = windows_of(expert_ids(320), 64)
        ends = {(row['window'], row['end_token']) for row in rows}
        self.assertEqual(ends, {(64, 64 * i) for i in range(1, 6)} | {(256, 256), (256, 320)})

    def test_the_same_experts_every_token_give_full_overlap_and_zero_entropy_spread(self):
        ids = np.tile(np.arange(TOP_K), (128, 1))
        row = windows_of(ids, 64)[-1]
        self.assertEqual(row['overlap_lag1'], 1.0)
        self.assertEqual(row['set_turnover'], 0.0)
        self.assertAlmostEqual(row['selection_entropy'], np.log(TOP_K))

    def test_a_window_only_uses_tokens_up_to_its_end(self):
        ids = expert_ids(256)
        changed = ids.copy()
        changed[200:] = expert_ids(56, seed=5)
        first, second = windows_of(ids, 64), windows_of(changed, 64)
        self.assertEqual([r for r in first if r['end_token'] <= 192], [r for r in second if r['end_token'] <= 192])
        self.assertNotEqual(first[-1], second[-1])

    def test_a_repeated_expert_in_one_token_is_refused(self):
        ids = expert_ids(8)
        ids[0, 1] = ids[0, 0]
        with self.assertRaisesRegex(ValueError, 'Invalid/duplicate expert IDs'):
            RoutingReducer(EXPERTS, TOP_K, layer=0, model='m').push(ids)

    def test_the_signals_must_stay_the_same_within_a_segment(self):
        reducer = RoutingReducer(EXPERTS, TOP_K, layer=0, model='m')
        ids = expert_ids(64)
        reducer.push(ids)
        weights = np.ones(ids.shape)
        with self.assertRaisesRegex(ValueError, 'Signal availability changed'):
            reducer.push(ids, weights=weights)

    def test_reduce_arrays_skips_the_gaps_between_reasoning_segments(self):
        ids = expert_ids(256)[None]  # one layer
        scores = np.random.default_rng(1).normal(size=(1, 256, EXPERTS))
        scores[0, np.arange(256)[:, None], ids[0]] += 50  # the dispatched experts are the top-k
        tokens = list(range(0, 128)) + list(range(192, 256))
        rows = reduce_arrays(ids, layers=[7], model='m', probabilities_array=probabilities_of(scores), weights=None,
                             selection_scores=scores, semantics={'deterministic_topk': True, 'num_experts': EXPERTS},
                             token_indices=tokens)
        self.assertEqual(sorted({(r['start_token'], r['end_token']) for r in rows if r['window'] == 64}),
                         [(0, 64), (64, 128), (192, 256)])
        self.assertTrue(all(r['layer'] == 7 for r in rows))
        self.assertFalse(any(r['start_token'] < 128 < r['end_token'] for r in rows))


def probabilities_of(scores):
    class Slices:
        shape = scores.shape

        def __getitem__(self, index):
            return probabilities(scores[index])
    return Slices()


if __name__ == '__main__':
    unittest.main()
