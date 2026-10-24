import unittest
import numpy as np
from steering.selection import matched_controls, count_set


class ScienceTests(unittest.TestCase):
    def test_count_controls_are_nested_disjoint_and_frequency_matched(self):
        freq = np.arange(64)
        sets = [[0], [0, 15], [0, 15, 30]]
        controls = matched_controls(sets, freq, ['model', 0, 'Verify'], 17)
        self.assertTrue(set(controls[0]) <= set(controls[1]) <= set(controls[2]))
        self.assertFalse(set(controls[-1]) & set(sets[-1]))
        quintile = lambda ids: sorted(min(4, e*5//64) for e in ids)
        self.assertEqual([quintile(x) for x in controls], [quintile(x) for x in sets])
        self.assertEqual(count_set([.1, .3, -1, np.nan], 2), [1, 0])
        self.assertEqual(count_set([.1, -1], 2), [])

    def test_hard_selection_noop_returns_original_tensors(self):
        import torch
        from steering.operators import hard_select
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        ids = torch.tensor([[5, 4, 3, 2]], dtype=torch.int32)
        weights = logits.softmax(-1).gather(1, ids.long())
        for direction, targets in [('activate', [5]), ('deactivate', [0])]:
            w, i, changed = hard_select(logits, weights, ids, targets, direction, renormalize=False)
            self.assertIs(w, weights)
            self.assertIs(i, ids)
            self.assertFalse(changed.any())

    def test_unavailable_outer_count_preserves_disjoint_central_controls(self):
        controls=matched_controls([[0],[0,15],[]],np.arange(64),['lfm25',0,'Analyze'],17)
        self.assertEqual(len(controls[1]),2)
        self.assertFalse(set(controls[1]) & {0,15})
        self.assertEqual(controls[2],[])

    def test_bias_selects_but_original_affinities_weight_with_scaling(self):
        import torch
        from steering.operators import hard_select
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        ids = torch.tensor([[5, 4, 3]], dtype=torch.int32)
        native = torch.tensor([[.6, .3, .1]])
        bias = torch.tensor([0., 0., 0., 0., 0., -100.])
        scales = torch.arange(1., 7.)
        w, i, changed = hard_select(logits, native, ids, [0], 'activate', scoring='sigmoid',
                                    correction_bias=bias, scale=2., expert_scale=scales)
        self.assertTrue(changed.item())
        self.assertIn(0, i[0].tolist())
        q = logits.sigmoid().gather(1, i.long())
        self.assertTrue(torch.allclose(w, q/q.sum(-1, keepdim=True)*2*scales[i.long()]))
        w2, i2, _ = hard_select(logits, native, ids, [0], 'activate', active=[False])
        self.assertIs(w2, native)
        self.assertIs(i2, ids)


if __name__ == '__main__':
    unittest.main()
