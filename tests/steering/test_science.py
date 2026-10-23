import unittest
import numpy as np
from moe_exp.steering.selection import matched_controls, count_set


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
        from moe_exp.steering.operators import hard_select
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        ids = torch.tensor([[5, 4, 3, 2]], dtype=torch.int32)
        weights = logits.softmax(-1).gather(1, ids.long())
        for direction, targets in [('activate', [5]), ('deactivate', [0])]:
            w, i, changed = hard_select(logits, weights, ids, targets, direction, renormalize=False)
            self.assertIs(w, weights)
            self.assertIs(i, ids)
            self.assertFalse(changed.any())

    def test_score_override_sets_extreme_log_probs_and_preserves_inactive_rows(self):
        import torch
        from moe_exp.steering.operators import score_override
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.], [0., 1., 2., 3., 4., 5.]])
        out, changed = score_override(logits, [0, 1], 'activate', active=[True, False])
        self.assertEqual(changed.tolist(), [True, False])
        reference = logits[0].log_softmax(-1)
        reference[[0, 1]] = reference.max()+.01
        self.assertTrue(torch.allclose(out[0], reference))
        self.assertTrue(torch.equal(out[1], logits[1]))
        out, changed = score_override(logits, [0, 1], 'deactivate', active=[True, False])
        reference = logits[0].log_softmax(-1)
        reference[[0, 1]] = reference.min()-.01
        self.assertTrue(torch.allclose(out[0], reference))
        self.assertFalse(changed[1])

    def test_score_override_is_invariant_to_a_constant_logit_shift(self):
        import torch
        from moe_exp.steering.operators import score_override
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        for direction in ('activate', 'deactivate'):
            original, _ = score_override(logits, [0, 1], direction)
            shifted, _ = score_override(logits+100., [0, 1], direction)
            self.assertTrue(torch.allclose(original, shifted))

    def test_executed_score_override_matches_reference_selection_and_weighting(self):
        import torch
        from moe_exp.steering.operators import score_override_select
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.], [-.4, .3, .1, .8, -.1, .6]])
        native_ids = logits.topk(4, dim=-1).indices.to(torch.int32)
        targets, active = [0, 2], torch.tensor([True, False])
        expert_scale = torch.tensor([1., 1.5, .5, 2., 3., .75])
        for direction in ('activate', 'deactivate'):
            for renormalize in (False, True):
                with self.subTest(direction=direction, renormalize=renormalize):
                    native_weights = logits.softmax(-1).gather(1, native_ids.long())
                    scores = logits.log_softmax(-1)
                    if direction == 'activate':
                        scores[0, targets] = scores[0].max()+.01
                    else:
                        scores[0, targets] = scores[0].min()-.01
                    reference_ids = scores.topk(4, dim=-1).indices
                    reference_weights = scores.softmax(-1).gather(1, reference_ids)
                    if renormalize:
                        native_weights /= native_weights.sum(-1, keepdim=True)
                        reference_weights /= reference_weights.sum(-1, keepdim=True)
                    native_weights *= 2.*expert_scale[native_ids.long()]
                    reference_weights *= 2.*expert_scale[reference_ids]
                    weights, ids, changed = score_override_select(
                        logits, native_weights, native_ids, targets, direction,
                        active=active, renormalize=renormalize, scale=2., expert_scale=expert_scale)
                    # Equal target scores may produce a different slot order in
                    # top-k; compare each expert's dispatched weight.
                    actual = torch.zeros_like(logits).scatter(1, ids.long(), weights)
                    expected = torch.zeros_like(logits).scatter(1, reference_ids, reference_weights)
                    self.assertTrue(torch.allclose(actual[0], expected[0]))
                    self.assertEqual(changed.tolist(), [True, False])
                    self.assertTrue(torch.equal(ids[1], native_ids[1]))
                    self.assertTrue(torch.equal(weights[1], native_weights[1]))

    def test_executed_score_override_with_no_active_rows_preserves_native_tensors(self):
        import torch
        from moe_exp.steering.operators import score_override_select
        logits = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        native_ids = logits.topk(4, dim=-1).indices.to(torch.int32)
        native_weights = torch.tensor([[.4, .3, .2, .1]])
        for direction in ('activate', 'deactivate'):
            weights, ids, changed = score_override_select(
                logits, native_weights, native_ids, [0], direction, active=[False])
            self.assertIs(weights, native_weights)
            self.assertIs(ids, native_ids)
            self.assertFalse(changed.any())

    def test_unavailable_outer_count_preserves_disjoint_central_controls(self):
        controls=matched_controls([[0],[0,15],[]],np.arange(64),['lfm25',0,'Analyze'],17)
        self.assertEqual(len(controls[1]),2)
        self.assertFalse(set(controls[1]) & {0,15})
        self.assertEqual(controls[2],[])

    def test_bias_selects_but_original_affinities_weight_with_scaling(self):
        import torch
        from moe_exp.steering.operators import hard_select
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
