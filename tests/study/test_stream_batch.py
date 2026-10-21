"""Batching preserves every row's dual operator, dose counter and audit ownership."""
import unittest

import torch

from moe_exp.final_routing.operator import score_override_dual, score_override_dual_masked
from moe_exp.study.stream import DOSE_KEYS, StreamCapture, batched_dose


class WholeBatchTests(unittest.TestCase):
    def test_segment_dose_equals_per_request_sums(self):
        for length, count in ((6, 2), (4, 4)):
            logits = torch.randn(length, 32)
            affinity = logits.softmax(-1)
            ni = affinity.topk(4, -1).indices
            nw = affinity.gather(1, ni)
            nw /= nw.sum(-1, keepdim=True)
            roles = torch.zeros((length, 2, 32), dtype=torch.bool)
            roles[:, 0, 0], roles[:, 1, 3] = True, True
            active, prompt = torch.ones(length, dtype=torch.bool), torch.zeros(length, dtype=torch.bool)
            weights, ids = score_override_dual_masked(logits, nw, ni, roles[:, 0], roles[:, 1], active)
            lengths = torch.full((count,), length//count, dtype=torch.long)
            totals = batched_dose(nw, ni, weights, ids, active, prompt, roles, lengths)
            size = length//count
            for group in range(count):
                part = slice(group*size, (group+1)*size)
                single = batched_dose(nw[part], ni[part], weights[part], ids[part], active[part], prompt[part],
                                      roles[part], torch.tensor([size]))
                torch.testing.assert_close(totals[group], single[0])

    def test_masked_operator_matches_individual_roles_and_inactive_bits(self):
        torch.manual_seed(725)
        for experts, native_k in ((32, 4), (256, 8)):
            logits = torch.randn(64, experts)
            affinity = logits.softmax(-1)
            native_ids = affinity.topk(native_k, -1).indices
            native_weights = affinity.gather(1, native_ids)
            native_weights /= native_weights.sum(-1, keepdim=True)
            roles = torch.zeros((64, 2, experts), dtype=torch.bool)
            active = torch.arange(64) % 4 != 0
            expected_weights, expected_ids = native_weights.clone(), native_ids.clone()
            for row in range(64):
                positive = [row % experts] if row % 3 else []
                negative = [(row+2) % experts, (row+3) % experts]
                roles[row, 0, positive] = True
                roles[row, 1, negative] = True
                weights, ids, _ = score_override_dual(logits[row:row+1], native_weights[row:row+1],
                    native_ids[row:row+1], positive, negative, active=[bool(active[row])])
                expected_weights[row], expected_ids[row] = weights[0], ids[0]
            weights, ids = score_override_dual_masked(logits, native_weights, native_ids,
                                                     roles[:, 0], roles[:, 1], active)
            self.assertTrue(torch.equal(ids, expected_ids))
            torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
            self.assertTrue(torch.equal(weights[~active], native_weights[~active]))

    def capture(self):
        """Two real request states sharing a layer, with prompt and generated token rows."""
        capture = object.__new__(StreamCapture)
        capture.contract = {'experts': 32, 'native_k': 4, 'hook_layers': [0, 1]}
        capture.whole_slots, capture.whole_totals, capture.whole_role_cache = {'a': 0, 'b': 1}, {}, {}
        capture.next_whole_slot = 2
        capture.whole_step, capture.whole_row_cache, capture.whole_audit_pending = 0, None, []
        capture.requests = {
            'a': {'actions': {0: {'activate': [0], 'deactivate': [3, 4]}, 1: {'activate': [], 'deactivate': [5]}},
                  'prompt_scope': True, 'audit_capture': True},
            'b': {'actions': {0: {'activate': [], 'deactivate': [8, 9]}},
                  'prompt_scope': False, 'audit_capture': True},
        }
        capture.records = {'a': {}, 'b': {}}
        capture.counts = {uid: {'dose': {}} for uid in capture.requests}
        return capture

    def test_batch_matches_reference_dose_and_full_audit_for_prefill_decode_and_reordering(self):
        torch.manual_seed(443)
        reference, batched = self.capture(), self.capture()
        work = [('a', [(0, {'reasoning': True, 'prompt': True, 'generated_position': -1}),
                       (1, {'reasoning': True, 'prompt': False, 'generated_position': 0})]),
                ('b', [(2, {'reasoning': True, 'prompt': True, 'generated_position': -1}),
                       (3, {'reasoning': True, 'prompt': False, 'generated_position': 0}),
                       (4, {'reasoning': True, 'prompt': False, 'generated_position': 1})])]
        for current in (work, list(reversed(work))):
            batched.whole_step += 1  # one engine step shares its row tensors across both hooked layers
            for layer in (0, 1):
                logits = torch.randn(5, 32)
                affinity = logits.softmax(-1)
                ni = affinity.topk(4, -1).indices
                nw = affinity.gather(1, ni)
                nw /= nw.sum(-1, keepdim=True)
                expected_weights, expected_ids = nw.clone(), ni.clone()
                for uid, rows in current:
                    reference._whole_layer(uid, layer, rows, logits, nw, ni, expected_weights, expected_ids, {})
                weights, ids = batched._whole_batch(layer, current, logits, nw, ni, {})
                self.assertTrue(torch.equal(ids, expected_ids))
                torch.testing.assert_close(weights, expected_weights)
                if layer == 1:  # request b has no target here: its rows keep the exact native bits
                    rows_b = [index for index, _ in dict(current)['b']]
                    self.assertTrue(torch.equal(weights[rows_b], nw[rows_b]) and torch.equal(ids[rows_b], ni[rows_b]))
            self.assertTrue(batched.whole_audit_pending)
            batched._flush_whole_audit()  # the next step's pre-hook moves audit rows to the host
            self.assertFalse(batched.whole_audit_pending)
        self.assertEqual(reference.records, batched.records)
        for uid in reference.requests:
            for layer in ('0', '1'):
                for key in DOSE_KEYS:
                    torch.testing.assert_close(batched.counts[uid]['dose'][layer][key].float(),
                                               reference.counts[uid]['dose'][layer][key].float())

    def test_growing_buffer_preserves_prior_counters_and_refreshes_live_views(self):
        capture = self.capture()
        initial = capture._whole_buffer(0, 'cpu')
        capture.counts['a']['dose']['0'] = dict(zip(DOSE_KEYS, initial[0].unbind()))
        initial[0, 0] = 7
        capture.next_whole_slot = 65
        grown = capture._whole_buffer(0, 'cpu')
        self.assertEqual(capture.counts['a']['dose']['0']['active_positions'], 7)
        grown[0, 0] += 1
        self.assertEqual(capture.counts['a']['dose']['0']['active_positions'], 8)


if __name__ == '__main__':
    unittest.main()
