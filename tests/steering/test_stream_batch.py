"""Batching preserves every row's dual operator, dose counter and audit ownership."""
import unittest

import torch

from steering.operators import score_override_dual, score_override_dual_masked, score_shift_masked
from steering.stream import DOSE_KEYS, StreamCapture, batched_dose


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
        capture.whole_ids_pending = []
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

    def test_native_id_capture_keeps_the_audit_native_ids_of_generated_tokens(self):
        torch.manual_seed(91)
        audit, slim = self.capture(), self.capture()
        for uid in slim.requests:
            slim.requests[uid].update(audit_capture=False, capture_mode='native_ids')
        steps = [  # prefill (prompt + first generated token), then two decode steps
            [('a', [(0, {'reasoning': True, 'prompt': True, 'generated_position': -1}),
                    (1, {'reasoning': True, 'prompt': False, 'generated_position': 0})]),
             ('b', [(2, {'reasoning': True, 'prompt': True, 'generated_position': -1}),
                    (3, {'reasoning': True, 'prompt': False, 'generated_position': 0})])],
            [('b', [(0, {'reasoning': True, 'prompt': False, 'generated_position': 1})]),
             ('a', [(1, {'reasoning': True, 'prompt': False, 'generated_position': 1})])],
            [('a', [(0, {'reasoning': True, 'prompt': False, 'generated_position': 2})])]]
        for work in steps:
            rows = sum(len(r) for _, r in work)
            for capture in (audit, slim):
                capture.whole_step += 1
            for layer in (0, 1):
                logits = torch.randn(rows, 32)
                affinity = logits.softmax(-1)
                ni = affinity.topk(4, -1).indices
                nw = affinity.gather(1, ni)
                nw /= nw.sum(-1, keepdim=True)
                for capture in (audit, slim):
                    capture._whole_batch(layer, work, logits, nw, ni, {})
            for capture in (audit, slim):
                capture._flush_whole_audit()
        for uid in ('a', 'b'):
            for layer in ('0', '1'):
                full, kept = audit.records[uid][layer], slim.records[uid][layer]
                self.assertEqual(kept['tokens'], len(full['positions']))
                ids = torch.as_tensor(__import__('numpy').concatenate(kept['native_ids']))
                weights = torch.as_tensor(__import__('numpy').concatenate(kept['native_weights']))
                self.assertTrue(torch.equal(ids.long(), torch.tensor(full['native_ids'])))
                torch.testing.assert_close(weights.float(), torch.tensor(full['native_weights']), rtol=1e-3, atol=1e-3)
        self.assertEqual(slim.records['a']['0']['tokens'], 3)

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


class NudgeTests(unittest.TestCase):
    """The gentle nudge lowers chosen logits and lets the router select natively."""

    def native(self, logits, k, scoring='softmax', bias=None):
        affinity = logits.softmax(-1) if scoring == 'softmax' else logits.sigmoid()
        priority = affinity if bias is None else affinity+bias
        ids = priority.topk(k, -1, sorted=True).indices
        weights = affinity.gather(1, ids)
        return weights/weights.sum(-1, keepdim=True), ids

    def test_shift_equals_native_routing_of_shifted_logits(self):
        torch.manual_seed(51)
        logits = torch.randn(16, 32)
        bias = torch.randn(32)*.1
        shift = torch.zeros(16, 32)
        shift[:, [2, 7]] = .8
        active = torch.arange(16) % 3 != 0
        for scoring, correction in (('softmax', None), ('sigmoid', bias)):
            nw, ni = self.native(logits, 4, scoring, correction)
            ew, ei = self.native(logits-shift, 4, scoring, correction)
            weights, ids = score_shift_masked(logits, nw, ni, shift, active, scoring=scoring,
                                              renormalize=True, correction_bias=correction)
            self.assertTrue(torch.equal(ids[active], ei[active]))
            torch.testing.assert_close(weights[active], ew[active], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(weights[~active], nw[~active]) and torch.equal(ids[~active], ni[~active]))

    def test_zero_shift_keeps_native_selection(self):
        torch.manual_seed(52)
        logits = torch.randn(8, 32)
        nw, ni = self.native(logits, 4)
        weights, ids = score_shift_masked(logits, nw, ni, torch.zeros(8, 32), torch.ones(8, dtype=torch.bool))
        self.assertTrue(torch.equal(ids, ni))
        torch.testing.assert_close(weights, nw)

    def test_shift_counters_record_weight_and_slots_of_nudged_experts(self):
        torch.manual_seed(53)
        logits = torch.randn(6, 32)
        logits[:, 5] += 4.                                   # expert 5 is natively always selected
        nw, ni = self.native(logits, 4)
        shift = torch.zeros(6, 32)
        shift[:, 5] = 50.                                    # a huge nudge removes it
        active, prompt = torch.ones(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)
        weights, ids = score_shift_masked(logits, nw, ni, shift, active)
        roles = torch.zeros((6, 2, 32), dtype=torch.bool)
        totals = batched_dose(nw, ni, weights, ids, active, prompt, roles, torch.tensor([6]), shift != 0)[0]
        values = dict(zip(DOSE_KEYS, totals.tolist()))
        self.assertEqual(values['native_shift_members'], 6)
        self.assertEqual(values['executed_shift_members'], 0)
        self.assertEqual(values['membership_failures'], 0)
        self.assertGreater(values['native_shift_weight'], 0)
        self.assertEqual(values['executed_shift_weight'], 0)

    def nudge_capture(self, actions):
        capture = WholeBatchTests.capture(self)
        capture.requests = {'a': {'actions': actions, 'prompt_scope': True, 'audit_capture': False}}
        capture.whole_slots, capture.next_whole_slot = {'a': 0}, 1
        capture.records, capture.counts = {'a': {}}, {'a': {'dose': {}}}
        return capture

    def test_batched_nudge_row_matches_operator_and_untargeted_layer_is_native(self):
        torch.manual_seed(54)
        capture = self.nudge_capture({0: {'activate': [], 'deactivate': [], 'shift': [1, 2], 'delta': .8}})
        work = [('a', [(0, {'reasoning': True, 'prompt': True, 'generated_position': -1}),
                       (1, {'reasoning': True, 'prompt': False, 'generated_position': 0})])]
        logits = torch.randn(2, 32)
        nw, ni = self.native(logits, 4)
        shift = torch.zeros(2, 32)
        shift[:, [1, 2]] = .8
        expected = score_shift_masked(logits, nw, ni, shift, torch.ones(2, dtype=torch.bool))
        weights, ids = capture._whole_batch(0, work, logits, nw, ni, {})
        self.assertTrue(torch.equal(ids, expected[1]))
        torch.testing.assert_close(weights, expected[0])
        weights, ids = capture._whole_batch(1, work, logits, nw, ni, {})
        self.assertTrue(torch.equal(ids, ni) and torch.equal(weights, nw))

    def test_a_batch_cannot_mix_nudged_and_forced_rows(self):
        capture = self.nudge_capture({0: {'activate': [0], 'deactivate': [], 'shift': [1], 'delta': .8}})
        with self.assertRaises(ValueError):
            capture._whole_roles([('a', [])], 'cpu')


if __name__ == '__main__':
    unittest.main()
