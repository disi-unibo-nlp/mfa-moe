"""The capture buffers, the memory budget, the token check and the agreement gates, on tiny synthetic tensors.

Needs torch, so it is skipped in the light environment and runs in the vLLM environment (CPU only).
"""
import copy
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - the light test environment has no torch
    torch = None

if torch is not None:
    from capture.agreement import DifferentialFailed, compare_backend_captures, compare_hf_vllm_capture, compare_native_captures
    from capture.buffers import CaptureBuffers, approved_capture_path, capture_memory_budget
    from capture.hf_chunked_capture import chunk_ranges
    from capture.native_selection import boundary_gaps, require_native_boundary
    from capture.vllm_capture import frame_entries

LAYERS, TOKENS, EXPERTS, TOP_K, HIDDEN, SCALE = [3, 5, 8], 40, 8, 2, 16, 1.5


def synthetic_capture(seed=0):
    """A capture whose dispatched experts are exactly the top-k of its selection scores."""
    generator = torch.Generator().manual_seed(seed)
    selection = torch.randn(len(LAYERS), TOKENS, EXPERTS, generator=generator)
    ids = selection.topk(TOP_K, dim=-1).indices
    weights = selection.gather(-1, ids).softmax(-1)
    return dict(router=selection.clone(), hidden=torch.randn(len(LAYERS), TOKENS, HIDDEN, generator=generator),
                ids=ids, weights=weights, selection=selection,
                audit={'native_dispatch': True, 'layers': LAYERS})


def vllm_side(seed=0):
    """The same capture as vLLM reports it: weights before the routed output scale, which the audit names."""
    data = synthetic_capture(seed)
    data['weights'] = data['weights'] / SCALE
    data['audit']['routed_output_scaling'] = {str(layer): SCALE for layer in LAYERS}
    return data


@unittest.skipIf(torch is None, 'torch is not installed')
class BufferTests(unittest.TestCase):
    def fill(self, buffers, data, rank=0):
        for index, layer in enumerate(LAYERS):
            values = {name: data[name][index] for name in ('ids', 'weights', 'router', 'selection')}
            if rank == 0:
                values['hidden'] = data['hidden'][index]
            buffers.write(layer, list(range(TOKENS)), values)

    def test_a_fully_written_capture_is_stored_with_fp16_weights(self):
        data = synthetic_capture()
        buffers = CaptureBuffers(LAYERS, TOKENS, rank=0)
        self.fill(buffers, data)
        stored = buffers.for_storage()
        self.assertEqual(stored['weights'].dtype, torch.float16)
        torch.testing.assert_close(stored['ids'], data['ids'])
        self.assertEqual(buffers.allocated_bytes, sum(len(LAYERS) * TOKENS * (v.shape[-1] * v.element_size() + 1)
                                                      for v in (data['ids'], data['weights'], data['router'],
                                                                data['selection'], data['hidden'])))

    def test_the_second_rank_has_no_hidden_states(self):
        buffers = CaptureBuffers(LAYERS, TOKENS, rank=1)
        self.fill(buffers, synthetic_capture(), rank=1)
        self.assertNotIn('hidden', buffers.require_complete())

    def test_missing_tokens_or_signals_are_refused(self):
        data = synthetic_capture()
        buffers = CaptureBuffers(LAYERS, TOKENS, rank=0)
        buffers.write(3, list(range(TOKENS)), {name: data[name][0] for name in ('ids', 'weights', 'router', 'selection', 'hidden')})
        with self.assertRaisesRegex(RuntimeError, 'Incomplete exact completion coverage'):
            buffers.require_complete()

    def test_recomputing_a_token_must_give_the_same_signal(self):
        data = synthetic_capture()
        buffers = CaptureBuffers(LAYERS, TOKENS, rank=1)
        buffers.write(3, [0, 1], {'router': data['router'][0, :2]})
        buffers.write(3, [1], {'router': data['router'][0, 1:2]})
        with self.assertRaisesRegex(RuntimeError, 'Recomputation changed'):
            buffers.write(3, [1], {'router': data['router'][0, 1:2] + 1})

    def test_unexpected_signals_and_positions_are_refused(self):
        buffers = CaptureBuffers(LAYERS, TOKENS, rank=1)
        with self.assertRaisesRegex(ValueError, 'Unexpected capture layer or signal'):
            buffers.write(3, [0], {'hidden': torch.zeros(1, HIDDEN)})
        with self.assertRaisesRegex(ValueError, 'outside completion'):
            buffers.write(3, [TOKENS], {'router': torch.zeros(1, EXPERTS)})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            buffers.write(3, [0], {'router': torch.full((1, EXPERTS), float('nan'))})

    def test_the_memory_budget_follows_the_stated_formulas(self):
        budget = capture_memory_budget(10, 2, 8, 4, 2)
        self.assertEqual(budget['native_tensor_bytes'], 10 * 2 * (8 * 2 + 4 * 2 + 8 * 4 + 4 * 8))
        self.assertLess(capture_memory_budget(10, 2, 8, 4, 2, rank=1)['worker_bytes'], budget['worker_bytes'])
        with self.assertRaisesRegex(ValueError, 'exceeds 240G'):
            capture_memory_budget(10 ** 9, 40, 2048, 128, 8)
        with self.assertRaisesRegex(ValueError, 'Invalid native capture dimensions'):
            capture_memory_budget(10, 2, 8, 4, 5)

    def test_captures_are_only_written_to_the_users_own_storage(self):
        self.assertEqual(approved_capture_path('/leonardo_work/IscrC_MIOSR/lmolfett/x/../y').name, 'y')
        with self.assertRaisesRegex(ValueError, 'Unapproved capture path'):
            approved_capture_path('/tmp/capture')


@unittest.skipIf(torch is None, 'torch is not installed')
class ReplayChecks(unittest.TestCase):
    def test_every_token_the_engine_runs_must_be_the_saved_token(self):
        requests = {'a': {'spec': {'tokens': [10, 11, 12, 13], 'prompt_length': 2}}}
        frame = frame_entries(requests, ['a'], [0, 4], [10, 11, 12, 13], [0, 1, 2, 3])
        self.assertEqual(frame, [None, None, ('a', 0), ('a', 1)])
        with self.assertRaisesRegex(ValueError, 'Exact replay token mismatch'):
            frame_entries(requests, ['a'], [0, 4], [10, 11, 12, 99], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'Unregistered request'):
            frame_entries(requests, ['b'], [0, 4], [10, 11, 12, 13], [0, 1, 2, 3])

    def test_the_dispatched_experts_must_be_the_top_k_of_the_scores(self):
        data = synthetic_capture()
        self.assertTrue(bool((require_native_boundary(data['selection'], data['ids']) > 0).all()))
        worst = data['selection'].topk(EXPERTS, dim=-1).indices[..., -TOP_K:]
        self.assertTrue(bool((boundary_gaps(data['selection'], worst) < 0).all()))
        with self.assertRaisesRegex(ValueError, 'disagree with scores'):
            require_native_boundary(data['selection'], worst)

    def test_chunks_cover_the_trace_and_the_prompt_tokens_come_first(self):
        self.assertEqual(list(chunk_ranges(10, 4, 4)), [(0, 4, 4), (4, 8, 0), (8, 10, 0)])
        self.assertEqual(list(chunk_ranges(10, 6, 4)), [(0, 4, 4), (4, 8, 2), (8, 10, 0)])
        with self.assertRaises(ValueError):
            list(chunk_ranges(10, 10, 4))


@unittest.skipIf(torch is None, 'torch is not installed')
class AgreementTests(unittest.TestCase):
    def compare(self, native):
        return compare_backend_captures(synthetic_capture(), native, total_input_tokens=100, layers=LAYERS)

    def test_identical_captures_agree(self):
        result = self.compare(vllm_side())
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['native_selected_sets_identical'])
        self.assertEqual(result['flips'], 0)
        self.assertEqual(result['vllm_weight_scale'], SCALE)

    def test_a_hidden_state_error_at_the_first_routed_layer_fails_with_a_full_report(self):
        native = vllm_side()
        native['hidden'][0] *= 1.5
        with self.assertRaises(DifferentialFailed) as caught:
            self.compare(native)
        report = caught.exception.report
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(any('first routed layer 3' in failure for failure in report['failures']))
        self.assertIn('per_layer', report)

    def test_a_change_of_the_selection_scores_beyond_the_noise_fails(self):
        native = vllm_side(1)
        with self.assertRaises(DifferentialFailed) as caught:
            self.compare(native)
        self.assertTrue(any('selection scores relative-L2' in failure for failure in caught.exception.report['failures']))

    def test_long_traces_must_also_keep_the_routing_dynamics(self):
        with self.assertRaisesRegex(ValueError, 'at most 32768'):
            compare_hf_vllm_capture(synthetic_capture(), synthetic_capture(), total_input_tokens=40000, layers=LAYERS)

    def test_the_vllm_capture_must_name_one_audited_scale_per_layer(self):
        native = vllm_side()
        native['audit']['routed_output_scaling']['5'] = 2.0
        with self.assertRaisesRegex(ValueError, 'homogeneous'):
            self.compare(native)
        del native['audit']['routed_output_scaling']['5']
        with self.assertRaisesRegex(ValueError, 'Missing audited'):
            self.compare(native)

    def test_chunked_replay_must_keep_experts_signals_and_weights(self):
        reference = synthetic_capture()
        self.assertEqual(compare_native_captures(reference, copy.deepcopy(reference))['status'], 'passed')
        other_experts = copy.deepcopy(reference)
        other_experts['ids'] = (other_experts['ids'] + 1) % EXPERTS
        with self.assertRaisesRegex(ValueError, 'selected expert sets'):
            compare_native_captures(reference, other_experts)
        other_hidden = copy.deepcopy(reference)
        other_hidden['hidden'] = other_hidden['hidden'] * 1.01
        with self.assertRaisesRegex(ValueError, 'exceeded tolerance: hidden'):
            compare_native_captures(reference, other_hidden)
        other_weights = copy.deepcopy(reference)
        other_weights['weights'] = other_weights['weights'] * 1.1
        with self.assertRaisesRegex(ValueError, 'executed weights'):
            compare_native_captures(reference, other_weights)


if __name__ == '__main__':
    unittest.main()
