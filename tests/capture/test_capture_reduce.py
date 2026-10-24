"""reduce_capture on a tiny synthetic trace: the packet and tensors that a capture stores.

Needs torch, tokenizers and transformers, so it runs in the vLLM environment (CPU only).
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
except ImportError:  # pragma: no cover - the light test environment has none of them
    torch = None

if torch is not None:
    import hashlib

    from labeling.spans import digest, sentence_spans, token_layout, trace_digest
    from models.token_replay import make_token_replay
    from schemas import TraceRecord

LAYERS, EXPERTS, TOP_K, HIDDEN = [1, 2], 8, 2, 12
REASONING = 'First read the problem. Then plan the steps carefully. Verify the result at the end.'
LABELS = ['Read', 'Plan', 'Verify']


def byte_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab['ab'] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, [('a', 'b')]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def scenario(directory, cohorts=('A', 'B'), seed=0):
    """(row, data, backend, manifest) of one saved trace in cohorts A (labelled) and B (unlabelled)."""
    tokenizer = byte_tokenizer()
    prompt_ids = tokenizer.encode('prompt<think>\n', add_special_tokens=False)
    completion_ids = tokenizer.encode(REASONING + '\n</think>\nFinal.', add_special_tokens=False)
    text, replay = make_token_replay(tokenizer, prompt_ids, completion_ids)
    trace = TraceRecord(dataset='math500', problem_id='p1', source_problem_id='p1', prompt='question',
                        gold_answer='Final.', model_id='m', model_answer='Final.', cot_text=text,
                        metadata={'token_replay': replay, 'reasoning_content': REASONING,
                                  'assistant_content': 'Final.'})
    raw = (json.dumps(trace.model_dump(mode='json')) + '\n').encode()
    path = Path(directory) / 'traces.jsonl'
    path.write_bytes(raw)
    location = dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, byte_offset=0,
                    line_bytes=len(raw), line_sha256=hashlib.sha256(raw).hexdigest())
    labelled = dict(schema_version=1, trace_sha256=trace_digest(trace), sentence_selection=None,
                    units=[dict(unit, label=label) for unit, label in zip(sentence_spans(trace), LABELS)])
    members = []
    for name in cohorts:
        member = dict(id=name, locations=[location])
        if name == 'A':
            member['annotation'] = dict(inline=labelled, sha256=digest(labelled))
        members.append(member)
    row = dict(id='attempt', trace_sha256=trace_digest(trace), completion_tokens=len(completion_ids), cohorts=members)
    generator = torch.Generator().manual_seed(seed)
    router = torch.randn(len(LAYERS), len(completion_ids), EXPERTS, generator=generator)
    ids = router.topk(TOP_K, dim=-1).indices
    data = dict(router=router, hidden=torch.randn(len(LAYERS), len(completion_ids), HIDDEN, generator=generator),
                ids=ids, weights=router.gather(-1, ids).softmax(-1), selection=router.clone(),
                semantics=dict(native_selection=True, executed_weights='native_expert_input_weights',
                               full_distribution='softmax_probability', deterministic_topk=True,
                               selection_scores='router_signal', num_experts=EXPERTS),
                audit=dict(native_dispatch=True, layers=LAYERS))
    backend = SimpleNamespace(tokenizer=tokenizer, provenance={'checkpoint': 'm'})
    manifest = dict(spec=dict(layers=LAYERS, top_k=TOP_K), binding='b' * 64,
                    position_reference=dict(model_id='m', mean_reasoning_tokens=60.0, bins=10))
    return row, data, backend, manifest, trace, tokenizer


@unittest.skipIf(torch is None, 'torch is not installed')
class ReduceCaptureTests(unittest.TestCase):
    def setUp(self):
        from capture.reduce import reduce_capture
        self.reduce = reduce_capture
        self.tmp = tempfile.TemporaryDirectory()
        self.row, self.data, self.backend, self.manifest, self.trace, self.tokenizer = scenario(self.tmp.name)
        self.tokens = self.row['completion_tokens']

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_packet_is_plain_json_and_names_both_cohorts(self):
        packet, _ = self.reduce(self.row, self.data, self.backend, self.manifest)
        json.dumps(packet, allow_nan=False)
        self.assertEqual((packet['id'], packet['schema_version'], packet['manifest_binding']), ('attempt', 2, 'b' * 64))
        self.assertEqual(sorted(packet['cohorts']), ['A', 'B'])
        self.assertEqual([sum(counts) for counts in packet['native_expert_counts']], [self.tokens * TOP_K] * 2)
        windows = packet['trace']['metadata']['routing_dynamics']['windows']
        self.assertTrue(windows and all(w['boundary_gap_mean'] is not None for w in windows))
        self.assertEqual(packet['trace']['model_logs']['layer_indices'], LAYERS)

    def test_the_tensors_are_stored_at_the_documented_precision(self):
        _, tensors = self.reduce(self.row, self.data, self.backend, self.manifest)
        layout = token_layout(self.trace, self.tokenizer)
        self.assertEqual((tensors['ids'].dtype, tensors['weights'].dtype), (torch.int16, torch.float16))
        self.assertEqual(tuple(tensors['ids'].shape), (len(LAYERS), self.tokens, TOP_K))
        self.assertTrue(bool((tensors['native_boundary_gaps'] >= 0).all()))
        self.assertEqual(tensors['reasoning_tokens'].tolist(), layout['reasoning_tokens'])
        self.assertEqual(tensors['reasoning_tokens'].dtype, torch.int32)

    def test_only_the_labelled_cohort_gets_class_views_and_class_counts(self):
        packet, _ = self.reduce(self.row, self.data, self.backend, self.manifest)
        labelled, unlabelled = packet['cohorts']['A'], packet['cohorts']['B']
        self.assertIsNone(unlabelled['expert_class'])
        self.assertNotIn('class', {scope['view'] for scope in unlabelled['views']['scopes']})
        classes = labelled['expert_class']['classes']
        layout = token_layout(self.trace, self.tokenizer)
        self.assertEqual({name: entry['tokens'] for name, entry in classes.items() if entry['tokens']},
                         dict(zip(LABELS, (len(tokens) for tokens in layout['unit_tokens']))))
        self.assertEqual(labelled['expert_class']['tokens'], len(layout['reasoning_tokens']))
        views = {(scope['view'], scope['name']) for scope in labelled['views']['scopes']}
        self.assertIn(('full', 'reasoning'), views)
        self.assertEqual({name for view, name in views if view == 'class'},
                         {'Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor'})
        self.assertIn(('position', 'overflow'), views)

    def test_a_trace_that_differs_from_the_manifest_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'Cohort changes replay or reasoning scope'):
            self.reduce({**self.row, 'trace_sha256': '0' * 64}, self.data, self.backend, self.manifest)

    def test_signals_must_cover_every_layer_and_token(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete dispatched IDs'):
            self.reduce(self.row, {**self.data, 'ids': self.data['ids'][:, :-1]}, self.backend, self.manifest)

    def test_dispatched_experts_that_are_not_the_top_k_are_refused(self):
        worst = self.data['selection'].topk(EXPERTS, dim=-1).indices[..., -TOP_K:]
        with self.assertRaisesRegex(ValueError, 'disagree with scores'):
            self.reduce(self.row, {**self.data, 'ids': worst}, self.backend, self.manifest)


if __name__ == '__main__':
    unittest.main()
