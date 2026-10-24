"""Focused whole-reasoning contracts, independent of model loading or scientific outcomes."""
from collections import Counter
from itertools import groupby
import math
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from steering.operators import score_override_dual
from steering.selection import global_pairs, layer_matched_controls
from store import BENCHMARKS, CLASSES, WORK
from steering.pulse_effects import Bootstrap, bounded_summary
from steering.stream import StreamCapture, dose_totals
from steering.whole import build_manifest, select_pairs
from steering.whole_effects import unit_bounds, contrast, primary_frame
from steering import pulse as driver, whole
from store import publish

LINK = driver.DRIVER/'EFFICIENCY_LINK_PREREG.json'  # the seven-model pre-registration


class NominationTests(unittest.TestCase):
    def inputs(self):
        """Two independent families in each fold; fixed integer inclusion probabilities."""
        sources = [{'fold': fold, 'benchmark': 'aime24', 'family': f'{fold}-{rep}'}
                   for fold in range(5) for rep in range(2)]
        tokens = np.full((10, 9), 100)
        counts = np.full((10, 9, 4, 32), 20.)
        counts[:, 0, :, 0] = 80
        counts[:, 0, :, 1:4] = 1
        return counts, tokens, sources

    def test_heldout_fold_is_excluded_from_full_fit_and_all_four_refits(self):
        counts, tokens, sources = self.inputs()
        before = global_pairs(counts, tokens, sources, list(range(4)), 4, 3, 6)[0]
        counts[:2] = 99-counts[:2]
        after = global_pairs(counts, tokens, sources, list(range(4)), 4, 3, 6)[0]
        self.assertEqual(before, after)
        self.assertEqual(len(before['selected']['activate']), 3)
        self.assertEqual(len(before['selected']['deactivate']), 6)
        self.assertEqual(len({p['layer'] for p in before['selected']['activate']}), 3)
        self.assertLessEqual(max(Counter(p['layer'] for p in
                             before['selected']['deactivate']).values()), 2)

    def test_sign_must_survive_each_inner_refit(self):
        counts, tokens, sources = self.inputs()
        counts[2:4, 0, :, 0] = 100
        counts[4:, 0, :, 0] = 10
        result = global_pairs(counts, tokens, sources, list(range(4)), 4, 3, 6)[0]
        self.assertFalse(any(pair['expert'] == 0 for pair in result['candidates']['positive']))

    def test_half_availability_rule_drops_below_half_without_substitution(self):
        candidates = [{'layer': i, 'expert': 0} for i in range(3)]
        used, status = select_pairs(candidates, 5, list(range(5)), 'activate')
        self.assertEqual(len(used), 3)
        self.assertTrue(status['qualified'])
        used, status = select_pairs(candidates[:2], 5, list(range(5)), 'activate')
        self.assertEqual(used, [])
        self.assertFalse(status['qualified'])

    def test_controls_match_layer_frequency_quintile_and_role_with_local_exclusions(self):
        layers = [2, 5]
        frequency = [list(range(50)), list(range(50))]
        activate = [{'layer': 2, 'expert': 1}, {'layer': 5, 'expert': 12}]
        deactivate = [{'layer': 2, 'expert': 22}, {'layer': 2, 'expert': 23}]
        excluded = {(p['layer'], p['expert']) for p in activate+deactivate} | {(2, 24)}
        controls = layer_matched_controls(activate, deactivate, frequency, layers, excluded, 'cell')
        self.assertEqual(controls, layer_matched_controls(activate, deactivate, frequency, layers, excluded, 'cell'))
        chosen = set()
        for role, targets in (('activate', activate), ('deactivate', deactivate)):
            self.assertEqual(sorted((p['layer'], p['expert']//10) for p in targets),
                             sorted((p['layer'], p['expert']//10) for p in controls[role]))
            for pair in controls[role]:
                key = pair['layer'], pair['expert']
                self.assertNotIn(key, excluded | chosen)
                chosen.add(key)


class OperatorTests(unittest.TestCase):
    def test_dual_matches_extrema_reference_and_preserves_inactive_bits(self):
        torch.manual_seed(42)
        logits = torch.randn(6, 32)
        q = logits.softmax(-1)
        ni = q.topk(4, -1).indices
        nw = q.gather(1, ni)
        nw /= nw.sum(-1, keepdim=True)
        active = [True, False, True, True, False, True]
        aw, ai, _ = score_override_dual(logits, nw, ni, [3], [1, 2, 5, 6, 7], active=active)
        reference = logits.log_softmax(-1)
        maximum, minimum = reference.max(-1, keepdim=True).values, reference.min(-1, keepdim=True).values
        reference[:, [3]] = maximum+.01
        reference[:, [1, 2, 5, 6, 7]] = minimum-.01
        weights = reference.softmax(-1)
        ids = weights.topk(4, -1).indices
        weights = weights.gather(1, ids)
        weights /= weights.sum(-1, keepdim=True)
        np.testing.assert_allclose(aw[active], weights[active], rtol=1e-6)
        self.assertTrue(torch.equal(ai[active], ids[active]))
        self.assertTrue(torch.equal(aw[[1, 4]], nw[[1, 4]]))
        self.assertTrue(torch.equal(ai[[1, 4]], ni[[1, 4]]))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            score_override_dual(logits, nw, ni, [1], [1])
        with self.assertRaisesRegex(ValueError, 'native k'):
            score_override_dual(logits, nw, ni, [1, 2, 3, 4], [])
        with self.assertRaisesRegex(ValueError, 'native k'):
            score_override_dual(logits, nw, ni, [], list(range(29)))

    def test_aggregate_dose_equals_direct_dense_computation(self):
        ni = torch.tensor([[0, 1], [2, 3], [0, 3]])
        nw = torch.tensor([[.3, .7], [.6, .4], [.2, .8]])
        ai = torch.tensor([[0, 2], [2, 3], [0, 2]])
        aw = torch.tensor([[.5, .5], [.6, .4], [.4, .6]])
        dose = dose_totals(nw, ni, aw, ai, [True, False, True], [True, False, False], [2], [1], 4)
        self.assertEqual(dose['active_prompt_positions'], 1)
        self.assertEqual(dose['active_generated_positions'], 1)
        self.assertEqual(dose['changed_membership'], 2)
        self.assertAlmostEqual(float(dose['executed_activate_weight']), 1.1, places=6)
        self.assertAlmostEqual(float(dose['native_deactivate_weight']), .7, places=6)
        self.assertAlmostEqual(float(dose['l1']), 3., places=6)


class PromptScopeTests(unittest.TestCase):
    def capture(self, prompt):
        """Attach the real clock/mask/operator to a tiny native router and runner."""
        class Model(torch.nn.Module):
            def forward(self, input_ids, positions):
                logits = torch.arange(32, dtype=torch.float32).repeat(len(input_ids), 1)/32
                return router._compute_routing(router_logits=logits)
        def native(router_logits):
            q = router_logits.softmax(-1)
            ids = q.topk(4, -1).indices
            weights = q.gather(1, ids)
            return weights/weights.sum(-1, keepdim=True), ids
        router = SimpleNamespace(top_k=4, global_num_experts=32, _compute_routing=native)
        model = Model()
        runner = SimpleNamespace(get_model=lambda: model, _prepare_inputs=lambda: None,
                                 input_batch=SimpleNamespace(req_ids=['internal']),
                                 query_start_loc=SimpleNamespace(np=np.array([0, 0])))
        with patch('steering.stream.discover', return_value={0: ('router', router, None)}), \
             patch('steering.stream.policy', return_value={'scoring': 'softmax', 'renormalize': True}):
            capture = StreamCapture(runner, {'layers': [0], 'hook_layers': [0], 'native_k': 4, 'experts': 32})
        capture.add([{'uid': 'uid', 'prompt': prompt, 'prefix': [], 'closure': [[9, 9]],
                      'actions': [{'layer': 0, 'activate': [0], 'deactivate': [31], 'pulse_tokens': 0}],
                      'capture_layers': [0], 'window': 50, 'whole': True, 'prompt_scope': True,
                      'audit_capture': True}], {'internal': 'uid'})
        self.addCleanup(capture.restore)
        return capture, runner, model

    @staticmethod
    def chunk(runner, model, ids, positions):
        """Drive one real stream prepare/forward pair in the fake runner."""
        runner.query_start_loc.np = np.array([0, len(ids)])
        runner._prepare_inputs()
        return model(torch.tensor(ids), torch.tensor(positions))

    def test_chunked_prefill_recompute_closure_and_generated_capture_zero(self):
        # The prompt contains the closure marker as a Harmony-like suffix.
        prompt = [1]*2498+[9, 9]
        capture, runner, model = self.capture(prompt)
        from functools import partial
        chunk = partial(self.chunk, runner, model)
        chunk(prompt[:2048], list(range(2048)))
        chunk(prompt[2048:], list(range(2048, 2500)))
        self.assertFalse(capture.clocks['uid'].closed)
        dose = capture.counts['uid']['dose']['0']
        self.assertEqual(dose['active_prompt_positions'], 2500)
        chunk([3, 9, 9, 4], list(range(2500, 2504)))
        active_before = float(capture.counts['uid']['dose']['0']['active_positions'])
        weights, ids = chunk([5], [2504])
        # Audit rows reach the host at the next step's pre-hook, after the device finished the step.
        self.assertEqual(capture.records['uid']['0']['positions'], [0, 1])
        self.assertNotIn(0, ids[0].tolist())
        self.assertEqual(float(capture.counts['uid']['dose']['0']['active_positions']), active_before)
        # Re-prefill must rebuild all counters, without counting both passes.
        chunk(prompt, list(range(2500)))
        self.assertEqual(capture.clocks['uid'].passes, 2)
        capture._flush_whole_audit()  # release() flushes the same way
        self.assertEqual(capture.records['uid']['0']['positions'], [])
        self.assertEqual(capture.counts['uid']['dose']['0']['active_prompt_positions'], 2500)
        chunk([3], [2500])
        capture._flush_whole_audit()
        self.assertEqual(capture.records['uid']['0']['positions'], [0])

    def test_original_prompt_runtime_allows_empty_prefix_only_when_flagged(self):
        from steering.runtime import prompt_prefix
        trace = {'metadata': {'token_replay': {'prompt_token_ids': [1, 2, 3], 'completion_token_ids': [4, 5]}}}
        binding = {'source_campaign': '/saved', 'root': '/saved'}
        row = {'source': {'attempt_id': 'source', 'prefix_tokens': 0}, 'original_prompt': True}
        with patch('steering.runtime.load', return_value={'records': [
                    {'attempt_id': 'source', 'source_location': 'saved'}]}), \
             patch('store.read_source', return_value=trace):
            self.assertEqual(prompt_prefix(binding, 'gpt', row, None)[:2], ([1, 2, 3], []))
            with self.assertRaisesRegex(ValueError, 'original-prompt'):
                prompt_prefix(binding, 'gpt', {**row, 'original_prompt': False}, None)

    def test_real_unix_socket_binds_short_work_path(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            path = str(Path(temporary)/'rpc.sock')
            self.assertLess(len(path.encode()), 107)
            with socket.socket(socket.AF_UNIX) as server:
                server.bind(path)


class AnalysisTests(unittest.TestCase):
    def test_first_endpoints_reject_incomplete_overlap_and_out_of_horizon_sentences(self):
        from steering.pulse_effects import endpoints
        outcome = {'pulse_tokens': 4, 'horizon': 8, 'reasoning_end': None, 'generated_tokens': 8,
                   'finish_reason': 'length', 'nonreasoning': []}
        spans = [{'token_start': 0, 'token_end': 3, 'complete': False},
                 {'token_start': 6, 'token_end': 10, 'complete': True}]
        for span in spans:
            span.update(lexical_label='Read', **{'p_'+c: float(c == 'Read') for c in CLASSES})
        value = endpoints(outcome, spans)
        self.assertFalse(value['first_post_available'])
        self.assertFalse(value['first_overlap_available'])

    def test_capped_and_missing_token_accuracy_bounds_use_actual_allowance(self):
        row = {'total_allowance': 9000, 'outcome_status': 'observed', 'reasoning_tokens': 4000,
               'reasoning_end': None, 'finish_reason': 'length', 'strict_correct': True}
        capped = unit_bounds(row)
        self.assertAlmostEqual(capped['log_reasoning:lower'], math.log1p(4000))
        self.assertAlmostEqual(capped['log_reasoning:upper'], math.log1p(9000))
        self.assertEqual(capped['accuracy:upper'], 0)
        self.assertAlmostEqual(unit_bounds({**row, 'reasoning_end': 4000})['log_reasoning:upper'], math.log1p(4000))
        missing = unit_bounds({'total_allowance': 7000, 'outcome_status': 'missing'})
        self.assertEqual(missing['log_reasoning:lower'], 0)
        self.assertEqual(missing['accuracy:lower'], 0)
        self.assertEqual(missing['accuracy:upper'], 1)
        benefit = contrast(capped, missing, 'log_reasoning', True)
        self.assertAlmostEqual(benefit[1], -math.log1p(9000))

    def test_one_missing_cell_widens_its_metric_without_poisoning_other_columns(self):
        frame = pd.DataFrame({'family': ['a', 'b'], 'x': [1., np.nan], 'x:lower': [1., -2.],
                              'x:upper': [1., 2.], 'y': [2., 4.]})
        bootstrap = Bootstrap(['a', 'b'], {'a': 'aime24', 'b': 'aime24'}, draws=100)
        result = bounded_summary(bootstrap, frame, ['x'], 'bounded')['x']
        self.assertEqual(result['status'], 'qualified_bounds')
        self.assertEqual(result['lower_bound'], -.5)
        self.assertEqual(result['upper_bound'], 1.5)
        self.assertEqual(bootstrap.estimate(frame, 'y')['estimate'], 3.)

    def test_missing_benchmark_is_unqualified_even_with_finite_other_values(self):
        bootstrap = Bootstrap(['a', 'b'], {'a': 'aime24', 'b': 'math500'}, draws=100)
        result = bootstrap.estimate(pd.DataFrame({'family': ['a'], 'x': [1.]}), 'x')
        self.assertEqual(result['status'], 'unqualified_missing_benchmark')

    def test_threaded_math_verify_uses_helper_process(self):
        from concurrent.futures import ThreadPoolExecutor
        from generation.scoring import score_completion_detailed, shutdown_scoring_pool
        self.addCleanup(shutdown_scoring_pool)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(score_completion_detailed, {'gold_answer': '2'}, answer_type='math',
                                     model_text=r'\boxed{2}').result(timeout=40)
        self.assertEqual(result['math_verify_status'], 'ok')
        self.assertTrue(result['is_correct'])

    def test_gpt_reasoning_count_comes_from_replay_even_when_usage_says_zero(self):
        from steering.pulse_outcomes import record
        text = ('<|channel|>analysis<|message|>Reason.<|end|><|start|>assistant'
                '<|channel|>final<|message|>2<|return|>')
        replay = {'completion_token_ids': list(range(len(text))),
                  'completion_offsets': [[i, i+1] for i in range(len(text))]}
        raw = {'token_ids': replay['completion_token_ids'], 'finish_reason': 'stop', 'reasoning_tokens': 0}
        original = {'prompt': 'Question', 'gold_answer': '2', 'metadata': {}}
        with patch('models.token_replay.make_token_replay', return_value=(text, replay)), \
             patch('generation.scoring.score_completion_detailed',
                   return_value={'is_correct': True, 'model_answer': '2'}):
            outcome = record({'models': {'gpt': {'key': 'gpt'}}}, 'gpt', {'horizon': 1000}, raw, None,
                             [[9000]], frozen_input=([1], [], original))
        self.assertEqual(outcome['reasoning_tokens'], len('Reason.'))
        self.assertNotEqual(outcome['reasoning_tokens'], raw['reasoning_tokens'])
        direct = '<|channel|>final<|message|>2<|return|>'
        direct_replay = {'completion_token_ids': list(range(len(direct))),
                         'completion_offsets': [[i, i+1] for i in range(len(direct))]}
        raw = {**raw, 'token_ids': direct_replay['completion_token_ids']}
        with patch('models.token_replay.make_token_replay', return_value=(direct, direct_replay)), \
             patch('generation.scoring.score_completion_detailed',
                   return_value={'is_correct': True, 'model_answer': '2'}):
            outcome = record({'models': {'gpt': {'key': 'gpt'}}}, 'gpt', {'horizon': 1000, 'whole': True},
                             raw, None, [list(range(len('<|channel|>final<|message|>')))],
                             frozen_input=([1], [], original))
        self.assertEqual(outcome['reasoning_tokens'], 0)
        self.assertEqual(outcome['sentences'], [])


class ManifestTests(unittest.TestCase):
    def test_cardinality_unique_uids_native_seeds_and_family_block_order(self):
        for model, expected, layer_count, experts, k in (('gpt', 3096, 24, 32, 4), ('qwen36', 2838, 40, 256, 8)):
            natives = [{'model': model, 'family': f'{i:03d}', 'benchmark': 'math500', 'fold': i % 5,
                        'seed': 100+i, 'source': {'attempt_id': str(i), 'prefix_tokens': 12},
                        'profile_sha256': 'profile'} for i in range(129)]
            profile = {'layers': list(range(layer_count)), 'native_k': k, 'experts': experts,
                       'card': {'max_model_len': 32768, 'max_tokens': None}}
            nominations = {'sha256': 'nomination', 'records': [{'fold': f, 'class': c} for f in range(5) for c in CLASSES]}
            def cell(nomination, profile, model, dose, operation):
                return {'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation,
                        'target': [], 'random': [], 'qualified': True}
            with patch('steering.whole.cell_actions', side_effect=cell):
                value = build_manifest(model, 'main', nominations, {'models': {model: profile}}, natives,
                                       {r['family']: [1, 2] for r in natives}, LINK)
            rows = value['records']
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r['uid'] for r in rows}), expected)
            self.assertEqual(sum(r['audit_capture'] for r in rows), 32)
            canary = whole.canary_rows({**value, 'sha256': 'synthetic-manifest'}, LINK)
            self.assertEqual(len(canary), 48)
            self.assertTrue(all(r['total_allowance'] == r['horizon'] == 32766 for r in canary))
            for family, members in groupby(rows, key=lambda r: r['family']):
                members = list(members)
                self.assertEqual([r['arm'] for r in members[:2]], ['native', 'native_repeat'])
                self.assertNotEqual(members[0]['seed'], members[1]['seed'])
                self.assertTrue(all(r['seed'] == members[0]['seed'] for r in members[2:]))
                self.assertTrue(all(r['source']['prefix_tokens'] == 0 for r in members))

    def test_whole_primary_equal_cell_and_equal_model_weights(self):
        rows = []
        for model, cells, value in (('gpt', 6, 2.), ('qwen36', 5, 4.)):
            for i in range(cells):
                row = {'model': model, 'dose': 'main', 'family': 'family', 'efficiency_directed': True}
                for metric in ('tokens', 'accuracy'):
                    for suffix in ('', ':lower', ':upper'):
                        row['target_random:'+metric+suffix] = value
                rows.append(row)
        frame = primary_frame(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(frame['gpt:tokens'], 2.)
        self.assertEqual(frame['qwen36:tokens'], 4.)
        self.assertEqual(frame['pooled:tokens'], 3.)


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        """Isolate the release receipts in user WORK, without accessing a scheduler."""
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_nonempty_unbound_store_rejects_before_any_engine_load(self):
        publish(self.root/'RELEASE.json', {'release_root': str(whole.REPO)})
        folder = self.root/'stores/gpt/main/units'
        folder.mkdir(parents=True)
        (folder/'PARTIAL.json').write_text('{}')
        with patch.object(whole, 'manifest', return_value={'sha256': 'manifest'}):
            with self.assertRaisesRegex(ValueError, 'nonempty unbound'):
                whole.bound_store(self.root, 'gpt', 'main')

    def test_production_needs_the_manifest_the_canary_or_the_sweep_selected(self):
        publish(self.root/'RELEASE.json', {'release_root': str(whole.REPO), 'models': ['gpt']})
        with patch.object(whole, 'verify_release', return_value={}):
            with self.assertRaises(FileNotFoundError):  # prepare publishes PREPARED.json only after the tests passed
                whole.run(self.root, 'gpt', (0, 1), 0.)
        publish(self.root/'PREPARED.json', {})
        publish(whole.manifest_path('gpt', 'main', self.root), {'dose': 'main', 'records': []})
        publish(self.root/'gpt/SELECTED.json', {'dose': 'main', 'manifest_sha256': 'another manifest',
                                                'canary_job': '1'})
        with patch.object(whole, 'verify_release', return_value={}):
            with self.assertRaisesRegex(ValueError, 'lacks a passing canary'):
                whole.run(self.root, 'gpt', (0, 1), 0.)
            manifest = whole.manifest(self.root, 'gpt', 'main')
            (self.root/'gpt/SELECTED.json').unlink()  # receipts are immutable; start over with the right manifest
            publish(self.root/'gpt/SELECTED.json', {'dose': 'main', 'manifest_sha256': manifest['sha256'],
                                                    'canary_job': '1'})
            publish(whole.canary_path(self.root, 'gpt', 'main'), {'gate': {'passed': False}, 'job': '1',
                                                                  'manifest_sha256': manifest['sha256']})
            with self.assertRaisesRegex(ValueError, 'canary qualification differs'):
                whole.run(self.root, 'gpt', (0, 1), 0.)

    def test_plan_names_the_snapshot_and_the_shards_of_each_model(self):
        publish(self.root/'RELEASE.json', {'release_root': '/snapshot', 'models': ['gpt', 'glm', 'qwen36'],
                                           'shards': {'gpt': 3, 'glm': 27}})
        with patch('builtins.print') as out:
            whole.plan(self.root)
        self.assertEqual([call.args for call in out.call_args_list],
                         [('/snapshot',), ('steering.whole combine',), ('gpt', 3), ('glm', 27)])
        publish(self.root/'OUTCOME.json', {})
        with patch('builtins.print') as out:
            whole.plan(self.root)
        self.assertEqual(out.call_args_list[1].args, ('steering.outcome analyze',))

    def test_the_link_receipt_comes_from_the_release(self):
        publish(self.root/'RELEASE.json', {'release_root': str(whole.REPO)})
        self.assertEqual(whole.link_receipt(self.root), driver.DRIVER/'EFFICIENCY_LINK_PREREG.json')
        link = publish(self.root/'LINK.json', {'models': ['gpt']})
        publish(self.root/'RELINK.json', {'link': str(self.root/'LINK.json'), 'link_sha256': link['sha256']})
        self.assertEqual(whole.link_receipt(self.root), self.root/'LINK.json')


class ClosedAnalysisTests(unittest.TestCase):
    def test_completed_whole_output_replays_and_extracts_in_worker_scope(self):
        from steering import whole_effects
        text = ('<|channel|>analysis<|message|>Reason.<|end|><|start|>assistant'
                '<|channel|>final<|message|>2<|return|>')
        tokens = list(range(len(text)))
        replay = {'completion_token_ids': tokens, 'completion_offsets': [[i, i+1] for i in tokens]}
        row = {'uid': 'a'*64, 'family': 'family', 'model': 'gpt', 'fold': 0, 'class': 'Verify',
               'arm': 'target_promote', 'horizon': 1000, 'whole': True, 'total_allowance': 1000,
               'profile_sha256': 'profile', 'seed': 7}
        original = {'prompt': 'Question', 'gold_answer': '2', 'metadata': {}}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            folder = Path(temporary)
            store = SimpleNamespace(state=lambda uid: {'status': 'complete'}, folder=lambda uid: folder)
            publish(folder/'OUTPUT.json', {'assignment_sha256': whole.digest(row), 'total_allowance': 1000,
                    'profile_sha256': 'profile', 'segments': [{'seed': 7}], 'token_ids': tokens,
                    'finish_reason': 'stop', 'reasoning_tokens': 0})
            publish(folder/'CAPTURE.json', {'stats': {'dose': {}}})
            with patch('models.token_replay.make_token_replay', return_value=(text, replay)), \
                 patch('generation.scoring.score_completion_detailed',
                       return_value={'is_correct': True, 'model_answer': '2'}):
                outcome, sentences, failure = whole_effects._extract_one(
                    row, b={'models': {'gpt': {'key': 'gpt'}}}, model='gpt', tokenizer=None,
                    close=[[9000]], store=store, inputs={'family': ([1], [], original)})
        self.assertIsNone(failure)
        self.assertEqual(outcome['reasoning_tokens'], len('Reason.'))
        self.assertTrue(outcome['strict_correct'])
        self.assertTrue(outcome['answer_present'])
        self.assertEqual(sentences[0]['sentence_text'], 'Reason.')

    def test_all_missing_production_still_exports_six_bounded_primary_contrasts(self):
        from steering import whole_effects
        models = ('gpt', 'qwen36')
        manifests, allocation = {}, []
        for i, benchmark in enumerate(BENCHMARKS):
            allocation.append({'family': str(i), 'benchmark': benchmark})
        for model in models:
            rows = []
            for family in allocation:
                base = {**family, 'model': model, 'dose': 'main', 'fold': int(family['family']) % 5,
                        'total_allowance': 1000, 'source': {}, 'actions': []}
                for arm in ('native', 'native_repeat'):
                    rows.append({**base, 'uid': model+family['family']+arm, 'class': None,
                                 'arm': arm, 'operation': 'native'})
                for operation in ('promote', 'suppress'):
                    for cls in (CLASSES if operation == 'promote' else whole.directions(model, LINK)['suppress']):
                        for role in ('target', 'random'):
                            rows.append({**base, 'uid': model+family['family']+operation+cls+role,
                                         'class': cls, 'arm': role+'_'+operation, 'operation': operation})
            manifests[model] = {'model': model, 'dose': 'main', 'records': rows,
                                'scope': 'prompt-and-generated-reasoning'}
        original_load = whole_effects.load
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary, \
             patch.object(whole_effects, 'load', side_effect=lambda path: {'records': allocation}
                          if Path(path).name == 'ALLOCATION.json' else original_load(path)):
            root = Path(temporary)
            for model in models:
                manifests[model] = publish(whole.manifest_path(model, 'main', root), manifests[model])
            publish(root/'RELEASE.json', {'role': 'bounded closeout fixture', 'classifier': {}})
            publish(root/'SOURCES.json', {'sources': [{'model': model, 'root': str(root), 'dose': 'main'}
                                                      for model in models]})
            publish(root/'gpt/extraction/EXTRACTION.json', {
                'manifest_sha256': manifests['gpt']['sha256'],
                'failures': [{'uid': 'failed-replay-unit', 'error': 'synthetic replay failure'}]})
            result = whole_effects.combine(root)
            self.assertTrue((root/'analysis-combined/EFFECTS_WHOLE.json').is_file())
        self.assertEqual(result['extraction_failures']['count'], 1)
        self.assertEqual(len(result['primary']), 6)
        self.assertTrue(all(value['status'] == 'qualified_bounds' for value in result['primary'].values()))
        self.assertEqual(result['primary']['gpt:accuracy']['bound_simultaneous'], [-1., 1.])
        self.assertTrue(all('inconclusive' in value for value in result['verdict'].values()))
        # Unassigned suppress classes have no cell, so no zero effect is imputed for them.
        self.assertNotIn('qwen36|main|suppress|Read', result['mechanism'])
        self.assertIn('qwen36|main|suppress|Analyze', result['mechanism'])


if __name__ == '__main__':
    unittest.main()
