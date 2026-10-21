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

from moe_exp.final_routing.operator import score_override_dual
from moe_exp.final_routing.selection import global_pairs, layer_matched_controls
from moe_exp.study.common import BENCHMARKS, CLASSES, WORK
from moe_exp.study.effects import Bootstrap, bounded_summary
from moe_exp.study.stream import Clock, StreamCapture, dose_totals
from moe_exp.study.whole import build_manifest, select_pairs
from moe_exp.study.whole_effects import unit_bounds, contrast, primary_frame
from moe_exp.study import whole
from moe_exp.study.common import publish


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
        with patch('moe_exp.study.stream.discover', return_value={0: ('router', router, None)}), \
             patch('moe_exp.study.stream.policy', return_value={'scoring': 'softmax', 'renormalize': True}):
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
        from moe_exp.final_routing.runtime import prompt_prefix
        trace = {'metadata': {'token_replay': {'prompt_token_ids': [1, 2, 3], 'completion_token_ids': [4, 5]}}}
        binding = {'source_campaign': '/saved', 'root': '/saved'}
        row = {'source': {'attempt_id': 'source', 'prefix_tokens': 0}, 'original_prompt': True}
        with patch('moe_exp.final_routing.runtime.load', return_value={'records': [
                    {'attempt_id': 'source', 'source_location': 'saved'}]}), \
             patch('moe_exp.research_campaign_v3_r6.inputs.read_source', return_value=trace):
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
    def test_native_only_pooled_planning_retains_cross_model_family_covariance(self):
        from moe_exp.study.whole_effects import _pooled_native_planning
        residuals = {
            'gpt': pd.DataFrame({'log_reasoning': [1., -1.], 'correct': [.5, -.5]}, index=['a', 'b']),
            'qwen36': pd.DataFrame({'log_reasoning': [-1., 1.], 'correct': [-.5, .5]}, index=['a', 'b']),
        }
        result = _pooled_native_planning(residuals, 3.48)
        self.assertEqual(result['mde_log_tokens'], 0.)
        self.assertEqual(result['mde_accuracy'], 0.)
        self.assertEqual(result['transport_status'], 'SUSPECTED')

    def test_first_endpoints_reject_incomplete_overlap_and_out_of_horizon_sentences(self):
        from moe_exp.study.effects import endpoints
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
        from moe_exp.correlation_pipeline.scoring import score_completion_detailed, shutdown_scoring_pool
        self.addCleanup(shutdown_scoring_pool)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(score_completion_detailed, {'gold_answer': '2'}, answer_type='math',
                                     model_text=r'\boxed{2}').result(timeout=40)
        self.assertEqual(result['math_verify_status'], 'ok')
        self.assertTrue(result['is_correct'])

    def test_gpt_reasoning_count_comes_from_replay_even_when_usage_says_zero(self):
        from moe_exp.study.outcomes import record
        text = ('<|channel|>analysis<|message|>Reason.<|end|><|start|>assistant'
                '<|channel|>final<|message|>2<|return|>')
        replay = {'completion_token_ids': list(range(len(text))),
                  'completion_offsets': [[i, i+1] for i in range(len(text))]}
        raw = {'token_ids': replay['completion_token_ids'], 'finish_reason': 'stop', 'reasoning_tokens': 0}
        original = {'prompt': 'Question', 'gold_answer': '2', 'metadata': {}}
        with patch('moe_exp.models.token_replay.make_token_replay', return_value=(text, replay)), \
             patch('moe_exp.correlation_pipeline.scoring.score_completion_detailed',
                   return_value={'is_correct': True, 'model_answer': '2'}):
            outcome = record({'models': {'gpt': {'key': 'gpt'}}}, 'gpt', {'horizon': 1000}, raw, None,
                             [[9000]], frozen_input=([1], [], original))
        self.assertEqual(outcome['reasoning_tokens'], len('Reason.'))
        self.assertNotEqual(outcome['reasoning_tokens'], raw['reasoning_tokens'])
        direct = '<|channel|>final<|message|>2<|return|>'
        direct_replay = {'completion_token_ids': list(range(len(direct))),
                         'completion_offsets': [[i, i+1] for i in range(len(direct))]}
        raw = {**raw, 'token_ids': direct_replay['completion_token_ids']}
        with patch('moe_exp.models.token_replay.make_token_replay', return_value=(direct, direct_replay)), \
             patch('moe_exp.correlation_pipeline.scoring.score_completion_detailed',
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
            with patch('moe_exp.study.whole.cell_actions', side_effect=cell):
                value = build_manifest(model, 'main', nominations, {'models': {model: profile}}, natives,
                                       {r['family']: [1, 2] for r in natives}, True)
            rows = value['records']
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r['uid'] for r in rows}), expected)
            self.assertEqual(sum(r['audit_capture'] for r in rows), 32)
            canary = whole.canary_rows({**value, 'sha256': 'synthetic-manifest'})
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
        """Isolate all gate receipts in user WORK, without accessing a scheduler."""
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(whole, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        # These gate regressions use the GPT/Qwen3.6 sizes of release r5 under its 72 GPU-hour ceiling.
        for name, values in (('SHARDS', {'gpt': 2, 'qwen36': 7}), ('RUN_MINUTES', {'gpt': 165, 'qwen36': 240}),
                             ('CANARY_MINUTES', {'gpt': 45, 'qwen36': 90})):
            sizes = patch.dict(getattr(whole, name), values)
            sizes.start()
            self.addCleanup(sizes.stop)
        publish(self.root/'RELEASE.json', {'release_root': str(whole.REPO),
                'prior_usage': {'new_gpu_hours': 2., 'new_cpu_core_hours': 3.}})

    def test_nonempty_unbound_store_rejects_before_any_engine_load(self):
        folder = self.root/'stores/gpt/main/units'
        folder.mkdir(parents=True)
        (folder/'PARTIAL.json').write_text('{}')
        with patch.object(whole, 'manifest', return_value={'sha256': 'manifest'}):
            with self.assertRaisesRegex(ValueError, 'nonempty unbound'):
                whole.bound_store('gpt', 'main')

    def test_shard_admission_charges_failures_and_allocation_maxima_before_submission(self):
        schedule = {'production': {m: [] for m in whole.MODELS},
                    'canaries': {m: [{'id': m, 'gpus': 2, 'minutes': whole.CANARY_MINUTES[m]}]
                                 for m in whole.MODELS}}
        whole.admit_shard_plan(schedule, 'gpt', {}, set())
        plan = whole.verified(self.root/'gpt/SHARD_PLAN.json')
        self.assertEqual(plan['minutes'], 142)
        self.assertLessEqual(whole.reserved_gpu_hours(schedule, {}, set()), whole.GPU_CEILING)

    def test_only_fluency_failure_selects_the_one_half_dose_fallback(self):
        schedule = {'canaries': {'gpt': [{'id': '1', 'dose': 'main', 'gpus': 2, 'minutes': 45}]},
                    'production': {'gpt': []}}
        publish(self.root/'gpt/CANARY.json', {'gate': {'passed': False, 'fluency_failures': ['cap'],
                                                    'software_failures': []}})
        publish(self.root/'gpt/WHOLE_HALF.json', {'dose': 'half', 'records': []})
        with patch.object(whole, 'submit_stage', return_value={'id': '2', 'dose': 'half'}) as submit, \
             patch.object(whole.driver, 'checkpoint'):
            whole.advance_canary(schedule, 'gpt', {'1': {'state': 'FAILED', 'elapsed': 60}}, set())
        self.assertEqual(submit.call_args.kwargs['dose'], 'half')
        self.assertEqual([job['dose'] for job in schedule['canaries']['gpt']], ['main', 'half'])
        self.assertEqual(whole.verified(self.root/'gpt/SELECTED.json')['dose'], 'half')

    def test_software_failure_never_becomes_a_dose_failure(self):
        schedule = {'canaries': {'gpt': [{'id': '1', 'dose': 'main'}]}, 'production': {'gpt': []}}
        publish(self.root/'gpt/CANARY.json', {'gate': {'passed': False, 'fluency_failures': ['cap'],
                                                    'software_failures': ['bad membership']}})
        stopped = set()
        with patch.object(whole, 'submit_stage') as submit:
            whole.advance_canary(schedule, 'gpt', {'1': {'state': 'FAILED'}}, stopped)
        submit.assert_not_called()
        self.assertEqual(stopped, {'gpt'})

    def test_passing_canaries_submit_nine_unheld_shards_and_dependent_cpu_work(self):
        schedule = {'canaries': {m: [{'id': str(i+1), 'dose': 'main'}] for i, m in enumerate(whole.MODELS)},
                    'production': {m: [] for m in whole.MODELS}}
        for model in whole.MODELS:
            publish(self.root/model/'CANARY.json', {'gate': {'passed': True}})
        calls = []
        def record_submission(command, **kwargs):
            calls.append({'command': command, **kwargs})
            return {'id': str(len(calls)+10), 'held': kwargs.get('held', False)}
        with patch.object(whole, 'submit_stage', side_effect=record_submission), \
             patch.object(whole, 'bound_store'), patch.object(whole, 'admit_shard_plan'), \
             patch.object(whole.driver, 'checkpoint'):
            for model in whole.MODELS:
                whole.advance_canary(schedule, model, {'1': {'state': 'COMPLETED'}, '2': {'state': 'COMPLETED'}}, set())
        production = [c for c in calls if c['command'] == 'run']
        self.assertEqual(len(production), 9)
        self.assertFalse(any(c.get('held') for c in calls))
        self.assertTrue(all(c['dependency'].startswith('afterok:') for c in production))
        self.assertEqual([c['command'] for c in calls].count('extract'), 2)
        self.assertEqual(calls[-1]['command'], 'analyze')
        self.assertTrue(all(c['dependency'].startswith('afterany:') for c in calls if c['command'] != 'run'))

    def test_quarter_pair_projection_cancels_exact_remaining_shards_above_ceiling(self):
        schedule = {'production': {'gpt': [{'id': '10', 'gpus': 2}], 'qwen36': []},
                    'canaries': {'gpt': [], 'qwen36': []}}
        stopped = set()
        with patch.object(whole, 'cancel_jobs') as cancel:
            whole.check_projection(schedule, 'gpt', {'10': {'elapsed': 18000}},
                                   {'pairs': 25, 'executable_pairs': 100}, stopped, 10.)
        self.assertTrue((self.root/'STOP.json').exists())
        self.assertEqual(stopped, {'gpt', 'qwen36'})
        self.assertEqual([job['id'] for job in cancel.call_args.args[0]], ['10'])
        self.assertAlmostEqual(whole.verified(self.root/'gpt/PROJECTION.json')['total_projected_gpu_hours'], 98.)

    def test_accounting_carries_all_earlier_terminal_costs(self):
        jobs = [{'id': 'gpu', 'gpus': 2, 'cpus': 16, 'submitted': 0},
                {'id': 'cpu', 'gpus': 0, 'cpus': 4, 'submitted': 0}]
        usage = {'gpu': {'state': 'COMPLETED', 'elapsed': 1800}, 'cpu': {'state': 'FAILED', 'elapsed': 900}}
        with patch.object(whole, 'native_accounting', return_value=usage), \
             patch.object(whole, 'work_remaining_bytes', return_value=20*2**30):
            _, gpu_hours = whole.accounting_snapshot(jobs)
        self.assertEqual(gpu_hours, 3.)
        self.assertEqual(whole.load(self.root/'ACCOUNTING.json')['new_cpu_core_hours'], 4.)

    def test_failed_validation_never_queues_empty_extraction_or_analysis(self):
        import time
        schedule = {'validation': {'id': 'cpu', 'gpus': 0, 'cpus': 4},
                    'canaries': {'gpt': [{'id': 'g', 'gpus': 2}], 'qwen36': [{'id': 'q', 'gpus': 2}]},
                    'production': {'gpt': [], 'qwen36': []}, 'extraction': {}}
        whole.driver.checkpoint(self.root/'JOBS.json', schedule)
        whole.driver.checkpoint(self.root/'ACCOUNTING.json', {'work_remaining_bytes': 20*2**30})
        usage = {'cpu': {'state': 'FAILED'}, 'g': {'state': 'CANCELLED'}, 'q': {'state': 'CANCELLED'}}
        with patch.object(whole.driver, 'deadline_from_slurm', return_value=time.time()+300), \
             patch.object(whole, 'accounting_snapshot', return_value=(usage, 0.)), \
             patch.object(whole, 'native_accounting', return_value=usage), \
             patch.object(whole, 'cancel_jobs') as cancel, patch.object(whole, 'advance_extraction') as extraction:
            whole.monitor()
        extraction.assert_not_called()
        self.assertFalse((self.root/'STOP.json').exists())
        self.assertEqual(whole.load(self.root/'VALIDATION_STOP.json')['reason'], 'cpu_validation_failed')
        self.assertTrue(all(job['gpus'] == 2 for job in cancel.call_args.args[0]))


class ClosedAnalysisTests(unittest.TestCase):
    def test_completed_whole_output_replays_and_extracts_in_worker_scope(self):
        from moe_exp.study import whole_effects
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
            with patch('moe_exp.models.token_replay.make_token_replay', return_value=(text, replay)), \
                 patch('moe_exp.correlation_pipeline.scoring.score_completion_detailed',
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
        from moe_exp.study import whole_effects
        manifests, allocation = {}, []
        for i, benchmark in enumerate(BENCHMARKS):
            allocation.append({'family': str(i), 'benchmark': benchmark})
        for model in whole.MODELS:
            rows = []
            for family in allocation:
                base = {**family, 'model': model, 'dose': 'main', 'fold': int(family['family']) % 5,
                        'total_allowance': 1000, 'source': {}, 'actions': []}
                for arm in ('native', 'native_repeat'):
                    rows.append({**base, 'uid': model+family['family']+arm, 'class': None,
                                 'arm': arm, 'operation': 'native'})
                for operation in ('promote', 'suppress'):
                    for cls in (CLASSES if operation == 'promote' else whole.directions(model)['suppress']):
                        for role in ('target', 'random'):
                            rows.append({**base, 'uid': model+family['family']+operation+cls+role,
                                         'class': cls, 'arm': role+'_'+operation, 'operation': operation})
            manifests[model] = {'model': model, 'dose': 'main', 'records': rows,
                                'scope': 'prompt-and-generated-reasoning'}
        original_load = whole_effects.load
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary, \
             patch.object(whole_effects, 'ROOT', Path(temporary)), \
             patch.object(whole_effects, 'manifest', side_effect=lambda model, dose=None: manifests[model]), \
             patch.object(whole_effects, 'load', side_effect=lambda path: {'records': allocation}
                          if Path(path).name == 'ALLOCATION.json' else original_load(path)):
            for model in whole.MODELS:
                manifests[model] = publish(whole.manifest_path(model, 'main', Path(temporary)), manifests[model])
            publish(Path(temporary)/'RELEASE.json', {'role': 'bounded closeout fixture'})
            publish(Path(temporary)/'gpt/extraction/EXTRACTION.json', {
                'manifest_sha256': manifests['gpt']['sha256'],
                'failures': [{'uid': 'failed-replay-unit', 'error': 'synthetic replay failure'}]})
            result = whole_effects.analyze()
            self.assertTrue((Path(temporary)/'analysis/EFFECTS.json').is_file())
            self.assertIn('Extraction failures: 1', (Path(temporary)/'analysis/REPORT.md').read_text())
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
