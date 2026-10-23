"""Every study model runs through the same whole-reasoning rules: doses, directions, routers, budgets."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from moe_exp.final_routing.operator import score_override_dual_masked
from moe_exp.study import driver, whole
from moe_exp.study.common import CLASSES, WORK, digest, publish, verified
from moe_exp.study.whole import build_manifest, dose_counts, directions


class DoseTests(unittest.TestCase):
    def test_published_counts_and_half_doses(self):
        gpt = {'layers': list(range(24)), 'experts': 32}
        qwen = {'layers': list(range(40)), 'experts': 256}
        self.assertEqual(dose_counts('gpt', gpt, 'main', 'promote'), (10, 50))
        self.assertEqual(dose_counts('gpt', gpt, 'half', 'promote'), (5, 25))
        self.assertEqual(dose_counts('gpt', gpt, 'half', 'suppress'), (0, 10))
        for model in ('qwen36', 'qwen35', 'qwen330b'):
            self.assertEqual(dose_counts(model, qwen, 'main', 'promote'), (15, 0))
            self.assertEqual(dose_counts(model, qwen, 'half', 'suppress'), (0, 250))

    def test_other_models_take_the_qwen3_share_of_their_own_pairs(self):
        glm = {'layers': list(range(46)), 'experts': 64}
        self.assertEqual(dose_counts('glm', glm, 'main', 'promote'), (round(15*46/48), 0))
        self.assertEqual(dose_counts('glm', glm, 'main', 'suppress'), (0, round(500*46*64/(48*128))))
        self.assertEqual(dose_counts('glm', glm, 'half', 'suppress'), (0, 120))
        lfm = {'layers': list(range(38)), 'experts': 64}
        activate, deactivate = dose_counts('lfm25', lfm, 'main', 'suppress')
        self.assertEqual(activate, 0)
        # Never more than half the routed experts of a layer, so native top-k always survives.
        self.assertLessEqual(math.ceil(deactivate/38), 64//2)


class DirectionTests(unittest.TestCase):
    def test_preregistered_signs_drive_promote_and_suppress(self):
        self.assertEqual(directions('gpt')['suppress'], ('Read', 'Analyze', 'Plan', 'Explore'))
        self.assertEqual(directions('qwen36')['suppress'], ('Analyze', 'Plan', 'Implement'))
        self.assertEqual(directions('gpt')['promote'], ('Verify', 'Monitor'))
        self.assertIn('Implement', directions('gemma')['promote'])
        self.assertEqual(directions('gemma')['source'], 'model')

    def test_lfm_uses_pooled_signs(self):
        lfm = directions('lfm25')
        self.assertEqual(lfm['source'], 'pooled')
        self.assertEqual(lfm['promote'], ('Verify', 'Monitor'))
        self.assertEqual(set(lfm['suppress']), {'Read', 'Analyze', 'Plan', 'Implement', 'Explore'})

    def test_eight_model_link_gives_lfm_its_own_signs(self):
        prereg = verified(whole.LINK)
        own = dict(zip(CLASSES, '0++0+--'))
        cells = prereg['cells']+[{'model': 'lfm25', 'class': c, 'tok_reason': {'sign': s}} for c, s in own.items()]
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            link = Path(tmp)/'EFFICIENCY_LINK_V2.json'
            publish(link, {**prereg, 'models': prereg['models']+['lfm25'], 'cells': cells})
            whole.directions.cache_clear()
            try:
                with patch.object(whole, 'LINK', link):
                    lfm, gpt = directions('lfm25'), directions('gpt')
            finally:
                whole.directions.cache_clear()
        self.assertEqual(lfm['source'], 'model')
        self.assertEqual(lfm['promote'], ('Verify', 'Monitor'))
        self.assertEqual(lfm['suppress'], ('Analyze', 'Plan', 'Explore'))
        self.assertEqual(gpt['suppress'], ('Read', 'Analyze', 'Plan', 'Explore'))

    def test_efficiency_directed_cells(self):
        self.assertTrue(whole.efficiency_directed('gemma', 'promote', 'Implement'))
        self.assertFalse(whole.efficiency_directed('gpt', 'promote', 'Implement'))
        self.assertTrue(whole.efficiency_directed('gpt', 'suppress', 'Read'))
        self.assertFalse(whole.efficiency_directed('gpt', 'native', None))


class SigmoidRouterTests(unittest.TestCase):
    def test_activation_and_deactivation_override_the_correction_bias(self):
        torch.manual_seed(5)
        logits = torch.randn(16, 64)
        bias = torch.zeros(64)
        bias[2], bias[7] = -10., 10.  # expert 2 can never win natively; expert 7 always would
        affinity = logits.sigmoid()
        native_ids = (affinity+bias).topk(4, -1).indices
        native_weights = affinity.gather(1, native_ids)
        native_weights = 1.8*native_weights/native_weights.sum(-1, keepdim=True)
        activate = torch.zeros(16, 64, dtype=torch.bool)
        deactivate = torch.zeros(16, 64, dtype=torch.bool)
        activate[:, 2], deactivate[:, 7] = True, True
        active = torch.arange(16) % 2 == 0
        weights, ids = score_override_dual_masked(logits, native_weights, native_ids, activate, deactivate, active,
                                                  scoring='sigmoid', correction_bias=bias, scale=1.8)
        steered = ids[active]
        self.assertTrue((steered == 2).any(-1).all())
        self.assertFalse((steered == 7).any())
        torch.testing.assert_close(weights[active].sum(-1), torch.full((8,), 1.8))
        # The activated expert carries the affinity of the row's highest logit plus epsilon.
        position = (steered == 2).float().argmax(-1)
        expected = (logits[active].max(-1).values+.01).sigmoid()
        chosen = affinity[active].gather(1, steered.long())
        chosen[torch.arange(8), position] = expected
        torch.testing.assert_close(weights[active].gather(1, position[:, None])[:, 0],
                                   1.8*expected/chosen.sum(-1))
        self.assertTrue(torch.equal(ids[~active], native_ids[~active]))
        self.assertTrue(torch.equal(weights[~active], native_weights[~active]))

    def test_softmax_path_is_unchanged(self):
        logits = torch.randn(4, 32)
        affinity = logits.softmax(-1)
        ni = affinity.topk(4, -1).indices
        nw = affinity.gather(1, ni)
        nw /= nw.sum(-1, keepdim=True)
        mask = torch.zeros(4, 32, dtype=torch.bool)
        weights, ids = score_override_dual_masked(logits, nw, ni, mask, mask, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.equal(ids.sort(-1).values, ni.sort(-1).values))


def synthetic_cell(nomination, profile, model, dose, operation):
    """A deterministic stand-in for cell_actions whose actions follow the dose counts."""
    counts = dose_counts(model, profile, dose, operation)
    return {'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation, 'dose': dose,
            'target': [{'layer': 0, 'activate': list(range(counts[0] % 3)), 'deactivate': list(range(counts[1] % 5))}],
            'random': [], 'qualified': True, 'mean_native_frequency': {'target': .1, 'random': .2}}


def synthetic_manifest(model, families, layers, experts, k, dose='main', sweep=False):
    """Build a manifest from synthetic natives, prompts and nominations."""
    natives = [{'model': model, 'family': f'{i:03d}', 'benchmark': 'math500', 'fold': i % 5,
                'seed': 100+i, 'source': {'attempt_id': str(i), 'prefix_tokens': 12},
                'profile_sha256': 'profile'} for i in range(families)]
    profile = {'layers': list(range(layers)), 'native_k': k, 'experts': experts,
               'card': {'max_model_len': 32768, 'max_tokens': None}}
    nominations = {'sha256': 'nomination', 'release_commit': 'abc',
                   'records': [{'fold': f, 'class': c} for f in range(5) for c in CLASSES]}
    with patch('moe_exp.study.whole.cell_actions', side_effect=synthetic_cell):
        return build_manifest(model, dose, nominations, {'models': {model: profile}}, natives,
                              {r['family']: [1, 2] for r in natives}, True, sweep=sweep)


class ManifestTests(unittest.TestCase):
    def test_row_counts_follow_each_models_suppress_classes_and_roster(self):
        for model, families, layers, experts, k in (('glm', 129, 46, 64, 4), ('lfm25', 112, 38, 64, 4),
                                                    ('gemma', 129, 30, 128, 8)):
            value = synthetic_manifest(model, families, layers, experts, k)
            expected = families*(2+2*(len(CLASSES)+len(directions(model)['suppress'])))
            self.assertEqual(len(value['records']), expected)
            self.assertEqual(len({r['uid'] for r in value['records']}), expected)
            canary = whole.canary_rows({**value, 'sha256': 'synthetic'})
            self.assertEqual(sum(r['operation'] != 'native' for r in canary), 32)

    def test_base_manifests_of_the_running_releases_are_unchanged(self):
        # Digests from the code that built the r7/r8 production manifests (commit a7df701), on the
        # same synthetic inputs: the strength sweep must not move any planned row or UID.
        frozen = {('gemma', 'main'): '171a90091a827d91d7f4e875d4559457f25f1ac3f5013f4019bdfe67082b7f9a',
                  ('glm', 'half'): '5e82f59ffad516d7bbdccb25f3d27dc300187dcb87bc3459c2be0ecac8632c9c',
                  ('gpt', 'half'): '8c0b31d3d9e96c6309921bddc8ef306f5ea223094865279a0a35afc5b098203f',
                  ('lfm25', 'main'): '9bf6583a95fc137768b8b2be5432cd00e2702da18504c0c765be6a291a2a41ce'}
        shapes = {'gemma': (129, 30, 128, 8), 'glm': (129, 46, 64, 4), 'gpt': (129, 24, 32, 4), 'lfm25': (112, 38, 64, 4)}
        for (model, dose), expected in frozen.items():
            self.assertEqual(digest(synthetic_manifest(model, *shapes[model], dose=dose)), expected)


class StrengthSweepTests(unittest.TestCase):
    def test_counts_grow_with_strength_and_keep_native_top_k(self):
        b = driver.binding()
        for model in whole.PLAN:
            profile = b['models'][model]
            layers, experts, k = len(profile['layers']), profile['experts'], profile['native_k']
            for operation in ('promote', 'suppress'):
                counts = [dose_counts(model, profile, dose, operation) for dose in ('quarter', 'half', 'main')]
                for role in (0, 1):
                    series = [count[role] for count in counts]
                    self.assertEqual(series, sorted(series), (model, operation, role))
                    # A role present at full strength is present at every strength.
                    self.assertTrue(series[-1] == 0 or series[0] > 0)
                activate, deactivate = counts[-1]
                self.assertLessEqual(activate, layers*(k-1))
                self.assertLessEqual(deactivate, layers*(experts-k))

    def test_sweep_manifest_runs_only_efficiency_cells_without_natives(self):
        model, shape = 'gemma', (129, 30, 128, 8)
        sweep = synthetic_manifest(model, *shape, dose='half', sweep=True)
        base = synthetic_manifest(model, *shape, dose='half')
        planned = len(directions(model)['promote'])+len(directions(model)['suppress'])
        self.assertEqual(len(sweep['records']), 129*2*planned)
        self.assertEqual(sweep['design'], 'efficiency-sweep')
        self.assertNotIn('design', base)
        self.assertFalse(any(r['operation'] == 'native' or r['audit_capture'] for r in sweep['records']))
        self.assertTrue(all(whole.efficiency_directed(model, r['operation'], r['class']) for r in sweep['records']))
        self.assertEqual(len({r['uid'] for r in sweep['records']}), len(sweep['records']))
        quarter = synthetic_manifest(model, *shape, dose='quarter', sweep=True)
        main = synthetic_manifest(model, *shape)
        self.assertFalse({r['uid'] for r in quarter['records']} & {r['uid'] for r in main['records']})
        # Sweep rows are the base rows of the same dose: same prompt, seed and actions (common random numbers).
        strip = lambda row: {k: v for k, v in row.items() if k not in ('uid', 'audit_capture')}
        same_dose = {(r['family'], r['arm'], r['class']): strip(r) for r in base['records']}
        for row in sweep['records']:
            self.assertEqual(strip(row), same_dose[row['family'], row['arm'], row['class']])
        seeds = {(r['family'], r['arm'], r['class']): r['seed'] for r in main['records']}
        self.assertTrue(all(seeds[r['family'], r['arm'], r['class']] == r['seed'] for r in quarter['records']))


class TopUpTests(unittest.TestCase):
    def test_topup_keeps_only_never_started_rows_of_the_failed_shards(self):
        from moe_exp.study.common import publish
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            parent, child = Path(directory)/'parent', Path(directory)/'child'
            rows = [{'uid': digest([family, arm]), 'family': family, 'arm': arm, 'nomination_status': 'nominated'}
                    for family in ('f0', 'f1', 'f2', 'f3') for arm in ('native', 'target_promote')]
            base = publish(whole.manifest_path('gemma', 'main', parent),
                           {'model': 'gemma', 'dose': 'main', 'scope': 'prompt-and-generated-reasoning', 'records': rows})
            units = parent/'stores/gemma/main/units'
            publish(units/digest(['f1', 'native'])/'RESULT.json', {'status': 'complete'})
            (units/digest(['f3', 'native'])).mkdir(parents=True)
            (units/digest(['f3', 'native'])/'PARTIAL.json').write_text(json.dumps({'tokens': [1, 2]}))
            with patch.object(whole, 'ROOT', child), patch.object(whole, 'SWEEP', {'gemma': 'main'}):
                result = whole.topup_manifest('gemma', {'parent': str(parent), 'shards': [[1, 2]]})
                value = whole.load(whole.manifest_path('gemma', 'main', child))
                selected = whole.load(child/'gemma/SELECTED.json')
        # Shard 1/2 holds f1 and f3; their started rows stay with the parent as complete or unknown.
        self.assertEqual({(r['family'], r['arm']) for r in value['records']},
                         {('f1', 'target_promote'), ('f3', 'target_promote')})
        self.assertEqual(result['planned'], 2)
        self.assertEqual((value['design'], value['parent_manifest_sha256']), ('top-up', base['sha256']))
        self.assertEqual((selected['sweep'], selected['manifest_sha256']), (True, value['sha256']))

    def test_second_topup_covers_all_shards_and_leaves_rows_started_by_the_first(self):
        from moe_exp.study.common import publish
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            parent, first, second = (Path(directory)/name for name in ('parent', 'first', 'second'))
            rows = [{'uid': digest([family, arm]), 'family': family, 'arm': arm, 'nomination_status': 'nominated'}
                    for family in ('f0', 'f1') for arm in ('native', 'target_promote')]
            base = publish(whole.manifest_path('gemma', 'main', parent), {'model': 'gemma', 'dose': 'main',
                           'scope': 'prompt-and-generated-reasoning', 'records': rows})
            publish(parent/'stores/gemma/main/units'/digest(['f0', 'native'])/'RESULT.json', {'status': 'complete'})
            # The first top-up planned f1's rows and started one of them; the other never started.
            publish(whole.manifest_path('gemma', 'main', first), {'records': rows[2:],
                    'parent_manifest_sha256': base['sha256']})
            publish(first/'stores/gemma/main/units'/digest(['f1', 'native'])/'RESULT.json', {'status': 'complete'})
            with patch.object(whole, 'ROOT', second), patch.object(whole, 'SWEEP', {'gemma': 'main'}):
                whole.topup_manifest('gemma', {'parent': str(parent), 'shards': 'all', 'siblings': [str(first)]})
                value = whole.load(whole.manifest_path('gemma', 'main', second))
        self.assertEqual({(r['family'], r['arm']) for r in value['records']},
                         {('f0', 'target_promote'), ('f1', 'target_promote')})


class CanaryDeadlineTests(unittest.TestCase):
    def test_rows_running_at_the_deadline_count_as_reaching_the_length_limit(self):
        from moe_exp.study import driver
        from moe_exp.study.common import digest, publish
        with tempfile.TemporaryDirectory(dir=WORK) as directory:
            store = driver.Store(Path(directory))
            rows = []
            for family in ('a', 'b'):
                for arm, operation in (('native', 'native'), ('native_repeat', 'native'),
                                       ('target_promote', 'promote'), ('random_promote', 'promote')):
                    rows.append({'uid': digest([family, arm]), 'family': family, 'arm': arm, 'operation': operation,
                                 'class': None if operation == 'native' else 'Verify', 'actions': [],
                                 'prompt_scope': True})
            unfinished = {digest(['b', 'native_repeat']), digest(['b', 'target_promote'])}
            for row in rows:
                folder = store.folder(row['uid'])
                folder.mkdir(parents=True)
                tokens = list(range(400 if row['uid'] in unfinished else 100))
                if row['uid'] in unfinished:
                    (folder/'PARTIAL.json').write_text(json.dumps({'tokens': tokens, 'segments': [{}]}))
                    continue
                publish(folder/'OUTPUT.json', {'token_ids': tokens, 'finish_reason': 'stop'})
                publish(folder/'CAPTURE.json', {'stats': {'dose': {}}})
                publish(folder/'RESULT.json', {'status': 'complete'})
            gate = whole.fluency(store, rows, require_complete=True, censor_unfinished=True)
            strict = whole.fluency(store, rows, require_complete=True)
        self.assertEqual((gate['completed'], gate['censored_at_deadline']), (6, 2))
        self.assertNotIn('incomplete_canary', gate['software_failures'])
        # One steered and one native row reached the limit in family b: cap rates move together.
        self.assertEqual(gate['cap_rate'], .25)
        self.assertEqual(gate['native_cap_rate'], .25)
        self.assertIn('incomplete_canary', strict['software_failures'])


class CanaryDecisionTests(unittest.TestCase):
    def test_decision_must_name_this_manifest_and_unchanged_evidence(self):
        from moe_exp.study.common import file_sha, publish
        with tempfile.TemporaryDirectory(dir=WORK) as directory, patch.object(whole, 'ROOT', Path(directory)):
            evidence = publish(Path(directory)/'evidence/CANARY_HALF.json', {'gate': {'passed': False}})
            path = Path(directory)/'evidence/CANARY_HALF.json'
            publish(Path(directory)/'gpt/CANARY_DECISION.json', {
                'decision': 'proceed', 'manifest_sha256': 'manifest', 'canary_receipt': str(path),
                'canary_receipt_sha256_file': file_sha(path), 'evidence_sha256': evidence['sha256']})
            self.assertEqual(whole.canary_decision('gpt', {'sha256': 'manifest'})['decision'], 'proceed')
            with self.assertRaisesRegex(ValueError, 'canary decision differs'):
                whole.canary_decision('gpt', {'sha256': 'another manifest'})

    def test_sweep_production_is_never_gated_or_projected(self):
        schedule = {'production': {'gpt': [{'id': '1', 'gpus': 2, 'minutes': 240}]}}
        progress = {'pairs': 90, 'executable_pairs': 100}
        with patch.object(whole, 'SWEEP', {'gpt': 'main'}), \
             patch.object(whole, 'production_progress', return_value=progress), \
             patch.object(whole, 'fluency') as gate, patch.object(whole, 'check_projection') as projection:
            whole.check_production(schedule, 'gpt', {'1': {'elapsed': 10000}}, set(), set(), set(), 1.)
        gate.assert_not_called()
        projection.assert_not_called()

    def test_models_without_a_canary_attempt_are_not_advanced(self):
        schedule = {'canaries': {'gpt': []}, 'production': {'gpt': []}}
        with patch.object(whole, 'submit_stage') as submit:
            whole.advance_canary(schedule, 'gpt', {}, set())
        submit.assert_not_called()


class BudgetTests(unittest.TestCase):
    def test_six_model_release_reserves_every_other_models_maxima(self):
        models = ('qwen35', 'qwen330b', 'gemma', 'glm', 'nemotron', 'lfm25')
        maxima = sum(2*(whole.CANARY_MINUTES[m]+whole.SHARDS[m]*whole.RUN_MINUTES[m])/60 for m in models)
        with tempfile.TemporaryDirectory(dir=WORK) as directory, \
             patch.object(whole, 'ROOT', Path(directory)), patch.object(whole, 'MODELS', models), \
             patch.object(whole, 'GPU_CEILING', 530.), \
             patch.object(whole, 'verified', side_effect=lambda path: (
                 {'prior_usage': {'new_gpu_hours': 0.}} if path.name == 'RELEASE.json' else whole.load(path))):
            schedule = {'production': {m: [] for m in models},
                        'canaries': {m: [{'id': m, 'gpus': 2, 'minutes': whole.CANARY_MINUTES[m]}] for m in models}}
            whole.admit_shard_plan(schedule, 'glm', {}, set())
            plan = whole.load(Path(directory)/'glm/SHARD_PLAN.json')
        self.assertLess(maxima, 530.)
        self.assertEqual(plan['minutes'], whole.RUN_MINUTES['glm'])


if __name__ == '__main__':
    unittest.main()
