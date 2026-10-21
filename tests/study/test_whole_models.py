"""Every study model runs through the same whole-reasoning rules: doses, directions, routers, budgets."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from moe_exp.final_routing.operator import score_override_dual_masked
from moe_exp.study import whole
from moe_exp.study.common import CLASSES, WORK
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


class ManifestTests(unittest.TestCase):
    def manifest(self, model, families, layers, experts, k):
        natives = [{'model': model, 'family': f'{i:03d}', 'benchmark': 'math500', 'fold': i % 5,
                    'seed': 100+i, 'source': {'attempt_id': str(i), 'prefix_tokens': 12},
                    'profile_sha256': 'profile'} for i in range(families)]
        profile = {'layers': list(range(layers)), 'native_k': k, 'experts': experts,
                   'card': {'max_model_len': 32768, 'max_tokens': None}}
        nominations = {'sha256': 'nomination', 'records': [{'fold': f, 'class': c} for f in range(5) for c in CLASSES]}

        def cell(nomination, profile, model, dose, operation):
            return {'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation,
                    'target': [], 'random': [], 'qualified': True}
        with patch('moe_exp.study.whole.cell_actions', side_effect=cell):
            return build_manifest(model, 'main', nominations, {'models': {model: profile}}, natives,
                                  {r['family']: [1, 2] for r in natives}, True)

    def test_row_counts_follow_each_models_suppress_classes_and_roster(self):
        for model, families, layers, experts, k in (('glm', 129, 46, 64, 4), ('lfm25', 112, 38, 64, 4),
                                                    ('gemma', 129, 30, 128, 8)):
            value = self.manifest(model, families, layers, experts, k)
            expected = families*(2+2*(len(CLASSES)+len(directions(model)['suppress'])))
            self.assertEqual(len(value['records']), expected)
            self.assertEqual(len({r['uid'] for r in value['records']}), expected)
            canary = whole.canary_rows({**value, 'sha256': 'synthetic'})
            self.assertEqual(sum(r['operation'] != 'native' for r in canary), 32)


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
