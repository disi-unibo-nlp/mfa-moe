"""r2 sweep: halves, rows, the per-half frame and the frozen pick rule."""
import unittest

import numpy as np
import pandas as pd

from moe_exp.outcome_steering import analysis, manifest, pick


def questions():
    return [f'aime24|a{i}' for i in range(30)]+[f'math500|m{i}' for i in range(50)]


class HalfTests(unittest.TestCase):
    def test_halves_are_disjoint_stratified_and_fixed(self):
        split = manifest.halves(questions())
        self.assertFalse(set(split['A']) & set(split['B']))
        self.assertEqual(sorted(split['A']+split['B']), sorted(questions()))
        self.assertEqual(sum(q.startswith('aime24') for q in split['A']), 15)
        self.assertEqual(split, manifest.halves(list(reversed(questions()))))


class RowTests(unittest.TestCase):
    def test_every_cell_gets_target_and_two_draws_from_the_held_out_set(self):
        qs = ['aime24|a1', 'aime24|a2']
        natives = {q: [{'uid': q+str(i), 'question': q, 'attempt': i, 'family': q, 'capture_mode': 'native_ids',
                        'seed': 5} for i in range(6)] for q in qs}
        held = {'targets': [{'layer': 12, 'expert': 3}], 'controls': [[{'layer': 12, 'expert': 4}], [{'layer': 12, 'expert': 5}]]}
        other = {'targets': [{'layer': 16, 'expert': 1}], 'controls': [[{'layer': 16, 'expert': 2}], [{'layer': 16, 'expert': 6}]]}
        selection = {'sets': {'Plan': {'sets': {'all': other, 'aime24|a1': held}}}}
        spec = {'questions': qs, 'half': 'A', 'cells': [{'set': 'Plan', 'delta': .8}, {'set': 'Plan', 'delta': .4}]}
        rows = manifest.sweep_rows('gpt', spec, selection, natives)
        self.assertEqual(len(rows), 2*2*3)
        first = [r for r in rows if r['question'] == 'aime24|a1' and r['delta'] == .8]
        self.assertEqual([(r['arm'], r['draw']) for r in first], [('target_nudge', None), ('random_nudge', 0), ('random_nudge', 1)])
        self.assertEqual([r['actions'][0]['shift'] for r in first], [[3], [4], [5]])
        self.assertTrue(all('capture_mode' not in r and r['native_uid'].endswith('0') for r in rows))


def bounded(log, acc):
    return {'log_reasoning': log, 'log_reasoning:lower': log, 'log_reasoning:upper': log,
            'accuracy': acc, 'accuracy:lower': acc, 'accuracy:upper': acc}


class FrameTests(unittest.TestCase):
    def test_target_against_mean_random_and_mean_native(self):
        natives = {'q': [{**bounded(np.log1p(1000), 1.), 'uid': 'n0', 'attempt': 0},
                         {**bounded(np.log1p(1000), 0.), 'uid': 'n1', 'attempt': 1}]}
        rows = [{**bounded(np.log1p(500), 1.), 'question': 'q', 'family': 'f', 'arm': 'target_nudge',
                 'expert_set': 'Plan', 'class': 'delta-0.8', 'native_uid': 'n0'},
                {**bounded(np.log1p(1000), 0.), 'question': 'q', 'family': 'f', 'arm': 'random_nudge',
                 'expert_set': 'Plan', 'class': 'delta-0.8', 'native_uid': 'n0'},
                {**bounded(np.log1p(2000), 1.), 'question': 'q', 'family': 'f', 'arm': 'random_nudge',
                 'expert_set': 'Plan', 'class': 'delta-0.8', 'native_uid': 'n0'}]
        frame = analysis.sweep_family_frame('gpt', pd.DataFrame(rows), natives)
        random = (np.log1p(1000)+np.log1p(2000))/2
        self.assertAlmostEqual(frame['gpt:Plan:delta-0.8:target_random:tokens'][0], random-np.log1p(500))
        self.assertAlmostEqual(frame['gpt:Plan:delta-0.8:target_random:accuracy'][0], .5)
        self.assertAlmostEqual(frame['gpt:Plan:delta-0.8:target_native:accuracy'][0], .5)
        self.assertAlmostEqual(frame['gpt:noise:accuracy'][0], -1.)


class PickTests(unittest.TestCase):
    def effects(self, cells):
        result = {'sha256': 'e', 'target_random': {}, 'target_native': {}}
        for name, tokens, versus_random, versus_native in cells:
            result['target_random'][f'gpt:{name}:target_random:tokens'] = {'estimate': tokens}
            result['target_random'][f'gpt:{name}:target_random:accuracy'] = {'estimate': versus_random}
            result['target_native'][f'gpt:{name}:target_native:accuracy'] = {'estimate': versus_native}
        return result

    def test_largest_token_benefit_among_cells_keeping_accuracy_plus_blind(self):
        effects = self.effects([('Plan:delta-1.2', .30, 0., -.08), ('Plan:delta-0.8', .20, 0., -.02),
                                ('Verify:delta-0.4', .10, .02, 0.), ('blind:delta-0.4', .05, 0., 0.),
                                ('blind:delta-0.8', .07, 0., -.01)])
        candidates = pick.cells(effects, 'gpt')
        self.assertEqual(pick.best(candidates)['set'], 'Plan')
        self.assertEqual(pick.best(candidates)['delta'], .8)
        self.assertEqual(pick.best([c for c in candidates if c['set'] == 'blind'])['delta'], .8)

    def test_fallback_is_best_accuracy_against_random(self):
        candidates = pick.cells(self.effects([('Plan:delta-0.8', .2, .01, -.2), ('Read:delta-0.4', .1, .03, -.1)]), 'gpt')
        self.assertEqual(pick.best(candidates)['set'], 'Read')


if __name__ == '__main__':
    unittest.main()
