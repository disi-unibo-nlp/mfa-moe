"""Outcome selection on synthetic routing: scores, stability, protection and the chosen sets."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from steering.outcome import select
from store import WORK


def toy_rates(questions=6, attempts=6, layers=4, experts=16, seed=0):
    """Rates where expert (1, 3) is used more in wrong attempts of every question."""
    rng = np.random.default_rng(seed)
    rates, names, strict = [], [], []
    for q in range(questions):
        level = rng.uniform(0.1, 0.4, size=(layers, experts))      # question-specific baseline
        for a in range(attempts):
            wrong = a % 2 == 0
            rate = level+rng.normal(0, .01, size=(layers, experts))
            rate[1, 3] += .2 if wrong else 0.
            rates.append(rate)
            names.append(f'q{q}')
            strict.append(not wrong)
    return np.asarray(rates), names, strict


def toy_nominations(layers=(10, 11, 12, 13), experts=16, k=2):
    frequency = np.tile(np.linspace(.01, .3, experts), (len(layers), 1))
    records = []
    for fold in range(5):
        for name in ('Read', 'Implement', 'Analyze'):
            positive = [{'layer': layers[0], 'expert': e, 'delta': .5-e/100} for e in (15, 14, 2, 1, 0, 5, 6, 7, 8, 9)]
            records.append({'fold': fold, 'class': name, 'frequency': frequency.tolist(), 'native_k': k,
                            'candidates': {'positive': positive, 'negative': []}})
    return {'layers': list(layers), 'experts': experts, 'records': records, 'sha256': 'x'}


class UsageFolderTests(unittest.TestCase):
    """`usage` writes MODEL/USAGE.json and USAGE.npz under the folder it is given, and touches no other."""
    class InlinePool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *error):
            return False

        def map(self, function, tasks):
            return map(function, tasks)

    def test_usage_lands_in_the_out_folder(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            base = Path(temporary)
            outcomes = base/'outcomes.parquet'
            outcomes.write_bytes(b'table')
            attempts = [{'attempt': f'a{i}', 'question': 'aime24|q', 'tensor': f't{i}', 'tensor_sha256': 'x'} for i in range(2)]
            with patch.object(select, 'cohort_attempts', return_value=attempts), \
                    patch.object(select, 'strict_outcomes', return_value=[True, False]), \
                    patch.object(select, 'class_nominations', return_value={'experts': 4, 'layers': [0, 1]}), \
                    patch.object(select, '_attempt_rate', return_value=np.ones((2, 4), dtype=np.float32)), \
                    patch.object(select, 'ProcessPoolExecutor', self.InlinePool), \
                    patch.object(select, 'OUTCOMES', outcomes), patch.object(select, 'BULK', base/'default'):
                receipt = select.usage('gpt', base/'reports', 1, base/'usage')
                self.assertEqual(receipt['shape'], [2, 2, 4])
                self.assertTrue((base/'usage/gpt/USAGE.json').is_file() and (base/'usage/gpt/USAGE.npz').is_file())
                self.assertFalse((base/'default').exists())
                select.usage('qwen330b', base/'reports', 1)  # without an out folder it writes to BULK, as the release reads it
                self.assertTrue((base/'default/qwen330b/USAGE.json').is_file())
                with self.assertRaisesRegex(ValueError, 'already measured'):  # published tables are never overwritten
                    select.usage('gpt', base/'reports', 1, base/'usage')


class ScoreTests(unittest.TestCase):
    def test_wrong_heavy_expert_scores_highest_and_is_stable(self):
        rates, names, strict = toy_rates()
        score, stable, used = select.stable_scores(rates, names, strict)
        self.assertEqual(len(used), 6)
        self.assertEqual(np.unravel_index(np.argmax(score), score.shape), (1, 3))
        self.assertTrue(stable[1, 3])

    def test_question_level_shift_cancels(self):
        # A question that is uniformly heavier in every attempt must not change any score.
        rates, names, strict = toy_rates()
        shifted = rates.copy()
        shifted[[i for i, n in enumerate(names) if n == 'q0']] += 5.
        a, _ = select.question_scores(rates, names, strict)
        b, _ = select.question_scores(shifted, names, strict)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_unknown_and_single_outcome_questions_are_ignored(self):
        rates, names, strict = toy_rates(questions=3)
        strict = [None if n == 'q0' else s for n, s in zip(names, strict)]     # q0 all unknown
        strict = [True if n == 'q1' else s for n, s in zip(names, strict)]     # q1 all right
        _, used = select.question_scores(rates, names, strict)
        self.assertEqual(used, ['q2'])

    def test_drop_leaves_question_out(self):
        rates, names, strict = toy_rates()
        _, used = select.question_scores(rates, names, strict, frozenset({'q2'}))
        self.assertNotIn('q2', used)


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        self.nominations = toy_nominations()
        self.frequency, self.committee, self.execution = select.protected_pairs(self.nominations)

    def test_protection_covers_core_and_execution(self):
        self.assertIn((10, 15), self.committee)                    # most used expert of layer 10
        self.assertEqual(self.execution, {(10, 15)})               # top 10% of 10 positive candidates
        self.assertEqual(len(self.committee), 4)                   # ceil(.05*16) per layer

    def test_choice_avoids_protected_and_matches_controls(self):
        rates, names, strict = toy_rates()
        score, stable, _ = select.stable_scores(rates, names, strict)
        protected = self.committee | self.execution
        layers = self.nominations['layers']
        out = select.choose(score, stable, self.frequency, layers, 2, protected, ['toy'])
        targets = {(p['layer'], p['expert']) for p in out['targets']}
        controls = {(p['layer'], p['expert']) for p in out['controls']}
        self.assertGreaterEqual(len(targets), select.MIN_PAIRS)
        self.assertIn((11, 3), targets)                            # the planted wrong-heavy expert
        self.assertFalse(targets & protected or controls & protected or targets & controls)
        self.assertEqual(sorted(p['layer'] for p in out['targets']), sorted(p['layer'] for p in out['controls']))

    def test_fill_flags_unstable_pairs_when_too_few_are_stable(self):
        rates, names, strict = toy_rates()
        score, _, _ = select.stable_scores(rates, names, strict)
        stable = np.zeros_like(score, dtype=bool)
        stable[1, 3] = True                                        # only one stable pair
        out = select.choose(score, stable, self.frequency, self.nominations['layers'], 2,
                            self.committee | self.execution, ['toy'])
        self.assertEqual(len(out['targets']), select.MIN_PAIRS)
        self.assertEqual(out['unstable_fill'], select.MIN_PAIRS-1)


if __name__ == '__main__':
    unittest.main()
