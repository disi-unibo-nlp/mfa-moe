"""Contrasts of the outcome-steering analysis on hand-made families."""
import math
import unittest

import pandas as pd

from moe_exp.outcome_steering import analysis


def unit(arm, tokens, correct, cls=None, family='f1', unknown=False):
    """One bounded row; `unknown` leaves the outcome open as an unfinished run would."""
    log = math.log1p(tokens)
    accuracy = (0., 1.) if unknown else (float(correct),)*2
    return {'family': family, 'arm': arm, 'class': cls, 'finish_reason': 'unknown' if unknown else 'stop',
            'log_reasoning': log, 'log_reasoning:lower': log, 'log_reasoning:upper': log,
            'accuracy': sum(accuracy)/2, 'accuracy:lower': accuracy[0], 'accuracy:upper': accuracy[1],
            'dose_json': '{}'}


class CurveTests(unittest.TestCase):
    def test_curve_range_interpolates_and_stays_flat_beyond_the_ends(self):
        points = [(1., 0., 0.), (3., 1., 1.)]
        self.assertEqual(analysis.curve_range(points, 2., 2.), (.5, .5))
        self.assertEqual(analysis.curve_range(points, 0., .5), (0., 0.))
        self.assertEqual(analysis.curve_range(points, 1.5, 2.5), (.25, .75))


class FamilyFrameTests(unittest.TestCase):
    def setUp(self):
        self.natives = pd.DataFrame([unit('native', 1000, True), unit('native_repeat', 900, True)])

    def test_shorter_equally_correct_target_has_positive_token_benefit(self):
        ours = pd.DataFrame([unit('target_nudge', 500, True, 'delta-0.8'), unit('random_nudge', 1000, True, 'delta-0.8'),
                             unit('budget', 500, False, 'fraction-0.5')])
        frame = analysis.family_frame('gpt', ours, self.natives).iloc[0]
        self.assertAlmostEqual(frame['gpt:delta-0.8:target_random:tokens'], math.log1p(1000)-math.log1p(500))
        self.assertEqual(frame['gpt:delta-0.8:target_random:accuracy'], 0.)
        self.assertEqual(frame['gpt:fraction-0.5:budget_native:accuracy'], -1.)
        # At 500 tokens the length-limit curve is wrong, the nudge right: +1 over the baseline.
        self.assertEqual(frame['gpt:delta-0.8:target_budget:accuracy'], 1.)

    def test_unknown_target_outcome_keeps_an_interval(self):
        ours = pd.DataFrame([unit('target_nudge', 500, True, 'delta-0.8', unknown=True),
                             unit('random_nudge', 1000, True, 'delta-0.8')])
        frame = analysis.family_frame('gpt', ours, self.natives).iloc[0]
        self.assertEqual(frame['gpt:delta-0.8:target_random:accuracy:lower'], -1.)
        self.assertEqual(frame['gpt:delta-0.8:target_random:accuracy:upper'], 0.)

    def test_family_without_a_native_pair_is_left_out(self):
        ours = pd.DataFrame([unit('target_nudge', 500, True, 'delta-0.8', family='f2'),
                             unit('random_nudge', 1000, True, 'delta-0.8', family='f2')])
        self.assertTrue(analysis.family_frame('gpt', ours, self.natives).empty)


if __name__ == '__main__':
    unittest.main()
