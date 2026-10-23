"""The declared extension rule on hand-made effect summaries."""
import unittest

from moe_exp.steering.outcome.decide import qualifying


def cell(estimate, low, high, status='qualified'):
    return {'status': status, 'estimate': estimate, 'simultaneous': [low, high]}


class RuleTests(unittest.TestCase):
    def effects(self, tokens, accuracy, native):
        return {'target_random': {'gpt:delta-0.8:target_random:tokens': tokens,
                                  'gpt:delta-0.8:target_random:accuracy': accuracy},
                'target_native': {'gpt:delta-0.8:target_native:accuracy': native}}

    def test_shorter_than_random_without_accuracy_loss_qualifies(self):
        cells = qualifying(self.effects(cell(.2, .05, .35), cell(0., -.05, .05), cell(-.005, -.04, .03)))
        self.assertEqual(cells, [{'model': 'gpt', 'delta': 'delta-0.8', 'shorter_without_loss': True,
                                  'more_accurate': False}])

    def test_shorter_but_less_accurate_than_native_does_not(self):
        self.assertEqual(qualifying(self.effects(cell(.2, .05, .35), cell(0., -.05, .05), cell(-.03, -.08, .02))), [])

    def test_more_accurate_than_random_qualifies_and_bounds_win(self):
        accuracy = {**cell(.06, .01, .11), 'bound_simultaneous': [-.02, .14]}
        self.assertEqual(qualifying(self.effects(cell(0., -.1, .1), accuracy, cell(0., -.05, .05))), [])
        cells = qualifying(self.effects(cell(0., -.1, .1), cell(.06, .01, .11), cell(0., -.05, .05)))
        self.assertTrue(cells and cells[0]['more_accurate'])


if __name__ == '__main__':
    unittest.main()
