import json
import unittest
import numpy as np
from moe_exp.study.common import CLASSES
from moe_exp.study.effects import Bootstrap, endpoints


def sentence(a, b, label, complete=True):
    p = {('p_'+c): (0.7 if c == label else 0.05) for c in CLASSES}
    return {'token_start': a, 'token_end': b, 'complete': complete, 'lexical_label': label, **p}


class Endpoints(unittest.TestCase):
    def test_window_occupancy_first_sentence_and_ending(self):
        o = {'pulse_tokens': 2, 'horizon': 12, 'reasoning_end': 10, 'generated_tokens': 40, 'finish_reason': 'stop',
             'nonreasoning': json.dumps([])}
        s = [sentence(0, 4, 'Verify', complete=False), sentence(4, 9, 'Explore'), sentence(9, 20, 'Monitor')]
        e = endpoints(o, s)
        occ = dict(zip(CLASSES, e['occupancy_hard']))
        # window positions 2..11 (10 tokens): Verify 2-3, Explore 4-8, Monitor 9; 10-11 ended
        self.assertAlmostEqual(occ['Verify'], .2)
        self.assertAlmostEqual(occ['Explore'], .5)
        self.assertAlmostEqual(occ['Monitor'], .1)
        self.assertAlmostEqual(e['ended'], .2)
        self.assertEqual(CLASSES[int(np.nanargmax(e['first_hard']))], 'Explore')
        self.assertAlmostEqual(e['occupancy_soft'].sum(), .8*(.7+6*.05))

    def test_no_complete_sentence_is_missing_not_zero(self):
        o = {'pulse_tokens': 0, 'horizon': 4, 'reasoning_end': None, 'generated_tokens': 4, 'finish_reason': 'length',
             'nonreasoning': json.dumps([])}
        e = endpoints(o, [sentence(0, 4, 'Plan', complete=False)])
        self.assertTrue(np.isnan(e['first_hard']).all())
        self.assertAlmostEqual(dict(zip(CLASSES, e['occupancy_hard']))['Plan'], 1.)


class BootstrapTest(unittest.TestCase):
    def test_benchmark_balanced_family_mean(self):
        import pandas as pd
        fams = {'a': 'aime24', 'b': 'aime24', 'c': 'math500'}
        boot = Bootstrap(sorted(fams), fams, draws=200)
        frame = pd.DataFrame({'family': ['a', 'a', 'b', 'c'], 'x': [1., 3., 0., 4.]})
        v = boot.estimate(frame, 'x')
        # aime24: mean(mean(1,3)=2, 0)=1; math500: 4; other benchmarks absent -> mean over present = 2.5
        self.assertAlmostEqual(v['estimate'], 2.5)
        self.assertEqual(v['families'], 3)


if __name__ == '__main__':
    unittest.main()
