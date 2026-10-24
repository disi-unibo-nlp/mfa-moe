"""r2 selection, signal table and attempt selection on synthetic captures with a planted effect."""
import unittest

import numpy as np

from moe_exp.outcome_steering import attempts, evidence, select

LAYERS, EXPERTS, K = list(range(24)), 32, 4


def synthetic(questions=40, per=6, planted=(16, 7), seed=3):
    """Usage where the planted pair is used more in wrong attempts; Plan tokens carry the effect."""
    rng = np.random.default_rng(seed)
    n = questions*per
    counts = np.zeros((n, 9, len(LAYERS), EXPERTS), np.float32)
    tokens = np.zeros((n, 9), np.int64)
    records = []
    for a in range(n):
        q, wrong = a//per, (a % per) < (per//2 if (a//per) % 4 else 0)   # every 4th question all right
        length = int(rng.integers(500, 1500))
        rate = rng.dirichlet(np.ones(EXPERTS)*5, size=len(LAYERS))*K
        if wrong:
            rate[planted[0], planted[1]] += .3
        for c in range(9):
            share = 1. if c >= 7 else (.3 if c == 2 else .1)
            tokens[a, c] = int(length*share)
            counts[a, c] = rate*tokens[a, c]
        records.append({'question': f'b|q{q}', 'attempt': a % per, 'usable': True, 'strict': not wrong,
                        'reasoning_tokens': length, 'model_answer': '7' if not wrong else f'1{a:03d}'})
    usage = {'layers': LAYERS, 'experts': EXPERTS, 'attempts': records, 'sha256': 'u'}
    return usage, {'counts': counts, 'tokens': tokens, 'weights': counts[:, 7]}


class WindowTests(unittest.TestCase):
    def test_window_layers_follow_beyond_the_trace_depths(self):
        self.assertEqual(select.window_layers(list(range(24))), [12, 16, 20])
        self.assertEqual(select.window_layers(list(range(48))), [24, 32, 40])

    def test_planted_pair_ranks_first_and_holdouts_drop_their_question(self):
        usage, arrays = synthetic()
        rates, present = select.class_rates(arrays['counts'], arrays['tokens'], 2)
        questions = [r['question'] for r in usage['attempts']]
        strict = [r['strict'] for r in usage['attempts']]
        differences, used = select.question_differences(rates, questions, strict, present)
        self.assertEqual(len(used), 30)                           # all-right questions are skipped
        scores = select.holdout_scores(differences, used, ['b|q1'])
        self.assertEqual(np.unravel_index(np.argmax(scores['all']), scores['all'].shape), (16, 7))
        again, _ = select.question_differences(rates, questions, strict,
                                               present & (np.array(questions) != 'b|q1'))
        np.testing.assert_allclose(scores['b|q1'], again.mean(0))

    def test_choice_stays_in_window_caps_pairs_and_draws_two_disjoint_matched_controls(self):
        usage, arrays = synthetic()
        frequency = select.class_rates(arrays['counts'], arrays['tokens'], 7)[0].mean(0)
        score = np.random.default_rng(1).normal(size=frequency.shape)
        protected = {(12, 0), (16, 1)}
        chosen = select.choose_window(score, frequency, LAYERS, [12, 16, 20], K, protected, ['m', 'x', 'all'])
        targets = {(p['layer'], p['expert']) for p in chosen['targets']}
        self.assertTrue(all(layer in (12, 16, 20) for layer, _ in targets))
        self.assertLessEqual(len(targets), select.R2_MAX_PAIRS)
        self.assertFalse(targets & protected)
        self.assertTrue(all(score[l, e] > 0 for l, e in targets))
        first, second = ({(p['layer'], p['expert']) for p in d} for d in chosen['controls'])
        self.assertEqual(len(first), len(targets))
        self.assertFalse(first & second or first & targets or second & targets or (first | second) & protected)
        def quintiles(pairs):
            result = []
            for p in pairs:
                ranks = sorted(range(EXPERTS), key=lambda e: (float(frequency[p['layer'], e]), e))
                result.append((p['layer'], ranks.index(p['expert'])*5//EXPERTS))
            return sorted(result)
        for draw in chosen['controls']:
            self.assertEqual(quintiles(draw), quintiles(chosen['targets']))   # same layer and usage quintile

    def test_set_size_is_the_window_gate_weight_share(self):
        usage, arrays = synthetic()
        frequency = select.class_rates(arrays['counts'], arrays['tokens'], 7)[0].mean(0)
        gate = np.full(frequency.shape, .01)
        score = np.zeros(frequency.shape)
        score[12, :8] = np.arange(8, 0, -1)
        chosen = select.choose_window(score, frequency, LAYERS, [12, 16, 20], K, set(), ['m', 'g', 'all'], gate=gate)
        self.assertEqual(len(chosen['targets']), select.R2_MIN_PAIRS)          # 5 x 1/96 already exceeds 1.2%
        self.assertAlmostEqual(chosen['mass'], 5/(3*EXPERTS))
        self.assertIn('slot_share', chosen)

    def test_fixed_set_with_a_crowded_quintile_falls_back_to_nearest_usage(self):
        frequency = np.tile(np.linspace(.01, .3, EXPERTS), (len(LAYERS), 1))
        fixed = [(12, 20), (12, 21), (12, 22), (12, 23), (16, 5)]           # four in one layer/quintile
        protected = {(12, e) for e in range(18, 26) if e not in (20, 21, 22, 23)}
        chosen = select.choose_window(np.zeros(frequency.shape), frequency, LAYERS, [12, 16, 20], K, protected,
                                      ['gpt', 'blind', 'all'], fixed=fixed)
        first, second = ({(p['layer'], p['expert']) for p in d} for d in chosen['controls'])
        self.assertIn('nearest_usage', chosen['control_matching'])
        self.assertFalse(first & second or (first | second) & (protected | set(fixed)))
        self.assertEqual(sorted(l for l, _ in first), sorted(l for l, _ in fixed))

    def test_fixed_set_is_kept_with_its_controls(self):
        usage, arrays = synthetic()
        frequency = select.class_rates(arrays['counts'], arrays['tokens'], 7)[0].mean(0)
        chosen = select.choose_window(np.zeros(frequency.shape), frequency, LAYERS, [12, 16, 20], K, set(),
                                      ['gpt', 'blind', 'all'], fixed=list(select.PAPER_FIVE))
        self.assertEqual([(p['layer'], p['expert']) for p in chosen['targets']], list(select.PAPER_FIVE))
        self.assertEqual(len(chosen['controls']), 2)


class SignalTests(unittest.TestCase):
    def test_planted_class_signal_beats_shuffles_and_replicates(self):
        usage, arrays = synthetic(planted=(16, 7))
        result = evidence.signal(usage, arrays, set(), 'test')['classes']
        plan = result['Plan']
        self.assertLess(plan['p_window'], .05)
        self.assertGreater(plan['heldout_window_mean'], 0)
        self.assertGreater(plan['real_window'], plan['null_window_95'])


class AttemptTests(unittest.TestCase):
    def test_majority_and_consensus(self):
        self.assertEqual(attempts.majority(['3', '7', ' 7', None, '\\boxed{7}'], 'q'), 1)
        self.assertIsNone(attempts.majority([None, ''], 'q'))
        rates = np.array([[[1., 0.]], [[.9, .1]], [[0., 1.]]])
        self.assertIn(attempts.jaccard_consensus(rates), (0, 1))

    def test_selection_receipt_reports_every_method(self):
        usage, arrays = synthetic(questions=12)
        selection = {'sha256': 's', 'sets': {'blind': {'sets': {'all': {'targets': [{'layer': 16, 'expert': 7}]}}}}}
        result = attempts.attempt_selection(usage, arrays, selection, 'test')
        methods = result['all_questions']['methods']
        self.assertEqual(methods['majority']['accuracy'], 1.)     # wrong attempts all disagree
        self.assertGreater(methods['outcome_blind']['accuracy'], methods['random']['accuracy'])


if __name__ == '__main__':
    unittest.main()
