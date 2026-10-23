import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from moe_exp.io import CLASSES
from moe_exp.steering.pulse_effects import (
    Bootstrap,
    analysis_assignments,
    bounded_summary,
    continuation_table,
    dynamics_metrics,
    endpoints,
    paired,
    restore_outcomes,
    summarize,
)


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

    def test_no_sentence_rows_are_unknown_not_a_zero_contrast(self):
        o = {'pulse_tokens': 2, 'horizon': 8, 'reasoning_end': None, 'generated_tokens': 8,
             'finish_reason': 'length', 'nonreasoning': json.dumps([])}
        e = endpoints(o, [])
        self.assertTrue((e['occupancy_lower'] == 0).all())
        self.assertTrue((e['occupancy_upper'] == 1).all())
        self.assertTrue((e['first_lower'] == 0).all())
        self.assertTrue((e['first_upper'] == 1).all())
        self.assertAlmostEqual(e['uncovered'], 1.)

    def test_target_dynamics_are_available(self):
        rows = [
            {'sentence_index': 0, 'complete': True, 'lexical_label': 'Analyze'},
            {'sentence_index': 1, 'complete': True, 'lexical_label': 'Verify'},
            {'sentence_index': 2, 'complete': True, 'lexical_label': 'Verify'},
            {'sentence_index': 3, 'complete': True, 'lexical_label': 'Analyze'},
        ]
        value = dynamics_metrics(rows, 'Verify')
        self.assertEqual(value['target_runs'], 1)
        self.assertAlmostEqual(value['target_persistence'], .5)
        self.assertAlmostEqual(value['target_entry'], 1.)
        self.assertAlmostEqual(value['target_exit'], .5)

    def test_gaps_and_unknown_sentences_cannot_create_transitions_or_runs(self):
        first = {'sentence_index': 0, 'complete': True, 'lexical_label': 'Verify'}
        last = {'sentence_index': 2, 'complete': True, 'lexical_label': 'Verify'}
        middle = {'sentence_index': 1, 'complete': True, 'lexical_label': 'Analyze'}
        sequences = (
            [first, last],
            [first, {**middle, 'complete': False}, last],
            [first, {**middle, 'lexical_label': None}, last],
        )
        for rows in sequences:
            with self.subTest(rows=rows):
                value = dynamics_metrics(rows, 'Verify')
                self.assertFalse(value['qualified'])
                for metric in ('ntrans', 'selfrate', 'target_runs', 'target_runmean'):
                    self.assertTrue(np.isnan(value[metric]), metric)


class PairingTest(unittest.TestCase):
    def test_signed_missingness_bounds_enclose_contrasts(self):
        import pandas as pd
        rows = []
        for arm, lower, upper in (
            ('native', .1, .4),
            ('target_activate', .2, .7),
            ('random_activate', .1, .5),
            ('target_deactivate', .2, .7),
            ('random_deactivate', .1, .5),
        ):
            row = {
                'model': 'gpt',
                'family': 'test-family',
                'class': None if arm == 'native' else 'Verify',
                'layer_regime': 'central',
                'benchmark': 'aime24',
                'layer': 1,
                'arm': arm,
                'generated_tokens': 10,
                'prefix_tokens': 2,
                'reasoning_tokens': 8,
                'strict_correct': False,
                'accuracy_lower': 0.,
                'accuracy_upper': 0.,
                'ended': 0.,
            }
            for base in ('occ_', 'first_', 'occs_', 'pulse_occ_', 'full_occ_', 'first_post_', 'first_overlap_'):
                for cls in CLASSES:
                    row[base+cls] = lower if cls == 'Verify' else 0.
                    row[base+'lower_'+cls] = lower if cls == 'Verify' else 0.
                    row[base+'upper_'+cls] = upper if cls == 'Verify' else 0.
            rows.append(row)
        contrasts = paired(pd.DataFrame(rows)).iloc[0]
        for kind in ('occupancy', 'first', 'soft_occupancy', 'pulse_occupancy',
                     'full_occupancy', 'first_post', 'first_overlap'):
            for direction, expected, expected_native in (
                ('activate', (-.3, .6), (-.2, .6)),
                ('deactivate', (-.6, .3), (-.6, .2)),
            ):
                for suffix, bounds in (('', expected), (':vs_native', expected_native)):
                    name = direction+':'+kind+suffix
                    with self.subTest(contrast=name):
                        self.assertAlmostEqual(contrasts[name+':lower'], bounds[0])
                        self.assertAlmostEqual(contrasts[name+':upper'], bounds[1])
                        self.assertLessEqual(contrasts[name+':lower'], contrasts[name])
                        self.assertLessEqual(contrasts[name], contrasts[name+':upper'])
        self.assertAlmostEqual(contrasts['deactivate:response:Verify:lower'], -.6)
        self.assertAlmostEqual(contrasts['deactivate:response:Verify:upper'], .3)
        self.assertAlmostEqual(contrasts['random_deactivate:occupancy:vs_native:lower'], -.4)
        self.assertAlmostEqual(contrasts['random_deactivate:occupancy:vs_native:upper'], .3)


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

    def test_single_family_benchmarks_never_become_zero_draws(self):
        import pandas as pd
        families = {'a': 'aime24', 'b': 'math500'}
        boot = Bootstrap(sorted(families), families, draws=32)
        frame = pd.DataFrame({'family': ['a', 'b'], 'x': [1., 4.]})
        value = boot.estimate(frame, 'x')
        np.testing.assert_array_equal(boot.weights, np.ones((32, 2), dtype=int))
        np.testing.assert_array_equal(value['draws'], np.full(32, 2.5))
        self.assertEqual(value['ci'], [2.5, 2.5])

    def test_missing_unbounded_outcome_is_unqualified_not_dropped(self):
        import pandas as pd
        families = {'a': 'aime24', 'b': 'aime24'}
        boot = Bootstrap(sorted(families), families, draws=16)
        frame = pd.DataFrame({'family': ['a', 'b'], 'x': [1., np.nan]})
        value = summarize(boot, frame, ['x'], 'test')['x']
        self.assertEqual(value['status'], 'unqualified_missing_outcomes')
        self.assertEqual(value['cells'], 2)
        self.assertEqual(value['missing_cells'], 1)
        self.assertNotIn('estimate', value)

    def test_sparse_subset_cannot_replace_an_empty_benchmark_by_zero(self):
        import pandas as pd
        families = {'a': 'aime24', 'b': 'aime24'}
        boot = Bootstrap(sorted(families), families, draws=64)
        value = boot.estimate(pd.DataFrame({'family': ['a'], 'x': [2.]}), 'x')
        self.assertEqual(value['status'], 'unqualified_sparse_benchmark')
        self.assertNotIn('ci', value)

    def test_bounded_summary_keeps_unknown_components(self):
        import pandas as pd
        fams = {'a': 'aime24', 'b': 'math500'}
        boot = Bootstrap(sorted(fams), fams, draws=50)
        frame = pd.DataFrame({
            'family': ['a', 'b'], 'x': [.1, .1], 'x:lower': [0., 0.], 'x:upper': [.2, .2],
        })
        value = bounded_summary(boot, frame, ['x'], 'test')['x']
        self.assertEqual(value['lower_bound'], 0.)
        self.assertEqual(value['upper_bound'], .2)
        self.assertLessEqual(value['bound_simultaneous'][0], value['bound_simultaneous'][1])


class PlannedOutcomesTest(unittest.TestCase):
    def setUp(self):
        """Freeze a tiny five-arm population with no observed continuation files."""
        self.assignments = []
        for arm in ('native', 'target_activate', 'random_activate', 'target_deactivate', 'random_deactivate'):
            self.assignments.append({
                'uid': arm,
                'model': 'gpt',
                'arm': arm,
                'class': None if arm == 'native' else 'Verify',
                'family': 'planned-family',
                'fold': 0,
                'benchmark': 'aime24',
                'source': {'prefix_tokens': 2},
                'prefix_status': 'enrolled',
                'pulse_tokens': 2,
                'horizon': 8,
                'action': None if arm == 'native' else {'layer': 1},
            })

    def absent_table(self):
        """Extract an entirely unavailable fixture without reading or writing data."""
        with patch('moe_exp.steering.pulse_effects.analysis_assignments', return_value=self.assignments), \
             patch('moe_exp.steering.pulse_effects.bulk', return_value=MagicMock()), \
             patch('moe_exp.steering.pulse_effects.log'):
            return continuation_table(['gpt'])

    def test_all_planned_missing_arms_remain_in_bounded_contrasts(self):
        table = self.absent_table()
        self.assertEqual(set(table.uid), {row['uid'] for row in self.assignments})
        self.assertTrue(table.generated_tokens.isna().all())
        self.assertTrue((table.outcome_status == 'outcome_unavailable').all())
        self.assertTrue(table.dyn_clsdist.isna().all())
        contrasts = paired(table)
        self.assertEqual(len(contrasts), 1)
        boot = Bootstrap(['planned-family'], {'planned-family': 'aime24'}, draws=16)
        result = bounded_summary(boot, contrasts, ['activate:occupancy'], 'test')['activate:occupancy']
        self.assertEqual(result['cells'], 1)
        self.assertEqual(result['bound_ci'], [-1., 1.])
        self.assertEqual(result['status'], 'qualified_bounds')
        self.assertEqual(result['unknown_cells'], 1)
        self.assertFalse(result['point_identified'])
        self.assertEqual(result['point_role'], 'compatibility_only')

    def test_accuracy_interval_is_exported_with_unknown_assignments(self):
        table = self.absent_table()
        native = table.arm == 'native'
        table.loc[native, ['strict_correct', 'accuracy_lower', 'accuracy_upper']] = [True, 1., 1.]
        contrasts = paired(table)
        boot = Bootstrap(['planned-family'], {'planned-family': 'aime24'}, draws=16)
        names = ['activate:correct', 'activate:correct:vs_native']
        result = summarize(boot, contrasts, names, 'efficiency')
        self.assertEqual(result[names[0]]['bound_ci'], [-1., 1.])
        self.assertEqual(result[names[1]]['bound_ci'], [-1., 0.])
        self.assertEqual(result[names[1]]['families'], 1)

    def test_zero_length_postpulse_window_is_not_a_null_effect(self):
        for assignment in self.assignments:
            assignment['pulse_tokens'] = assignment['horizon']
        contrasts = paired(self.absent_table())
        self.assertTrue(np.isnan(contrasts['activate:occupancy'].iloc[0]))
        self.assertEqual(contrasts['activate:full_occupancy:lower'].iloc[0], -1.)
        self.assertEqual(contrasts['activate:full_occupancy:upper'].iloc[0], 1.)

    def test_unavailable_prefix_is_distinguished_from_missing_execution(self):
        import pandas as pd
        assignments = [dict(self.assignments[0], source=None, prefix_status='unavailable')]
        result = restore_outcomes(pd.DataFrame(), assignments).iloc[0]
        self.assertEqual(result.outcome_status, 'unavailable_prefix')
        self.assertTrue(np.isnan(result.generated_tokens))
        self.assertTrue(np.isnan(result.prefix_tokens))
        self.assertEqual([result.accuracy_lower, result.accuracy_upper], [0., 1.])

    def test_unavailable_original_assignments_are_restored_without_new_draws(self):
        manifest = {
            'records': self.assignments[:1],
            'unavailable': ['target_activate'],
            'source_assignments_sha256': 'sealed-source',
        }
        original = {'sha256': 'sealed-source', 'records': self.assignments}
        with patch('moe_exp.steering.pulse_effects.verified', side_effect=[manifest, original]):
            rows = analysis_assignments('gpt', 'main')
        self.assertEqual(rows, self.assignments[:2])

    def test_unqualified_rescue_is_rejected_before_reading_outcomes(self):
        with patch('pandas.read_parquet') as read:
            with self.assertRaisesRegex(ValueError, 'requires --cohort'):
                continuation_table(['gpt'], 'rescue')
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
