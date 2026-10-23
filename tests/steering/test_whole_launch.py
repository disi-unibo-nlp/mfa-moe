"""Launch regressions: actual nomination optima, continuous admission, relative gates and the job graph."""
import math
import os
from pathlib import Path
import tempfile
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from steering import pulse as driver, whole, whole_effects
from store import BENCHMARKS, REPO, WORK, load, publish
from steering.pulse_effects import Bootstrap
from steering.selection import select_pairs
from test_driver import FakeEngine

LINK = driver.DRIVER/'EFFICIENCY_LINK_PREREG.json'  # the seven-model pre-registration


class ActualNominationTests(unittest.TestCase):
    def test_greedy_matches_all_70_dp_optima_and_preserves_other_exact_rank_cells(self):
        """Use saved discovery evidence; neither nominations nor tests read whole outcomes."""
        history = driver.ROOT.parent/'whole-reasoning-20261020-r2'
        proposal = load(history/'CONTROL_FEASIBLE_RD_BOTH_DOSES.json')
        references = {(r['dose'], r['fold'], r['class']): r for r in proposal['records']}
        for model, layers, experts, native_k in (('gpt', 24, 32, 4), ('qwen36', 40, 256, 8)):
            path = history/'gpt/WHOLE_NOMINATIONS.json' if model == 'gpt' else history/'qwen-feasibility/qwen36/WHOLE_NOMINATIONS.json'
            nominations = load(path)
            profile = {'layers': list(range(layers)), 'experts': experts, 'native_k': native_k}
            for dose in ('main', 'half'):
                for nomination in nominations['records']:
                    for operation in ('promote', 'suppress'):
                        if operation == 'suppress' and nomination['class'] not in whole.directions(model, LINK)['suppress']:
                            continue
                        cell = whole.cell_actions(nomination, profile, model, dose, operation)
                        targets = {role: {(a['layer'], e) for a in cell['target'] for e in a[role]}
                                   for role in ('activate', 'deactivate')}
                        counts = whole.dose_counts(model, profile, dose, operation)
                        if model == 'gpt' and operation == 'promote':
                            reference = references[dose, nomination['fold'], nomination['class']]
                            for role in targets:
                                self.assertEqual(targets[role], {(p['layer'], p['expert']) for p in reference[role]})
                            self.assertAlmostEqual(cell['retained_rd_fraction'], reference['rd_fraction'])
                            controls = {role: {(a['layer'], e) for a in cell['random'] for e in a[role]}
                                        for role in targets}
                            self.assertEqual(controls, {role: {(p['layer'], p['expert']) for p in reference['controls'][role]}
                                                        for role in targets})
                        else:
                            for role, count in zip(('activate', 'deactivate'), counts):
                                polarity = 'negative' if role == 'deactivate' and operation == 'promote' else 'positive'
                                exact, _ = select_pairs(nomination['candidates'][polarity], count, profile['layers'], role)
                                self.assertEqual(targets[role], {(p['layer'], p['expert']) for p in exact})


class ContinuousAdmissionTests(unittest.TestCase):
    def test_tiers_order_all_primary_families_before_other_promote(self):
        rows = [
            {'model': 'gpt', 'family': f, 'arm': arm, 'class': cls, 'operation': operation}
            for f in ('a', 'b') for arm, cls, operation in (
                ('target_promote', 'Read', 'promote'), ('target_promote', 'Verify', 'promote'),
                ('random_suppress', 'Analyze', 'suppress'), ('native_repeat', None, 'native'),
                ('native', None, 'native'), ('target_promote', 'Monitor', 'promote'))
        ]
        ordered = whole.admission_order(list(reversed(rows)), LINK)
        self.assertTrue(all(r['class'] != 'Read' for r in ordered[:-2]))
        self.assertEqual([r['family'] for r in ordered[-2:]], ['a', 'b'])
        self.assertEqual(ordered[:5], sorted(ordered[:5], key=lambda r: (r['family'], r['arm'], r['class'] or '')))

    def test_engine_admits_64_across_families_and_refills_before_long_row_finishes(self):
        rows = [{'uid': f'{i:064x}', 'family': str(i//2)} for i in range(70)]
        fake = FakeEngine({r['uid']: 200 if i == 0 else 5 for i, r in enumerate(rows)})
        snapshots = []
        original_step = fake.step
        def step():
            snapshots.append(set(fake.live))
            return original_step()
        fake.step = step
        def build(row, history, segment):
            return {'prompt_ids': [0], 'params': SimpleNamespace(seed=7, max_tokens=250),
                    'request': None, 'total': 250}
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            engine = driver.Engine(SimpleNamespace(llm_engine=fake), driver.Store(Path(temporary)), time.time()+30,
                                   hooked=False, max_inflight=64, drain_seconds=0)
            engine.run(rows, build, lambda *args: None)
        self.assertEqual(len(snapshots[0]), 64)
        self.assertTrue(any(rows[0]['uid'] in snapshot and rows[-1]['uid'] in snapshot for snapshot in snapshots))


class RelativeGateTests(unittest.TestCase):
    def rows(self, native_cap=True, steered_cap=True, ratio=1., native_rep=.3, steered_rep=.3):
        return [{'family': 'a', 'arm': arm, 'operation': operation, 'tokens': int(6000*length),
                 'cap': cap, 'repetition': repetition} for arm, operation, length, cap, repetition in (
            ('native', 'native', 1., native_cap, native_rep),
            ('native_repeat', 'native', 1., native_cap, native_rep),
            ('target_promote', 'promote', ratio, steered_cap, steered_rep),
            ('random_promote', 'promote', ratio, steered_cap, steered_rep))]

    def test_historically_long_and_repetitive_native_does_not_fail_equal_steering(self):
        self.assertEqual(whole.relative_fluency(self.rows())['fluency_failures'], [])

    def test_only_relative_cap_and_repetition_excess_fail(self):
        self.assertEqual(whole.relative_fluency(self.rows(native_cap=False))['fluency_failures'], ['cap'])
        # Longer but fluent reasoning is the measured outcome, never a gate.
        value = whole.relative_fluency(self.rows(ratio=3.))
        self.assertEqual(value['fluency_failures'], [])
        self.assertGreater(value['geometric_length_ratio_to_native'], 1.4)
        self.assertEqual(whole.relative_fluency(self.rows(steered_rep=.41))['fluency_failures'], ['repetition'])

    def test_early_comparison_waits_for_both_same_family_native_references(self):
        value = whole.relative_fluency(self.rows()[1:])
        self.assertEqual(value['steered_units'], 0)


class DesignAndCoverageTests(unittest.TestCase):
    def frame(self):
        """A complete synthetic design with distinct non-primary Read perturbations."""
        records = []
        for model in whole.MODELS:
            base = {'model': model, 'dose': 'main', 'family': 'family', 'nomination_status': 'nominated',
                    **{metric+suffix: value for metric, value in (('log_reasoning', 2.), ('accuracy', 1.))
                       for suffix in ('', ':lower', ':upper')},
                    **{'occ_'+cls+suffix: 0. for cls in whole.CLASSES for suffix in ('', ':lower', ':upper')}}
            for arm in ('native', 'native_repeat'):
                records.append({**base, 'arm': arm, 'class': None, 'operation': 'native'})
            for operation in ('promote', 'suppress'):
                for cls in (whole.CLASSES if operation == 'promote' else whole.directions(model, LINK)['suppress']):
                    for role in ('target', 'random'):
                        records.append({**base, 'arm': role+'_'+operation, 'class': cls, 'operation': operation})
        return pd.DataFrame(records)

    def test_dropped_promote_cells_are_excluded_and_primary_is_unchanged(self):
        table = self.frame()
        baseline, _ = whole_effects.family_contrasts(table, LINK)
        drop = (table.model == 'gpt') & (table['class'] == 'Read') & (table.operation == 'promote')
        table.loc[drop, 'nomination_status'] = 'dropped_availability'
        table.loc[drop, 'log_reasoning:lower'] = 0.
        table.loc[drop, 'log_reasoning:upper'] = 20.
        cells, _ = whole_effects.family_contrasts(table, LINK)
        self.assertFalse(((cells.model == 'gpt') & (cells['class'] == 'Read') & (cells.operation == 'promote')).any())
        pd.testing.assert_frame_equal(whole_effects.primary_frame(cells), whole_effects.primary_frame(baseline))

    def test_an_unnominated_efficiency_cell_is_a_design_error(self):
        table = self.frame()
        table.loc[table['class'] == 'Verify', 'nomination_status'] = 'dropped_availability'
        with self.assertRaisesRegex(ValueError, 'efficiency-directed'):
            whole_effects.family_contrasts(table, LINK)

    def test_sentence_coverage_failure_bounds_one_unit_and_keeps_the_other_observed(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            root = Path(temporary)
            records = [{'uid': uid, 'family': uid, 'model': 'gpt', 'dose': 'main', 'total_allowance': 100,
                        'nomination_status': 'nominated'} for uid in ('valid', 'invalid')]
            folder = root/'gpt/extraction'
            folder.mkdir(parents=True)
            outcomes = [{**row, 'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5,
                         'strict_correct': True, 'finish_reason': 'stop'} for row in records]
            spans = [{'uid': uid, 'token_start': 0, 'token_end': end, 'lexical_label': 'Read'}
                     for uid, end in (('valid', 5), ('invalid', 6))]
            pd.DataFrame(outcomes).to_parquet(folder/'outcomes.parquet')
            pd.DataFrame(spans).to_parquet(folder/'sentences.parquet')
            publish(root/'RELEASE.json', {'classifier': {}})
            publish(whole.manifest_path('gpt', 'main', root), {'records': records})
            with patch('routing.lexical.predict', side_effect=lambda frame, *args, **kwargs: frame):
                table = whole_effects.load_table([{'model': 'gpt', 'root': str(root), 'dose': 'main'}]).set_index('uid')
        self.assertEqual(table.loc['valid', 'outcome_status'], 'observed')
        self.assertEqual(table.loc['invalid', 'outcome_status'], 'unknown_sentence_coverage')
        self.assertEqual(table.loc['invalid', 'accuracy:lower'], 0.)
        self.assertEqual(table.loc['invalid', 'accuracy:upper'], 1.)
        self.assertAlmostEqual(table.loc['invalid', 'log_reasoning:upper'], math.log1p(100))


class LaunchGraphTests(unittest.TestCase):
    """scripts/steering/launch.sh on a sealed release prints the sbatch commands of its graph (DRY_RUN=1)."""
    PY = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python')

    def launch(self, kind_file: str, shards: dict) -> tuple[list[list[str]], str]:
        """Lay out a repository-like folder around a made-up release, run the launcher, return its commands."""
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            repo = Path(temporary)
            for folder in ('src', 'scripts'):
                os.symlink(REPO/folder, repo/folder)
            root = repo/'report/release'
            publish(root/'RELEASE.json', {'release_root': str(repo), 'models': list(shards), 'shards': shards})
            publish(root/kind_file, {})
            run = subprocess.run(['bash', str(repo/'scripts/steering/launch.sh'), 'release'], cwd=repo,
                                 env={**os.environ, 'DRY_RUN': '1'}, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            return [line.split() for line in run.stderr.splitlines() if line.startswith('sbatch')], str(root)

    @staticmethod
    def option(command, name):
        return next(word.split('=', 1)[1] for word in command if word.startswith(name+'='))

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_sweep_release_runs_tests_and_manifests_then_shards_then_one_extraction_per_model(self):
        jobs, root = self.launch('COMPLEMENT.json', {'gpt': 2, 'glm': 1})
        names = [command[command.index('-J')+1] for command in jobs]
        self.assertEqual(names, ['whole-tests', 'whole-prepare', 'whole-run-gpt', 'whole-run-gpt', 'whole-extract-gpt',
                                 'whole-run-glm', 'whole-extract-glm', 'whole-combine'])
        gate = 'afterok:whole-tests:whole-prepare'
        shards, extract, combine = jobs[3], jobs[4], jobs[-1]
        self.assertEqual(self.option(jobs[2], '--dependency'), gate)
        self.assertEqual(self.option(extract, '--dependency'), gate+',afterany:whole-run-gpt:whole-run-gpt')
        self.assertEqual(self.option(combine, '--dependency'), gate+',afterany:whole-extract-gpt:whole-extract-glm')
        self.assertEqual(shards[-6:], ['--release', root, '--model', 'gpt', '--shard', '1/2'])
        self.assertIn('--gres=gpu:2', shards)
        self.assertEqual(shards[shards.index('-t')+1], '240')  # a 2-GPU shard of 4 hours
        self.assertEqual(combine[-4:], ['steering.whole', 'combine', '--release', root])
        self.assertIn('--export=ALL,MOE_EXP_SCORING_PROCESSES=8', extract)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_relink_release_runs_only_the_tests_and_the_combine(self):
        jobs, _ = self.launch('RELINK.json', {'gpt': 0})
        self.assertEqual([command[command.index('-J')+1] for command in jobs], ['whole-tests', 'whole-combine'])
        self.assertEqual(self.option(jobs[1], '--dependency'), 'afterok:whole-tests')

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_an_outcome_release_ends_in_its_own_analysis(self):
        jobs, root = self.launch('OUTCOME.json', {'gpt': 1})
        self.assertEqual(jobs[-1][-4:], ['steering.outcome', 'analyze', '--release', root])


class DoseResponseTests(unittest.TestCase):
    # Eight families per benchmark, so no bootstrap draw of a benchmark misses every family a model ran.
    families = [(str(i), benchmark) for i, benchmark in enumerate(BENCHMARKS*8)]

    def cells(self, values: dict, model='gpt', families=None):
        """Efficiency cells whose target-minus-random benefit is fixed per dose."""
        rows = []
        for dose, value in values.items():
            for family, _ in families or self.families:
                row = {'model': model, 'dose': dose, 'family': family, 'efficiency_directed': True,
                       'nomination_status': 'nominated'}
                for metric in ('tokens', 'accuracy'):
                    for suffix in ('', ':lower', ':upper'):
                        row['target_random:'+metric+suffix] = value
                rows.append(row)
        return pd.DataFrame(rows)

    def bootstrap(self):
        return Bootstrap([f for f, _ in self.families], dict(self.families), draws=200, seed=1)

    def test_slope_on_log2_strength_with_point_identified_bounds(self):
        cells = pd.concat([self.cells({'main': 3., 'half': 2., 'quarter': 1.}),
                           self.cells({'main': 1., 'half': 1., 'quarter': 1.}, model='qwen36')])
        result = whole_effects.dose_response(self.bootstrap(), cells)
        contrasts = result['contrasts']
        self.assertAlmostEqual(contrasts['gpt:slope:tokens']['estimate'], 1.)
        self.assertAlmostEqual(contrasts['qwen36:slope:tokens']['estimate'], 0.)
        self.assertAlmostEqual(contrasts['pooled:slope:tokens']['estimate'], .5)
        self.assertAlmostEqual(contrasts['gpt@half:tokens']['estimate'], 2.)
        self.assertTrue(contrasts['gpt:slope:tokens']['point_identified'])
        self.assertEqual(result['slope_models'], ['gpt', 'qwen36'])

    def test_a_model_missing_a_strength_has_no_slope(self):
        cells = pd.concat([self.cells({'main': 3., 'half': 2., 'quarter': 1.}),
                           self.cells({'half': 1., 'quarter': 1.}, model='glm')])
        result = whole_effects.dose_response(self.bootstrap(), cells)
        self.assertEqual(result['slope_models'], ['gpt'])
        self.assertNotIn('glm:slope:tokens', result['contrasts'])
        self.assertEqual(result['models_by_dose']['main'], ['gpt'])

    def test_a_model_with_fewer_families_is_summarized_over_its_own(self):
        lfm_families = self.families[:-1]
        cells = pd.concat([self.cells({'main': 1.}), self.cells({'main': 2.}, model='lfm25', families=lfm_families)])
        frame = whole_effects.primary_frame(cells, dose='main')
        result = whole_effects.summarize_present(self.bootstrap(), frame, 'test')
        self.assertIn(result['lfm25:tokens']['status'], ('qualified', 'qualified_bounds'))
        self.assertEqual(result['lfm25:tokens']['families'], len(lfm_families))
        self.assertAlmostEqual(result['lfm25:tokens']['estimate'], 2.)
        self.assertEqual(result['pooled:tokens']['families'], len(self.families))

    def test_sweep_doses_pair_with_the_base_release_natives(self):
        table = DesignAndCoverageTests().frame()
        steered = table[(table.model == 'gpt') & (table.operation != 'native')]
        quarter = steered[[whole.efficiency_directed('gpt', op, c, LINK) for op, c in zip(steered.operation, steered['class'])]].copy()
        quarter['dose'] = 'quarter'
        for suffix in ('', ':lower', ':upper'):
            quarter.loc[quarter.arm.str.startswith('target_'), 'log_reasoning'+suffix] = 1.
        cells, noise = whole_effects.family_contrasts(pd.concat([table, quarter], ignore_index=True), LINK)
        self.assertEqual(len(noise), len(whole.MODELS))
        with self.assertRaisesRegex(ValueError, 'several doses'):
            whole_effects.primary_frame(cells)
        frame = whole_effects.primary_frame(cells, 'target_native', 'quarter').iloc[0]
        self.assertEqual(frame['gpt:tokens'], 1.)
        self.assertEqual(set(cells[cells.dose == 'quarter'].model), {'gpt'})

    def test_topup_outcomes_join_their_base_manifest_by_uid(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            base, topup = Path(temporary)/'base', Path(temporary)/'topup'
            rows = [{'uid': uid, 'family': uid, 'model': 'gpt', 'dose': 'half', 'total_allowance': 100,
                     'nomination_status': 'nominated'} for uid in ('early', 'late')]
            planned = publish(whole.manifest_path('gpt', 'half', base), {'records': rows})
            publish(whole.manifest_path('gpt', 'half', topup), {'records': rows[1:],
                                                                'parent_manifest_sha256': planned['sha256']})
            for root, uid in ((base, 'early'), (topup, 'late')):
                (root/'gpt/extraction').mkdir(parents=True)
                pd.DataFrame([{'uid': uid, 'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5,
                               'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                    root/'gpt/extraction/outcomes.parquet')
            source = {'model': 'gpt', 'root': str(base), 'dose': 'half', 'topups': [str(topup)]}
            table = whole_effects.load_table([source]).set_index('uid')
            self.assertEqual(set(table.outcome_status), {'observed'})
            # A row observed in both releases would be run twice: refused.
            pd.DataFrame([{'uid': 'early', 'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5,
                           'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                topup/'gpt/extraction/outcomes.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate whole extracted UID'):
                whole_effects.load_table([source])

    def test_combine_reads_base_and_sweep_releases(self):
        allocation = [{'family': family, 'benchmark': benchmark} for family, benchmark in self.families]
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            base, sweep = Path(temporary)/'base', Path(temporary)/'sweep'
            for root, dose, natives in ((base, 'main', True), (sweep, 'quarter', False)):
                rows = []
                for family in allocation:
                    row = {**family, 'model': 'gpt', 'dose': dose, 'fold': 0, 'total_allowance': 1000,
                           'nomination_status': 'nominated'}
                    for arm in (('native', 'native_repeat') if natives else ()):
                        rows.append({**row, 'uid': dose+family['family']+arm, 'arm': arm, 'class': None,
                                     'operation': 'native'})
                    for operation, classes in (('promote', whole.directions('gpt', LINK)['promote']),
                                               ('suppress', whole.directions('gpt', LINK)['suppress'])):
                        for cls in classes:
                            for role in ('target', 'random'):
                                rows.append({**row, 'uid': dose+family['family']+operation+cls+role, 'class': cls,
                                             'arm': role+'_'+operation, 'operation': operation})
                publish(whole.manifest_path('gpt', dose, root), {'model': 'gpt', 'dose': dose, 'records': rows})
                publish(root/'RELEASE.json', {'classifier': {}})
            sources = publish(sweep/'SOURCES.json', {'sources': [
                {'model': 'gpt', 'root': str(base), 'dose': 'main'},
                {'model': 'gpt', 'root': str(sweep), 'dose': 'quarter'}]})
            original_load = whole_effects.load
            with patch.object(whole_effects, 'load', side_effect=lambda path: {'records': allocation}
                              if Path(path).name == 'ALLOCATION.json' else original_load(path)):
                result = whole_effects.combine(sweep)
            self.assertTrue((sweep/'analysis-combined/EFFECTS_WHOLE.json').is_file())
        self.assertEqual(set(result['primary']), {'gpt:tokens', 'gpt:accuracy', 'pooled:tokens', 'pooled:accuracy'})
        self.assertIn('gpt@quarter:tokens', result['dose_response']['contrasts'])
        self.assertEqual(result['dose_response']['slope_models'], [])
        self.assertIn('gpt|quarter|promote|Verify', result['cell_effects'])
        self.assertEqual({(level['dose'], level['arm']) for level in result['levels'] if level['arm'] == 'native'},
                         {('main', 'native')})
        self.assertEqual(len(result['sources']), 2)
        self.assertEqual(sources['sources'][1]['dose'], 'quarter')


class RankAndCanaryTests(unittest.TestCase):
    def test_ranks_may_differ_only_in_float_telemetry(self):
        def payload(l1, changed=3., executed_ids=(1, 2)):
            dose = {'0': {'active_positions': 5., 'changed_membership': changed, 'l1': l1, 'finite': 1.}}
            capture = {'0': {'positions': [0], 'executed_ids': [list(executed_ids)]}}
            return {'capture': capture, 'stats': {'dose': dose}, 'closed': True, 'passes': 1,
                    'observed_tokens': 9, 'sha256': str(l1)+str(changed)+str(executed_ids)}
        self.assertTrue(driver.ranks_agree(payload(1.0), payload(1.0+1e-9)))
        self.assertFalse(driver.ranks_agree(payload(1.0), payload(1.01)))
        self.assertFalse(driver.ranks_agree(payload(1.0), payload(1.0, changed=4.)))
        self.assertFalse(driver.ranks_agree(payload(1.0), payload(1.0, executed_ids=(1, 3))))

    def test_canary_rows_keep_production_audit_flags(self):
        records = []
        for index in range(8):
            family = f'f{index}'
            records += [{'family': family, 'benchmark': f'b{index%6}', 'arm': arm, 'operation': 'native', 'class': None,
                         'nomination_status': 'nominated', 'audit_capture': False} for arm in ('native', 'native_repeat')]
            records += [{'family': family, 'benchmark': f'b{index%6}', 'arm': role+'_'+operation, 'operation': operation,
                         'class': name, 'nomination_status': 'nominated', 'audit_capture': role == 'target'}
                        for operation, name in (('promote', 'Verify'), ('suppress', whole.directions('gpt', LINK)['suppress'][0]))
                        for role in ('target', 'random')]
        rows = whole.canary_rows({'model': 'gpt', 'sha256': 'm', 'records': records}, LINK)
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(r['audit_capture'] for r in rows), 16)
        self.assertTrue(all(r['canary'] for r in rows))


if __name__ == '__main__':
    unittest.main()
