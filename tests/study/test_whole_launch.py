"""Launch regressions: actual nomination optima, continuous admission and relative gates."""
import math
from pathlib import Path
import tempfile
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from moe_exp.study import driver, whole, whole_effects
from moe_exp.study.common import BENCHMARKS, REPO, WORK, load, publish
from moe_exp.study.effects import Bootstrap
from moe_exp.final_routing.selection import select_pairs
from test_driver import FakeEngine


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
                        if operation == 'suppress' and nomination['class'] not in whole.directions(model)['suppress']:
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
        ordered = whole.admission_order(list(reversed(rows)))
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


class MonitorSurvivalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(whole, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_own_deadline_neither_writes_stop_nor_cancels_jobs(self):
        with patch.object(driver, 'deadline_from_slurm', return_value=time.time()-1), \
             patch.object(whole, 'cancel_jobs') as cancel:
            whole.monitor()
        self.assertFalse((self.root/'STOP.json').exists())
        cancel.assert_not_called()

    def test_transient_accounting_or_storage_exception_retries_without_stop(self):
        driver.checkpoint(self.root/'JOBS.json', {})
        for error in (RuntimeError('sacct transient'), RuntimeError('lfs quota transient')):
            now = time.time()
            with patch.object(driver, 'deadline_from_slurm', return_value=now+1), \
                 patch.object(whole, 'all_jobs', return_value=[]), \
                 patch.object(whole, 'accounting_snapshot', side_effect=error), \
                 patch.object(whole.time, 'sleep', side_effect=lambda seconds: None) as sleep, \
                 patch.object(whole.time, 'time', side_effect=[now, now+2]), \
                 patch.object(whole, 'cancel_jobs') as cancel:
                whole.monitor()
            sleep.assert_called_once_with(60)
            cancel.assert_not_called()
            self.assertFalse((self.root/'STOP.json').exists())

    def test_transient_scontrol_readback_never_duplicates_sbatch(self):
        publish(self.root/'RELEASE.json', {'release_root': str(REPO)})
        minutes = whole.CANARY_MINUTES['gpt']
        readback = ('UserId=lmolfett(1) Account=iscrc_miosr Partition=boost_usr_prod '
                    f'NumCPUs=16 TRES=cpu=16,gres/gpu=2 TimeLimit={minutes//60:02d}:{minutes%60:02d}:00 ')
        responses = [SimpleNamespace(stdout='12345;leonardo\n'),
                     subprocess.CalledProcessError(1, ['scontrol']), SimpleNamespace(stdout=readback)]
        with patch.object(whole.subprocess, 'run', side_effect=responses) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                whole.submit_stage('canary', model='gpt', dose='main')
            accepted = whole.submit_stage('canary', model='gpt', dose='main')
        self.assertEqual(accepted['id'], '12345')
        self.assertEqual(sum(call.args[0][0] == 'sbatch' for call in run.call_args_list), 1)

    def test_cpu_budget_exhaustion_never_starts_another_cpu_worker_or_stops_gpus(self):
        publish(self.root/'RELEASE.json', {'prior_usage': {'new_cpu_core_hours': 16.}})
        driver.checkpoint(self.root/'JOBS.json', {})
        with patch.object(whole, 'all_jobs', return_value=[]), patch.object(whole, 'native_accounting', return_value={}), \
             patch.object(whole.subprocess, 'Popen') as worker:
            with self.assertRaisesRegex(ValueError, 'CPU allowance exhausted'):
                whole.cpu_stage('analyze', None)
        worker.assert_not_called()
        self.assertFalse((self.root/'STOP.json').exists())

    def test_all_model_allocation_maxima_fit_the_release_ceiling(self):
        # Every canary (and one half-dose fallback each) plus every production shard at its maximum walltime.
        reserved = sum(2*(2*whole.CANARY_MINUTES[m]+whole.SHARDS[m]*whole.RUN_MINUTES[m])/60 for m in whole.PLAN)
        self.assertLess(reserved, 720.)

    def test_purged_dependency_requires_native_success_before_afterok_is_satisfied(self):
        gone = SimpleNamespace(returncode=1, stderr='slurm_load_jobs error: Invalid job id specified')
        with patch.object(whole.subprocess, 'run', return_value=gone), \
             patch.object(whole, 'native_accounting', return_value={'old': {'state': 'COMPLETED', 'exit_code': '0:0'}}):
            self.assertIsNone(whole.scheduler_dependency('afterok:old'))
        with patch.object(whole.subprocess, 'run', return_value=gone), \
             patch.object(whole, 'native_accounting', return_value={'failed': {'state': 'FAILED', 'exit_code': '1:0'}}):
            with self.assertRaisesRegex(ValueError, 'required native'):
                whole.scheduler_dependency('afterok:failed')
            self.assertIsNone(whole.scheduler_dependency('afterany:failed'))


class PairedPlanningTests(unittest.TestCase):
    def test_planning_noise_uses_repeat_differences_and_driver_accuracy(self):
        rows = []
        for family, native, repeat, correct in (('a', 1, 3, True), ('b', 7, 7, False), ('c', 9, 4, True)):
            for arm, tokens, accuracy in (('native', native, False), ('native_repeat', repeat, correct)):
                rows.append({'family': family, 'arm': arm, 'reasoning_tokens': tokens,
                             'reasoning_end': tokens, 'finish_reason': 'stop', 'strict_correct': accuracy})
        difference = whole_effects.native_repeat_differences(pd.DataFrame(rows))
        expected = np.array([math.log(4/2), 0., math.log(5/10)])
        np.testing.assert_allclose(difference.log_reasoning, expected)
        self.assertAlmostEqual(difference.log_reasoning.std()/math.sqrt(6), expected.std(ddof=1)/math.sqrt(6))
        self.assertEqual(difference.correct.tolist(), [1., 0., 1.])


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
                for cls in (whole.CLASSES if operation == 'promote' else whole.directions(model)['suppress']):
                    for role in ('target', 'random'):
                        records.append({**base, 'arm': role+'_'+operation, 'class': cls, 'operation': operation})
        return pd.DataFrame(records)

    def test_dropped_promote_cells_are_excluded_and_primary_is_unchanged(self):
        table = self.frame()
        baseline, _ = whole_effects.family_contrasts(table)
        drop = (table.model == 'gpt') & (table['class'] == 'Read') & (table.operation == 'promote')
        table.loc[drop, 'nomination_status'] = 'dropped_availability'
        table.loc[drop, 'log_reasoning:lower'] = 0.
        table.loc[drop, 'log_reasoning:upper'] = 20.
        cells, _ = whole_effects.family_contrasts(table)
        self.assertFalse(((cells.model == 'gpt') & (cells['class'] == 'Read') & (cells.operation == 'promote')).any())
        pd.testing.assert_frame_equal(whole_effects.primary_frame(cells), whole_effects.primary_frame(baseline))

    def test_an_unnominated_efficiency_cell_is_a_design_error(self):
        table = self.frame()
        table.loc[table['class'] == 'Verify', 'nomination_status'] = 'dropped_availability'
        with self.assertRaisesRegex(ValueError, 'efficiency-directed'):
            whole_effects.family_contrasts(table)

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
            with patch('moe_exp.study.lexical.predict', side_effect=lambda frame, *args, **kwargs: frame):
                table = whole_effects.load_table([{'model': 'gpt', 'root': str(root), 'dose': 'main'}]).set_index('uid')
        self.assertEqual(table.loc['valid', 'outcome_status'], 'observed')
        self.assertEqual(table.loc['invalid', 'outcome_status'], 'unknown_sentence_coverage')
        self.assertEqual(table.loc['invalid', 'accuracy:lower'], 0.)
        self.assertEqual(table.loc['invalid', 'accuracy:upper'], 1.)
        self.assertAlmostEqual(table.loc['invalid', 'log_reasoning:upper'], math.log1p(100))


class SweepLaunchTests(unittest.TestCase):
    def test_sweep_release_queues_production_after_validation_without_a_canary(self):
        submitted = []

        def submit(command, *, model=None, dose=None, shard=None, dependency=None, held=False):
            job = {'id': str(len(submitted)+1), 'command': command, 'model': model, 'dose': dose, 'shard': shard,
                   'dependency': dependency, 'gpus': 2 if command == 'run' else 0, 'cpus': 1, 'minutes': 240,
                   'submitted': time.time()}
            submitted.append(job)
            return job
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            root, base = Path(temporary)/'sweep', Path(temporary)/'base'
            publish(root/'RELEASE.json', {'release_root': str(REPO)})
            driver.checkpoint(base/'JOBS.json', {'extraction': {'gpt': {'id': '900'}}})
            publish(root/'SOURCES.json', {'sources': [{'model': 'gpt', 'root': str(base), 'dose': 'half'},
                                                      {'model': 'gpt', 'root': str(root), 'dose': 'quarter'}]})
            with patch.object(whole, 'ROOT', root), patch.object(whole, 'MODELS', ('gpt',)), \
                 patch.object(whole, 'SWEEP', {'gpt': 'quarter'}), patch.object(whole, 'SHARDS', {'gpt': 2}), \
                 patch.object(whole, 'GPU_CEILING', 100.), patch.object(whole, 'verify_release'), \
                 patch.object(whole.driver, 'log'), patch.object(whole, 'submit_stage', side_effect=submit):
                whole.main(['launch'])
            schedule = load(root/'JOBS.json')
            plan = load(root/'gpt/SHARD_PLAN.json')
        self.assertEqual([job['command'] for job in submitted], ['validate', 'run', 'run', 'extract', 'combine', 'monitor'])
        self.assertEqual(schedule['canaries'], {'gpt': []})
        self.assertTrue(all(job['dependency'] == 'afterok:1' and job['dose'] == 'quarter'
                            for job in submitted if job['command'] == 'run'))
        self.assertEqual(submitted[3]['dependency'], 'afterany:2:3')
        # The combined analysis waits for this release's and the base release's extractions.
        self.assertEqual(submitted[4]['dependency'], 'afterany:4:900')
        self.assertEqual(schedule['analysis']['command'], 'combine')
        self.assertEqual(plan['minutes'], whole.RUN_MINUTES['gpt'])

    def test_combined_analysis_allocation_and_release_settings(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            root = Path(temporary)
            publish(root/'RELEASE.json', {'release_root': str(REPO)})
            readback = ('UserId=lmolfett(1) Account=iscrc_miosr Partition=boost_usr_prod '
                        'NumCPUs=8 TimeLimit=06:00:00 ')
            answers = [SimpleNamespace(stdout='123\n'), SimpleNamespace(stdout=readback)]
            with patch.object(whole, 'ROOT', root), patch.object(whole, 'SWEEP', {'gpt': 'quarter'}), \
                 patch.object(whole.subprocess, 'run', side_effect=answers) as run:
                job = whole.submit_stage('combine')
            environment = run.call_args_list[0].kwargs['env']
        self.assertIn('--mem=96G', job['args'])
        self.assertIn('--time=360', job['args'])
        self.assertEqual(environment['WHOLE_SWEEP'], 'gpt:quarter')
        self.assertEqual(float(environment['WHOLE_CUTOFF']), whole.CUTOFF)


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
        quarter = steered[[whole.efficiency_directed('gpt', op, c) for op, c in zip(steered.operation, steered['class'])]].copy()
        quarter['dose'] = 'quarter'
        for suffix in ('', ':lower', ':upper'):
            quarter.loc[quarter.arm.str.startswith('target_'), 'log_reasoning'+suffix] = 1.
        cells, noise = whole_effects.family_contrasts(pd.concat([table, quarter], ignore_index=True))
        self.assertEqual(len(noise), len(whole.MODELS))
        with self.assertRaisesRegex(ValueError, 'several doses'):
            whole_effects.primary_frame(cells)
        frame = whole_effects.primary_frame(cells, 'target_native', 'quarter').iloc[0]
        self.assertEqual(frame['gpt:tokens'], 1.)
        self.assertEqual(set(cells[cells.dose == 'quarter'].model), {'gpt'})

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
                    for operation, classes in (('promote', whole.directions('gpt')['promote']),
                                               ('suppress', whole.directions('gpt')['suppress'])):
                        for cls in classes:
                            for role in ('target', 'random'):
                                rows.append({**row, 'uid': dose+family['family']+operation+cls+role, 'class': cls,
                                             'arm': role+'_'+operation, 'operation': operation})
                publish(whole.manifest_path('gpt', dose, root), {'model': 'gpt', 'dose': dose, 'records': rows})
                publish(root/'RELEASE.json', {'classifier': {}})
            sources = publish(Path(temporary)/'SOURCES.json', {'sources': [
                {'model': 'gpt', 'root': str(base), 'dose': 'main'},
                {'model': 'gpt', 'root': str(sweep), 'dose': 'quarter'}]})
            original_load = whole_effects.load
            with patch.object(whole_effects, 'ROOT', sweep), \
                 patch.object(whole_effects, 'load', side_effect=lambda path: {'records': allocation}
                              if Path(path).name == 'ALLOCATION.json' else original_load(path)):
                result = whole_effects.combine(Path(temporary)/'SOURCES.json')
            self.assertTrue((sweep/'analysis-combined/EFFECTS_WHOLE.json').is_file())
        self.assertEqual(set(result['primary']), {'gpt:tokens', 'gpt:accuracy', 'pooled:tokens', 'pooled:accuracy'})
        self.assertIn('gpt@quarter:tokens', result['dose_response']['contrasts'])
        self.assertEqual(result['dose_response']['slope_models'], [])
        self.assertIn('gpt|quarter|promote|Verify', result['cell_effects'])
        self.assertEqual({(level['dose'], level['arm']) for level in result['levels'] if level['arm'] == 'native'},
                         {('main', 'native')})
        self.assertEqual(len(result['sources']), 2)
        self.assertEqual(sources['sources'][1]['dose'], 'quarter')


class NestedSubmissionAndRankTests(unittest.TestCase):
    def test_stage_environment_drops_caller_slurm_state_and_keeps_stage_exports(self):
        caller = {'SLURM_JOB_GPUS': '0,1', 'SLURM_JOB_ID': '1', 'SBATCH_ACCOUNT': 'x', 'SRUN_CPUS_PER_TASK': '16',
                  'MOE_EXP_SCORING_PROCESSES': '2', 'PATH': '/bin'}
        with patch.dict('os.environ', caller, clear=True):
            env = whole.submission_environment({'MOE_EXP_SCORING_PROCESSES': '8', 'WHOLE_ROOT': '/w'})
        self.assertFalse([k for k in env if k.startswith(('SLURM_', 'SBATCH_', 'SRUN_', 'SALLOC_'))])
        self.assertEqual((env['MOE_EXP_SCORING_PROCESSES'], env['WHOLE_ROOT'], env['PATH']), ('8', '/w', '/bin'))

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
                        for operation, name in (('promote', 'Verify'), ('suppress', whole.directions('gpt')['suppress'][0]))
                        for role in ('target', 'random')]
        rows = whole.canary_rows({'model': 'gpt', 'sha256': 'm', 'records': records})
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(r['audit_capture'] for r in rows), 16)
        self.assertTrue(all(r['canary'] for r in rows))


if __name__ == '__main__':
    unittest.main()
