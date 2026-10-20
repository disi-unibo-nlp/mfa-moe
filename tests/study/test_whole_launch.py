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
from moe_exp.study.common import REPO, WORK, load, publish
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
                        if operation == 'suppress' and nomination['class'] not in whole.SUPPRESS[model]:
                            continue
                        cell = whole.cell_actions(nomination, profile, model, dose, operation)
                        targets = {role: {(a['layer'], e) for a in cell['target'] for e in a[role]}
                                   for role in ('activate', 'deactivate')}
                        counts = whole.DOSES[model][dose][operation]
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
            {'family': f, 'arm': arm, 'class': cls, 'operation': operation}
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

    def test_only_relative_cap_length_and_repetition_excess_fail(self):
        self.assertEqual(whole.relative_fluency(self.rows(native_cap=False))['fluency_failures'], ['cap'])
        self.assertEqual(whole.relative_fluency(self.rows(ratio=1.41))['fluency_failures'], ['length'])
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
        readback = ('UserId=lmolfett(1) Account=iscrc_miosr Partition=boost_usr_prod '
                    'NumCPUs=16 TRES=cpu=16,gres/gpu=2 TimeLimit=00:45:00 ')
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

    def test_main_allocation_maxima_reserve_71_point_5_gpu_hours(self):
        reserved = sum(2*(whole.CANARY_MINUTES[m]+whole.SHARDS[m]*whole.RUN_MINUTES[m])/60 for m in whole.MODELS)
        self.assertEqual(reserved, 71.5)
        self.assertLess(reserved, whole.GPU_CEILING)

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
            base = {'model': model, 'family': 'family', 'nomination_status': 'nominated',
                    **{metric+suffix: value for metric, value in (('log_reasoning', 2.), ('accuracy', 1.))
                       for suffix in ('', ':lower', ':upper')},
                    **{'occ_'+cls+suffix: 0. for cls in whole.CLASSES for suffix in ('', ':lower', ':upper')}}
            for arm in ('native', 'native_repeat'):
                records.append({**base, 'arm': arm, 'class': None, 'operation': 'native'})
            for operation in ('promote', 'suppress'):
                for cls in (whole.CLASSES if operation == 'promote' else whole.SUPPRESS[model]):
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
            records = [{'uid': uid, 'family': uid, 'model': 'gpt', 'total_allowance': 100,
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
            with patch.object(whole_effects, 'ROOT', root), patch.object(whole_effects, 'MODELS', ('gpt',)), \
                 patch.object(whole_effects, 'manifest', return_value={'records': records}), \
                 patch('moe_exp.study.lexical.predict', side_effect=lambda frame, *args, **kwargs: frame):
                table = whole_effects.load_table().set_index('uid')
        self.assertEqual(table.loc['valid', 'outcome_status'], 'observed')
        self.assertEqual(table.loc['invalid', 'outcome_status'], 'unknown_sentence_coverage')
        self.assertEqual(table.loc['invalid', 'accuracy:lower'], 0.)
        self.assertEqual(table.loc['invalid', 'accuracy:upper'], 1.)
        self.assertAlmostEqual(table.loc['invalid', 'log_reasoning:upper'], math.log1p(100))


class EngineeringDiagnosticsTests(unittest.TestCase):
    def test_diagnostic_command_never_queues_production(self):
        with patch.object(whole, 'verify_release'), patch.object(whole, 'inference_stage') as stage, \
             patch.object(whole, 'queue_after_canary') as queue, \
             patch.dict('os.environ', {'SLURM_JOB_ID': '123'}):
            whole.main(['diagnose', '--model', 'gpt', '--dose', 'main'])
        self.assertEqual(stage.call_args.args[0].command, 'diagnose')
        queue.assert_not_called()

    def test_diagnostic_allocation_is_two_gpus_fifteen_minutes_and_journaled(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            root = Path(temporary)
            publish(root/'RELEASE.json', {'release_root': str(REPO)})
            readback = ('UserId=lmolfett(1) Account=iscrc_miosr Partition=boost_usr_prod '
                        'NumCPUs=16 gres/gpu=2 TimeLimit=00:15:00 ')
            answers = [SimpleNamespace(stdout='123\n'), SimpleNamespace(stdout=readback)]
            with patch.object(whole, 'ROOT', root), patch.object(whole.subprocess, 'run', side_effect=answers) as run:
                job = whole.submit_stage('diagnose', model='gpt', dose='main')
                repeated = whole.submit_stage('diagnose', model='gpt', dose='main')
            self.assertEqual(job['id'], repeated['id'])
            self.assertEqual(job['minutes']*job['gpus']/60, .5)
            self.assertEqual(run.call_count, 2)
            self.assertIn('--time=15', job['args'])


if __name__ == '__main__':
    unittest.main()
