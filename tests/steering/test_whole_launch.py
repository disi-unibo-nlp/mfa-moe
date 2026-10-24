"""Launch checks: actual nomination optima, continuous admission, relative gates and the job graph."""
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from steering import pulsed_steering as pulse, whole_reasoning as whole, whole_reasoning_effects
from store import BENCHMARKS, REPO, WORK, load, publish
from steering.pulsed_effects import Bootstrap
from steering.selection import select_pairs
from test_driver import FakeEngine

LINK = pulse.DRIVER/'EFFICIENCY_LINK_PREREG.json'  # the seven-model pre-registration


class ActualNominationTests(unittest.TestCase):
    def test_greedy_matches_all_70_dp_optima_and_preserves_other_exact_rank_cells(self):
        """Use saved discovery evidence; neither nominations nor tests read whole outcomes."""
        saved = pulse.STUDY.parent/'whole-reasoning-20261020-r2'
        proposal = load(saved/'CONTROL_FEASIBLE_RD_BOTH_DOSES.json')
        references = {(r['dose'], r['fold'], r['class']): r for r in proposal['records']}
        for model, layers, experts, native_k in (('gpt', 24, 32, 4), ('qwen36', 40, 256, 8)):
            path = saved/'gpt/WHOLE_NOMINATIONS.json' if model == 'gpt' else saved/'qwen-feasibility/qwen36/WHOLE_NOMINATIONS.json'
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
            engine = pulse.Engine(SimpleNamespace(llm_engine=fake), pulse.Store(Path(temporary)), time.time()+30,
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

    def test_long_and_repetitive_native_does_not_fail_equal_steering(self):
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
        baseline, _ = whole_reasoning_effects.family_contrasts(table, LINK)
        drop = (table.model == 'gpt') & (table['class'] == 'Read') & (table.operation == 'promote')
        table.loc[drop, 'nomination_status'] = 'dropped_availability'
        table.loc[drop, 'log_reasoning:lower'] = 0.
        table.loc[drop, 'log_reasoning:upper'] = 20.
        cells, _ = whole_reasoning_effects.family_contrasts(table, LINK)
        self.assertFalse(((cells.model == 'gpt') & (cells['class'] == 'Read') & (cells.operation == 'promote')).any())
        pd.testing.assert_frame_equal(whole_reasoning_effects.primary_frame(cells), whole_reasoning_effects.primary_frame(baseline))

    def test_an_unnominated_efficiency_cell_is_a_design_error(self):
        table = self.frame()
        table.loc[table['class'] == 'Verify', 'nomination_status'] = 'dropped_availability'
        with self.assertRaisesRegex(ValueError, 'efficiency-directed'):
            whole_reasoning_effects.family_contrasts(table, LINK)

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
            planned = publish(whole.manifest_path('gpt', 'main', root), {'records': records})
            publish(folder/'EXTRACTION.json', {'manifest_sha256': planned['sha256']})
            with patch('routing.lexical.predict', side_effect=lambda frame, *args, **kwargs: frame):
                table = whole_reasoning_effects.load_table([{'model': 'gpt', 'root': str(root), 'dose': 'main'}]).set_index('uid')
        self.assertEqual(table.loc['valid', 'outcome_status'], 'observed')
        self.assertEqual(table.loc['invalid', 'outcome_status'], 'unknown_sentence_coverage')
        self.assertEqual(table.loc['invalid', 'accuracy:lower'], 0.)
        self.assertEqual(table.loc['invalid', 'accuracy:upper'], 1.)
        self.assertAlmostEqual(table.loc['invalid', 'log_reasoning:upper'], math.log1p(100))


class LaunchGraphTests(unittest.TestCase):
    """scripts/steering/submit_release_jobs.sh on a made-up sealed release, with Slurm commands that only keep a record."""
    PY = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python')
    FAKE = {  # job numbers count up from 1001; FAIL_AT and HANG_AT name the sbatch call that fails or hangs
        'sbatch': 'n=$(( $(cat "$FAKE_SLURM/count" 2>/dev/null || echo 0) + 1 )); echo $n > "$FAKE_SLURM/count"\n'
                  'test "$n" != "${FAIL_AT:-}" || { echo "sbatch: error: invalid account" >&2; exit 1; }\n'
                  'test "$n" != "${HANG_AT:-}" || exec sleep 30\n'
                  'echo $((1000 + n))\n',
        'scancel': 'echo "$@" >> "$FAKE_SLURM/cancelled"\n',
        'sacct': 'echo "${JOB_STATE:-RUNNING}"\n',
    }

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        for folder in ('src', 'scripts'):
            os.symlink(REPO/folder, self.repo/folder)
        (self.repo/'bin').mkdir()
        for name, body in self.FAKE.items():
            (self.repo/'bin'/name).write_text('#!/bin/bash\n'+body)
            (self.repo/'bin'/name).chmod(0o755)
        self.root = self.repo/'report/release'
        self.study = self.repo/'study'  # the launcher wants a study folder and hands it to the jobs
        publish(self.study/'BINDING.json', {})

    def seal(self, kind_file: str, shards: dict, cpu_ceiling=1000., gpu_ceiling=1000., **release):
        publish(self.root/'RELEASE.json', {'release_root': str(self.repo), 'models': list(shards), 'shards': shards,
                                           'cpu_core_hour_ceiling': cpu_ceiling, 'new_gpu_hour_ceiling': gpu_ceiling,
                                           **release})
        publish(self.root/kind_file, {})

    def environment(self, **extra):
        return {**os.environ, 'PATH': f"{self.repo/'bin'}:{os.environ['PATH']}", 'FAKE_SLURM': str(self.repo),
                'STUDY_DATA_ROOT': str(self.study), **extra}

    def launch(self, *jobs, **extra):
        return subprocess.run(['bash', str(self.repo/'scripts/steering/submit_release_jobs.sh'), str(self.root), *jobs],
                              cwd=self.repo, env=self.environment(**extra), capture_output=True, text=True)

    def dry_run(self, kind_file: str, shards: dict, *jobs, **release) -> list[list[str]]:
        """The sbatch commands the launcher prints for a release."""
        self.seal(kind_file, shards, **release)
        run = self.launch(*jobs, DRY_RUN='1')
        self.assertEqual(run.returncode, 0, run.stderr)
        return [line.split() for line in run.stderr.splitlines() if line.startswith('sbatch')]

    @staticmethod
    def option(command, name):
        return next(word.split('=', 1)[1] for word in command if word.startswith(name+'='))

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_sweep_release_runs_tests_then_manifests_then_shards_then_one_extraction_per_model(self):
        jobs, root = self.dry_run('COMPLEMENT.json', {'gpt': 2, 'glm': 1}), str(self.root)
        names = [command[command.index('-J')+1] for command in jobs]
        self.assertEqual(names, ['whole-tests', 'whole-prepare', 'whole-run-gpt', 'whole-run-gpt', 'whole-extract-gpt',
                                 'whole-run-glm', 'whole-extract-glm', 'whole-combine', 'whole-export'])
        tests, prepare, first_shard, last_shard, extract, combine, export = (
            jobs[0], jobs[1], jobs[2], jobs[3], jobs[4], jobs[7], jobs[8])
        self.assertEqual(self.option(prepare, '--dependency'), 'afterok:whole-tests')  # the manifests wait for the tests
        gate = 'afterok:whole-prepare'
        self.assertEqual(self.option(first_shard, '--dependency'), gate)
        self.assertEqual(self.option(extract, '--dependency'), gate+',afterany:whole-run-gpt:whole-run-gpt')
        self.assertEqual(self.option(combine, '--dependency'), gate+',afterany:whole-extract-gpt:whole-extract-glm')
        self.assertEqual(self.option(export, '--dependency'), 'afterok:whole-combine')
        self.assertEqual(last_shard[-6:], ['--release', root, '--model', 'gpt', '--shard', '1/2'])
        self.assertEqual(last_shard[-8:-6], ['steering.whole_reasoning', 'run'])
        self.assertTrue(last_shard[-9].endswith('scripts/slurm/run_module_gpu.sbatch'))
        for cpu_job in (tests, prepare, extract, combine):  # the serial partition would reject them
            self.assertEqual(cpu_job[cpu_job.index('-p')+1], 'boost_usr_prod')
        self.assertEqual(tests[-4:], ['steering.whole_reasoning', 'validate', '--release', root])
        self.assertEqual(combine[-4:], ['steering.whole_reasoning', 'combine', '--release', root])
        self.assertIn('--export=ALL,MOE_EXP_SCORING_PROCESSES=8', extract)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_every_job_states_its_cpus_and_time_and_a_shard_its_two_gpus(self):
        jobs = {command[command.index('-J')+1]: command for command in self.dry_run('COMPLEMENT.json', {'gpt': 1})}
        for name, cpus, minutes in (('whole-tests', '4', '60'), ('whole-prepare', '4', '60'), ('whole-extract-gpt', '8', '90'),
                                    ('whole-combine', '8', '120'), ('whole-export', '2', '60'), ('whole-run-gpt', '16', '240')):
            command = jobs[name]
            self.assertEqual((command[command.index('-c')+1], command[command.index('-t')+1]), (cpus, minutes), name)
        self.assertIn('--gres=gpu:2', jobs['whole-run-gpt'])
        self.assertLess(jobs['whole-run-gpt'].index('--gres=gpu:2'), jobs['whole-run-gpt'].index('steering.whole_reasoning'))
        self.assertFalse(any('--gres' in command for name, command in jobs.items() if name != 'whole-run-gpt'))

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_the_export_runs_from_the_checkout_that_holds_report(self):
        jobs = {command[command.index('-J')+1]: command for command in self.dry_run('COMPLEMENT.json', {'gpt': 1})}
        export = jobs['whole-export']
        self.assertEqual(self.option(export, '--chdir'), str(self.repo))
        self.assertEqual(export[export.index('paper.export_whole_results')-1], str(self.repo/'scripts/slurm/run_module_cpu.sbatch'))
        self.assertIn(str(self.root/'analysis-combined/EFFECTS_WHOLE.json'), export)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_relink_release_runs_only_the_tests_the_combine_and_the_export(self):
        jobs = self.dry_run('RELINK.json', {'gpt': 0}, cpu_ceiling=22., gpu_ceiling=0.)  # the ceilings sealed for a relink
        self.assertEqual([command[command.index('-J')+1] for command in jobs],
                         ['whole-tests', 'whole-combine', 'whole-export'])
        self.assertEqual(self.option(jobs[1], '--dependency'), 'afterok:whole-tests')

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_relink_waits_for_the_jobs_named_after_the_release(self):
        self.seal('RELINK.json', {'gpt': 0})
        run = self.launch('4242', DRY_RUN='1', JOB_STATE='RUNNING')
        self.assertEqual(run.returncode, 0, run.stderr)
        combine = next(line.split() for line in run.stderr.splitlines() if '-J whole-combine' in line)
        self.assertEqual(self.option(combine, '--dependency'), 'afterok:whole-tests,afterok:4242')
        run = self.launch('4242', DRY_RUN='1', JOB_STATE='FAILED')  # a failed job is no reason to start
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('did not complete', run.stderr)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_an_outcome_release_ends_in_its_own_analysis(self):
        jobs = self.dry_run('OUTCOME.json', {'gpt': 1})
        self.assertEqual(jobs[-1][-4:], ['steering.outcome', 'analyze', '--release', str(self.root)])
        self.assertEqual((jobs[-1][jobs[-1].index('-c')+1], jobs[-1][jobs[-1].index('-t')+1]), ('8', '120'))
        self.assertNotIn('whole-export', [command[command.index('-J')+1] for command in jobs])

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_capture_release_ends_with_its_extractions_and_names_its_jobs_oc(self):
        jobs = self.dry_run('CAPTURE_NATIVE.json', {'gpt': 2, 'glm': 1}, job_prefix='oc')
        self.assertEqual([command[command.index('-J')+1] for command in jobs],
                         ['oc-tests', 'oc-prepare', 'oc-run-gpt', 'oc-run-gpt', 'oc-extract-gpt', 'oc-run-glm',
                          'oc-extract-glm'])

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_an_outcome_sweep_names_its_jobs_os_and_ends_in_its_analysis(self):
        jobs = self.dry_run('OUTCOME_SWEEP.json', {'gpt': 1}, job_prefix='os')
        self.assertEqual([command[command.index('-J')+1] for command in jobs],
                         ['os-tests', 'os-prepare', 'os-run-gpt', 'os-extract-gpt', 'os-analyze'])
        self.assertEqual(jobs[-1][-4:], ['steering.outcome', 'analyze', '--release', str(self.root)])

    def derive(self, *models):
        return subprocess.run(['bash', str(self.repo/'scripts/steering/derive_outcome_sets.sh'), str(self.root), *models],
                              cwd=self.repo, env=self.environment(DRY_RUN='1'), capture_output=True, text=True)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_the_derive_chain_waits_for_the_extraction_of_each_model_it_is_given(self):
        self.seal('CAPTURE_NATIVE.json', {'gpt': 1, 'glm': 1}, job_prefix='oc')
        (self.root/'JOBS.txt').write_text('oc-tests 11 x\noc-extract-gpt 21 x\noc-extract-glm 22 x\n')
        run = self.derive('glm', 'openai/gpt-oss-20b')  # a short key and a Hugging Face id
        self.assertEqual(run.returncode, 0, run.stderr)
        glm, gpt = (line.split() for line in run.stderr.splitlines() if line.startswith('sbatch'))
        self.assertEqual((glm[glm.index('-J')+1], self.option(glm, '--dependency')), ('oc-derive-glm', 'afterok:22'))
        self.assertEqual((gpt[gpt.index('-J')+1], self.option(gpt, '--dependency')), ('oc-derive-gpt', 'afterok:21'))
        self.assertEqual(gpt[-6:], ['--release', str(self.root), '--models', 'gpt', '--workers', '4'])
        self.assertEqual(self.option(gpt, '--chdir'), str(self.repo))  # the snapshot of the release
        self.assertIn('--kill-on-invalid-dep=yes', gpt)
        self.assertIn(f'--export=ALL,STUDY_DATA_ROOT={self.study}', gpt)
        missing = self.derive('qwen35')
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('no queued extraction of qwen35', missing.stderr)
        unqueued = self.derive('zai-org/GLM-4.7-Flash', 'Qwen/Qwen3-30B-A3B')  # in the study, but not in this release
        self.assertIn('no queued extraction of Qwen/Qwen3-30B-A3B', unqueued.stderr)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_nothing_is_submitted_when_the_time_limits_exceed_a_ceiling(self):
        # Complement of two models, three shards: 4+4+24+16+2 = 50 CPU core-hours and 3 x 2 GPUs x 4 hours = 24 GPU-hours.
        self.dry_run('COMPLEMENT.json', {'gpt': 2, 'glm': 1}, cpu_ceiling=50., gpu_ceiling=24.)
        for ceilings in ({'cpu_core_hour_ceiling': 49.9, 'new_gpu_hour_ceiling': 24.},
                         {'cpu_core_hour_ceiling': 50., 'new_gpu_hour_ceiling': 23.9}):
            shutil.rmtree(self.root)
            publish(self.root/'RELEASE.json', {'release_root': str(self.repo), 'models': ['gpt', 'glm'],
                                               'shards': {'gpt': 2, 'glm': 1}, **ceilings})
            publish(self.root/'COMPLEMENT.json', {})
            run = self.launch()
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('the release allows', run.stderr)
            self.assertFalse((self.repo/'count').exists(), 'sbatch was called')
            self.assertFalse((self.root/'JOBS.txt').exists())

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_a_failed_launch_cancels_its_jobs_keeps_their_record_and_can_be_repeated(self):
        self.seal('RELINK.json', {'gpt': 0})
        run = self.launch(FAIL_AT='3')  # the export is refused
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual((self.repo/'cancelled').read_text().split(), ['1001', '1002'])
        self.assertFalse((self.root/'JOBS.txt').exists())
        (record,) = self.root.glob('JOBS.cancelled-*.txt')
        self.assertEqual([line.split()[:2] for line in record.read_text().splitlines()],
                         [['whole-tests', '1001'], ['whole-combine', '1002']])
        self.assertEqual(self.launch().returncode, 0)  # a new launch is allowed
        self.assertEqual(len((self.root/'JOBS.txt').read_text().splitlines()), 3)
        self.assertIn('launched already', self.launch().stderr)
        self.assertEqual(len(list(self.root.glob('JOBS.cancelled-*.txt'))), 1)

    @unittest.skipUnless(PY.exists(), 'needs the cluster environment')
    def test_an_interrupted_launch_cancels_its_jobs_too(self):
        self.seal('RELINK.json', {'gpt': 0})
        process = subprocess.Popen(['bash', str(self.repo/'scripts/steering/submit_release_jobs.sh'), str(self.root)], cwd=self.repo,
                                   env=self.environment(HANG_AT='3'), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)  # its own process group, as a terminal gives it
        for _ in range(100):  # the third sbatch call is under way: two jobs are queued
            if (self.repo/'count').exists() and (self.repo/'count').read_text().strip() == '3':
                break
            time.sleep(.2)
        os.killpg(process.pid, signal.SIGTERM)
        _, error = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0, error)
        self.assertEqual((self.repo/'cancelled').read_text().split(), ['1001', '1002'])
        self.assertFalse((self.root/'JOBS.txt').exists())
        self.assertEqual(len(list(self.root.glob('JOBS.cancelled-*.txt'))), 1)


class ReleaseStageTests(unittest.TestCase):
    """The checks around a release's stages: the unit-test receipt, the ceilings and the extractions."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def suites(self, returncode=0, text='Ran 5 tests in 0.1s\n\nOK\n'):
        """Stand-ins for the three unittest runs of validate."""
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=returncode, stderr=text)
        self.enterContext(patch.object(whole.subprocess, 'run', run))
        self.enterContext(patch.object(whole, 'verify_release', return_value={'commit': 'abc123'}))
        self.enterContext(patch.dict(os.environ, {'SLURM_JOB_ID': '77'}))
        return calls

    def test_validate_seals_the_tests_run_and_keeps_that_receipt_on_a_second_launch(self):
        calls = self.suites()
        receipt = whole.validate(self.root)
        self.assertEqual((receipt['passed'], receipt['tests'], receipt['job'], receipt['commit']), (True, 15, '77', 'abc123'))
        self.assertEqual(len(calls), 3)
        whole.validated(self.root)
        self.assertEqual(whole.validate(self.root), receipt)
        self.assertEqual(len(calls), 3)  # the same snapshot is not tested twice

    def test_a_failing_suite_is_sealed_as_failed_and_stops_the_stages(self):
        self.suites(returncode=1, text='Ran 5 tests in 0.1s\n\nFAILED (failures=1)\n')
        self.assertFalse(whole.validate(self.root)['passed'])
        with self.assertRaisesRegex(ValueError, 'did not pass'):
            whole.validated(self.root)

    def test_a_suite_that_ran_nothing_does_not_pass(self):
        self.suites(text='ImportError: cannot import name\n')
        self.assertFalse(whole.validate(self.root)['passed'])

    def test_the_stages_need_the_receipt_to_exist(self):
        with self.assertRaises(FileNotFoundError):
            whole.validated(self.root)

    def test_budget_compares_the_committed_hours_with_both_ceilings(self):
        publish(self.root/'RELEASE.json', {'cpu_core_hour_ceiling': 24., 'new_gpu_hour_ceiling': 10.})
        whole.budget(self.root, 24., 10.)
        with self.assertRaisesRegex(ValueError, 'CPU core-hours; the release allows 24.0'):
            whole.budget(self.root, 24.1, 10.)
        with self.assertRaisesRegex(ValueError, 'GPU-hours; the release allows 10.0'):
            whole.budget(self.root, 1., 10.1)

    def test_a_release_whose_model_has_no_extraction_cannot_be_combined(self):
        publish(self.root/'RELEASE.json', {'classifier': {}})
        value = publish(whole.manifest_path('gpt', 'main', self.root), {'records': []})
        with self.assertRaisesRegex(ValueError, 'has no gpt extraction'):
            whole_reasoning_effects._extracted(self.root, 'gpt', value)
        publish(self.root/'gpt/extraction/EXTRACTION.json', {'manifest_sha256': 'another'})
        with self.assertRaisesRegex(ValueError, 'another manifest'):
            whole_reasoning_effects._extracted(self.root, 'gpt', value)


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
        result = whole_reasoning_effects.dose_response(self.bootstrap(), cells)
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
        result = whole_reasoning_effects.dose_response(self.bootstrap(), cells)
        self.assertEqual(result['slope_models'], ['gpt'])
        self.assertNotIn('glm:slope:tokens', result['contrasts'])
        self.assertEqual(result['models_by_dose']['main'], ['gpt'])

    def test_a_model_with_fewer_families_is_summarized_over_its_own(self):
        lfm_families = self.families[:-1]
        cells = pd.concat([self.cells({'main': 1.}), self.cells({'main': 2.}, model='lfm25', families=lfm_families)])
        frame = whole_reasoning_effects.primary_frame(cells, dose='main')
        result = whole_reasoning_effects.summarize_present(self.bootstrap(), frame, 'test')
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
        cells, noise = whole_reasoning_effects.family_contrasts(pd.concat([table, quarter], ignore_index=True), LINK)
        self.assertEqual(len(noise), len(whole.MODELS))
        with self.assertRaisesRegex(ValueError, 'several doses'):
            whole_reasoning_effects.primary_frame(cells)
        frame = whole_reasoning_effects.primary_frame(cells, 'target_native', 'quarter').iloc[0]
        self.assertEqual(frame['gpt:tokens'], 1.)
        self.assertEqual(set(cells[cells.dose == 'quarter'].model), {'gpt'})

    def test_topup_outcomes_join_their_base_manifest_by_uid(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as temporary:
            base, topup = Path(temporary)/'base', Path(temporary)/'topup'
            rows = [{'uid': uid, 'family': uid, 'model': 'gpt', 'dose': 'half', 'total_allowance': 100,
                     'nomination_status': 'nominated'} for uid in ('early', 'late')]
            planned = publish(whole.manifest_path('gpt', 'half', base), {'records': rows})
            follow_up = publish(whole.manifest_path('gpt', 'half', topup), {'records': rows[1:],
                                                                            'parent_manifest_sha256': planned['sha256']})
            for root, uid, manifest in ((base, 'early', planned), (topup, 'late', follow_up)):
                (root/'gpt/extraction').mkdir(parents=True)
                publish(root/'gpt/extraction/EXTRACTION.json', {'manifest_sha256': manifest['sha256']})
                pd.DataFrame([{'uid': uid, 'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5,
                               'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                    root/'gpt/extraction/outcomes.parquet')
            source = {'model': 'gpt', 'root': str(base), 'dose': 'half', 'topups': [str(topup)]}
            table = whole_reasoning_effects.load_table([source]).set_index('uid')
            self.assertEqual(set(table.outcome_status), {'observed'})
            # A row observed in both releases would be run twice: refused.
            pd.DataFrame([{'uid': 'early', 'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5,
                           'strict_correct': True, 'finish_reason': 'stop'}]).to_parquet(
                topup/'gpt/extraction/outcomes.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate whole extracted UID'):
                whole_reasoning_effects.load_table([source])

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
                planned = publish(whole.manifest_path('gpt', dose, root), {'model': 'gpt', 'dose': dose, 'records': rows})
                publish(root/'gpt/extraction/EXTRACTION.json', {'manifest_sha256': planned['sha256'], 'failures': []})
                publish(root/'RELEASE.json', {'classifier': {}})
            sources = publish(sweep/'SOURCES.json', {'sources': [
                {'model': 'gpt', 'root': str(base), 'dose': 'main'},
                {'model': 'gpt', 'root': str(sweep), 'dose': 'quarter'}]})
            original_load = whole_reasoning_effects.load
            with patch.object(whole_reasoning_effects, 'load', side_effect=lambda path: {'records': allocation}
                              if Path(path).name == 'ALLOCATION.json' else original_load(path)):
                result = whole_reasoning_effects.combine(sweep)
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
        self.assertTrue(pulse.ranks_agree(payload(1.0), payload(1.0+1e-9)))
        self.assertFalse(pulse.ranks_agree(payload(1.0), payload(1.01)))
        self.assertFalse(pulse.ranks_agree(payload(1.0), payload(1.0, changed=4.)))
        self.assertFalse(pulse.ranks_agree(payload(1.0), payload(1.0, executed_ids=(1, 3))))

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
