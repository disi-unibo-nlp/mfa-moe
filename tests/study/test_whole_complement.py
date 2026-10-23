"""Complement releases: the suppress cells a base release left out, joined to its rows by family.

Also their follow-up: a continuation release that runs the rows the complement never started and
continues the ones it left partial, sealed from a synthetic complement with finished, partial, output
and never-started rows.
"""
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from moe_exp.study import driver, whole, whole_complement, whole_effects
from moe_exp.study.common import CLASSES, MODELS, WORK, digest, file_sha, load, publish

TMP = WORK/'tools/tmp'
SUPPRESSED = ('Plan', 'Implement')  # classes the synthetic base release already suppresses


def native(family: str) -> dict:
    return {'model': 'gemma', 'family': family, 'benchmark': 'aime', 'fold': 0, 'seed': 7, 'total_allowance': 100,
            'arm': 'native', 'class': None, 'operation': 'native', 'actions': [], 'nomination_status': 'nominated',
            'audit_capture': False, 'uid': 'native-'+family}


def cell(nomination, profile, model, dose, operation):
    """A fake cell_actions result; Monitor has no available experts."""
    return {'fold': nomination['fold'], 'class': nomination['class'], 'operation': operation, 'dose': dose,
            'qualified': nomination['class'] != 'Monitor', 'target': [{'layer': 0}], 'random': [{'layer': 1}],
            'mean_native_frequency': {'target': .2, 'random': .3}}


class ComplementManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        self.parent, self.root = Path(self.directory.name)/'parent', Path(self.directory.name)/'child'
        self.nominations = publish(self.parent/'gemma/WHOLE_NOMINATIONS.json', {
            'records': [{'fold': 0, 'class': c} for c in CLASSES]})
        self.natives = [native('f1'), native('f2')]
        steered = [{**native('f1'), 'arm': 'target_suppress', 'class': 'Plan', 'operation': 'suppress',
                    'uid': 'steered'}]
        self.base = publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'scope': 'prompt-and-generated-reasoning',
            'nominations_sha256': self.nominations['sha256'], 'cells': [], 'records': self.natives+steered})
        self.patches = [patch.object(whole, 'ROOT', self.root), patch.object(whole, 'SWEEP', {'gemma': 'main'}),
                        patch.object(whole, 'directions', return_value={'suppress': SUPPRESSED}),
                        patch.object(whole, 'cell_actions', side_effect=cell),
                        patch.object(driver, 'binding', return_value={'models': {'gemma': {}}})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_only_the_missing_suppress_cells_are_planned_on_the_native_rows(self):
        result = whole.complement_manifest('gemma', {'parent': str(self.parent)})
        value = load(whole.manifest_path('gemma', 'main'))
        classes = [c for c in CLASSES if c not in SUPPRESSED]
        self.assertEqual(value['complement_classes'], classes)
        self.assertEqual(result['planned'], 2*2*len(classes))
        self.assertEqual({(r['family'], r['class'], r['arm']) for r in value['records']},
                         {(f, c, a) for f in ('f1', 'f2') for c in classes
                          for a in ('target_suppress', 'random_suppress')})
        self.assertTrue(all(r['seed'] == 7 and r['total_allowance'] == 100 and r['complement']
                            and not r['audit_capture'] for r in value['records']))
        self.assertEqual(len({r['uid'] for r in value['records']}), result['planned'])
        dropped = {r['class'] for r in value['records'] if r['nomination_status'] != 'nominated'}
        self.assertEqual(dropped, {'Monitor'})
        self.assertEqual(value['parent_manifest_sha256'], self.base['sha256'])
        self.assertTrue(load(self.root/'gemma/SELECTED.json')['complement'])

    def test_nominations_of_another_manifest_are_refused(self):
        other = Path(self.directory.name)/'other'
        publish(other/'gemma/WHOLE_NOMINATIONS.json', {'records': []})
        publish(whole.manifest_path('gemma', 'main', other), {**self.base, 'sha256': None})
        with self.assertRaisesRegex(ValueError, 'parent nominations differ'):
            whole.complement_manifest('gemma', {'parent': str(other)})

    def test_prepare_dispatches_a_complement_release(self):
        publish(self.root/'COMPLEMENT.json', {'models': {'gemma': {'parent': str(self.parent)}}})
        result = whole.prepare_model({}, 'gemma', True)
        self.assertEqual(result['gemma:main']['planned'], 20)

    def test_the_combined_table_joins_complement_rows_and_refuses_a_foreign_complement(self):
        whole.complement_manifest('gemma', {'parent': str(self.parent)})
        source = {'model': 'gemma', 'dose': 'main', 'root': str(self.parent), 'complements': [str(self.root)]}
        table = whole_effects._source_table(source)
        self.assertEqual(len(table), 3+20)
        self.assertEqual(int(table.complement.eq(True).sum()), 20)
        foreign = Path(self.directory.name)/'foreign'
        publish(whole.manifest_path('gemma', 'main', foreign), {'design': 'complement', 'records': [],
                                                                'parent_manifest_sha256': 'other'})
        with self.assertRaisesRegex(ValueError, 'complement belongs to another manifest'):
            whole_effects._source_table({**source, 'complements': [str(foreign)]})


class ContinuationTests(unittest.TestCase):
    """A synthetic complement release (rows: finished, partial, output-only, never started, not nominated)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.base_release, self.complement, self.child = base/'base', base/'complement', base/'child'
        nominations = publish(self.base_release/'gemma/WHOLE_NOMINATIONS.json', {
            'records': [{'fold': 0, 'class': c} for c in CLASSES]})
        self.base = publish(whole.manifest_path('gemma', 'main', self.base_release), {
            'model': 'gemma', 'dose': 'main', 'scope': 'prompt-and-generated-reasoning',
            'nominations_sha256': nominations['sha256'], 'cells': [], 'records': [native('f1'), native('f2')]})
        self.patches = [patch.object(whole, 'ROOT', self.child), patch.object(whole, 'SWEEP', {'gemma': 'main'}),
                        patch.object(whole, 'directions', return_value={'suppress': SUPPRESSED}),
                        patch.object(whole, 'cell_actions', side_effect=cell),
                        patch.object(driver, 'binding', return_value={'models': {'gemma': {}}})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(whole, 'ROOT', self.complement):
            whole.complement_manifest('gemma', {'parent': str(self.base_release)})
        self.complement_manifest = load(whole.manifest_path('gemma', 'main', self.complement))
        self.nominated = [r for r in self.complement_manifest['records'] if r['nomination_status'] == 'nominated']
        self.store = driver.Store(self.complement/'stores/gemma/main/units')
        # Row 0 finished, row 1 crashed after its output, rows 2 and 3 are partial, the rest never started.
        self.finished, self.crashed, self.partial_rows = self.nominated[0], self.nominated[1], self.nominated[2:4]
        publish(self.store.folder(self.finished['uid'])/'RESULT.json', {'status': 'complete'})
        publish(self.store.folder(self.crashed['uid'])/'OUTPUT.json', {'token_ids': [1]})
        for index, row in enumerate(self.partial_rows):
            self.save_partial(row, [index, 1, 2])
        self.never_started = self.nominated[4:]

    def save_partial(self, row, tokens, **changes):
        driver.checkpoint(self.store.folder(row['uid'])/'PARTIAL.json', {
            'uid': row['uid'], 'assignment_sha256': digest(row), 'tokens': tokens,
            'segments': [{'job': '1', 'history': 0, 'seed': 7}], 'total_allowance': 100, **changes})

    def continuation(self):
        whole.continuation_manifest('gemma', {'parent': str(self.complement)})
        return load(whole.manifest_path('gemma', 'main'))

    def test_a_continuation_holds_the_partial_and_never_started_rows_only(self):
        value = self.continuation()
        self.assertEqual(value['records'], self.partial_rows+self.never_started)
        self.assertEqual(value['resume_sources'], {r['uid']: str(self.complement) for r in self.partial_rows})
        self.assertEqual(value['parent_states'], {'complete': 1, 'output': 1, 'partial': 2,
                                                  'never_started': len(self.never_started), 'not_nominated': 4})
        self.assertEqual((value['design'], value['parent_root'], value['parent_manifest_sha256']),
                         ('continuation', str(self.complement), self.complement_manifest['sha256']))
        self.assertEqual((value['planned'], value['executable']), (len(value['records']),)*2)
        # Cells and classes of the complement survive; the parent's counts and hash are replaced.
        self.assertEqual(value['complement_classes'], self.complement_manifest['complement_classes'])
        self.assertNotEqual(value['sha256'], self.complement_manifest['sha256'])
        selected = load(self.child/'gemma/SELECTED.json')
        self.assertTrue(selected['sweep'] and selected['continuation'] and selected['canary_job'] is None)
        self.assertEqual(selected['manifest_sha256'], value['sha256'])

    def test_the_parent_store_and_a_foreign_partial_are_guarded(self):
        self.save_partial(self.partial_rows[0], [9], assignment_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'another assignment'):
            self.continuation()
        for row in self.nominated:
            publish(self.store.folder(row['uid'])/'RESULT.json', {'status': 'complete'})
        with self.assertRaisesRegex(ValueError, 'no unfinished rows'):
            self.continuation()

    def test_prepare_dispatches_a_continuation_and_never_two_designs(self):
        publish(self.child/'CONTINUATION.json', {'models': {'gemma': {'parent': str(self.complement)}}})
        result = whole.prepare_model({}, 'gemma', True)
        self.assertEqual(result['gemma:main']['planned'], 2+len(self.never_started))
        publish(self.child/'TOPUP.json', {'models': {}})
        with self.assertRaisesRegex(ValueError, 'either a top-up'):
            whole.prepare_model({}, 'gemma', True)

    def test_seeding_copies_the_partials_and_leaves_never_started_rows_fresh(self):
        value = self.continuation()
        child_store = driver.Store(self.child/'stores/gemma/main/units')
        whole.check_saved_state(value, value['records'], child_store)
        receipt = whole.seed_resume('gemma', value, value['records'], child_store, (0, 1))
        self.assertEqual([r['uid'] for r in receipt['rows']], [r['uid'] for r in self.partial_rows])
        self.assertEqual(child_store.state(self.partial_rows[1]['uid'])['tokens'], [1, 1, 2])
        self.assertTrue(all(child_store.state(r['uid']) is None for r in self.never_started))
        # A restarted shard passes the gate again; a partial of another assignment does not.
        whole.check_saved_state(value, value['records'], child_store)
        driver.checkpoint(child_store.folder(self.never_started[0]['uid'])/'PARTIAL.json', {
            'uid': self.never_started[0]['uid'], 'assignment_sha256': '0'*64, 'tokens': [5]})
        with self.assertRaisesRegex(ValueError, 'fresh only'):
            whole.check_saved_state(value, value['records'], child_store)
        # A plain resume manifest still needs a partial for every row.
        with self.assertRaisesRegex(ValueError, 'no saved partial'):
            whole.seed_resume('gemma', {**value, 'design': 'resume'}, value['records'], child_store, (0, 1))

    def test_validation_binds_the_store_before_any_shard_can_race_for_it(self):
        self.continuation()
        publish(self.child/'RELEASE.json', {'release_root': str(whole.REPO)})
        units = self.child/'stores/gemma/main/units'
        self.assertFalse(units.exists())
        whole.bind_sweep_stores()
        self.assertEqual(sorted(p.name for p in units.iterdir()), ['EXECUTION.json'])
        # Shards then find a bound store, and a second validation run changes nothing.
        before = load(units/'EXECUTION.json')
        self.assertEqual(whole.bound_store('gemma', 'main').root, units)
        whole.bind_sweep_stores()
        self.assertEqual(load(units/'EXECUTION.json'), before)

    def test_a_model_whose_preparation_failed_is_not_bound(self):
        publish(self.child/'RELEASE.json', {'release_root': str(whole.REPO)})
        whole.bind_sweep_stores()
        self.assertFalse((self.child/'stores').exists())

    def test_a_shard_starting_after_the_cutoff_exits_cleanly_without_seeding(self):
        value = self.continuation()
        child_store = driver.Store(self.child/'stores/gemma/main/units')
        publish(self.child/'CPU_VALIDATION.json', {'passed': True})
        events = []
        environment = {'SLURM_JOB_ID': '1', 'SLURM_JOB_GPUS': '0,1'}
        with patch.dict(os.environ, environment), patch.object(whole, 'MODELS', ('gemma',)), \
             patch.object(whole, 'CUTOFF', 1.), patch.object(whole, 'verify_release'), \
             patch.object(whole, 'bound_store', return_value=child_store), \
             patch.object(driver, 'deadline_from_slurm', return_value=0.), \
             patch.object(driver, 'log', side_effect=events.append):
            # main returns None, so the job ends with exit code 0 and its dependents are not blocked.
            self.assertIsNone(whole.main(['run', '--model', 'gemma', '--shard', '0/2']))
        self.assertEqual([e['event'] for e in events], ['whole-shard-after-cutoff'])
        self.assertEqual(value['design'], 'continuation')
        self.assertTrue(all(child_store.state(r['uid']) is None for r in value['records']))

    def test_the_combined_table_joins_the_follow_up_to_the_complement_by_uid(self):
        value = self.continuation()
        observed = {'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5, 'strict_correct': True,
                    'finish_reason': 'stop'}
        for root, uids in ((self.base_release, ['native-f1', 'native-f2']), (self.complement, [self.finished['uid']]),
                           (self.child, [self.partial_rows[0]['uid'], self.never_started[0]['uid']])):
            (root/'gemma/extraction').mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{'uid': uid, **observed} for uid in uids]).to_parquet(root/'gemma/extraction/outcomes.parquet')
        source = {'model': 'gemma', 'root': str(self.base_release), 'dose': 'main',
                  'complements': [str(self.complement), str(self.child)]}
        table = whole_effects._source_table(source)
        # The follow-up adds outcomes, never rows: 2 natives plus the 20 complement rows.
        self.assertEqual(len(table), 2+len(self.complement_manifest['records']))
        self.assertEqual(table.uid.is_unique, True)
        status = table.set_index('uid').outcome_status
        self.assertEqual(status[self.partial_rows[0]['uid']], 'observed')
        self.assertEqual(status[self.never_started[0]['uid']], 'observed')
        self.assertEqual(status[self.partial_rows[1]['uid']], 'missing')
        self.assertEqual(status[self.crashed['uid']], 'missing')
        self.assertEqual(int((status == 'not_in_design').sum()), 4)
        self.assertEqual(table.set_index('uid').release[self.never_started[0]['uid']], self.base_release.name)
        # A follow-up whose parent is not listed before it, or a row run by both releases, is refused.
        with self.assertRaisesRegex(ValueError, 'continuation belongs to another manifest'):
            whole_effects._source_table({**source, 'complements': [str(self.child)]})
        pd.DataFrame([{'uid': self.finished['uid'], **observed}]).to_parquet(self.child/'gemma/extraction/outcomes.parquet')
        with self.assertRaisesRegex(ValueError, 'duplicate whole extracted UID'):
            whole_effects._source_table(source)

    def test_the_combine_waits_for_the_follow_up_extraction_and_every_earlier_one(self):
        source = {'model': 'gemma', 'root': str(self.base_release), 'dose': 'main',
                  'complements': [str(self.complement), str(self.child)]}
        publish(self.child/'SOURCES.json', {'sources': [source]})
        driver.checkpoint(self.base_release/'JOBS.json', {'extraction': {'gemma': {'id': '3'}}})
        driver.checkpoint(self.complement/'JOBS.json', {'extraction': {'gemma': {'id': '1'}}})
        self.assertEqual(whole.source_extractions({'extraction': {'gemma': {'id': '2'}}}), ['1', '2', '3'])


class StoreBindingTests(unittest.TestCase):
    def test_shards_starting_together_bind_one_store_without_a_nonempty_failure(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temporary:
            root = Path(temporary)
            publish(root/'RELEASE.json', {'release_root': str(whole.REPO)})
            inside, proceed, results = threading.Event(), threading.Event(), []

            def slow_publish(path, value):
                # The first shard has created the folder and holds its pending receipt when the second starts.
                if not inside.is_set():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    (path.parent/'.pending-x').write_text('{}')
                    inside.set()
                    proceed.wait(30)
                    (path.parent/'.pending-x').unlink()
                return publish(path, value)

            def bind():
                try:
                    results.append(whole.bound_store('gemma', 'main').root)
                except ValueError as error:
                    results.append(error)

            with patch.object(whole, 'ROOT', root), patch.object(whole, 'manifest', return_value={
                    'sha256': 'manifest'}), patch.object(whole, 'publish', side_effect=slow_publish):
                first = threading.Thread(target=bind)
                first.start()
                self.assertTrue(inside.wait(30))
                second = threading.Thread(target=bind)
                second.start()
                second.join(.5)
                self.assertTrue(second.is_alive(), 'the second shard must wait for the first to bind')
                proceed.set()
                first.join(10)
                second.join(10)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, Path) for r in results), results)


class FollowupPlanTests(unittest.TestCase):
    """The sealer's read-only plan and its fail-fast checks, on a synthetic complement release."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.directory.cleanup)
        self.reports = Path(self.directory.name)/'report'
        self.parent = self.reports/'parent'
        rows = [{**native(f'f{i}'), 'arm': arm, 'class': 'Verify', 'operation': 'suppress',
                 'uid': digest(['row', i, arm]), 'nomination_status': 'nominated'}
                for i in range(4) for arm in ('target_suppress', 'random_suppress')]
        self.rows = rows
        manifest = publish(whole.manifest_path('gemma', 'main', self.parent), {
            'model': 'gemma', 'dose': 'main', 'records': rows, 'planned': 8, 'executable': 8})
        self.input_file = self.reports.parent/'input.json'
        self.input_file.write_text('{}')
        self.release = publish(self.parent/'RELEASE.json', {
            'models': ['gemma'], 'sweep': {'gemma': 'main'}, 'shards': {'gemma': 4}, 'complement_sha256': 'x',
            'cutoff': 1000., 'release_root': '/old/snapshot', 'inputs': {str(self.input_file): file_sha(self.input_file)},
            'classifier': {}, 'nomination_reference': 'n', 'template_release_sha256': 't'})
        self.cutoff = time.time()+86400.
        store = driver.Store(self.parent/'stores/gemma/main/units')
        publish(store.folder(rows[0]['uid'])/'RESULT.json', {'status': 'complete'})
        driver.checkpoint(store.folder(rows[1]['uid'])/'PARTIAL.json', {'uid': rows[1]['uid']})
        publish(store.folder(rows[2]['uid'])/'OUTPUT.json', {'token_ids': []})
        # Shard i takes family f<i>: shard 0 finished, shard 1 failed after starting f1, shards 2 and 3
        # (f2, f3) ended without touching a row.
        self.parent_jobs = [{'id': str(i), 'model': 'gemma', 'shard': [i, 4], 'state': state}
                            for i, state in enumerate(('COMPLETED', 'FAILED', 'FAILED', 'CANCELLED'))]
        self.patches = [patch.object(whole_complement, 'REPORTS', self.reports),
                        patch.object(whole_complement, 'parent_shard_jobs', return_value=self.parent_jobs)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.manifest = manifest

    def test_rows_by_state_and_shards_follow_the_parent_capacity(self):
        plan = whole_complement.plan_followup('parent')
        value = plan['models']['gemma']
        self.assertEqual({k: value[k] for k in ('never_started', 'partial', 'complete', 'output', 'unfinished')},
                         {'never_started': 5, 'partial': 1, 'complete': 1, 'output': 1, 'unfinished': 6})
        # The parent ran 8 rows in 4 shards: 2 rows per shard, so 6 rows need 3 shards (4 families cover them).
        self.assertEqual((value['rows_per_shard'], value['shards_needed'], plan['shards']), (2., 3, {'gemma': 3}))
        self.assertEqual((plan['gpu_ceiling'], plan['rows_beyond_shard_capacity']), (34., 0))
        self.assertEqual(value['parent_shards'], {'jobs': 4, 'failed': 3, 'failed_before_start': 2,
                                                   'states': {'COMPLETED': 1, 'FAILED': 2, 'CANCELLED': 1}})
        self.assertEqual(plan['live_parent_jobs'], [])
        table = whole_complement.format_plan(plan)
        self.assertIn('gemma', table)
        self.assertIn('3 (  2) of   4', table)

    def test_jobs_that_are_not_terminal_are_live(self):
        jobs = [{**job, 'state': state} for job, state in zip(self.parent_jobs, ('RUNNING', 'PENDING', 'UNKNOWN', 'TIMEOUT'))]
        with patch.object(whole_complement, 'parent_shard_jobs', return_value=jobs):
            plan = whole_complement.plan_followup('parent')
        self.assertEqual(plan['live_parent_jobs'], ['0', '1', '2'])

    def test_the_gpu_limit_scales_shards_down_and_names_the_rows_left_over(self):
        plan = whole_complement.plan_followup('parent', max_gpu_hours=18.)
        self.assertEqual((plan['shards'], plan['gpu_ceiling'], plan['rows_beyond_shard_capacity']),
                         ({'gemma': 1}, 18., 4))
        with self.assertRaisesRegex(ValueError, 'exceed'):
            whole_complement.plan_followup('parent', max_gpu_hours=18., shards={'gemma': 2})
        with self.assertRaisesRegex(ValueError, 'exactly the models'):
            whole_complement.plan_followup('parent', shards={'glm': 2})

    def test_only_a_complement_with_unfinished_rows_has_a_follow_up(self):
        publish(self.reports/'base/RELEASE.json', {k: v for k, v in self.release.items()
                                                   if k not in ('sha256', 'complement_sha256')})
        with self.assertRaisesRegex(ValueError, 'not a complement release'):
            whole_complement.plan_followup('base')
        store = driver.Store(self.parent/'stores/gemma/main/units')
        for row in self.rows:
            folder = store.folder(row['uid'])
            for name in ('OUTPUT.json', 'PARTIAL.json'):
                (folder/name).unlink(missing_ok=True)
            publish(folder/'RESULT.json', {'status': 'complete'})
        with self.assertRaisesRegex(ValueError, 'no unfinished rows'):
            whole_complement.plan_followup('parent')

    def test_fit_shards_keeps_one_per_model_and_never_exceeds_the_limit(self):
        needed = {'a': 27, 'b': 12, 'c': 3, 'd': 1}
        self.assertEqual(whole_complement.fit_shards(needed, 43), needed)
        for limit in (4, 5, 12, 30, 42):
            fitted = whole_complement.fit_shards(needed, limit)
            self.assertEqual(sum(fitted.values()), limit)
            self.assertTrue(all(1 <= fitted[m] <= needed[m] for m in needed), fitted)
        with self.assertRaisesRegex(ValueError, 'allows 3 shards for 4 models'):
            whole_complement.fit_shards(needed, 3)

    def test_the_cutoff_is_read_in_rome_time_unless_it_names_a_zone(self):
        self.assertEqual(whole_complement.parse_cutoff('2026-10-24T06:00'), whole_complement.CUTOFF)
        self.assertEqual(whole_complement.parse_cutoff('2026-10-24T04:00+00:00'), whole_complement.CUTOFF)
        self.assertEqual(whole_complement.parse_cutoff('1792814400'), whole_complement.CUTOFF)

    def seal_environment(self):
        """Everything followup() reads outside the report folder, replaced by small stand-ins."""
        publish(self.reports.parent/'LINK.json', {'models': list(MODELS)})
        chain = [{'model': 'gemma', 'root': str(self.reports/'base'), 'dose': 'main', 'topups': [],
                  'complements': [str(self.parent)]},
                 {'model': 'glm', 'root': str(self.reports/'base'), 'dose': 'half', 'complements': [str(self.parent)]}]
        publish(self.reports/'chain/SOURCES.json', {'sources': chain})
        snapshot = self.reports.parent/'snapshot'
        (snapshot/'src').mkdir(parents=True)
        (snapshot/'src/code.py').write_text('x = 1\n')

        def command(arguments, **kwargs):
            answers = {'status': '', 'rev-parse': 'abc123\n'}
            return SimpleNamespace(stdout=answers.get(arguments[1], ''))

        def materialize(root, commit):
            (root/'SOURCE.tar.gz').write_text('archive')
            return root/'SOURCE.tar.gz', snapshot
        patches = [patch.object(whole_complement.subprocess, 'run', side_effect=command),
                   patch.object(whole_complement, 'materialize', side_effect=materialize),
                   patch.object(whole_complement, 'whole_usage', return_value={'gpu_hours': 1.})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_a_follow_up_is_sealed_with_the_chain_the_link_and_the_parent_manifests(self):
        self.seal_environment()
        result = whole_complement.followup('child', 12, 'parent', self.cutoff, chain='chain',
                                           link=self.reports.parent/'LINK.json')
        root = self.reports/'child'
        release = load(root/'RELEASE.json')
        self.assertEqual((result['shards'], release['cutoff'], release['shards'], release['sweep']),
                         ({'gemma': 3}, self.cutoff, {'gemma': 3}, {'gemma': 'main'}))
        self.assertEqual(release['parent_release_sha256'], load(self.parent/'RELEASE.json')['sha256'])
        self.assertEqual(release['continuation_sha256'], load(root/'CONTINUATION.json')['sha256'])
        self.assertIn(str(self.reports.parent/'LINK.json'), release['inputs'])
        self.assertIn(str(self.input_file), release['inputs'])
        self.assertIn(str(whole.manifest_path('gemma', 'main', self.parent)), release['inputs'])
        self.assertEqual(load(root/'CONTINUATION.json')['models'], {'gemma': {'parent': str(self.parent)}})
        sources = load(root/'SOURCES.json')['sources']
        self.assertEqual([s['complements'] for s in sources if s['model'] == 'gemma'],
                         [[str(self.parent), str(root)]])
        self.assertEqual(next(s for s in sources if s['model'] == 'glm')['complements'], [str(self.parent)])
        launch = dict(line.split('=', 1) for line in (root/'launch.env').read_text().splitlines())
        self.assertEqual((launch['MODELS'], launch['SHARDS'], launch['CUTOFF'], launch['LINK'], launch['COMMIT']),
                         ('gemma', 'gemma:3', str(int(self.cutoff)), str(self.reports.parent/'LINK.json'), 'abc123'))
        self.assertEqual(float(launch['GPU_CEILING']), 34.)
        # The chain release itself is untouched.
        self.assertEqual(load(self.reports/'chain/SOURCES.json')['sources'][0]['complements'], [str(self.parent)])

    def test_a_follow_up_waits_for_the_parent_and_for_a_later_cutoff_and_leaves_nothing_behind(self):
        self.seal_environment()
        arguments = ('child', 12, 'parent')
        options = {'chain': 'chain', 'link': self.reports.parent/'LINK.json'}
        live = [{'id': '61927000', 'model': 'gemma', 'shard': [3, 4], 'state': 'RUNNING'}]
        with patch.object(whole_complement, 'parent_shard_jobs', return_value=live):
            with self.assertRaisesRegex(ValueError, 'not ended'):
                whole_complement.followup(*arguments, self.cutoff, **options)
        with self.assertRaisesRegex(ValueError, 'cutoff must be later'):
            whole_complement.followup(*arguments, time.time()-5, **options)
        self.assertFalse((self.reports/'child').exists())


if __name__ == '__main__':
    unittest.main()
