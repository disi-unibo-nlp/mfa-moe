"""Complement releases: the suppress cells a base release left out, joined to its rows by family.

Also their continuation: a release that runs the rows the complement never started and continues the ones it
left partial, sealed from a synthetic complement with finished, partial, output-only and never-started rows.
"""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd

from steering import pulse as driver, whole, whole_effects
from store import CLASSES, WORK, digest, load, publish

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
        publish(self.root/'RELEASE.json', {'models': ['gemma'], 'sweep': {'gemma': 'main'}})
        self.patches = [patch.object(whole, 'directions', return_value={'suppress': SUPPRESSED}),
                        patch.object(whole, 'cell_actions', side_effect=cell),
                        patch.object(driver, 'binding', return_value={'models': {'gemma': {}}})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_only_the_missing_suppress_cells_are_planned_on_the_native_rows(self):
        result = whole.complement_manifest(self.root, 'gemma', {'parent': str(self.parent)})
        value = load(whole.manifest_path('gemma', 'main', self.root))
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
            whole.complement_manifest(self.root, 'gemma', {'parent': str(other)})

    def test_prepare_dispatches_a_complement_release(self):
        publish(self.root/'COMPLEMENT.json', {'models': {'gemma': {'parent': str(self.parent)}}})
        result = whole.prepare_model(self.root, {}, 'gemma')
        self.assertEqual(result['gemma:main']['planned'], 20)

    def test_the_combined_table_joins_complement_rows_and_refuses_a_foreign_complement(self):
        whole.complement_manifest(self.root, 'gemma', {'parent': str(self.parent)})
        for root in (self.parent, self.root):  # both releases extracted their rows (none of them is observed here)
            publish(root/'gemma/extraction/EXTRACTION.json',
                    {'manifest_sha256': load(whole.manifest_path('gemma', 'main', root))['sha256']})
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
        for root in (self.complement, self.child):
            publish(root/'RELEASE.json', {'models': ['gemma'], 'sweep': {'gemma': 'main'}, 'release_root': str(whole.REPO)})
        self.patches = [patch.object(whole, 'directions', return_value={'suppress': SUPPRESSED}),
                        patch.object(whole, 'cell_actions', side_effect=cell),
                        patch.object(driver, 'binding', return_value={'models': {'gemma': {}}})]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        whole.complement_manifest(self.complement, 'gemma', {'parent': str(self.base_release)})
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
        whole.continuation_manifest(self.child, 'gemma', {'parent': str(self.complement)})
        return load(whole.manifest_path('gemma', 'main', self.child))

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
        result = whole.prepare_model(self.child, {}, 'gemma')
        self.assertEqual(result['gemma:main']['planned'], 2+len(self.never_started))
        publish(self.child/'TOPUP.json', {'models': {}})
        with self.assertRaisesRegex(ValueError, 'either a top-up'):
            whole.prepare_model(self.child, {}, 'gemma')

    def test_seeding_copies_the_partials_and_leaves_never_started_rows_fresh(self):
        value = self.continuation()
        child_store = driver.Store(self.child/'stores/gemma/main/units')
        whole.check_saved_state(value, value['records'], child_store)
        receipt = whole.seed_resume(self.child, 'gemma', value, value['records'], child_store, (0, 1))
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
            whole.seed_resume(self.child, 'gemma', {**value, 'design': 'resume'}, value['records'], child_store, (0, 1))

    def test_preparation_binds_the_store_before_any_shard_can_race_for_it(self):
        self.continuation()
        units = self.child/'stores/gemma/main/units'
        self.assertFalse(units.exists())
        whole.bind_sweep_stores(self.child)
        self.assertEqual(sorted(p.name for p in units.iterdir()), ['EXECUTION.json'])
        # Shards then find a bound store, and binding again changes nothing.
        before = load(units/'EXECUTION.json')
        self.assertEqual(whole.bound_store(self.child, 'gemma', 'main').root, units)
        whole.bind_sweep_stores(self.child)
        self.assertEqual(load(units/'EXECUTION.json'), before)

    def test_prepare_binds_every_sweep_store_once_the_manifests_exist(self):
        publish(self.child/'CONTINUATION.json', {'models': {'gemma': {'parent': str(self.complement)}}})
        with patch.object(whole, 'verify_release', return_value={'models': ['gemma']}), \
                patch.object(whole, 'audit_nominations'):
            whole.prepare(self.child)
        self.assertTrue((self.child/'stores/gemma/main/units/EXECUTION.json').exists())
        self.assertTrue((self.child/'PREPARED.json').exists())

    def test_a_model_whose_preparation_failed_is_not_bound(self):
        whole.bind_sweep_stores(self.child)
        self.assertFalse((self.child/'stores').exists())

    def test_a_shard_starting_after_the_cutoff_exits_cleanly_without_seeding(self):
        value = self.continuation()
        child_store = driver.Store(self.child/'stores/gemma/main/units')
        publish(self.child/'PREPARED.json', {})
        events = []
        environment = {'SLURM_JOB_ID': '1', 'SLURM_JOB_GPUS': '0,1'}
        with patch.dict(os.environ, environment), patch.object(whole, 'verify_release', return_value={'cutoff': 1.}), \
             patch.object(whole, 'bound_store', return_value=child_store), \
             patch.object(driver, 'deadline_from_slurm', return_value=0.), \
             patch.object(driver, 'log', side_effect=events.append):
            # main returns None, so the job ends with exit code 0 and its dependents are not blocked.
            self.assertIsNone(whole.main(['run', '--release', str(self.child), '--model', 'gemma', '--shard', '0/2']))
        self.assertEqual([e['event'] for e in events], ['whole-shard-after-cutoff'])
        self.assertEqual(value['design'], 'continuation')
        self.assertTrue(all(child_store.state(r['uid']) is None for r in value['records']))

    def test_the_link_of_a_continuation_comes_from_its_release_file(self):
        self.assertEqual(whole.link_receipt(self.child), driver.DRIVER/'EFFICIENCY_LINK_PREREG.json')
        publish(self.child/'CONTINUATION.json', {'models': {}, 'link': '/some/EFFICIENCY_LINK_V2.json'})
        self.assertEqual(whole.link_receipt(self.child), Path('/some/EFFICIENCY_LINK_V2.json'))

    def test_the_combined_table_joins_the_follow_up_to_the_complement_by_uid(self):
        self.continuation()
        observed = {'outcome_status': 'observed', 'reasoning_tokens': 5, 'reasoning_end': 5, 'strict_correct': True,
                    'finish_reason': 'stop'}
        extracted = {self.base_release: ['native-f1', 'native-f2'], self.complement: [self.finished['uid']],
                     self.child: [self.partial_rows[0]['uid'], self.never_started[0]['uid']]}
        for root, uids in extracted.items():
            (root/'gemma/extraction').mkdir(parents=True, exist_ok=True)
            publish(root/'gemma/extraction/EXTRACTION.json',
                    {'manifest_sha256': load(whole.manifest_path('gemma', 'main', root))['sha256']})
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
                    results.append(whole.bound_store(root, 'gemma', 'main').root)
                except ValueError as error:
                    results.append(error)

            with patch.object(whole, 'manifest', return_value={'sha256': 'manifest'}), \
                    patch.object(whole, 'publish', side_effect=slow_publish):
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


if __name__ == '__main__':
    unittest.main()
