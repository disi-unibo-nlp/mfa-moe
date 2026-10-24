"""Sealing a release: one mechanism for complement, relink, outcome, capture, sweep and continuation releases."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from steering import release, whole
from steering.outcome import capture, manifest
from store import WORK, file_sha, publish, verified

LINK_PATH = 'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'
ARGS = SimpleNamespace(models=None, shards=None, cutoff=None)


class SealTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.reports, self.snapshots, old = base/'report', base/'releases', base/'old-snapshot'
        (old/'configs').mkdir(parents=True)
        (old/'configs/study.yaml').write_text('seed: 1\n')
        self.link = self.reports/LINK_PATH
        publish(self.link, {'models': list(release.MODELS)})
        self.parent = publish(self.reports/'whole-reasoning-20261021-r18/RELEASE.json', {
            'release_root': str(old), 'cutoff': 1792814400., 'classifier': {'root': 'c'},
            'nomination_reference': 'n.json', 'inputs': {str(old/'configs/study.yaml'): file_sha(old/'configs/study.yaml')}})
        publish(self.reports/'whole-reasoning-20261021-r18/SOURCES.json',
                {'sources': [{'model': 'gpt', 'root': 'r8', 'dose': 'half'}]})
        self.config = 'seed: 1\n'

        def snapshot(root, commit):
            folder = self.snapshots/commit
            (folder/'configs').mkdir(parents=True)
            (folder/'configs/study.yaml').write_text(self.config)
            archive = root/'SOURCE.tar.gz'
            archive.write_bytes(b'archive')
            return archive, folder

        def run(command, **kwargs):
            return SimpleNamespace(stdout='' if command[1] == 'status' else 'abc123\n')
        for patcher in (patch.object(release, 'REPORTS', self.reports), patch.object(release, 'snapshot', snapshot),
                        patch.object(release.subprocess, 'run', run)):
            self.enterContext(patcher)

    def test_a_relink_release_names_its_link_and_runs_no_shards(self):
        folder = release.seal('relink', 'whole-reasoning-test', 12, ARGS)
        root = self.reports/'whole-reasoning-test'
        sealed = verified(root/'RELEASE.json')
        self.assertEqual(folder, self.snapshots/'abc123')
        self.assertEqual((sealed['commit'], sealed['release_root']), ('abc123', str(folder)))
        self.assertEqual((sealed['models'], sealed['shards'], sealed['sweep'], sealed['cutoff']),
                         (list(release.MODELS), {}, {}, 1792814400.))
        self.assertEqual(sealed['parent_release_sha256'], self.parent['sha256'])  # the key a relink names its parent by
        self.assertNotIn('template_release_sha256', sealed)
        # The release records what the earlier jobs cost, since its ceilings do not reset the budget.
        usage = verified(root/'PRIOR_USAGE.json')
        self.assertEqual(sealed['prior_usage'], usage)
        self.assertEqual(usage['ledger_totals'], {'whole-': 'abc123', 'os-': 'abc123', 'oc-': 'abc123'})
        self.assertEqual((sealed['cpu_core_hour_ceiling'], sealed['new_gpu_hour_ceiling']), (24., 0.))
        self.assertEqual(sealed['relink_sha256'], verified(root/'RELINK.json')['sha256'])
        # The parent's pinned input moves to the new snapshot, and the link receipt is pinned too.
        self.assertIn(str(folder/'configs/study.yaml'), sealed['inputs'])
        self.assertEqual(sealed['inputs'][str(self.link)], file_sha(self.link))
        self.assertEqual(verified(root/'SOURCES.json')['sources'][0]['model'], 'gpt')
        # The stages read their settings back from the release folder.
        self.assertEqual(whole.link_receipt(root), self.link)
        self.assertEqual(whole.sweep_doses(root), {})
        with self.assertRaisesRegex(ValueError, 'exists'):
            release.seal('relink', 'whole-reasoning-test', 12, ARGS)

    def test_a_failed_seal_leaves_no_release_behind(self):
        self.config = 'seed: 2\n'  # the snapshot no longer holds the file the parent pinned
        with self.assertRaisesRegex(ValueError, 'bound input changed'):
            release.seal('relink', 'whole-reasoning-bad', 1, ARGS)
        self.assertFalse((self.reports/'whole-reasoning-bad').exists())

    def test_gpu_releases_need_a_cutoff_and_the_shards_of_exactly_their_models(self):
        with self.assertRaisesRegex(ValueError, '--cutoff is required'):
            release.complement(self.reports/'x', ARGS)
        with self.assertRaisesRegex(ValueError, '--cutoff is required'):
            release.outcome(self.reports/'x', ARGS)
        with self.assertRaisesRegex(ValueError, 'exactly the --models'):
            release.outcome(self.reports/'x', SimpleNamespace(models='gemma,glm', shards='glm:12', cutoff='2026-10-24T06:00'))

    def test_the_rome_times_of_the_first_releases(self):
        self.assertEqual(release.rome('2026-10-24T06:00'), 1792814400.)  # complement r18
        self.assertEqual(release.rome('2026-10-23T23:00'), 1792789200.)  # outcome r1

    def test_a_cutoff_names_its_zone_or_is_epoch_seconds(self):
        self.assertEqual(release.rome('2026-10-24T04:00+00:00'), 1792814400.)  # the same moment as 06:00 in Rome
        self.assertEqual(release.rome('1792814400'), 1792814400.)


class OutcomeReleaseTests(unittest.TestCase):
    """The capture release and the two halves of the r2 sweep, on a made-up report folder holding one model."""
    def setUp(self):
        """A report/ folder with the template release, the folds and one model's source table; Git and the snapshot faked."""
        temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.reports, self.snapshots, self.fast, old = base/'report', base/'releases', base/'fast', base/'old-snapshot'
        (old/'configs').mkdir(parents=True)
        (old/'configs/study.yaml').write_text('seed: 1\n')
        publish(self.reports/'whole-reasoning-20261021-r7/RELEASE.json', {
            'release_root': str(old), 'cutoff': 1792814400., 'classifier': {'root': 'c'},
            'nomination_reference': 'n.json', 'inputs': {str(old/'configs/study.yaml'): file_sha(old/'configs/study.yaml')}})
        questions = [f'aime24|a{i}' for i in range(6)]
        families = {q: 'f-'+q for q in questions}
        folds = self.reports/capture.FOLDS
        folds.parent.mkdir(parents=True)
        folds.write_text(json.dumps({'question_families': families, 'folds': {f: 1 for f in families.values()}}))
        table = capture.sources_path(self.reports, 'gpt')
        table.parent.mkdir(parents=True)
        table.write_text(json.dumps({'records': [{'question': q, 'attempt_id': 'a-'+q} for q in questions]}))
        self.config = 'seed: 1\n'
        for patcher in (patch.object(release, 'REPORTS', self.reports), patch.object(release, 'FAST', self.fast),
                        patch.object(release, 'snapshot', self.write_snapshot),
                        patch.object(release.subprocess, 'run', self.fake_run),
                        patch.object(capture, 'PER_BENCHMARK', {'aime24': 4})):
            self.enterContext(patcher)

    def write_snapshot(self, root, commit):
        """Stand in for the git archive: a snapshot folder holding the one file the template pinned."""
        folder = self.snapshots/commit
        (folder/'configs').mkdir(parents=True)
        (folder/'configs/study.yaml').write_text(self.config)
        archive = root/'SOURCE.tar.gz'
        archive.write_bytes(b'archive')
        return archive, folder

    @staticmethod
    def fake_run(command, **kwargs):
        """Stand in for git and the billing ledger: a clean tree and the commit abc123."""
        return SimpleNamespace(stdout='' if command[1] == 'status' else 'abc123\n')

    def capture_args(self, **extra):
        return SimpleNamespace(models='gpt', shards='gpt:2', cutoff='2026-10-30T23:00', half=None, capture=None,
                               pick=None, **extra)

    def test_a_capture_release_links_its_stores_to_fast_names_its_jobs_oc_and_pins_its_inputs(self):
        release.seal('capture', 'outcome-capture-test', 5, self.capture_args())
        root = self.reports/'outcome-capture-test'
        sealed = verified(root/'RELEASE.json')
        spec = verified(root/'CAPTURE_NATIVE.json')['models']['gpt']
        self.assertEqual((sealed['job_prefix'], sealed['stores']), ('oc', str(self.fast/'outcome-capture-test/stores')))
        self.assertEqual((root/'stores').resolve(), (self.fast/'outcome-capture-test/stores').resolve())
        self.assertEqual((spec['attempts'], spec['cap_tokens'], len(spec['questions'])), (6, 32768, 4))
        self.assertEqual((sealed['new_gpu_hour_ceiling'], sealed['cpu_core_hour_ceiling'], sealed['cutoff']),
                         (26., 200., 1793397600.))
        self.assertEqual(sealed['capture_sha256'], verified(root/'CAPTURE_NATIVE.json')['sha256'])
        self.assertEqual(sealed['sweep'], {'gpt': 'main'})
        for pinned in capture.pinned_inputs(self.reports, ['gpt']):
            self.assertEqual(sealed['inputs'][str(pinned)], file_sha(pinned))
        with self.assertRaisesRegex(ValueError, 'exists'):
            release.seal('capture', 'outcome-capture-test', 5, self.capture_args())

    def test_a_failed_capture_seal_leaves_neither_a_release_nor_a_stores_folder(self):
        self.config = 'seed: 2\n'  # the snapshot no longer holds the file the template pinned
        with self.assertRaisesRegex(ValueError, 'bound input changed'):
            release.seal('capture', 'outcome-capture-bad', 5, self.capture_args())
        self.assertFalse((self.reports/'outcome-capture-bad').exists())
        self.assertFalse((self.fast/'outcome-capture-bad').exists() and any((self.fast/'outcome-capture-bad').iterdir()))

    def sweep_args(self, half, pick=None):
        return SimpleNamespace(models='gpt', shards='1', cutoff='2026-10-30T23:00', half=half,
                               capture='outcome-capture-test', pick=pick)

    def seal_capture_results(self):
        """The files of a finished capture release that a sweep reads: its roster, its expert sets and its extraction."""
        root = self.reports/'outcome-capture-test'
        questions = [f'aime24|a{i}' for i in range(6)]
        publish(root/'CAPTURE_NATIVE.json', {'models': {'gpt': {'questions': questions}}})
        publish(root/'gpt/SELECTION_R2.json', {'sets': {
            'Plan': {'sets': {'all': {}}}, 'blind': {'sets': {'all': {}}}, 'Read': {'status': 'too_few_mixed_questions'}}})
        publish(root/'gpt/WHOLE.json', {})
        (root/'gpt/extraction').mkdir()
        (root/'gpt/extraction/outcomes.parquet').write_bytes(b'rows')
        return questions

    def test_half_a_runs_every_available_set_at_every_shift_on_its_half_of_the_questions(self):
        questions = self.seal_capture_results()
        release.seal('sweep', 'outcome-steering-test', 5, self.sweep_args('A'))
        root = self.reports/'outcome-steering-test'
        spec = verified(root/'OUTCOME_SWEEP.json')['models']['gpt']
        self.assertEqual(sorted({c['set'] for c in spec['cells']}), ['Plan', 'blind'])  # Read has no set to nudge
        self.assertEqual(sorted({c['delta'] for c in spec['cells']}), [.4, .8, 1.2])
        self.assertEqual(len(spec['cells']), 6)
        self.assertEqual(spec['questions'], manifest.halves(questions)['A'])
        self.assertEqual((spec['half'], spec['pick']), ('A', None))
        sealed = verified(root/'RELEASE.json')
        self.assertEqual((sealed['job_prefix'], sealed['cpu_core_hour_ceiling'], sealed['new_gpu_hour_ceiling']),
                         ('os', 60., 18.))

    def test_half_b_runs_only_the_cells_the_pick_confirmed(self):
        questions = self.seal_capture_results()
        pick = publish(self.reports/'outcome-steering-test/analysis/PICK.json',
                       {'confirm': [{'set': 'Plan', 'delta': .8}, {'set': 'Read', 'delta': .4}]})
        spec = release.sweep(self.reports/'outcome-steering-b', self.sweep_args('B', self.reports/'outcome-steering-test/analysis/PICK.json'))
        entry = spec['kind_file']['models']['gpt']
        self.assertEqual(entry['cells'], [{'set': 'Plan', 'delta': .8}])
        self.assertEqual(entry['questions'], manifest.halves(questions)['B'])
        self.assertEqual(entry['pick']['sha256'], pick['sha256'])

    def test_a_half_is_a_or_b_and_only_half_b_has_a_pick(self):
        self.seal_capture_results()
        with self.assertRaisesRegex(ValueError, 'half A has no --pick'):
            release.sweep(self.reports/'x', self.sweep_args('A', self.reports/'PICK.json'))
        with self.assertRaisesRegex(ValueError, 'half A has no --pick'):
            release.sweep(self.reports/'x', self.sweep_args('B'))


if __name__ == '__main__':
    unittest.main()
