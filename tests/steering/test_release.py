"""Sealing a release: one mechanism for complement, relink and outcome releases."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from steering import release, whole
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
        self.assertEqual(sealed['parent_release_sha256'], self.parent['sha256'])  # the key the earlier relinks used
        self.assertNotIn('template_release_sha256', sealed)
        # The release records what the earlier jobs cost, since its ceilings do not reset the budget.
        usage = verified(root/'PRIOR_USAGE.json')
        self.assertEqual(sealed['prior_usage'], usage)
        self.assertEqual(usage['ledger_totals'], {'whole-': 'abc123', 'os-': 'abc123'})
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


if __name__ == '__main__':
    unittest.main()
