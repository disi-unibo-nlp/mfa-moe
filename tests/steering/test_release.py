"""Sealing a release: one mechanism for complement, relink and outcome releases."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from steering import release, whole
from store import WORK, file_sha, publish, verified


class SealTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=WORK/'tools/tmp')
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.reports, self.snapshots, old = base/'report', base/'releases', base/'old-snapshot'
        (old/'configs').mkdir(parents=True)
        (old/'configs/study.yaml').write_text('seed: 1\n')
        link = publish(self.reports/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json',
                       {'models': list(release.ALL_MODELS)})
        self.parent = publish(self.reports/'whole-reasoning-20261021-r18/RELEASE.json', {
            'release_root': str(old), 'cutoff': 1792814400., 'classifier': {'root': 'c'},
            'nomination_reference': 'n.json', 'inputs': {str(old/'configs/study.yaml'): file_sha(old/'configs/study.yaml')}})
        publish(self.reports/'whole-reasoning-20261021-r18/SOURCES.json',
                {'sources': [{'model': 'gpt', 'root': 'r8', 'dose': 'half'}]})
        self.link = link

        def snapshot(root, commit):
            folder = self.snapshots/commit
            (folder/'configs').mkdir(parents=True)
            (folder/'configs/study.yaml').write_text('seed: 1\n')
            archive = root/'SOURCE.tar.gz'
            archive.write_bytes(b'archive')
            return archive, folder

        def run(command, **kwargs):
            return SimpleNamespace(stdout='' if command[1] == 'status' else 'abc123\n')
        for patcher in (patch.object(release, 'REPORTS', self.reports), patch.object(release, 'snapshot', snapshot),
                        patch.object(release.subprocess, 'run', run)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_relink_release_names_its_link_and_runs_no_shards(self):
        folder = release.seal('relink', 'whole-reasoning-test', 12,
                              SimpleNamespace(models=None, shards=None, cutoff=None))
        root = self.reports/'whole-reasoning-test'
        sealed = verified(root/'RELEASE.json')
        self.assertEqual(folder, self.snapshots/'abc123')
        self.assertEqual((sealed['commit'], sealed['release_root']), ('abc123', str(folder)))
        self.assertEqual((sealed['models'], sealed['shards'], sealed['sweep'], sealed['cutoff']),
                         (list(release.ALL_MODELS), {}, {}, 1792814400.))
        self.assertEqual(sealed['parent_release_sha256'], self.parent['sha256'])
        self.assertEqual(sealed['relink_sha256'], verified(root/'RELINK.json')['sha256'])
        # The parent's pinned input moves to the new snapshot, and the link receipt is pinned too.
        self.assertIn(str(folder/'configs/study.yaml'), sealed['inputs'])
        self.assertEqual(sealed['inputs'][str(self.reports/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json')],
                         file_sha(self.reports/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'))
        self.assertEqual(verified(root/'SOURCES.json')['sources'][0]['model'], 'gpt')
        # The stages read their settings back from the release folder.
        self.assertEqual(whole.link_receipt(root), self.reports/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json')
        self.assertEqual(whole.sweep_doses(root), {})
        with self.assertRaisesRegex(ValueError, 'exists'):
            release.seal('relink', 'whole-reasoning-test', 12, SimpleNamespace(models=None, shards=None, cutoff=None))

    def test_a_changed_pinned_input_refuses_the_seal(self):
        (self.reports/'final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json').write_text('{}')
        with self.assertRaises(ValueError):
            release.seal('relink', 'whole-reasoning-bad', 1, SimpleNamespace(models=None, shards=None, cutoff=None))

    def test_the_default_cutoffs_are_the_times_of_the_first_releases(self):
        self.assertEqual(release.rome('2026-10-24T06:00'), 1792814400.)  # complement r18
        self.assertEqual(release.rome('2026-10-23T23:00'), 1792789200.)  # outcome r1


if __name__ == '__main__':
    unittest.main()
