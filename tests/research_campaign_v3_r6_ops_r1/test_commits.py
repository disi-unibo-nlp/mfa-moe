"""Actual isolated Git commits of compact reports; preserve dirty manuscript state."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.research_campaign_v3_r6.contracts import STAGING, file_sha, load, publish
from moe_exp.research_campaign_v3_r6_ops_r1 import commits


class ReportCommits(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r6-ops-git-', dir=STAGING)
        self.repo = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Report contract')
        self.git('config', 'user.email', 'report-contract@example.invalid')
        (self.repo / '.gitignore').write_text('*.csv\n')
        self.git('add', '.gitignore'); self.git('commit', '-qm', 'Initial contract state')
        paper = self.repo / 'paper'; paper.mkdir(); (paper / 'main.tex').write_text('Untouched dirty source\n')
        self.directory = self.repo / 'report/test/output'
        self.report = publish(self.directory / 'REPORT.json', {'SCIENCE_COMPLETE': False, 'phase': 'memory'})
        (self.directory / 'report.md').write_text('Observed incomplete science\n')
        (self.directory / 'cells.csv').write_text('Bulk data must not be staged\n')
        self.campaign = {'root': str(self.repo / 'report/test'), 'campaign_id': 'r6-contract'}

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, stderr=subprocess.STDOUT, text=True).strip()

    def prepare(self):
        return commits.prepare_report(self.campaign, self.directory, self.report, repo=self.repo)

    def test_real_commit_excludes_bulk_and_preserves_head_index_and_dirty_paper(self):
        head, index = self.git('rev-parse', 'HEAD'), file_sha(self.repo / '.git/index')
        before = (self.repo / 'paper/main.tex').read_bytes()
        result = self.prepare()
        self.assertEqual(self.git('rev-parse', 'HEAD'), head)
        self.assertEqual(file_sha(self.repo / '.git/index'), index)
        self.assertEqual((self.repo / 'paper/main.tex').read_bytes(), before)
        changed = self.git('diff-tree', '--no-commit-id', '--name-only', '-r', result['commit']).splitlines()
        self.assertEqual(changed, ['report/test/output/REPORT.json', 'report/test/output/report.md'])
        self.assertEqual(self.git('rev-parse', result['ref']), result['commit'])
        self.assertFalse(result['remote_delivery_verified'])

    def test_interruption_after_commit_object_reuses_same_commit(self):
        original = commits.git
        def interrupted(args, *a, **k):
            if args[0] == 'update-ref':
                raise RuntimeError('Interrupted ref update')
            return original(args, *a, **k)
        with patch.object(commits, 'git', interrupted):
            with self.assertRaises(RuntimeError):
                self.prepare()
        directory = Path(self.campaign['root']) / 'ops_r1/report_commits' / self.report['sha256']
        obj = load(directory / 'COMMIT_OBJECT.json')
        self.assertEqual(self.prepare()['commit'], obj['commit'])
        self.assertEqual(self.prepare()['commit'], obj['commit'])

    def test_remote_git_write_is_denied(self):
        with self.assertRaises(ValueError):
            commits.git(['push', 'origin'], self.repo)


if __name__ == '__main__':
    unittest.main()
