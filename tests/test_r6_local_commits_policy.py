"""Real local Git preparation; no scheduler or remote network mutation."""
import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r6_local_commits', REPO / 'scripts/research_campaign_v3_r6_local_commits.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from moe_exp.research_campaign_v3_r6.contracts import STAGING, publish, load


class LocalCommits(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r6-local-commits-', dir=STAGING)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.run_git('init', '-q')
        self.run_git('config', 'user.name', 'Policy test')
        self.run_git('config', 'user.email', 'policy-test@example.invalid')
        (self.repo / '.gitignore').write_text('*.tex\n')
        self.run_git('add', '.gitignore')
        self.run_git('commit', '-qm', 'Initial test state')
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        (self.snapshot / 'ignored.tex').write_text('Source-bound manuscript\n')
        self.files = {'ignored.tex': hashlib.sha256((self.snapshot / 'ignored.tex').read_bytes()).hexdigest()}
        self.directory = self.root / 'publication' / ('a' * 64) / 'github'

    def tearDown(self):
        self.temp.cleanup()

    def run_git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, stderr=subprocess.STDOUT).decode().strip()

    def prepare(self):
        return M.prepare_commit(self.repo, self.snapshot, self.files, self.directory,
            campaign_id='r6-test', phase='memory', destination='github')

    def test_ignored_source_is_committed_without_changing_user_index_or_head(self):
        index = (self.repo / '.git/index').read_bytes()
        head = self.run_git('rev-parse', 'HEAD')
        result = self.prepare()
        self.assertEqual(self.run_git('show', result['commit'] + ':ignored.tex'), 'Source-bound manuscript')
        self.assertEqual((self.repo / '.git/index').read_bytes(), index)
        self.assertEqual(self.run_git('rev-parse', 'HEAD'), head)
        self.assertEqual(self.run_git('rev-parse', result['ref']), result['commit'])
        self.assertFalse(result['remote_delivery_verified'])
        self.assertEqual(result['remote_writes'], 0)

    def test_repeated_preparation_reuses_exact_commit(self):
        first = self.prepare()
        self.assertEqual(self.prepare(), first)

    def test_interruption_after_commit_recovers_same_object(self):
        original = M.git
        def interrupted(args, directory, **kwargs):
            if args[0] == 'update-ref':
                raise RuntimeError('Interrupted after durable commit')
            return original(args, directory, **kwargs)
        with patch.object(M, 'git', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
                self.prepare()
        commit = load(self.directory / 'LOCAL_COMMIT_OBJECT.json')['commit']
        self.assertEqual(self.prepare()['commit'], commit)

    def test_changed_bound_source_is_rejected(self):
        self.prepare()
        (self.snapshot / 'ignored.tex').write_text('Changed input\n')
        with self.assertRaisesRegex(ValueError, 'Bound publication source changed'):
            self.prepare()

    def test_network_push_commands_are_rejected(self):
        for command in ('push', 'send-pack'):
            with self.assertRaisesRegex(RuntimeError, 'Remote writes'):
                M.git([command, 'origin'], self.repo)

    def test_future_workers_use_bound_policy_and_remote_functions_are_disabled(self):
        from moe_exp.research_campaign_v3_r6 import publication, scheduler
        root = self.root / 'policy'
        campaign = {'root': str(root), 'sha256': 'fixed-scientific-seal'}
        policy = publish(root / 'PUBLICATION_POLICY.json', {'remote_writes_allowed': False})
        publish(root / 'LOCAL_COMMITS_OPERATIONAL_BINDING.json', {'files': {},
            'campaign_sha256': campaign['sha256'], 'policy_sha256': policy['sha256']})
        captured = []
        def command(self, args):
            captured.append(args)
            return 'scheduler boundary simulated'
        def verify(self, jid, value):
            return {'fields': {'Command': str(M.WRAPPER)}}
        with patch.object(scheduler.Slurm, 'command', command), \
             patch.object(scheduler.Slurm, 'verify', verify), \
             patch.object(publication, 'deliver'), patch.object(publication, 'push'), \
             patch.object(publication, 'push_snapshot'):
            M.install_policy(campaign)
            original = str(REPO / 'scripts/research_campaign_v3_r6/job.sbatch')
            scheduler.Slurm().command(['sbatch', '--cpus-per-task=16', original, 'original-card.json'])
            self.assertEqual(captured[-1], ['sbatch', '--cpus-per-task=16', str(M.WRAPPER), 'original-card.json'])
            scheduler.Slurm().verify('1', {'campaign_sha256': campaign['sha256'], 'stage': 'memory_qualify'})
            with self.assertRaisesRegex(RuntimeError, 'deferred to the user'):
                publication.push_snapshot(campaign, {}, 'github')

    def test_publication_worker_finishes_without_remote_delivery_stamp(self):
        from moe_exp.research_campaign_v3_r6 import publication
        paper = self.repo / 'paper/6a510c6db0d9b08aa438c142'
        paper.mkdir(parents=True)
        subprocess.run(['git', 'init', '-q', str(paper)], check=True)
        subprocess.run(['git', '-C', str(paper), 'config', 'user.name', 'Policy test'], check=True)
        subprocess.run(['git', '-C', str(paper), 'config', 'user.email', 'policy-test@example.invalid'], check=True)
        (paper / 'base.tex').write_text('Original source\n')
        subprocess.run(['git', '-C', str(paper), 'add', 'base.tex'], check=True)
        subprocess.run(['git', '-C', str(paper), 'commit', '-qm', 'Initial paper'], check=True)
        relative = 'paper/6a510c6db0d9b08aa438c142/main.tex'
        target = self.snapshot / relative
        target.parent.mkdir(parents=True)
        target.write_text('New local snapshot\n')
        files = {relative: hashlib.sha256(target.read_bytes()).hexdigest()}
        root = self.root / 'campaign'
        policy = publish(root / 'PUBLICATION_POLICY.json', {'remote_writes_allowed': False})
        report = publish(root / 'report.json', {'phase': 'final', 'snapshot': False})
        directory = root / 'publication' / report['sha256']
        publish(directory / 'LOCAL_PDF.json', {'snapshot': str(self.snapshot), 'files': files})
        campaign = {'root': str(root), 'campaign_id': 'r6-test'}
        with patch.object(M, 'REPO', self.repo), patch.object(publication, 'PAPER', paper):
            result = M.local_delivery(campaign, report)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['publication_status'], 'manual_delivery_pending')
        self.assertFalse(result['PAPER_DELIVERED'])
        self.assertFalse((root / 'PAPER_DELIVERY.json').exists())
        self.assertFalse(load(root / 'PAPER_REMOTE_DELIVERY.json')['complete'])
        self.assertEqual(load(directory / 'PUBLICATION.json')['policy_sha256'], policy['sha256'])


if __name__ == '__main__':
    unittest.main()
