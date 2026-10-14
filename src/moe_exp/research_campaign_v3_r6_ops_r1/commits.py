"""Compact report commits on private refs; working HEAD/index and remotes untouched."""
from pathlib import Path
import os
import subprocess

from moe_exp.research_campaign_v3_r6.contracts import REPO, check_files, file_sha, load, locked, publish, require

ALLOWED = ('REPORT.json', 'report.md', 'UNCERTAINTY.json', 'paper_tables.csv', 'measurement_ledger.csv')


def git(args, repo, env=None):
    require(args[0] not in ('push', 'send-pack', 'fetch'), 'Remote Git operations are not authorized')
    return subprocess.check_output(['git', *args], cwd=repo, text=True, stderr=subprocess.PIPE,
                                   env=env, timeout=30).strip()


def prepare_report(campaign, directory, report, *, repo=REPO):
    repo, directory = Path(repo), Path(directory)
    relative = directory.resolve().relative_to(repo.resolve())
    require(relative.parts[0] == 'report', 'Only compact research report files may be prepared')
    files = {str(relative / name): file_sha(directory / name) for name in ALLOWED
             if (directory / name).is_file() and (directory / name).stat().st_size <= 8 * 1024**2}
    require(str(relative / 'REPORT.json') in files and str(relative / 'report.md') in files,
            'Compact report commit lacks its report and prose')
    check_files({str(repo / name): sha for name, sha in files.items()})
    target = Path(campaign['root']) / 'ops_r1/report_commits' / report['sha256']
    with locked(target, 'commit'):
        done = target / 'LOCAL_COMMIT.json'
        if done.exists():
            value = load(done)
            require(value['files'] == files, 'Committed report inputs changed')
            require(git(['rev-parse', value['ref']], repo) == value['commit'], 'Prepared report ref differs')
            return value
        intentpath = target / 'INTENT.json'
        if intentpath.exists():
            intent = load(intentpath)
            require(intent['files'] == files, 'Report commit intent changed')
        else:
            intent = publish(intentpath, {'parent': git(['rev-parse', 'HEAD'], repo), 'files': files})
        objectpath = target / 'COMMIT_OBJECT.json'
        if objectpath.exists():
            commit = load(objectpath)['commit']
        else:
            env = {**os.environ, 'GIT_INDEX_FILE': str((target / 'isolated.git-index').resolve())}
            git(['read-tree', intent['parent']], repo, env)
            for name in sorted(files):
                blob = git(['hash-object', '-w', '--', str(repo / name)], repo, env)
                git(['update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name], repo, env)
            tree = git(['write-tree'], repo, env)
            message = 'Record ' + campaign['campaign_id'] + ' ' + report.get('phase', 'final') + ' report and explicit scientific status'
            commit = git(['commit-tree', tree, '-p', intent['parent'], '-m', message], repo, env)
            publish(objectpath, {'commit': commit, 'tree': tree, 'parent': intent['parent'], 'message': message})
        ref = 'refs/heads/prepared/' + campaign['campaign_id'] + '/ops-report-' + report['sha256'][:16]
        probe = subprocess.run(['git', 'rev-parse', '--verify', ref], cwd=repo,
                               capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            require(probe.stdout.strip() == commit, 'Prepared report ref already owns another commit')
        else:
            git(['update-ref', ref, commit, '0' * 40], repo)
        import hashlib
        for name, sha in files.items():
            content = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo, timeout=30)
            require(hashlib.sha256(content).hexdigest() == sha, 'Prepared report commit readback differs')
        return publish(done, {'commit': commit, 'ref': ref, 'files': files,
            'report_sha256': report['sha256'], 'remote_writes': 0, 'remote_delivery_verified': False,
            'working_head_unchanged': True, 'working_index_unchanged': True})
