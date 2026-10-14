#!/usr/bin/env python3
"""Separately bound operational policy: prepare commits; the user performs pushes."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
WRAPPER = REPO / 'scripts/research_campaign_v3_r6_local_commits.sbatch'


def git(args, directory, *, env=None):
    if args and args[0] in ('push', 'send-pack'):
        raise RuntimeError('Remote writes are reserved for the user')
    result = subprocess.run(['git', *args], cwd=directory, env=env,
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Local Git preparation failed: ' + result.stderr[-2000:])
    return result.stdout.strip()


def prepare_commit(source, snapshot, files, directory, *, campaign_id, phase, destination):
    """Use an isolated index and a prepared ref; leave HEAD and the user index intact."""
    from moe_exp.research_campaign_v3_r6.contracts import load, publish, require, file_sha
    source, snapshot, directory = map(Path, (source, snapshot, directory))
    directory.mkdir(parents=True, exist_ok=True)
    for relative, expected in files.items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts
                and '\n' not in relative, 'Publication path escapes its allowlist')
        require(file_sha(snapshot / path) == expected, 'Bound publication source changed')
    intent_path = directory / 'LOCAL_COMMIT_INTENT.json'
    if intent_path.exists():
        intent = load(intent_path)
        require(intent['files'] == files, 'Interrupted local commit inputs changed')
    else:
        intent = publish(intent_path, {'parent': git(['rev-parse', 'HEAD'], source),
            'files': files, 'destination': destination, 'campaign_id': campaign_id, 'phase': phase})
    commit_path = directory / 'LOCAL_COMMIT_OBJECT.json'
    if commit_path.exists():
        commit = load(commit_path)['commit']
    else:
        index = directory / 'local.git-index'
        env = {**os.environ, 'GIT_INDEX_FILE': str(index.resolve())}
        git(['read-tree', intent['parent']], source, env=env)
        for relative in sorted(files):
            blob = git(['hash-object', '-w', '--', str((snapshot / relative).resolve())], source, env=env)
            git(['update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + relative], source, env=env)
        tree = git(['write-tree'], source, env=env)
        message = f'{campaign_id}: prepare {phase} manuscript and figures for manual {destination} delivery'
        commit = git(['commit-tree', tree, '-p', intent['parent'], '-m', message], source, env=env)
        publish(commit_path, {'commit': commit, 'parent': intent['parent'], 'tree': tree,
                             'message': message, 'files': files})
    ref = 'refs/heads/prepared/' + campaign_id + '/' + directory.parent.name[:16] + '-' + destination
    probe = subprocess.run(['git', 'rev-parse', '--verify', ref], cwd=source,
                           capture_output=True, text=True, timeout=30)
    if probe.returncode == 0:
        require(probe.stdout.strip() == commit, 'Prepared publication ref already owns another commit')
    else:
        git(['update-ref', ref, commit, '0' * 40], source)
    for relative, expected in files.items():
        raw = subprocess.run(['git', 'show', commit + ':' + relative], cwd=source,
                             capture_output=True, timeout=30, check=True).stdout
        import hashlib
        require(hashlib.sha256(raw).hexdigest() == expected, 'Prepared local commit readback differs')
    return publish(directory / 'LOCAL_COMMIT.json', {'status': 'local_commit_prepared',
        'commit': commit, 'parent': intent['parent'], 'ref': ref, 'files': files,
        'working_index_unchanged': True, 'working_head_unchanged': True,
        'remote_writes': 0, 'remote_delivery_verified': False})


def local_delivery(campaign, report, binding=None):
    from moe_exp.research_campaign_v3_r6.contracts import load, locked, publish, require
    from moe_exp.research_campaign_v3_r6.publication import PAPER
    root = Path(campaign['root'])
    policy = load(root / 'PUBLICATION_POLICY.json')
    require(policy['remote_writes_allowed'] is False, 'Local-only publication policy differs')
    directory = root / 'publication' / report['sha256']
    local = load(directory / 'LOCAL_PDF.json')
    with locked(root, 'publication'):
        outcomes = {}
        for destination in ('github', 'overleaf'):
            snapshot = Path(local['snapshot'])
            if destination == 'github':
                source, files = REPO, local['files']
            else:
                source, files = PAPER, {}
                prefix = Path('paper/6a510c6db0d9b08aa438c142')
                for name, sha in local['files'].items():
                    path = Path(name)
                    if path.is_relative_to(prefix):
                        files[str(path.relative_to(prefix))] = sha
                snapshot = snapshot / prefix
            receipt = prepare_commit(source, snapshot, files, directory / destination,
                campaign_id=campaign['campaign_id'], phase=report.get('phase', 'final'), destination=destination)
            outcomes[destination] = publish(directory / destination / 'DESTINATION.json', {
                'destination': destination, 'status': 'manual_delivery_pending',
                'local_commit': receipt, 'remote_writes': 0, 'remote_delivery_verified': False})
        final = publish(directory / 'PUBLICATION.json', {'status': 'manual_delivery_pending',
            'destinations': outcomes, 'report_sha256': report['sha256'],
            'policy_sha256': policy['sha256'], 'scientific_progression_affected': False,
            'remote_writes': 0, 'remote_delivery_verified': False})
        if report.get('phase') == 'final' and not report.get('snapshot'):
            publish(root / 'PAPER_REMOTE_DELIVERY.json', {'complete': False,
                'report_sha256': report['sha256'], 'publication_sha256': final['sha256'],
                'github': 'manual_delivery_pending', 'overleaf': 'manual_delivery_pending'})
        # The preparation worker finishes; neither destination is called delivered.
        return {**final, 'status': 'complete', 'publication_status': final['status'],
                'PAPER_DELIVERED': False}


def install_policy(campaign):
    from moe_exp.research_campaign_v3_r6.contracts import load, check_files, require
    from moe_exp.research_campaign_v3_r6 import publication, scheduler
    root = Path(campaign['root'])
    policy = load(root / 'PUBLICATION_POLICY.json')
    binding = load(root / 'LOCAL_COMMITS_OPERATIONAL_BINDING.json')
    check_files(binding['files'])
    require(binding['campaign_sha256'] == campaign['sha256'] and
            binding['policy_sha256'] == policy['sha256'], 'Operational publication binding differs')
    require(policy['remote_writes_allowed'] is False, 'Remote writes are not locally authorized')
    publication.deliver = local_delivery
    def denied(*args, **kwargs):
        raise RuntimeError('Remote publication is deferred to the user')
    publication.push = publication.push_snapshot = denied
    original_command = scheduler.Slurm.command
    original_verify = scheduler.Slurm.verify
    def command(self, args):
        args = list(args)
        if args[0] == 'sbatch':
            old = str(REPO / 'scripts/research_campaign_v3_r6/job.sbatch')
            if old in args:
                args[args.index(old)] = str(WRAPPER)
        return original_command(self, args)
    def verify(self, jid, value):
        proof = original_verify(self, jid, value)
        if value['campaign_sha256'] == campaign['sha256'] and value['stage'] != 'validate':
            require(proof['fields']['Command'] == str(WRAPPER), 'Accepted worker lacks bound local-only policy')
        return proof
    scheduler.Slurm.command = command
    scheduler.Slurm.verify = verify


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    from moe_exp.research_campaign_v3_r6.contracts import load, require, check_files
    require(args.campaign is not None or args.card is not None, 'Campaign or immutable card is required')
    value = load(args.card) if args.card else None
    root = args.campaign or Path(value['root'])
    campaign = load(root / 'CAMPAIGN.json')
    check_files(campaign['code_files'])
    install_policy(campaign)
    spec = importlib.util.spec_from_file_location('sealed_r6_launcher', REPO / 'scripts/research_campaign_v3_r6/run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.main()


if __name__ == '__main__':
    main()
