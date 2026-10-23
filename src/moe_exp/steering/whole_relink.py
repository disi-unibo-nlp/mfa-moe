"""Seal and launch a CPU-only re-analysis release: the final whole-reasoning combine under another link.

The whole-reasoning releases r7-r18 chose each model's efficiency-directed cells from the seven-model
efficiency-link pre-registration, so LFM2.5 used the pooled signs. The eight-model receipt
(driver/EFFICIENCY_LINK_V2.json, efficiency_link) gives LFM its own signs; its directed cells are
a subset of the pooled ones and every one of them already ran at every strength. A relink release
therefore runs no steering: an allocated test run, then the combined analysis of the same sources as
the parent (its SOURCES.json) with WHOLE_LINK pointing at the new receipt.

    scripts/steering/whole_relink.sh whole-reasoning-20261021-r19 LOCAL_TESTS
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from moe_exp.io import file_sha, publish, require, verified

REPO = Path(__file__).resolve().parents[3]
REPORTS = REPO/'report'
PARENT = 'whole-reasoning-20261021-r18'       # its SOURCES.json lists every release of the final combine
LINK = REPO/'report/final-routing-20261015-r1/driver/EFFICIENCY_LINK_V2.json'
MODELS = ('gpt', 'nemotron', 'qwen35', 'qwen36', 'gemma', 'glm', 'qwen330b', 'lfm25')
CPU_CEILING = 24.  # one combine (~25 min x 8 CPUs) plus the 4-CPU test run, with retries
RUN_PATHS = ('src/moe_exp', 'tests/steering', 'tests/routing', 'scripts/slurm', 'scripts/steering')


def materialize(root: Path, commit: str) -> tuple[Path, Path]:
    """Archive the commit into the release and unpack it once under mfa-moe/releases/<commit>."""
    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, 'src', 'scripts',
                    'configs', 'tests', 'docs/ledger', 'pyproject.toml', 'requirements', 'AGENTS.md'],
                   cwd=REPO, check=True)
    snapshot = REPO.parent/'releases'/commit
    if not snapshot.exists():
        snapshot.mkdir(mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(snapshot)], check=True)
        (snapshot/'RELEASE_COMMIT').write_text(commit+'\n')
    return archive, snapshot


def release(name: str, local_tests: int) -> dict:
    """Write the receipts of a new relink release and materialize its code snapshot.

    Args:
        name: Release directory under report/ (must not exist).
        local_tests: Tests that passed locally on this commit (the allocated run repeats the suites).
    """
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    parent = verified(REPORTS/PARENT/'RELEASE.json')
    link = verified(LINK)
    require(link['models'] == list(MODELS), 'the link receipt does not cover the eight models')
    root.mkdir(mode=0o700)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': local_tests, 'commit': commit,
                                           'scope': 'tests/steering/test_whole*.py; the allocated run repeats '
                                                    'tests/steering, tests/steering/outcome and tests/routing in full'})
    publish(root/'RESOURCE_PLAN.json', {'models': list(MODELS), 'new_gpu_hour_ceiling': 0.,
                                        'cpu_core_hour_ceiling': CPU_CEILING, 'no_budget_reset': True,
                                        'jobs': 'allocated tests, then one combine'})
    publish(root/'RELINK.json', {'parent': str(REPORTS/PARENT), 'link': str(LINK), 'link_sha256': link['sha256'],
                                 'reason': 'LFM2.5 uses its own native efficiency-link signs (eight-model '
                                           'receipt) instead of the pooled seven-model signs'})
    publish(root/'SOURCES.json', {'sources': verified(REPORTS/PARENT/'SOURCES.json')['sources']})
    (root/'README.md').write_text(
        f'# Whole-reasoning relink ({name})\n\n'
        f'Re-analysis only, no steering. The combined analysis of {PARENT} (r7-r18) is rerun with the\n'
        'eight-model efficiency link (driver/EFFICIENCY_LINK_V2.json), in which LFM2.5 has its own native\n'
        'signs instead of the pooled seven-model ones. Its directed cells are a subset of the pooled ones,\n'
        'all of which ran at every strength, so only LFM2.5 and the pooled rows can change. The seven other\n'
        'models keep their pre-registered signs. Output: analysis-combined/EFFECTS_WHOLE.json.\n')

    archive, snapshot = materialize(root, commit)
    files = {str(p): file_sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    inputs = {k.replace(parent['release_root'], str(snapshot)): v for k, v in parent['inputs'].items()}
    inputs[str(LINK)] = file_sha(LINK)
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    value = publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(snapshot), 'models': list(MODELS), 'sweep': {}, 'shards': {},
        'cutoff': parent['cutoff'], 'relink_sha256': verified(root/'RELINK.json')['sha256'],
        'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
        'code_files': files, 'inputs': inputs, 'classifier': parent['classifier'],
        'nomination_reference': parent['nomination_reference'],
        'local_validation_sha256': verified(root/'LOCAL_VALIDATION.json')['sha256'],
        'resource_plan_sha256': verified(root/'RESOURCE_PLAN.json')['sha256'],
        'new_gpu_hour_ceiling': 0., 'cpu_core_hour_ceiling': CPU_CEILING, 'engineering_only': False,
        'parent_release_sha256': parent['sha256']})
    (root/'launch.env').write_text(f"MODELS={','.join(MODELS)}\nCPU_CEILING={CPU_CEILING}\nCOMMIT={commit}\n"
                                   f"LINK={LINK}\n")
    return {'release': name, 'commit': commit, 'sha256': value['sha256'], 'snapshot': str(snapshot)}


def launch(after: str) -> dict:
    """Submit the allocated tests and the combine (after the tests and the parent's combine).

    Runs from the release snapshot with WHOLE_ROOT and WHOLE_LINK exported (scripts/steering/whole_relink.sh).
    """
    from moe_exp.steering import whole
    whole.verify_release()
    root = whole.ROOT
    require(not (root/'JOBS.json').exists(), 'relink graph already accepted; supervise the existing jobs')
    # The snapshot has no report/, so the receipt path comes from RELINK.json, not from this module.
    relink = verified(root/'RELINK.json')
    require(whole.LINK == Path(relink['link']) and verified(whole.LINK)['sha256'] == relink['link_sha256'],
            'WHOLE_LINK must name the release link receipt')
    (root/'logs').mkdir(parents=True, exist_ok=True, mode=0o700)
    release_root = verified(root/'RELEASE.json')['release_root']
    # The test suites run in the snapshot through the study job wrapper, with unittest as the module.
    cmd = ['sbatch', '--parsable', '--account=iscrc_miosr', '--partition=boost_usr_prod', '--qos=normal',
           '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=24G', '--time=60', '--job-name=whole-relink-tests',
           '--chdir='+release_root, '--output='+str(root/'logs/%x-%j.out'), '--error='+str(root/'logs/%x-%j.err'),
           '--export=ALL', '--wrap',
           f'set -e; for s in tests/steering tests/steering/outcome tests/routing; do STUDY_MODULE=unittest '
           f'{release_root}/scripts/steering/gpu.sh discover -s $s -t $s; done']
    env = whole.submission_environment({'RELEASE_ROOT': release_root, 'STUDY_DATA_ROOT': str(whole.driver.ROOT)})
    for key in ('WHOLE_MODELS', 'WHOLE_GPU_CEILING', 'WHOLE_CPU_CEILING', 'WHOLE_CUTOFF', 'WHOLE_SWEEP',
                'WHOLE_SHARDS', 'WHOLE_ROOT', 'WHOLE_LINK'):
        env.pop(key, None)  # the suites test the default configuration in a fresh interpreter
    job = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout.strip().split(';')[0]
    require(job.isdecimal(), 'sbatch did not return a job ID')
    validation = {'id': job, 'command': 'tests', 'model': None, 'gpus': 0, 'cpus': 4, 'minutes': 60,
                  'submitted': time.time()}
    schedule = {'validation': validation, 'canaries': {m: [] for m in whole.MODELS},
                'production': {m: [] for m in whole.MODELS}, 'extraction': {}}
    whole.driver.checkpoint(root/'JOBS.json', schedule)
    schedule['analysis'] = whole.submit_stage('combine', dependency=f'afterok:{job}:{after}')
    whole.driver.checkpoint(root/'JOBS.json', schedule)
    return {'tests': job, 'combine': schedule['analysis']['id']}


def main() -> None:
    parser = argparse.ArgumentParser(prog='python -m moe_exp.steering.whole_relink')
    sub = parser.add_subparsers(dest='command', required=True)
    seal = sub.add_parser('seal')
    seal.add_argument('--name', required=True)
    seal.add_argument('--local-tests', type=int, required=True)
    start = sub.add_parser('launch')
    start.add_argument('--after', required=True, help="job ID of the parent's combine")
    args = parser.parse_args()
    print(json.dumps(release(args.name, args.local_tests) if args.command == 'seal' else launch(args.after)))


if __name__ == '__main__':
    main()
