"""Seal an outcome-steering release: receipts, a frozen code snapshot and the launch environment.

The release is an ordinary whole-reasoning sweep release (moe_exp.study.whole runs, extracts and
monitors it) whose OUTCOME.json makes `prepare` build the nudge and length-limit rows. It reuses the
template release r7 for pinned inputs, the sentence classifier and the nomination reference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from moe_exp.final_routing.common import file_sha, publish, require, verified

from . import REPORTS
from .manifest import default_spec
from .select import BULK

CUTOFF = 1792789200            # 2026-10-23 23:00 Europe/Rome: no GPU work starts after it
DELTAS = (0.4, 0.8, 1.6)       # logit shifts (user decision, D-0003)
FRACTIONS = (0.5, 0.7, 0.85)   # where the length-limit baseline stops the native reasoning
ANSWER_TOKENS = 2048           # answer allowance of a length-limit row
TEMPLATE = 'whole-reasoning-20261021-r7'
RUN_PATHS = ('src/moe_exp', 'tests/study', 'tests/outcome_steering', 'scripts/study', 'scripts/outcome_steering')


def whole_usage() -> dict:
    """GPU and CPU hours of every earlier whole-* and os-* job, for the release's prior-usage receipt."""
    lines = subprocess.run(['sacct', '-u', 'lmolfett', '-S', '2026-10-20T18:00', '-X', '-n', '-P',
                            '-o', 'JobIDRaw,JobName,ElapsedRaw,AllocTRES'], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    used = {'gpu_hours': 0., 'cpu_core_hours': 0.}
    for line in lines:
        _, name, elapsed, tres = line.split('|')
        if name.startswith(('whole-', 'os-')):
            fields = dict(f.split('=', 1) for f in tres.split(',') if '=' in f)
            gpus, cpus, seconds = int(fields.get('gres/gpu', 0)), int(fields.get('cpu', 0)), int(elapsed or 0)
            used['gpu_hours'] += seconds*gpus/3600
            used['cpu_core_hours'] += 0 if gpus else seconds*cpus/3600
    return used


def release(name: str, models: list[str], shards: dict[str, int], local_tests: int) -> dict:
    """Write every receipt of a new release and materialize its code snapshot.

    Args:
        name: Release directory under report/ (must not exist).
        models: Models to steer.
        shards: 2-GPU shards per model (4-hour walltime each).
        local_tests: Tests that passed locally on this commit (the allocated validation reruns all).
    """
    repo = REPORTS.parent
    # Only the code a release runs must be committed; unrelated work elsewhere in the tree may be open.
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=repo,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    root.mkdir(mode=0o700)
    template = verified(REPORTS/TEMPLATE/'RELEASE.json')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True,
                            check=True).stdout.strip()
    gpu_ceiling, cpu_ceiling = 8.*sum(shards[m] for m in models)+10, 60.

    publish(root/'PRIOR_USAGE.json', {'earlier_whole_usage_snapshot': whole_usage(), 'snapshot_time': time.time(),
                                      'budget_reset': False,
                                      'separate_budget': f'{gpu_ceiling} GPU-h and {cpu_ceiling} CPU core-h'})
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': local_tests, 'commit': commit,
                                           'scope': 'tests/outcome_steering plus tests/study/test_stream_batch.py; '
                                                    'the allocated validation runs both suites in full'})
    publish(root/'RESOURCE_PLAN.json', {'models': models, 'shards': {m: shards[m] for m in models},
                                        'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': cpu_ceiling,
                                        'cutoff': float(CUTOFF), 'no_budget_reset': True})
    publish(root/'PREFLIGHT.json', {
        'host': os.uname().nodename, 'user': 'lmolfett', 'account': 'iscrc_miosr', 'time': time.time(),
        'template_release_sha256': template['sha256'],
        'balance': subprocess.run(['/cineca/bin/saldo', '-b'], capture_output=True, text=True).stdout})
    spec = {m: {**default_spec(m), 'deltas': list(DELTAS), 'fractions': list(FRACTIONS),
                'answer_tokens': ANSWER_TOKENS} for m in models}
    publish(root/'OUTCOME.json', {'models': spec})        # makes whole.prepare build the outcome rows
    (root/'README.md').write_text(
        f'# Outcome-selected gentle steering ({name})\n\n'
        'Nudges a small set of experts chosen by outcome (used more in wrong than right attempts of the same\n'
        'question, campaign-v3 cohort B) by subtracting a delta from their router logits at every prompt and\n'
        'reasoning token, against matched random experts, and compares both with unedited routing (the base\n'
        'releases\' native runs) and a stop-thinking-earlier baseline cut from those native runs.\n'
        f'Models: {", ".join(models)}; deltas {list(DELTAS)}; length fractions {list(FRACTIONS)}.\n'
        'Code: src/moe_exp/outcome_steering (README there). Ledger F-0004, D-0003.\n'
        'GPU work stops at 2026-10-23 23:00 Rome.\n')

    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, 'src', 'scripts',
                    'configs', 'tests', 'docs/ledger', 'pyproject.toml', 'requirements', 'AGENTS.md'],
                   cwd=repo, check=True)
    snapshot = repo.parent/'releases'/commit
    if not snapshot.exists():
        snapshot.mkdir(mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(snapshot)], check=True)
        (snapshot/'RELEASE_COMMIT').write_text(commit+'\n')
    files = {str(p): file_sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    inputs = {k.replace(template['release_root'], str(snapshot)): v for k, v in template['inputs'].items()}
    # The selection reads these outside the snapshot: per-attempt usage and the class nominations.
    for model in models:
        for path in (BULK/model/'USAGE.json', BULK/model/'USAGE.npz',
                     REPORTS/TEMPLATE/model/'WHOLE_NOMINATIONS.json'):
            inputs[str(path)] = file_sha(path)
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    value = publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(snapshot), 'models': models,
        'sweep': {m: 'main' for m in models}, 'shards': {m: shards[m] for m in models}, 'cutoff': float(CUTOFF),
        'outcome_sha256': verified(root/'OUTCOME.json')['sha256'],
        'source_archive': {'path': str(archive), 'sha256_file': file_sha(archive)},
        'code_files': files, 'inputs': inputs, 'classifier': template['classifier'],
        'nomination_reference': template['nomination_reference'],
        'preflight_sha256': verified(root/'PREFLIGHT.json')['sha256'],
        'local_validation_sha256': verified(root/'LOCAL_VALIDATION.json')['sha256'],
        'resource_plan_sha256': verified(root/'RESOURCE_PLAN.json')['sha256'],
        'prior_usage': verified(root/'PRIOR_USAGE.json'), 'new_gpu_hour_ceiling': gpu_ceiling,
        'cpu_core_hour_ceiling': cpu_ceiling, 'engineering_only': False, 'template_release_sha256': template['sha256']})
    (root/'launch.env').write_text(
        f"MODELS={','.join(models)}\nSWEEP={','.join(m+':main' for m in models)}\n"
        f"SHARDS={','.join(f'{m}:{shards[m]}' for m in models)}\nGPU_CEILING={gpu_ceiling}\nCPU_CEILING={cpu_ceiling}\n"
        f"CUTOFF={CUTOFF}\nCOMMIT={commit}\n")
    return {'release': name, 'commit': commit, 'sha256': value['sha256'], 'snapshot': str(snapshot)}
