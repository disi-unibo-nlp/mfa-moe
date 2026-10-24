"""Seal an outcome-steering release: receipts, a frozen code snapshot and the launch environment.

The release is an ordinary whole-reasoning sweep release (moe_exp.study.whole runs, extracts and
monitors it) whose spec file makes `prepare` build its rows:

- OUTCOME.json (r1): nudge and length-limit rows on the base releases' families (`release`);
- CAPTURE_NATIVE.json (r2 data): six native attempts per roster question with routing saved
  (`capture_release`).

It reuses the template release r7 for pinned inputs, the sentence classifier and the nomination
reference. Receipts go to report/<name> of the working repository (REPORTS); the code snapshot is
the commit checked out where this file lives (CODE), which may be a separate worktree.
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
CODE = Path(__file__).resolve().parents[3]      # repository (or worktree) holding this code
CAPTURE_CUTOFF = 1793397600    # 2026-10-30 23:00 Europe/Rome: no capture shard starts after it


def whole_usage() -> dict:
    """GPU and CPU hours of every earlier whole-* and os-* job, for the release's prior-usage receipt."""
    lines = subprocess.run(['sacct', '-u', 'lmolfett', '-S', '2026-10-20T18:00', '-X', '-n', '-P',
                            '-o', 'JobIDRaw,JobName,ElapsedRaw,AllocTRES'], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    used = {'gpu_hours': 0., 'cpu_core_hours': 0.}
    for line in lines:
        _, name, elapsed, tres = line.split('|')
        if name.startswith(('whole-', 'os-', 'oc-')):
            fields = dict(f.split('=', 1) for f in tres.split(',') if '=' in f)
            gpus, cpus, seconds = int(fields.get('gres/gpu', 0)), int(fields.get('cpu', 0)), int(elapsed or 0)
            used['gpu_hours'] += seconds*gpus/3600
            used['cpu_core_hours'] += 0 if gpus else seconds*cpus/3600
    return used


def release(name: str, models: list[str], shards: dict[str, int], local_tests: int) -> dict:
    """Seal an r1-style outcome release (OUTCOME.json; see module docstring)."""
    spec = {m: {**default_spec(m), 'deltas': list(DELTAS), 'fractions': list(FRACTIONS),
                'answer_tokens': ANSWER_TOKENS} for m in models}
    readme = (f'# Outcome-selected gentle steering ({name})\n\n'
              'Nudges a small set of experts chosen by outcome (used more in wrong than right attempts of the same\n'
              'question, campaign-v3 cohort B) by subtracting a delta from their router logits at every prompt and\n'
              'reasoning token, against matched random experts, and compares both with unedited routing (the base\n'
              'releases\' native runs) and a stop-thinking-earlier baseline cut from those native runs.\n'
              f'Models: {", ".join(models)}; deltas {list(DELTAS)}; length fractions {list(FRACTIONS)}.\n'
              'Code: src/moe_exp/outcome_steering (README there). Ledger F-0004, D-0003.\n'
              'GPU work stops at 2026-10-23 23:00 Rome.\n')
    # The selection reads these outside the snapshot: per-attempt usage and the class nominations.
    extra = [path for model in models for path in (BULK/model/'USAGE.json', BULK/model/'USAGE.npz',
                                                   REPORTS/TEMPLATE/model/'WHOLE_NOMINATIONS.json')]
    return seal(name, models, shards, local_tests, spec_name='OUTCOME.json', spec=spec, readme=readme,
                extra_inputs=extra, cutoff=CUTOFF, cpu_ceiling=60., job_prefix='whole')


def capture_release(name: str, models: list[str], shards: dict[str, int], local_tests: int) -> dict:
    """Seal a capture release: ATTEMPTS native attempts per roster question, native routing ids saved.

    Stores go to FAST (report/<name>/stores is a link), because WORK cannot hold the captures.
    """
    from .capture import ATTEMPTS, CAP_TOKENS, FOLDS, SOURCE_CAMPAIGN, SOURCES, roster
    questions = roster(models)
    spec = {m: {'questions': questions, 'attempts': ATTEMPTS, 'cap_tokens': CAP_TOKENS} for m in models}
    readme = (f'# Native attempts with routing ({name})\n\n'
              f'Every model answers the same {len(questions)} questions {ATTEMPTS} times with unedited routing; every\n'
              'reasoning token keeps its native top-k expert ids and weights at every routed layer (AUDIT.npz per\n'
              f'unit). Completion allowance {CAP_TOKENS} tokens. These attempts feed the class-conditioned outcome\n'
              'selection, the attempt-selection analysis and serve as the unedited arm of the nudge releases.\n'
              f'Models: {", ".join(models)}. Code: src/moe_exp/outcome_steering (capture.py). Ledger F-0013, D-0006.\n')
    extra = [FOLDS, *[SOURCES.get(m, SOURCE_CAMPAIGN/m/'SOURCES.json') for m in models]]
    stores = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs')/name/'stores'
    return seal(name, models, shards, local_tests, spec_name='CAPTURE_NATIVE.json', spec=spec, readme=readme,
                extra_inputs=extra, cutoff=CAPTURE_CUTOFF, cpu_ceiling=200., job_prefix='oc', stores=stores)


CAPTURE_RELEASE = 'outcome-capture-20261023-r2'   # r1's allocated validation failed on a test message
R2_DELTAS = (0.4, 0.8, 1.2)    # logit shifts of the r2 sweep (D-0006)


def sweep_release(name: str, models: list[str], shards: dict[str, int], local_tests: int, half: str,
                  pick_path: str | None = None) -> dict:
    """Seal one half of the r2 class-conditioned sweep (OUTCOME_SWEEP.json).

    Half A runs every available expert set at every R2_DELTAS shift; half B runs the cells of PICK.json
    (pick.py). Both read the capture release: its manifests (prompts, seeds, unedited arm), the
    SELECTION_R2.json sets and, for the stop-early rows, its finished attempts.
    """
    from .manifest import halves
    capture = REPORTS/CAPTURE_RELEASE
    require(half in ('A', 'B') and (half == 'A') == (pick_path is None), 'half A has no pick; half B needs one')
    picked = verified(pick_path) if pick_path else None
    spec, extra = {}, []
    for model in models:
        selection_path = capture/model/'SELECTION_R2.json'
        selection = verified(selection_path)
        questions = verified(capture/'CAPTURE_NATIVE.json')['models'][model]['questions']
        available = [name_ for name_, entry in selection['sets'].items() if 'sets' in entry]
        cells = ([{'set': s, 'delta': d} for s in available for d in R2_DELTAS] if half == 'A' else
                 [c for c in picked['confirm'] if c['set'] in available])
        require(cells, f'no cell to run for {model}')
        spec[model] = {'capture': str(capture), 'selection': str(selection_path),
                       'selection_sha256': selection['sha256'], 'half': half, 'questions': halves(questions)[half],
                       'cells': cells, 'fractions': list(FRACTIONS), 'answer_tokens': ANSWER_TOKENS,
                       'pick': {'path': pick_path, 'sha256': picked['sha256']} if picked else None}
        extra += [selection_path, capture/model/'WHOLE.json', capture/model/'extraction'/'outcomes.parquet']
    readme = (f'# Class-conditioned outcome nudge, half {half} ({name})\n\n'
              'Experts used more in wrong than in right attempts of the same question, on the tokens of one\n'
              'reasoning class (or on all reasoning, the class-blind set), are nudged down by a logit shift at\n'
              'every prompt and reasoning token, against two disjoint matched random sets. The unedited arm is\n'
              f'the six attempts per question of {CAPTURE_RELEASE}; stop-early rows cut one of them.\n'
              f'Cells: {len(spec[models[0]]["cells"])} per model; shifts {list(R2_DELTAS)}. Ledger F-0013, D-0006.\n')
    stores = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs')/name/'stores'
    return seal(name, models, shards, local_tests, spec_name='OUTCOME_SWEEP.json', spec=spec, readme=readme,
                extra_inputs=extra, cutoff=CAPTURE_CUTOFF, cpu_ceiling=60., job_prefix='os', stores=stores)


def seal(name: str, models: list[str], shards: dict[str, int], local_tests: int, *, spec_name: str, spec: dict,
         readme: str, extra_inputs: list, cutoff: int, cpu_ceiling: float, job_prefix: str,
         stores: Path | None = None) -> dict:
    """Write every receipt of a new release and materialize its code snapshot.

    Args:
        name: Release directory under report/ (must not exist).
        models: Models of the release.
        shards: 2-GPU shards per model (4-hour walltime each).
        local_tests: Tests that passed locally on this commit (the allocated validation reruns all).
        spec_name, spec: The spec file that makes whole.prepare build this release's rows, and its models.
        readme: README.md of the release directory.
        extra_inputs: Files read outside the snapshot, pinned by hash.
        cutoff: Unix time after which no shard starts.
        cpu_ceiling: CPU core-hour ceiling (GPU ceiling: every shard's full walltime plus 10 h).
        job_prefix: Slurm job-name prefix (whole.JOB_PREFIX).
        stores: If given, report/<name>/stores links to this directory (created empty).
    """
    # Only the code a release runs must be committed; unrelated work elsewhere in the tree may be open.
    require(not subprocess.run(['git', 'status', '--porcelain', '--', *RUN_PATHS], cwd=CODE,
                               capture_output=True, text=True, check=True).stdout, 'commit the code first')
    root = REPORTS/name
    require(not root.exists(), f'{root} exists')
    root.mkdir(mode=0o700)
    if stores is not None:
        require(not stores.exists(), f'{stores} exists')
        stores.mkdir(parents=True, mode=0o700)
        (root/'stores').symlink_to(stores)
    template = verified(REPORTS/TEMPLATE/'RELEASE.json')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CODE, capture_output=True, text=True,
                            check=True).stdout.strip()
    gpu_ceiling = 8.*sum(shards[m] for m in models)+10

    publish(root/'PRIOR_USAGE.json', {'earlier_whole_usage_snapshot': whole_usage(), 'snapshot_time': time.time(),
                                      'budget_reset': False,
                                      'separate_budget': f'{gpu_ceiling} GPU-h and {cpu_ceiling} CPU core-h'})
    publish(root/'LOCAL_VALIDATION.json', {'passed': True, 'tests': local_tests, 'commit': commit,
                                           'scope': 'tests/outcome_steering plus tests/study/test_stream_batch.py; '
                                                    'the allocated validation runs both suites in full'})
    publish(root/'RESOURCE_PLAN.json', {'models': models, 'shards': {m: shards[m] for m in models},
                                        'new_gpu_hour_ceiling': gpu_ceiling, 'cpu_core_hour_ceiling': cpu_ceiling,
                                        'cutoff': float(cutoff), 'no_budget_reset': True})
    publish(root/'PREFLIGHT.json', {
        'host': os.uname().nodename, 'user': 'lmolfett', 'account': 'iscrc_miosr', 'time': time.time(),
        'template_release_sha256': template['sha256'],
        'balance': subprocess.run(['/cineca/bin/saldo', '-b'], capture_output=True, text=True).stdout})
    publish(root/spec_name, {'models': spec})        # makes whole.prepare build this release's rows
    (root/'README.md').write_text(readme)

    archive = root/'SOURCE.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(archive), commit, 'src', 'scripts',
                    'configs', 'tests', 'docs/ledger', 'pyproject.toml', 'requirements', 'AGENTS.md'],
                   cwd=CODE, check=True)
    snapshot = REPORTS.parent.parent/'releases'/commit
    if not snapshot.exists():
        snapshot.mkdir(mode=0o700)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(snapshot)], check=True)
        (snapshot/'RELEASE_COMMIT').write_text(commit+'\n')
    files = {str(p): file_sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    inputs = {k.replace(template['release_root'], str(snapshot)): v for k, v in template['inputs'].items()}
    for path in extra_inputs:
        inputs[str(path)] = file_sha(path)
    for path, expected in inputs.items():
        require(file_sha(Path(path)) == expected, 'input mismatch: '+path)
    value = publish(root/'RELEASE.json', {
        'commit': commit, 'release_root': str(snapshot), 'models': models,
        'sweep': {m: 'main' for m in models}, 'shards': {m: shards[m] for m in models}, 'cutoff': float(cutoff),
        'spec_file': spec_name, 'spec_sha256': verified(root/spec_name)['sha256'],
        'stores': str(stores) if stores is not None else None, 'job_prefix': job_prefix,
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
        f"CUTOFF={cutoff}\nCOMMIT={commit}\nJOB_PREFIX={job_prefix}\n")
    return {'release': name, 'commit': commit, 'sha256': value['sha256'], 'snapshot': str(snapshot)}
