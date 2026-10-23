"""Frozen scientific configuration and one source manifest per released commit."""
from pathlib import Path
import os
import subprocess
from moe_exp.io import (
    REPO,
    MODELS,
    CLASSES,
    check_files,
    digest,
    file_sha,
    load,
    locked,
    private,
    publish,
    require,
    verified,
)


def read_config(path):
    import yaml
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    require(tuple(value['models']) == MODELS, 'the final study requires all eight models in fixed order')
    require(tuple(value['design']['classes']) == CLASSES, 'reasoning classes changed')
    for key in ('root', 'source_campaign', 'bulk_root', 'environment'):
        p = Path(value[key])
        value[key] = str(private(p if p.is_absolute() else REPO/p))
    value['budget'].setdefault('gpu_repair_hours', 760.)
    require(0 < value['budget']['gpu_repair_hours'] <= 760., 'GPU repair authorization exceeded')
    value['config_path'] = str(path)
    value['config_file_sha256'] = file_sha(path)
    return value



def study_files():
    paths = [* (REPO / 'src/moe_exp/study').glob('*.py'),
             * (REPO / 'tests/study').glob('*.py'),
             * (REPO / 'scripts/study').glob('*.sh'),
             * (REPO / 'scripts/inference').glob('*.sh')]
    return {str(path): file_sha(path) for path in sorted(paths)}


def active(config):
    pointer = Path(config['root']) / 'CURRENT_STUDY.json'
    if not pointer.exists():
        return None
    return verified(verified(pointer)['binding_path'])


def terminal_release(binding):
    """Require closed native allocations and no unresolved submission intent."""
    from .resources import accounting
    from .reservations import TERMINAL
    control = Path(binding['control_root'])
    terminal = verified(control / 'TERMINAL.json')
    require(terminal['collection_terminal'], 'previous release is not terminal')
    usage = accounting(binding['root'])
    for intent in (control / 'submissions').glob('*/INTENT.json'):
        acceptance = intent.parent / 'ACCEPTED.json'
        require(acceptance.exists(), 'reconcile ambiguous submission before replacement')
        jid = verified(acceptance)['job_id']
        require(jid in usage and usage[jid]['state'] in TERMINAL,
                'previous release still has an outstanding allocation: ' + jid)
    return terminal


def initialize(config, replace_terminal=False):
    with locked(config['root'], 'source-release'):
        return initialize_locked(config, replace_terminal)


def initialize_locked(config, replace_terminal):
    old = active(config)
    if old is not None:
        require(old['config_file_sha256'] == config['config_file_sha256'], 'released configuration differs; use the frozen configuration')
        check_files(old['study_files'])
        if not replace_terminal:
            return old
    require(old is not None or not replace_terminal, 'no previous release to replace')
    root = Path(config['root'])
    files = study_files()
    if replace_terminal:
        terminal = terminal_release(old)
        proof = verified(root / 'study-reviews' / ('release-' + digest(files) + '.json'))
        require(proof['status'] == 'complete' and proof['source_files'] == files,
                'corrected source lacks matching readiness and reviews')
        for name in ('readiness', 'reviews', 'authorization'):
            receipt = verified(proof[name])
            require(receipt['status'] == 'complete', name + ' is incomplete')
            if name == 'authorization':
                require(receipt['previous_binding_sha256'] == old['sha256'] and
                        receipt['previous_terminal_sha256'] == terminal['sha256'],
                        'authorization names a different prior release')
            else:
                require(receipt['source_files'] == files, name + ' names a different source')
        proof_names = ('STUDY_INSTALLED_REQUEST_API.json',)
    else:
        proof_names = ('STUDY_READY.json', 'STUDY_REVIEWS.json', 'STUDY_INSTALLED_REQUEST_API.json')
    for name in proof_names:
        evidence = verified(root / name)
        require(evidence['status'] == 'complete', name + ' is incomplete')
        check_files(evidence['source_files'])
    for name in files:
        relative = str(Path(name).relative_to(REPO))
        subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=REPO,
                       capture_output=True, check=True)
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', relative], cwd=REPO,
                       capture_output=True, check=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
                            capture_output=True, check=True).stdout.strip()
    require(not old or commit != old['source_commit'], 'replacement requires a new committed source')
    control = root / 'study' / commit
    base = verified(root / 'BINDING.json')
    require(base['config_file_sha256'] == config['config_file_sha256'], 'scientific configuration changed')
    checkout = control / 'source'
    if not checkout.exists():
        control.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(['git', 'worktree', 'add', '--detach', str(checkout), commit],
                       cwd=REPO, capture_output=True, check=True)
    frozen_files = {}
    runtime_files = verified(root / 'STUDY_INSTALLED_REQUEST_API.json')['source_files']
    for name, expected in {**base['code_files'], **runtime_files, **files}.items():
        original = Path(name)
        candidate = checkout / original.relative_to(REPO) if original.is_relative_to(REPO) else original
        frozen = candidate if candidate.is_file() else original
        require(file_sha(frozen) == expected, 'source snapshot differs: ' + str(frozen))
        frozen_files[str(frozen)] = expected
    cfg = {**config, 'control_root': str(control), 'source_commit': commit, 'code_root': str(checkout)}
    binding = {k: value for k, value in base.items() if k != 'sha256'}
    binding.update(config=cfg, control_root=str(control), source_commit=commit,
                   code_files=frozen_files, study_files={str(checkout / Path(name).relative_to(REPO)): value for name, value in files.items()},
                   inherited_price_jobs=['60643321', '60643322'],
                   base_binding_sha256=base['sha256'], scientific_configuration_unchanged=True,
                   approved_gpu_repair_margin_inclusive_hours=config['budget']['gpu_repair_hours'],
                   canonical_manuscript_edits=False, remote_pushes=False)
    bound = publish(control / 'BINDING.json', binding)
    pointer = {'binding_path': str(control / 'BINDING.json'),
               'source_commit': commit, 'binding_sha256': bound['sha256']}
    if old:
        publish(control / 'REPLACES.json', {'previous_pointer': verified(root / 'CURRENT_STUDY.json'),
            'previous_terminal_sha256': terminal['sha256'], 'release_proof_sha256': proof['sha256'],
            'scientific_assignments_unchanged': True, 'prior_failures_preserved': True})
        pending = root / ('.CURRENT_STUDY-' + commit + '.pending.json')
        publish(pending, pointer)
        os.replace(pending, root / 'CURRENT_STUDY.json')
        require(verified(root / 'CURRENT_STUDY.json')['binding_sha256'] == bound['sha256'],
                'active release readback differs')
    else:
        publish(root / 'CURRENT_STUDY.json', pointer)
    return bound
