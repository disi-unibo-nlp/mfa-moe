"""Frozen scientific configuration and one source manifest per released commit."""
from pathlib import Path
import subprocess
from .common import REPO, MODELS, CLASSES, check_files, file_sha, load, private, publish, require, verified


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


def initialize(config):
    old = active(config)
    if old is not None:
        require(old['config_file_sha256'] == config['config_file_sha256'], 'released configuration differs; use the frozen configuration')
        check_files(old['study_files'])
        return old
    root = Path(config['root'])
    for name in ('STUDY_READY.json', 'STUDY_REVIEWS.json', 'STUDY_INSTALLED_REQUEST_API.json'):
        proof = verified(root / name)
        require(proof['status'] == 'complete', name + ' is incomplete')
        check_files(proof['source_files'])
    files = study_files()
    for name in files:
        relative = str(Path(name).relative_to(REPO))
        subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=REPO,
                       capture_output=True, check=True)
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', relative], cwd=REPO,
                       capture_output=True, check=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
                            capture_output=True, check=True).stdout.strip()
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
    publish(root / 'CURRENT_STUDY.json', {'binding_path': str(control / 'BINDING.json'),
            'source_commit': commit, 'binding_sha256': bound['sha256']})
    return bound
