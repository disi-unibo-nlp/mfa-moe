"""Frozen campaign-v3 source closure and import verification (stdlib only)."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

# Top-level modules and packages of the project: everything directly under src/.
SRC = Path(__file__).resolve().parents[1]
PROJECT = {path.stem for path in SRC.iterdir() if path.suffix == '.py' or (path / '__init__.py').exists()}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle path escapes root: ' + str(relative))
    return path


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = bundle / 'SHA256.json'
    hashes = json.loads(manifest.read_text())
    if not hashes or not isinstance(hashes, dict):
        raise ValueError('Empty or invalid bundle manifest')
    for rel, expected in hashes.items():
        path = _inside(bundle, rel)
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError('Frozen code changed: ' + rel)
    campaign_path = bundle / 'campaign-v3.json'
    if campaign_path.exists():
        if 'campaign-v3.json' not in hashes:
            raise ValueError('Campaign metadata is not bound to source manifest')
        campaign = json.loads(campaign_path.read_text())
        if campaign.get('schema_version') != 3:
            raise ValueError('Expected campaign-v3 schema')
        for rel, expected in campaign.get('runtime_code_hashes', {}).items():
            if hashes.get(rel) != expected or file_sha(_inside(bundle, rel)) != expected:
                raise ValueError('Unbound runtime source: ' + rel)
        for path, expected in campaign.get('external_runtime_hashes', {}).items():
            if file_sha(path) != expected:
                raise ValueError('Installed runtime dependency changed: ' + path)
    else:
        campaign_path = bundle / 'campaign.json'
        campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    return {'bundle_sha256': file_sha(manifest),
            'campaign_sha256': file_sha(campaign_path) if campaign_path.exists() else None,
            'files': hashes, 'campaign': campaign}


def verify_imports(bundle, modules):
    bundle = Path(bundle).resolve()
    hashes = json.loads((bundle / 'SHA256.json').read_text())
    origins = {}
    for name in modules:
        module = importlib.import_module(name) if isinstance(name, str) else name
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError('Import escaped frozen bundle: ' + module.__name__ + ': ' + str(path))
        rel = path.relative_to(bundle).as_posix()
        if rel not in hashes or file_sha(path) != hashes[rel]:
            raise ValueError('Imported module hash changed: ' + module.__name__)
        origins[module.__name__] = {'path': str(path), 'sha256': hashes[rel]}
    return origins


def verify_loaded_imports(bundle):
    """Check the entire imported project closure, not just an entrypoint."""
    names = [name for name, module in list(sys.modules.items())
             if module is not None and getattr(module, '__file__', None)
             and (name.split('.')[0] in PROJECT or name == 'measurement_study' or name.startswith('measurement_study.')
                  or name in ('analyze_available_metrics', 'analyze_pilot_metrics'))]
    return verify_imports(bundle, names)


def worker_init(bundle):
    bundle = Path(bundle).resolve()
    for path in (bundle / 'src', bundle / 'scripts', bundle):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    os.environ['MEASUREMENT_BUNDLE'] = str(bundle)
    result = verify_bundle(bundle)
    if (bundle / 'campaign-v3.json').exists():
        result['imports'] = verify_loaded_imports(bundle)
    return result


def import_frozen_helper(name, bundle=None):
    if not name.isidentifier():
        raise ValueError('Expected a simple helper module name')
    root = Path(bundle or os.environ.get('MEASUREMENT_BUNDLE') or Path(__file__).resolve().parents[2]).resolve()
    script = root / 'scripts' / (name + '.py')
    if not script.is_file():
        raise ValueError('Missing bundle-local helper: ' + str(script))
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() != script:
        raise ValueError('Helper already imported from another source: ' + name)
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    module = importlib.import_module(name)
    if (root / 'campaign-v3.json').exists():
        verify_imports(root, [name])
    return module


def source_tree_hash(root):
    root = Path(root).resolve()
    files = {path.relative_to(root).as_posix(): file_sha(path)
             for path in sorted(root.rglob('*'))
             if path.is_file() and path.suffix in ('.py', '.sh') and '__pycache__' not in path.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze_bundle(source_root, repo_src, destination, campaign, t1_receipt):
    """Publish a new immutable closure only after an explicit T1 receipt."""
    source_root, repo_src, destination = map(lambda p: Path(p).resolve(), (source_root, repo_src, destination))
    receipt = json.loads(Path(t1_receipt).read_text())
    if receipt.get('status') not in ('approved', 'passed', 'complete') or str(receipt.get('reviewer', '')).lower() != 'claude':
        raise ValueError('Claude source-final T1 receipt required before freezing')
    if receipt.get('repo_src_sha256') != source_tree_hash(repo_src):
        raise ValueError('T1 receipt does not bind the current reviewed repo/src tree')
    if destination.exists():
        raise FileExistsError('Never modify an existing bundle: ' + str(destination))
    if campaign.get('schema_version') != 3:
        raise ValueError('Campaign schema_version=3 required')
    repo_tree_before = source_tree_hash(repo_src)
    sources = {}
    for prefix, tree in [('measurement_study', source_root / 'measurement_study'), ('scripts', source_root / 'scripts'), ('src', repo_src)]:
        for path in sorted(tree.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh') and '__pycache__' not in path.parts:
                if path.is_symlink():
                    raise ValueError('Source symlink must be resolved explicitly: ' + str(path))
                sources[(Path(prefix) / path.relative_to(tree)).as_posix()] = (path, file_sha(path))
    # These packages belong to the routing-study worktree. Repo/src supplies
    # generation/scoring/native adapters; the worktree versions overlay them.
    # JSON files are part of the runtime closure: code_hashes() binds
    # routing_study/checkpoint_files.json and the guiding packages are imported
    # while loading the frozen routing-study package.
    overlays = ('routing_study', 'moe_margin_guiding', 'moe_identity_guiding')
    overlay_before = {}
    for name in overlays:
        overlay = source_root / 'src' / name
        if not overlay.is_dir():
            raise ValueError('Missing routing-study package: src/' + name)
        overlay_before[name] = source_tree_hash(overlay)
        for path in sorted(overlay.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh', '.json') and '__pycache__' not in path.parts:
                if path.is_symlink():
                    raise ValueError('Source symlink must be resolved explicitly: ' + str(path))
                rel = (Path('src') / name / path.relative_to(overlay)).as_posix()
                sources[rel] = (path, file_sha(path))
    # Root files that src/routing_study/common.py code_hashes() binds.
    # scripts/* already came from the worktree scripts tree above.
    for name in ('pyproject.toml', 'scripts/run_routing_study.py', 'scripts/verify_routing_study.py',
                 'scripts/routing_study_env.sh', 'sbatch/routing_study_smoke.sbatch'):
        path = source_root / name
        if not path.is_file():
            raise ValueError('Missing bundle closure file required by code_hashes(): ' + name)
        sources.setdefault(name, (path, file_sha(path)))
    required = {'src/routing_study/checkpoint_files.json',
                'src/routing_study/reducer_provenance.json',
                'src/moe_identity_guiding/execution.py',
                'src/moe_identity_guiding/calibration.py',
                'src/moe_identity_guiding/routing.py',
                'src/moe_margin_guiding/calibration.py',
                'src/moe_margin_guiding/routing.py',
                'src/correlation_pipeline/scoring.py',
                'src/utils.py'}
    missing = required - set(sources)
    if missing:
        raise ValueError('Missing runtime closure files: ' + ','.join(sorted(missing)))
    if not any(k.startswith('src/') for k in sources):
        raise ValueError('Missing repo/src source closure')
    staging = destination.with_name(destination.name + '.partial')
    if staging.exists():
        raise FileExistsError('Existing interrupted bundle staging must be reviewed: ' + str(staging))
    staging.mkdir(parents=True)
    try:
        for rel, (source, expected) in sources.items():
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_sha(target) != expected or file_sha(source) != expected:
                raise ValueError('Source changed during freeze: ' + str(source))
        # Detect edits to files copied early while later files were being copied.
        if (source_tree_hash(repo_src) != repo_tree_before
                or any(source_tree_hash(source_root / 'src' / name) != digest
                       for name, digest in overlay_before.items())
                or any(file_sha(path) != expected for path, expected in sources.values())):
            raise ValueError('Source tree changed during freeze')
        metadata = dict(campaign, source_root=str(source_root), repo_src=str(repo_src),
                        t1_receipt_sha256=file_sha(t1_receipt), reviewed_repo_src_sha256=repo_tree_before,
                        routing_study_overlay_sha256=overlay_before['routing_study'],
                        guiding_overlay_sha256={name: digest for name, digest in overlay_before.items() if name != 'routing_study'},
                        source_origins={rel: str(path) for rel, (path, _) in sources.items()},
                        runtime_code_hashes={rel: digest for rel, (_, digest) in sources.items()})
        (staging / 'campaign-v3.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Compatibility name is frozen too; it never resolves to the old live campaign.
        (staging / 'campaign.json').write_text(json.dumps(metadata, indent=2) + '\n')
        hashes = {rel: digest for rel, (_, digest) in sources.items()}
        for rel in ('campaign-v3.json', 'campaign.json'):
            hashes[rel] = file_sha(staging / rel)
        (staging / 'SHA256.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
        verify_bundle(staging)
        staging.rename(destination)
    except BaseException:
        # Retain interrupted staging for review; never disturb an older bundle.
        raise
    return verify_bundle(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--repo-src', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--t1-receipt', type=Path, required=True)
    a = p.parse_args()
    result = freeze_bundle(a.source_root, a.repo_src, a.destination, json.loads(a.campaign.read_text()), a.t1_receipt)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'campaign')}))

if __name__ == '__main__':
    main()
