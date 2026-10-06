"""Freeze the corrected saved-results report driver without changing the first snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/paper/resume-v1')


def run():
    files = {'paper_package.py': REPO/'scripts/experimental_resume/paper_package.py',
             'uncertainty.py': REPO/'src/moe_exp/routing_control/uncertainty.py'}
    hashes = {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in files.items()}
    tree = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    directory = ROOT / ('code-final-' + tree[:16])
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        destination = directory / name
        data = source.read_bytes()
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError('immutable final paper driver changed: ' + name)
        if not destination.exists():
            destination.write_bytes(data)
    manifest = {'schema': 'resume-paper-final-code-v1', 'directory': str(directory),
                'driver': str(directory/'paper_package.py'), 'tree_sha256': tree,
                'files': hashes}
    path = ROOT/'CODE_FINAL.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('different final report code already frozen')
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=1)+'\n')
    print(json.dumps({'path': str(path), 'tree_sha256': tree, 'driver': manifest['driver']}))


if __name__ == '__main__':
    run()
