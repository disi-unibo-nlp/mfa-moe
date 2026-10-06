"""Freeze authorized code addenda while preserving the old keeper's latest pointer.

This performs no Slurm submission and does not wake the drained keeper.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path(__file__).resolve().parents[2]


def freeze():
    if not (ROOT / 'swarm/DRAIN').exists() or 'DRAIN=yes' not in (ROOT / 'swarm/STATUS.md').read_text():
        raise ValueError('old keeper must remain drained')
    campaign = ROOT / 'steering-v1'
    snap = campaign / 'code/s1-f2ded3957eb54fd5'
    spec = importlib.util.spec_from_file_location('freeze_runtime', snap / 'scripts/freeze_steer.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    freezer.verify(snap)
    pointer = campaign / 'code/latest.json'
    original_pointer = pointer.read_bytes()
    frozen = {}
    try:
        for name, directory in [('s2', campaign / 'addenda/s2'),
                                ('h14', campaign / 'qualification/h14/proposal')]:
            proposal = json.loads((directory / 'PROPOSED_TREE.json').read_text())
            result = freezer.freeze(directory / 'code-root', snap / 'moe_exp_src', campaign)
            frozen[name] = freezer.verify(result['snapshot'], proposal['proposal_tree'])
    finally:
        pointer.write_bytes(original_pointer)
    assert pointer.read_bytes() == original_pointer
    source = {'native_nll.py': REPO / 'scripts/experimental_resume/native_nll.py',
              'nll_helpers.py': REPO / 'src/moe_exp/routing_control/nll.py',
              'receipts.py': REPO / 'src/moe_exp/routing_control/receipts.py'}
    table = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source.items()}
    nll_tree = freezer.digest(table)
    dest = campaign / 'addenda/nll' / ('nll-' + nll_tree[:16])
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in source.items():
        target = dest / name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('immutable NLL addendum content differs')
        if not target.exists():
            shutil.copy2(path, target)
            target.chmod(0o400)
    manifest = {'schema': 'native-nll-addendum-v1', 'tree_sha256': nll_tree, 'files': table,
        'generation_tree': frozen['s2']['tree_sha256'], 'measurement': 'native plugin-free logprobs',
        'qualification': 'PENDING actual Q3 fixture rescore',
        'authorization_source': 'current user instruction to implement the experimental resume plan'}
    manifest = freezer.seal(manifest)
    path = dest / 'MANIFEST.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('immutable NLL inventory differs')
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=1) + '\n')
        path.chmod(0o400)
    frozen['nll'] = {'path': str(dest), 'tree_sha256': nll_tree, 'seal': manifest['sha256']}
    dest.chmod(0o500)
    output = campaign / 'runs/s2prop/FROZEN_ADDENDA.json'
    output.write_text(json.dumps({'snapshots': frozen, 'latest_pointer_preserved': True,
        'authorization': 'Implementation of named legacy work within the ceilings in the current user plan; '
                         'the separate 10.5 GPU-hour study remains proposed.'}, indent=1) + '\n')
    assert json.loads(output.read_text())['snapshots'] == frozen
    print(json.dumps(frozen, indent=1))


if __name__ == '__main__':
    freeze()
