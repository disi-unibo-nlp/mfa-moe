"""Preserve every hashed input of a completed report before mutable audits refresh."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
POINTER = REPO/'report/experimental-resume-v1/PAPER_SNAPSHOT.json'


def run():
    pointer = json.loads(POINTER.read_text())
    out = Path(pointer['path'])
    frozen = json.loads((out/'FROZEN.json').read_text())
    if str(frozen['job_id']) != str(pointer['job_id']):
        raise ValueError('report pointer and job binding differ')
    directory = out/'input-archives'
    directory.mkdir(exist_ok=True)
    entries = {}
    for index, (source, expected) in enumerate(frozen['inputs'].items()):
        original = Path(source)
        data = original.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('paper input changed before archival: ' + source)
        target = directory/(f'{index:02d}-{expected[:12]}-{original.name}')
        if target.exists() and target.read_bytes() != data:
            raise ValueError('archived paper input changed: ' + str(target))
        if not target.exists():
            target.write_bytes(data)
        entries[source] = {'archive': str(target), 'sha256': expected, 'bytes': len(data)}
    body = {'schema': 'resume-paper-input-archives-v1', 'paper_job_id': pointer['job_id'],
            'paper_frozen_sha256': hashlib.sha256((out/'FROZEN.json').read_bytes()).hexdigest(),
            'inputs': entries}
    path = out/'INPUT_ARCHIVES.json'
    if path.exists() and json.loads(path.read_text()) != body:
        raise ValueError('a different input archive inventory already exists')
    if not path.exists():
        path.write_text(json.dumps(body, indent=1)+'\n')
    print(json.dumps({'path': str(path), 'files': len(entries),
                      'bytes': sum(e['bytes'] for e in entries.values())}))


if __name__ == '__main__':
    run()
