"""Identify changed files in the frozen R3-D readout specification on CPU Slurm."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
FROZEN = ROOT / 'forum/tests/r3_context/FROZEN.readout.v3.json'
OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/R3D_FROZEN_INPUT_AUDIT.json')


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('audit frozen inputs in a CPU Slurm allocation')
    specification = json.loads(FROZEN.read_text())
    mismatches = []
    count = 0
    for kind in ('inputs', 'code'):
        for name, expected in specification[kind].items():
            path = Path(name)
            observed = sha(path) if path.is_file() else None
            count += 1
            if observed != expected:
                mismatches.append({'kind': kind, 'path': name, 'expected': expected,
                                   'observed': observed})
    result = {'schema': 'r3d-frozen-input-audit-v1', 'job_id': os.environ['SLURM_JOB_ID'],
              'frozen_sha256': sha(FROZEN), 'files_checked': count, 'mismatches': mismatches}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'files_checked': count, 'mismatches': mismatches}))


if __name__ == '__main__':
    main()
