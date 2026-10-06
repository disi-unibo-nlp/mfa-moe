"""Compare freshly constructed R3-D specification before any fit starts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
CODE = ROOT / 'forum/tests/r3_context/code'
FROZEN = ROOT / 'forum/tests/r3_context/FROZEN.readout.v3.json'
OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/R3D_SPEC_AUDIT.json')


class Captured(Exception):
    pass


def diff_mapping(expected, actual):
    return {'added': sorted(set(actual) - set(expected)),
            'removed': sorted(set(expected) - set(actual)),
            'changed': sorted(key for key in set(actual) & set(expected)
                              if actual[key] != expected[key])}


def main() -> None:
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('spec audit requires CPU Slurm')
    sys.path.insert(0, str(CODE))
    import r3d_readout as readout
    frozen = json.loads(FROZEN.read_text())
    real_seal = readout.seal

    def capture(value):
        if value.get('schema') != 'r3d-clean-readout-v1':
            return real_seal(value)
        current = real_seal(value)
        result = {'schema': 'r3d-spec-audit-v1', 'job_id': os.environ['SLURM_JOB_ID'],
                  'frozen_sha256': frozen['sha256'], 'current_sha256': current['sha256'],
                  'differences': {key: diff_mapping(frozen[key], current[key])
                                  if isinstance(frozen.get(key), dict) and isinstance(current.get(key), dict)
                                  else {'expected': frozen.get(key), 'actual': current.get(key)}
                                  for key in sorted(set(frozen) | set(current))
                                  if frozen.get(key) != current.get(key)}}
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(result, indent=1) + '\n')
        print(json.dumps(result))
        raise Captured

    readout.seal = capture
    try:
        readout.run(4)
    except Captured:
        pass
    else:
        raise RuntimeError('readout did not construct a specification')


if __name__ == '__main__':
    main()
