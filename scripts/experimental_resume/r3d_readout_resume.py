"""Resume the frozen R3-D driver with a JSON-equivalent model-order value.

The frozen v3 driver has a tuple for MODEL_ORDER. Its specification is saved as
JSON (a list), then compared with in-memory Python equality on resume. The
canonical specification seal is identical; the tuple/list equality alone fails.
This wrapper changes that in-memory representation before invoking the same
frozen driver. Its code and checkpoint binding remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


CODE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code')
FROZEN = CODE.parent / 'FROZEN.readout.v3.json'
DRIVER_SHA256 = '1de08eae6e8d85e352d091c829b33598d2a848b2a9b28f1937a14d687d749037'


def main() -> None:
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('R3-D readout requires CPU Slurm')
    driver = CODE / 'r3d_readout.py'
    if hashlib.sha256(driver.read_bytes()).hexdigest() != DRIVER_SHA256:
        raise ValueError('frozen R3-D driver changed')
    sys.path.insert(0, str(CODE))
    import r3d_readout as readout
    original = readout.MODEL_ORDER
    if not isinstance(original, tuple):
        raise ValueError('expected historical tuple/list serialization bug')
    frozen = json.loads(FROZEN.read_text())
    if list(original) != frozen['models']:
        raise ValueError('model order differs from frozen specification')
    readout.MODEL_ORDER = list(original)
    readout.run(4)


if __name__ == '__main__':
    main()
