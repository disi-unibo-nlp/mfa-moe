#!/usr/bin/env python3
"""Run one frozen v3 CPU entrypoint with verified parent/worker import origins."""
import argparse
import importlib
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--module', choices=('measurement_study.production_plan',
        'measurement_study.production_analyze', 'measurement_study.attach_labels',
        'measurement_study.capture_gate'), required=True)
    a, remaining = p.parse_known_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Production CPU work requires Slurm')
    for path in (a.bundle / 'src', a.bundle / 'scripts', a.bundle):
        sys.path.insert(0, str(path.resolve()))
    from measurement_study.bundle import worker_init, verify_loaded_imports
    worker_init(a.bundle)
    module = importlib.import_module(a.module)
    verify_loaded_imports(a.bundle)
    # The outer parser consumes --bundle; forward it to entrypoints that
    # require their own explicit bundle argument.
    if a.module != 'measurement_study.attach_labels':
        remaining += ['--bundle', str(a.bundle.resolve())]
    sys.argv = [a.module, *remaining]
    result = module.main()
    verify_loaded_imports(a.bundle)
    return result

if __name__ == '__main__':
    raise SystemExit(main())
