"""Pin the qualified worker package before importing the frozen runner.

The v1 traceback identified a cached moe_exp package outside the qualified
overlay. This real Python entry point imports and verifies the overlay first,
then runs the unchanged manifest-bound runner with traceback-on-exit logging.
"""
from __future__ import annotations


def pin_qualified_worker(overlay):
    """Load the worker package and verify its frozen source before other imports."""
    import importlib.util
    from pathlib import Path
    import sys

    overlay = Path(overlay).resolve()
    if 'moe_exp' in sys.modules:
        raise RuntimeError('moe_exp was cached before qualified overlay pin')
    sys.path[:] = [str(overlay), *(path for path in sys.path if path != str(overlay))]
    import moe_exp
    expected_package = overlay / 'moe_exp'
    expected_worker = expected_package / 'routing_control/ordered_vllm.py'
    worker_spec = importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
    if (Path(moe_exp.__file__).resolve().parent != expected_package or
            worker_spec is None or worker_spec.origin is None or
            Path(worker_spec.origin).resolve() != expected_worker):
        raise RuntimeError('qualified worker package did not resolve from overlay')
    return expected_package


def main():
    import os
    from pathlib import Path
    import socket
    import sys
    import traceback

    overlay = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
                   'claude-analysis-2026-09-24/steering-v1/addenda/ordered/'
                   '9727c10299b71e7a/moe_exp_src').resolve()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('qualified worker import requires GPU Slurm compute node')
    if '--overlay' not in sys.argv or Path(sys.argv[sys.argv.index('--overlay') + 1]).resolve() != overlay:
        raise ValueError('runner overlay argument differs from qualified worker')
    expected_package = pin_qualified_worker(overlay)

    from moe_steer import qualify
    original = qualify.finish_child

    def show_exception_then_finish(code, driver=None):
        if code:
            kind, error, trace = sys.exc_info()
            if error is None:
                print('diagnostic: nonzero child exit without active Python exception',
                      file=sys.stderr, flush=True)
            else:
                print('diagnostic: caught generation exception:',
                      file=sys.stderr, flush=True)
                traceback.print_exception(kind, error, trace, file=sys.stderr)
                sys.stderr.flush()
        return original(code, driver)

    qualify.finish_child = show_exception_then_finish
    import run_mechanism_validation_v1 as runner
    if Path(sys.modules['moe_exp'].__file__).resolve().parent != expected_package:
        raise RuntimeError('runner imports rebound qualified moe_exp package')
    runner.main()


if __name__ == '__main__':
    main()
