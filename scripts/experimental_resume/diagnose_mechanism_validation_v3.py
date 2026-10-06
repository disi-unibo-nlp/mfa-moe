"""Pin the qualified worker while exposing repository-only validation modules.

The qualified overlay is intentionally sparse. Its worker must resolve from
that overlay, while manifest validation needs repository-only detector code.
This entry point makes both locations visible in that order, verifies the
resolved files, then runs the unchanged manifest-bound generation driver.
"""
from __future__ import annotations


def pin_qualified_worker(overlay):
    """Keep the worker frozen and permit repository-only validation imports."""
    import importlib
    import importlib.util
    from pathlib import Path
    import sys

    overlay = Path(overlay).resolve()
    source_package = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/src/moe_exp').resolve()
    if 'moe_exp' in sys.modules:
        raise RuntimeError('moe_exp was cached before qualified overlay pin')
    sys.path[:] = [str(overlay), *(path for path in sys.path if path != str(overlay))]
    import moe_exp
    expected_package = overlay / 'moe_exp'
    expected_worker = expected_package / 'routing_control/ordered_vllm.py'
    if Path(moe_exp.__file__).resolve().parent != expected_package:
        raise RuntimeError('qualified package did not resolve from overlay')
    # The overlay contains the qualified worker but predates the frozen
    # detector. Extend only package search paths; never replace worker files.
    moe_exp.__path__.append(str(source_package))
    routing = importlib.import_module('moe_exp.routing_control')
    expected_routing = expected_package / 'routing_control'
    if Path(routing.__file__).resolve().parent != expected_routing:
        raise RuntimeError('qualified routing package did not resolve from overlay')
    routing.__path__.append(str(source_package / 'routing_control'))
    worker_spec = importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
    detector_spec = importlib.util.find_spec('moe_exp.routing_control.transitions_v2')
    if (worker_spec is None or worker_spec.origin is None or
            Path(worker_spec.origin).resolve() != expected_worker):
        raise RuntimeError('qualified worker package did not resolve from overlay')
    if (detector_spec is None or detector_spec.origin is None or
            Path(detector_spec.origin).resolve() != source_package / 'routing_control/transitions_v2.py'):
        raise RuntimeError('repository detector cannot be resolved for manifest validation')
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
