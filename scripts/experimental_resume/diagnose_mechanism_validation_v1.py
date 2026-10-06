"""Expose a swallowed generation exception without changing sealed drivers.

This real .py entry point is required because vLLM v1 re-executes the entry
script. It imports the frozen runner as a module, so its __file__ and all
manifest-bound source hashes remain unchanged. On failure, the qualified
finish_child still performs its normal shutdown and exit after the traceback.
"""
from __future__ import annotations


def main():
    import sys
    import traceback

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
    runner.main()


if __name__ == '__main__':
    main()
