"""CPU-only reproduction of vLLM-style spawned worker overlay import ordering."""
from __future__ import annotations

import multiprocessing as mp
import importlib.util
import os
from pathlib import Path
import sys

BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24')
OVERLAY = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
               'claude-analysis-2026-09-24/steering-v1/addenda/ordered/'
               '9727c10299b71e7a/moe_exp_src')


def probe(queue):
    try:
        import moe_exp
        import moe_exp.routing_control
        queue.put({'ok': True, 'source': str(Path(moe_exp.__file__).resolve())})
    except Exception as error:
        import moe_exp
        queue.put({'ok': False, 'error': type(error).__name__ + ': ' + str(error),
                   'source': str(Path(moe_exp.__file__).resolve())})


def spawned_result():
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    process = ctx.Process(target=probe, args=(queue,))
    process.start()
    result = queue.get(timeout=20)
    process.join(timeout=20)
    if process.exitcode != 0:
        raise RuntimeError('spawned worker exited abnormally')
    return result


if __name__ == '__main__':
    sys.path.insert(0, str(BASE))
    from moe_steer import engine
    runner_path = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/run_boundary_micro_screen.py'
    spec = importlib.util.spec_from_file_location('boundary_micro_spawn_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    env = engine.engine_env(None, None, plugin=False)
    env['PYTHONPATH'] = str(OVERLAY) + os.pathsep + env['PYTHONPATH']
    engine.apply_env(env)
    observed = spawned_result()
    if observed['ok'] or str(OVERLAY) in observed['source']:
        raise AssertionError('qualified engine env ordering did not reproduce worker failure: ' + str(observed))
    runner.prepare_worker_import_path(engine, env, OVERLAY)
    fixed = spawned_result()
    if not fixed['ok'] or str(OVERLAY) not in fixed['source']:
        raise AssertionError('overlay-first spawn repair failed: ' + str(fixed))
    print({'qualified_engine_env_reproduction': observed, 'overlay_first_fix': fixed})
