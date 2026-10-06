"""Separately bound serial/eager replay of the frozen six-arm engineering pilot.

This qualifies a new execution profile only. Existing MARLIN model weights,
ordered worker and action table remain exact; semantic pilot starts are invalid
and are not used for an efficacy claim. One vLLM sequence executes at a time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
PILOT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
PILOT_RUN = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
                 'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
                 'micro-screen-qual4-ac4c9651e71fe067')
ORIGINAL_VALIDATE = base.validate_manifest
ENGINE_OVERRIDES = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}


def serial_kwargs(original, *args, **kwargs):
    return original(*args, **{**kwargs, 'max_num_seqs': 1, 'enforce_eager': True})


def validate_serial(manifest, driver_path):
    if manifest.get('schema') != 'routing-boundary-serial-engine-qual-v1':
        raise ValueError('unknown serial qualification manifest')
    if manifest.get('entry_driver_sha256') != base.file_sha(__file__):
        raise ValueError('serial entry code changed')
    if manifest.get('engine_overrides') != ENGINE_OVERRIDES:
        raise ValueError('serial/eager execution profile changed')
    if os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0':
        raise ValueError('serial qualification retains original non-BI MARLIN backend')
    pilot = base.sealed(PILOT)
    if manifest.get('source_pilot_sha256') != pilot['sha256']:
        raise ValueError('frozen pilot source changed')
    for key, value in pilot.items():
        if key in ('sha256', 'schema', 'code_files'):
            continue
        if manifest.get(key) != value:
            raise ValueError(f'frozen pilot field changed: {key}')
    codes = manifest.get('code_files', {})
    if codes.get(str(Path(__file__))) != base.file_sha(__file__) or any(
            codes.get(path) != expected for path, expected in pilot['code_files'].items()):
        raise ValueError('serial entry, base driver or qualified worker source changed')
    original_view = {**manifest, 'schema': pilot['schema']}
    rows, actions, arms = ORIGINAL_VALIDATE(original_view, driver_path)
    table = base.build_policy_table(actions)
    if table.hooked_layers() != (28,) or table.sealed() != json.loads((PILOT_RUN / 'policy-table.json').read_text()):
        raise ValueError('serial qualification must use exact frozen pilot action table')
    return rows, actions, arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate_serial(manifest, base.__file__)
    from moe_steer import engine
    kwargs = serial_kwargs(engine.engine_kwargs, plugin=True, max_num_seqs=48,
                           return_routed_experts=True)
    if kwargs['max_num_seqs'] != 1 or not kwargs['enforce_eager']:
        raise ValueError('vLLM serial/eager kwargs differ')
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_SERIAL_EAGER_CPU_PREFLIGHT',
                          'requests': len(rows) * len(arms) * len(manifest['seeds']),
                          'manifest_sha256': manifest['sha256'],
                          'engine_overrides': ENGINE_OVERRIDES}))
        return
    original = engine.engine_kwargs
    base.validate_manifest = validate_serial
    engine.engine_kwargs = lambda *a, **kw: serial_kwargs(original, *a, **kw)
    try:
        base.run(args)
    finally:
        engine.engine_kwargs = original
        base.validate_manifest = ORIGINAL_VALIDATE


if __name__ == '__main__':
    main()
