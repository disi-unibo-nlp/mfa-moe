"""Separately bound serial/eager execution of the unchanged v3 score fixture."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def serial_kwargs(original, *args, **kwargs):
    return original(*args, **{**kwargs, 'max_num_seqs': 1, 'enforce_eager': True})


def validate(args):
    value = json.loads(args.manifest.read_text())
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('serial calibration manifest changed')
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if value['entry_driver_sha256'] != actual or value['engine_overrides'] != {
        'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}:
        raise ValueError('serial/eager execution binding differs')
    if os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0':
        raise ValueError('serial calibration retains the original non-batch-invariant backend')
    for path, expected in value['code_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('bound serial calibration source changed')
    spec = importlib.util.spec_from_file_location('score_base_v3', value['driver'])
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    if len(base.cases(value)) != 80:
        raise ValueError('same 80 score assignments required')
    return value, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    value, base = validate(args)
    from moe_steer import engine, policies as P
    actual = serial_kwargs(engine.engine_kwargs, plugin=True, max_num_seqs=16, return_routed_experts=True)
    table = P.PolicyTable.from_sealed(base.H.sealed(value['policy_table']))
    if actual['max_num_seqs'] != 1 or not actual['enforce_eager'] or table.hooked_layers() != (28,):
        raise ValueError('serial execution or native hook table invalid')
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_SERIAL_EAGER_CPU_PREFLIGHT', 'requests': 80,
            'max_num_seqs': actual['max_num_seqs'], 'enforce_eager': actual['enforce_eager'],
            'policy_table_sha256': table.digest(), 'manifest_sha256': value['sha256']}))
        return
    original = engine.engine_kwargs
    engine.engine_kwargs = lambda *a, **kw: serial_kwargs(original, *a, **kw)
    try:
        base.run(args)
    finally:
        engine.engine_kwargs = original
        path = args.out / 'QUALIFICATION.json'
        if path.exists():
            result = base.H.sealed(path)
            body = {k: v for k, v in result.items() if k != 'sha256'}
            body.update(schema='routing-score-api-calibration-result-v4-serial',
                engine_overrides=value['engine_overrides'], entry_driver_sha256=value['entry_driver_sha256'],
                preserved_v3_failure_sha256=value['preserved_v3_failure_sha256'])
            base.R.atomic_json(path, {**body, 'sha256': digest(body)})


if __name__ == '__main__':
    main()
