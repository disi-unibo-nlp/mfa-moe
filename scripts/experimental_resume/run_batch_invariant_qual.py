"""Isolated vLLM batch-invariant engineering qualification on frozen pilot cases.

The source six-arm runner and ordered worker are unchanged. This wrapper binds
VLLM_BATCH_INVARIANT=1 to a new manifest/output and checks the environment
before model load. Passing this stage would qualify a distinct engine regime.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
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


def preflight():
    if os.environ.get('VLLM_BATCH_INVARIANT') != '1':
        raise ValueError('batch-invariant qualification flag is not enabled')
    if importlib.metadata.version('vllm') != '0.29.0+cu129':
        raise ValueError('installed vLLM wheel changed')
    # A package import pulls in Transformers/torchvision and is too expensive
    # for a login-node preflight. Inspect the exact installed wheel's source;
    # actual backend choice and correctness are tested on Slurm GPUs.
    vllm = Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    expert_dir = vllm / 'model_executor/layers/fused_moe/experts'
    for filename, classname in (('fused_humming_moe.py', 'HummingExpertsBase'),
                                ('triton_moe.py', 'TritonExperts')):
        tree = ast.parse((expert_dir / filename).read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == classname]
        supported = any(isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                        method.name == '_supports_batch_invariance' and
                        any(isinstance(ret, ast.Return) and isinstance(ret.value, ast.Constant)
                            and ret.value.value is True for ret in ast.walk(method))
                        for node in classes for method in node.body)
        if not supported:
            raise ValueError(f'{classname} batch-invariant declaration missing')
    if importlib.util.find_spec('humming') is None:
        raise ValueError('Humming package not installed')


def validate_bi(manifest, driver_path):
    if manifest.get('schema') != 'routing-batch-invariant-qual-v1':
        raise ValueError('unknown batch-invariant qualification manifest')
    if manifest.get('bi_driver_sha256') != base.file_sha(__file__):
        raise ValueError('batch-invariant wrapper code changed')
    if manifest.get('batch_invariant_flag') != '1':
        raise ValueError('batch-invariant manifest flag differs')
    pilot = base.sealed(PILOT)
    if manifest.get('source_pilot_sha256') != pilot['sha256']:
        raise ValueError('qualified pilot source changed')
    for key, value in pilot.items():
        if key in ('sha256', 'schema', 'code_files'):
            continue
        if manifest.get(key) != value:
            raise ValueError(f'frozen six-arm pilot field changed: {key}')
    codes = manifest.get('code_files', {})
    if codes.get(str(Path(__file__))) != base.file_sha(__file__):
        raise ValueError('wrapper absent from exact source hashes')
    if any(codes.get(path) != expected for path, expected in pilot['code_files'].items()):
        raise ValueError('pilot code or ordered overlay changed')
    preflight()
    original_view = {**manifest, 'schema': pilot['schema']}
    rows, actions, arms = ORIGINAL_VALIDATE(original_view, driver_path)
    table = base.build_policy_table(actions)
    source_table = json.loads((PILOT_RUN / 'policy-table.json').read_text())
    if table.hooked_layers() != (28,) or table.sealed() != source_table:
        raise ValueError('batch-invariant qualifier did not retain pilot hook table')
    return rows, actions, arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()
    base.validate_manifest = validate_bi
    base.run(args)


if __name__ == '__main__':
    main()
