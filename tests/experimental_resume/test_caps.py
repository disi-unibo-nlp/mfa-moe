"""T1/T2 cap regression against real sealed questions (run on the proposal PYTHONPATH)."""
import copy
import json

import pytest


def test_production_completion_requires_cap_or_eos_and_matching_rows():
    import numpy as np
    from types import SimpleNamespace
    from guarded_runner import check_completion
    from moe_steer import runner, results
    request = {'uid': 'test', 'max_tokens': 4}
    flight = SimpleNamespace(request=request)
    policy = SimpleNamespace(operator=SimpleNamespace(kind='none'))
    fin = SimpleNamespace(token_ids=[1, 2, 3, 4], finish_reason='length',
                          routed_experts=np.zeros((4, 40, 8), np.uint8))
    check_completion(fin, flight, policy=policy, fsm_factory=None, runner=runner, results=results)
    fin.token_ids = [1, 248046]
    fin.finish_reason = 'stop'
    fin.routed_experts = np.zeros((2, 40, 8), np.uint8)
    check_completion(fin, flight, policy=policy, fsm_factory=None, runner=runner, results=results)
    fin.token_ids = [1, 2]
    with pytest.raises(runner.RunnerError, match='cap'):
        check_completion(fin, flight, policy=policy, fsm_factory=None, runner=runner, results=results)
    fin.token_ids = [1, 248046]
    fin.routed_experts = np.zeros((1, 40, 8), np.uint8)
    with pytest.raises(runner.RunnerError, match='routed rows'):
        check_completion(fin, flight, policy=policy, fsm_factory=None, runner=runner, results=results)

pytest.importorskip('moe_steer', reason='cap qualification requires the external s2 proposal PYTHONPATH')
from moe_steer import manifests as M, policies
from moe_steer.spec import seal
import x2_build as X


def test_T4_output_binding_rejects_changed_cap_code_and_unbound_records(workdir):
    import importlib.util
    from pathlib import Path
    path = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/guarded_runner.py')
    spec = importlib.util.spec_from_file_location('guarded_runner', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    manifest = M.load_manifest(policies.CAMPAIGN / 'manifests/q11-v2.json')
    guard.bind_output(manifest, workdir, wrapper_sha256='a' * 64)
    guard.bind_output(manifest, workdir, wrapper_sha256='a' * 64)
    changed = copy.deepcopy(manifest)
    changed.pop('sha256')
    changed['requests'][0]['max_tokens'] = 1024
    changed = seal(changed)
    assert changed['requests'][0]['uid'] == manifest['requests'][0]['uid']
    with pytest.raises(ValueError, match='different manifest'):
        guard.bind_output(changed, workdir, wrapper_sha256='a' * 64)
    with pytest.raises(ValueError, match='different manifest'):
        guard.bind_output(manifest, workdir, wrapper_sha256='b' * 64)
    dirty = workdir / 'dirty'
    dirty.mkdir()
    (dirty / 'shard-0.complete.json').write_text('{}')
    with pytest.raises(ValueError, match='unbound nonempty'):
        guard.bind_output(manifest, dirty, wrapper_sha256='a' * 64)
    interrupted = workdir / 'interrupted'
    interrupted.mkdir()
    (interrupted / '.RUN_BINDING.pending').write_text('{')
    guard.bind_output(manifest, interrupted, wrapper_sha256='a' * 64)
    assert json.loads((interrupted / 'RUN_BINDING.json').read_text())['manifest_sha256'] == manifest['sha256']


@pytest.mark.parametrize('name', ['x1-v1', 'q11-v1', 'q11-v2'])
def test_T1_sealed_manifests(name):
    path = policies.CAMPAIGN / 'manifests' / f'{name}.json'
    before = path.read_bytes()
    manifest = M.load_manifest(path)
    M.validate_manifest(manifest)
    stored = json.loads(before)
    assert [(r['uid'], r['max_tokens']) for r in manifest['requests']] == [
        (r['uid'], r['max_tokens']) for r in stored['requests']]
    assert path.read_bytes() == before


def real_requests():
    world = M.load_world()
    table, _ = X.x2_table(world.inputs)
    rows = [r for r in json.loads((policies.CAMPAIGN / 'runs/x2prep/eligibility.json').read_text())['rows']
            if r['eligible']][:3]
    for row in [None, *rows]:
        info = world.infos[rows[0]['question'] if row is None else row['question']]
        kwargs = dict(arm='N', policy_name='zero', seed_k=0)
        if row is not None:
            kwargs['parent'] = {'trace_ref': row['trace_ref'], 'prefix_len': row['prefix_len']}
        yield table, info, kwargs


def test_T2_caps_preserve_every_other_field():
    for table, info, kwargs in real_requests():
        native = M.make_request('caps', table, info, **kwargs)
        assert M.make_request('caps', table, info, max_new_tokens=None, **kwargs) == native
        cap = 1024 if kwargs.get('parent') else 32768
        capped = M.make_request('caps', table, info, max_new_tokens=cap, **kwargs)
        assert capped['max_tokens'] == cap
        assert {**capped, 'max_tokens': native['max_tokens']} == native
        large = M.make_request('caps', table, info, max_new_tokens=10**9, **kwargs)
        assert large == native


@pytest.mark.parametrize('cap', [True, False, 0, -1, 1.5, '1024'])
def test_T2_invalid_builder_caps(cap):
    table, info, kwargs = next(real_requests())
    with pytest.raises(M.ManifestError, match='integer'):
        M.make_request('caps', table, info, max_new_tokens=cap, **kwargs)


@pytest.mark.parametrize('cap', [True, 0, -1, 1.5, '1024', 10**9])
def test_T2_invalid_manifest_caps(cap):
    original = M.load_manifest(policies.CAMPAIGN / 'manifests/q11-v2.json')
    bad = copy.deepcopy(original)
    bad.pop('sha256')
    bad['requests'][0]['max_tokens'] = cap
    with pytest.raises(M.ManifestError):
        M.validate_manifest(seal(bad))
