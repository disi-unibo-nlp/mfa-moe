"""Prepare the two-hunk s2 proposal; never freeze or submit a job.

Run with the existing correlation-client Python, PYTHONDONTWRITEBYTECODE=1.
All historical files are preserved. The keeper must already acknowledge DRAIN.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data: str) -> None:
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f'refusing to overwrite differing artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def patch_manifests(text: str) -> str:
    substitutions = [
        ('    expected_len: int | None = None,\n) -> dict[str, Any]:',
         '    expected_len: int | None = None,\n    max_new_tokens: int | None = None,\n) -> dict[str, Any]:'),
        ('    request: dict[str, Any] = {\n',
         '    cap = request_cap(info["prompt_tokens"], prefix_len)\n'
         '    if max_new_tokens is not None:\n'
         '        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:\n'
         '            raise ManifestError("max_new_tokens must be an integer >= 1")\n'
         '        cap = min(cap, max_new_tokens)\n'
         '    request: dict[str, Any] = {\n'),
        ('        "max_tokens": request_cap(info["prompt_tokens"], prefix_len),',
         '        "max_tokens": cap,'),
        ('        if r["max_tokens"] != request_cap(r["prompt_tokens"], prefix):\n'
         '            raise ManifestError(f"{where}: max_tokens {r[\'max_tokens\']} is not the cumulative cap")',
         '        cap = request_cap(r["prompt_tokens"], prefix)\n'
         '        if isinstance(r["max_tokens"], bool) or not isinstance(r["max_tokens"], int) or not 1 <= r["max_tokens"] <= cap:\n'
         '            raise ManifestError(f"{where}: max_tokens {r[\'max_tokens\']} outside [1, cumulative cap {cap}]")'),
    ]
    for old, new in substitutions:
        if text.count(old) != 1:
            raise ValueError(f'expected exactly one patch anchor: {old[:70]}')
        text = text.replace(old, new)
    compile(text, 'manifests.py', 'exec')
    return text


def prepare(root: Path = ROOT) -> dict:
    swarm = root / 'swarm'
    status = (swarm / 'STATUS.md').read_text()
    if not (swarm / 'DRAIN').exists() or 'DRAIN=yes' not in status or 'Controller: cycle 6; drain' not in status:
        raise ValueError('keeper must acknowledge DRAIN before shared writes')
    for path in (swarm / 'state3').glob('*.json'):
        row = json.loads(path.read_text())
        if row.get('start_ts', 0) > row.get('end_ts', 0):
            raise ValueError(f'unsettled executor: {path}')
    campaign = root / 'steering-v1'
    snap = campaign / 'code/s1-f2ded3957eb54fd5'
    spec = importlib.util.spec_from_file_location('freeze_s1', snap / 'scripts/freeze_steer.py')
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    freezer.verify(snap)
    proposal = campaign / 'addenda/s2'
    code = proposal / 'code-root'
    for tree in freezer.TREES:
        if not (code / tree).exists():
            shutil.copytree(snap / tree, code / tree)
    original = (snap / 'moe_steer/manifests.py').read_text()
    patched = patch_manifests(original)
    target = code / 'moe_steer/manifests.py'
    target.chmod(0o600)
    target.write_text(patched)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                      fromfile='s1/moe_steer/manifests.py',
                                      tofile='s2/moe_steer/manifests.py', n=40))
    if sum(line.startswith('@@') for line in diff.splitlines()) != 2:
        raise ValueError('expected two diff hunks')
    write_once(proposal / 'manifests.py.patch', diff)
    files = freezer.source_files(code, snap / 'moe_exp_src')
    baseline = json.loads((snap / 'MANIFEST.json').read_text())['files']
    changed = [key for key in files if files[key] != baseline.get(key)]
    assert changed == ['moe_steer/manifests.py'] and files.keys() == baseline.keys()
    inventory = {
        'snap_tree': freezer.digest(baseline), 'proposal_tree': freezer.digest(files),
        'would_be_snapshot': 's1-' + freezer.digest(files)[:16], 'files_total': len(files),
        'added': [], 'changed': changed, 'removed': [], 'moe_exp_src_identical': True,
        'patch_sha256': sha(proposal / 'manifests.py.patch'),
        'file_table': {key: {'snap': baseline[key], 'proposal': value,
                            'identical': baseline[key] == value} for key, value in files.items()},
    }
    write_once(proposal / 'PROPOSED_TREE.json', json.dumps(inventory, indent=1) + '\n')
    x2 = campaign / 'addenda/x2'
    old_builder = x2 / 'x2_build.py'
    backup = x2 / 'attic/x2_build.s1-dryrun.py'
    if not backup.exists():
        write_once(backup, old_builder.read_text())
    builder = (REPO / 'scripts/experimental_resume/x2_build.py').read_text()
    old_builder.write_text(builder)
    tests = x2 / 'tests'
    attic_tests = x2 / 'attic/tests-s1'
    if not attic_tests.exists():
        shutil.copytree(tests, attic_tests)
    test = (attic_tests / 'test_x2_build.py').read_text()
    start = test.index('def test_horizon_cannot_be_encoded_in_the_manifest_gap():')
    end = test.index('\ndef test_determinism_same_inputs_same_seal():', start)
    test = test[:start] + '''def test_horizon_is_enforced_by_manifest():
    _, _, _, manifest, _ = build()
    assert all(r["max_tokens"] == 1024 for r in manifest["requests"])
    M.validate_manifest(manifest)

''' + test[end:]
    test = test.replace('"after_pulse": 256}', '"after_pulse": 1024}')
    test = test.replace('max_new_tokens=1024)\n    tampered',
                        'max_new_tokens=1024, expect_tree="0" * 64)\n    tampered')
    test = test.replace('"--out", "/tmp/x"]',
                        '"--out", "/tmp/x", "--expect-tree", "0" * 64]')
    test = test.replace('args.n_policy == "zero"', 'args.n_policy == "sham"')
    (tests / 'test_x2_build.py').write_text(test)
    # Keep all outputs of this follow-up out of historical x2prep/.
    conf = (attic_tests / 'conftest.py').read_text().replace('runs/x2prep/_pytest_tmp', 'runs/s2prop/_pytest_tmp')
    (tests / 'conftest.py').write_text(conf)
    runner_test = (attic_tests / 'test_runner_acceptance.py').read_text()
    runner_test = runner_test.replace('@pytest.mark.parametrize("n_policy", ["zero", "sham"])',
                                    '@pytest.mark.parametrize("n_policy", ["zero", "sham"])\n'
                                    '@pytest.mark.parametrize("natural", [False, True])')
    runner_test = runner_test.replace('workdir, n_policy):', 'workdir, n_policy, natural):')
    runner_test = runner_test.replace('driver = fakes.FakeDriver(table.digest())',
                                    'driver = fakes.FakeDriver(table.digest(), n_tokens=300 if natural else 1024)\n'
                                    '    if natural:\n'
                                    '        original_step = driver.step\n'
                                    '        def natural_step():\n'
                                    '            done = original_step()\n'
                                    '            for fin in done:\n'
                                    '                fin.finish_reason = "stop"\n'
                                    '                fin.token_ids[-1] = 248046\n'
                                    '            return done\n'
                                    '        driver.step = natural_step')
    runner_test = runner_test.replace('assert sampling["max_tokens"] == request["max_tokens"] > 125_000',
                                    'assert sampling["max_tokens"] == request["max_tokens"] == 1024')
    runner_test = runner_test.replace('{"length": 170}', '{"stop" if natural else "length": 170}')
    runner_test = runner_test.replace('== prefix + 300', '== prefix + (300 if natural else 1024)')
    runner_test = runner_test.replace('== 300 and meta["row_offset"]',
                                    '== (300 if natural else 1024) and meta["row_offset"]')
    runner_test = runner_test.replace('== {256}', '== {300 if natural else 1024}')
    runner_test = runner_test.replace('else {256})', 'else {300 if natural else 1024})')
    runner_test += '''
    resumed_driver = fakes.FakeDriver(table.digest())
    resumed = runner.run_shard(
        manifest, cfg, driver_factory=lambda kwargs, env: resumed_driver, decode=fakes.decode,
        fingerprint_fn=lambda: {"combined": fakes.FINGERPRINT},
        code_tree_fn=lambda: synth.CODE_TREE, trace_store=store, log=lambda message: None)
    assert resumed["status"] == "complete" and resumed["n_done"] == 170
    assert not resumed_driver.prompts
    again = results.read_shard_records(workdir, 0)
    assert len(again) == 170 and {r["uid"] for r in again} == set(by_uid)
'''
    (tests / 'test_runner_acceptance.py').write_text(runner_test)
    for src in (REPO / 'tests/experimental_resume').glob('test_caps*.py'):
        shutil.copy2(src, tests / src.name)
    sums = {str(path.relative_to(x2)): sha(path) for path in sorted(x2.glob('tests/*.py'))}
    sums['x2_build.py'] = sha(old_builder)
    (x2 / 'SHA256SUMS').write_text(''.join(f'{value}  {key}\n' for key, value in sums.items()))
    # Copy the existing eligibility cache: no corpus scan is needed for the dry build.
    out = campaign / 'runs/s2prop'
    out.mkdir(exist_ok=True)
    shutil.copy2(campaign / 'runs/x2prep/eligibility.json', out / 'eligibility.json')
    shutil.copytree(campaign / 'runs/x2prep/trace-offsets', out / 'trace-offsets', dirs_exist_ok=True)
    result = {'tree': inventory['proposal_tree'], 'snapshot': inventory['would_be_snapshot'],
              'patch_sha256': inventory['patch_sha256'], 'old_builder': sha(backup),
              'new_builder': sha(old_builder), 'status': 'PREPARED; tests and dry manifest pending'}
    (out / 'preparation.json').write_text(json.dumps(result, indent=1) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=1))
