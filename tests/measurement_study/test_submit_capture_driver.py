import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

DRIVER = Path(__file__).resolve().parents[2] / 'scripts/submit_capture_v3.py'


@pytest.fixture
def driver(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('submit_capture_v3', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.D, module.STORAGE, module.STATE = tmp_path / 'D', tmp_path / 'fast/campaign-v3', tmp_path / 'D/capture'
    (module.D / 'acceptance').mkdir(parents=True)
    calls = []

    def fake_sbatch(args, model, stage):
        calls.append((stage, args))
        return 1000 + len(calls)

    monkeypatch.setattr(module, 'sbatch', fake_sbatch)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    module.calls = calls
    return module


def args(**kw):
    return SimpleNamespace(**{'model': 'glm', 'bundle': Path('/bundle'), 'cohort': None, 'round': 1, **kw})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_start_b_plans_one_shared_verify_and_a_waits_for_labels(driver):
    driver.start(args(cohort='B'))
    assert [c[0] for c in driver.calls] == ['plan-verify']
    assert json.loads(driver.pending('glm').read_text()) == ['B']
    with pytest.raises(SystemExit, match='label gate'):
        driver.start(args(cohort='A'))
    write(driver.D / 'acceptance/labels-glm.json', {'status': 'complete'})
    write(driver.verify_manifest('glm'), {'requested_seconds': 57600})
    driver.queued = lambda name: name == 'v3-capverify-glm'
    driver.start(args(cohort='A'))  # verify already queued: only queue A behind it
    assert [c[0] for c in driver.calls] == ['plan-verify']
    assert json.loads(driver.pending('glm').read_text()) == ['A', 'B']
    driver.queued = lambda name: False  # planned but its chain died: re-arm the verify
    driver.start(args(cohort='A'))
    assert [c[0] for c in driver.calls] == ['plan-verify', 'verify-r1', 'after-verify']
    assert '--time=16:00:00' in driver.calls[1][1]


def test_after_verify_retries_boundedly_then_plans_pending_cohorts(driver):
    write(driver.verify_manifest('glm'), {'requested_seconds': 18000})
    driver.mark_pending('glm', 'B')
    # a hard failure (no drain marker) is deterministic: stop instead of reloading the model
    with pytest.raises(SystemExit, match='without draining'):
        driver.after_verify(args(round=1))
    drained = driver.verify_receipt('glm').with_name('drained.json')
    write(drained, {'status': 'drained'})
    driver.after_verify(args(round=1))
    stages = [c[0] for c in driver.calls]
    assert stages == ['verify-r2', 'after-verify']
    assert '--time=05:00:00' in driver.calls[0][1]
    assert not drained.exists() and drained.with_name('drained-r1.json').exists()
    write(drained, {'status': 'drained'})
    with pytest.raises(SystemExit, match='did not pass'):
        driver.after_verify(args(round=driver.MAX_ROUNDS))
    write(driver.verify_receipt('glm'), {'status': 'passed', 'measurements': {}})
    driver.calls.clear()
    driver.after_verify(args(round=2))
    assert [c[0] for c in driver.calls] == ['plan-cohort']
    assert not driver.pending('glm').exists()


def shards(driver, complete, attempts_done):
    root = driver.STORAGE / 'manifests/glm/B'
    write(root / 'ready.json', {'shards': [{'index': i, 'requested_seconds': 7200} for i in range(3)]})
    for i in range(3):
        out = driver.STORAGE / f'capture/glm/B/{i}'
        write(root / f'{i}.json', {'capture_dir': str(out), 'requested_seconds': 7200})
        out.mkdir(parents=True, exist_ok=True)
        for n in range(attempts_done[i]):
            (out / f'a{n}.complete.json').write_text('{}')
        if i in complete:
            write(out / 'result.json', {'status': 'complete_native_collection'})


def test_gate_resubmits_only_incomplete_shards_while_progressing(driver, monkeypatch):
    shards(driver, complete={0}, attempts_done=[3, 1, 0])
    driver.gate(args(cohort='B', round=1))
    submitted = [c for c in driver.calls if c[0].startswith('B')]
    assert [c[0] for c in submitted] == ['B1-r2', 'B2-r2']
    assert all('--dependency=singleton' in c[1] for c in submitted)
    # shared campaign pool: consecutive slots from the counter, bounded by gpu_slots.json
    assert [a for c in submitted for a in c[1] if a.startswith('--job-name')] == [
        '--job-name=v3-gpu-0', '--job-name=v3-gpu-1']
    write(driver.D / 'gpu_slots.json', {'slots': 2})
    assert driver.gpu_slot()[0] == '--job-name=v3-gpu-0'
    # no new .complete.json since the last round: stop instead of looping
    with pytest.raises(SystemExit, match='no progress'):
        driver.gate(args(cohort='B', round=2))
    ran = []
    monkeypatch.setattr(driver, 'bundle_module', lambda *a: ran.append(a) or SimpleNamespace(returncode=0))
    shards(driver, complete={0, 1, 2}, attempts_done=[3, 2, 2])
    driver.gate(args(cohort='B', round=3))
    assert ran and ran[0][1] == 'measurement_study.capture_gate'


def test_walltime_is_clamped_to_the_partition_limit(driver):
    assert driver.walltime(100) == '01:00:00'
    assert driver.walltime(5 * 3600 + 1800) == '05:30:00'
    assert driver.walltime(40 * 3600) == '24:00:00'


def test_sbatch_scrubs_inherited_slurm_environment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('submit_capture_env', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.D, module.STATE = tmp_path, tmp_path / 'capture'
    seen = {}

    def run(argv, **kw):
        seen.update(kw['env'])
        return SimpleNamespace(stdout='4242\n')

    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setenv('SLURM_CPU_BIND', 'quiet,mask_cpu:0xF000')
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setenv('KEEP_ME', 'yes')
    assert module.sbatch(['x.sbatch'], 'glm', 'test') == 4242
    assert 'SLURM_CPU_BIND' not in seen and 'SLURM_JOB_ID' not in seen and seen['KEEP_ME'] == 'yes'
