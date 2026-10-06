"""No real sbatch calls: verify the exact-price qualification/replay/analysis DAG."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/experimental_resume'))
import dispatch_mechanism_semantic_cap_recovery_v1 as dispatch


def fixture(monkeypatch, workdir):
    plan = {'sha256': 'test-recovery-plan', 'status': 'FROZEN_PRICED_REQUIRES_GPU_QUALIFICATION',
            'qualification_price': {'ratings': 4, 'gpus': 2, 'max_wall_seconds': 3600,
                                    'estimated_complete_gpu_hours': 1.2},
            'generation_price': {'ratings': 180, 'gpus': 2, 'max_wall_seconds': 7200,
                                 'shards': [{'start': i*40, 'end': min((i+1)*40, 180)} for i in range(5)],
                                 'estimated_complete_gpu_hours': 14.9},
            'estimated_complete_gpu_hours': 16.1, 'complete_gpu_hour_ceiling': 20.,
            'code_files': {str(ROOT/'scripts/experimental_resume'/name): 'source-hash' for name in
                           ('run_mechanism_semantic_cap_recovery_v1.sbatch',
                            'analyze_mechanism_semantic_cap_recovery_v1.sbatch')}}
    monkeypatch.setenv('SLURM_JOB_ID', 'test-cpu'); monkeypatch.setenv('SLURM_JOB_PARTITION', 'lrd_all_viz')
    monkeypatch.setattr(dispatch.recovery, 'DOC', workdir)
    monkeypatch.setattr(dispatch.shared.base, 'sealed', lambda path: plan)
    monkeypatch.setattr(dispatch.shared.base, 'file_sha', lambda path: 'source-hash')
    validated = []
    monkeypatch.setattr(dispatch.recovery, 'validate', lambda value: validated.append(value))
    monkeypatch.setattr(dispatch.subprocess, 'run', lambda *a, **kw: None)
    calls = []; saved = {}
    def submit(directory, name, args, env, binding):
        job = str(7001 + len(calls)); calls.append({'name': name, 'args': args, 'env': dict(env), 'binding': binding})
        return job
    monkeypatch.setattr(dispatch.shared, 'submit', submit)
    monkeypatch.setattr(dispatch.shared, 'save', lambda path, body: saved.update({Path(path).name: body}))
    return plan, calls, saved, validated


def test_pilot_then_complete_array_then_sensitivity_analysis(monkeypatch, workdir):
    plan, calls, saved, validated = fixture(monkeypatch, workdir)
    dispatch.main()
    assert validated == [plan]
    assert [c['name'] for c in calls] == ['qualification', 'generation', 'analysis']
    assert '--time=01:00:00' in calls[0]['args'] and calls[0]['env']['CAP_RECOVERY_MODE'] == 'qualification'
    assert '--array=0-4' in calls[1]['args'] and '--time=02:00:00' in calls[1]['args']
    assert '--dependency=afterok:7001' in calls[1]['args']
    assert calls[1]['env']['CAP_RECOVERY_MODE'] == 'generation'
    assert '--dependency=afterok:7002' in calls[2]['args']
    assert saved['CHAIN.json']['original_primary_unchanged'] is True
    assert all(c['env']['CAP_RECOVERY_RESUME_UNCOMMITTED'] == '0' for c in calls)


@pytest.mark.parametrize('change', ['partial_population', 'missing_pilot_cost', 'over_budget', 'empty_array', 'source_change'])
def test_unpriced_or_changed_chain_fails_before_submission(monkeypatch, workdir, change):
    plan, calls, _, _ = fixture(monkeypatch, workdir)
    if change == 'partial_population': plan['generation_price']['ratings'] = 179
    elif change == 'missing_pilot_cost': plan['estimated_complete_gpu_hours'] = 14.9
    elif change == 'over_budget': plan['complete_gpu_hour_ceiling'] = 15.
    elif change == 'empty_array': plan['generation_price']['shards'] = []
    else: plan['code_files'][str(ROOT/'scripts/experimental_resume/run_mechanism_semantic_cap_recovery_v1.sbatch')] = 'changed'
    with pytest.raises(ValueError): dispatch.main()
    assert not calls
