import numpy as np
import pytest
from moe_exp.moe_identity_guiding.bootstrap import paired_accuracy_bootstrap


def pairs():
    a, b = {}, {}
    for dataset, problem, n, delta in [('a', 'p', 32, 1), ('a', 'q', 32, -1),
                                      ('b', 'p', 1, 0), ('b', 'q', 1, 1)]:
        for i in range(n):
            key = f'{dataset}/{problem}/{i}'
            data = dict(dataset=dataset, source_problem_id=problem)
            a[key] = dict(input=data, is_correct=delta < 0)
            b[key] = dict(input=data, is_correct=delta > 0)
    return a, b


def test_cluster_sampling_and_worker_independence():
    a, b = pairs()
    serial = paired_accuracy_bootstrap(a, b, replicates=300, workers=1)
    parallel = paired_accuracy_bootstrap(a, b, replicates=300, workers=2)
    assert parallel['workers'] == 2
    serial.pop('workers'); parallel.pop('workers')
    assert serial == parallel
    assert serial['overall']['num_problems'] == 4
    assert serial['overall']['accuracy_delta'] == 1/66
    # Sampling whole clusters allows both all-positive and all-negative draws.
    # An independent-attempt bootstrap would yield much narrower intervals.
    assert serial['datasets']['a']['ci95'] == [-1, 1]
    assert serial['datasets']['b']['ci95'] == [0, 1]
    assert np.allclose(serial['overall']['ci95'], [-64/66, 1])


def test_single_problem_and_missing_identity():
    a = {'x': dict(input=dict(dataset='a', problem_id='p'), is_correct=False)}
    b = {'x': dict(input=a['x']['input'], is_correct=True)}
    result = paired_accuracy_bootstrap(a, b, replicates=10, workers=1)
    assert result['overall']['ci95'] is None
    assert result['datasets']['a']['limited_problem_support']
    a['x']['input'].pop('problem_id')
    with pytest.raises(ValueError, match='identity'):
        paired_accuracy_bootstrap(a, b, workers=1)


def test_compare_includes_intervals_and_rejects_incomplete(tmp_path):
    import json
    from moe_exp.moe_identity_guiding.run import compare
    a, b = pairs()
    for condition, records in [('baseline', a), ('guided', b)]:
        root = tmp_path/condition
        root.mkdir()
        manifest = dict(status='complete', condition=condition, prompts_sha256='p',
                        policy_sha256='q', engine_args={}, sampling_args={}, versions={},
                        scoring_contract='test', calibration_overlap=[])
        (root/'manifest.json').write_text(json.dumps(manifest))
        rows = [dict(id=k, **r, prompt_token_ids=[1], sampling_args={},
                     generated_token_count=10, finish_reason='stop') for k, r in records.items()]
        (root/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = compare(tmp_path/'baseline', tmp_path/'guided', bootstrap_replicates=300,
                     bootstrap_workers=2)
    assert result['bootstrap']['workers'] == 2
    assert result['accuracy_delta'] == pytest.approx(result['bootstrap']['overall']['accuracy_delta'])
    manifest['status'] = 'running'
    (tmp_path/'guided'/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete'):
        compare(tmp_path/'baseline', tmp_path/'guided')
