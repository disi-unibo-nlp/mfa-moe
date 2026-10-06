"""No scientific generation/submission: exact contracts and failure cases."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import native_finalization_v1 as N
import native_finalization_audit_v1 as A
import native_finalization_outcomes_v1 as O
import operator_panel_outcomes_v1 as frozen


def factorial():
    families = ['family-' + str(i) for i in range(8)]
    reps = {f: 'question-' + str(i) for i, f in enumerate(families)}
    prompts = {q: {'original': [7, 8], 'finalization': [7, 9, 8]} for q in reps.values()}
    return families, N.assignment_rows(families, reps, prompts, {'test_binding': True})


def endpoint_rows():
    families, assigned = factorial()
    rows = []
    for r in assigned:
        a = r['assignment']; i = families.index(a['family'])
        offset = (i + 1) * (a['seed'] + 1) if a['arm'] == 'finalization' else 0
        rows.append({**a, 'operational_correct': True, 'reasoning_tokens': 100 - offset, 'tokens': 120 - offset})
    return families, rows


@pytest.mark.parametrize('k', [1, 2, 4, 6, 8])
def test_architecture_count_nonranked_ids_masks_weights(k):
    ids = np.arange(k)[::-1][None, :]
    weights = np.full((1, k), 1 / k)
    affinity = np.arange(10, 0, -1, dtype=float)[None, :]
    obs = A.routing_observation(k, 10, ids, weights, affinity, affinity, True, np.array([False]))
    assert obs['native_ids'] == ids.tolist()
    assert obs['executed_weights'] == weights.tolist()
    assert obs['active_mask'] == [False]
    assert obs['genuine_selection_cutoff_gap'] == [1.]
    assert A.architecture({'text_config': {'num_experts_per_tok': k, 'num_experts': 10}})['routed_experts_per_token'] == k


def test_gap_semantics_and_missing_native_are_unknown():
    affinity = [[.9, .5, .3, .2, .01]]
    obs = A.routing_observation(4, 5, affinity=affinity, selection_scores=affinity, deterministic_topk=True)
    assert obs['native_ids'] is None and obs['native_selection_status'] == 'UNKNOWN'
    assert obs['top1_top2_affinity_gap'] == [.4]
    assert obs['affinity_k_gap'] == [.19]
    assert obs['genuine_selection_cutoff_gap'] is None
    obs = A.routing_observation(2, 5, native_ids=[[4, 2]], affinity=affinity, selection_scores=affinity)
    assert obs['genuine_selection_cutoff_gap'] is None  # non-ranked selector
    with pytest.raises(ValueError):
        A.routing_observation(2, 5, native_ids=[[4, 2]], selection_scores=affinity, deterministic_topk=True)
    with pytest.raises(ValueError):
        A.routing_observation(2, 5, executed_weights=[[.5, .5]])
    with pytest.raises(ValueError):
        A.routing_observation(2, 5, native_ids=[[1, 1]])
    with pytest.raises(ValueError):
        A.routing_observation(2, 5, native_ids=[[1, 2]], executed_weights=[[1, -1]])


def test_prediction_rows_prefill_and_presence_union():
    assert A.prediction_rows(5, [0, 3, 4, 5, 6]) == [None, None, 0, 1, 2]
    assert A.presence_restored([1, 1, 2], [2, 3]) == {1: 1.5, 2: 1.5, 3: 1.5}


def test_actual_frozen_pulse_stops_after_closure():
    # Pin qualified imports exactly as production before loading the frozen adapter.
    from utility_outcomes_v3 import scoring_modules
    scoring_modules()
    from utility_routing_worker_v2 import rows_for_episode
    episode = {'absolute_slots': [2], 'action_names': ['force']}
    table = SimpleNamespace(index_of=lambda name: 1)
    mask, indices = rows_for_episode(episode, table, np.arange(-3, 270), [7, 7, 7, 248069, 8])
    assert np.arange(-3, 270)[mask].tolist() == [2, 3]
    assert set(indices[mask]) == {1}
    mask, _ = rows_for_episode(episode, table, np.arange(270), [7] * 270)
    assert np.flatnonzero(mask).tolist() == list(range(2, 258))


def test_counterbalancing_exact_factorial():
    families, rows = factorial()
    assert len(rows) == len({r['assignment']['uid'] for r in rows}) == 32
    counts = {(arm, pos): 0 for arm in N.ARMS for pos in (0, 1)}
    for i, f in enumerate(families):
        block = [r['assignment'] for r in rows if r['assignment']['family'] == f]
        assert [a['seed'] for a in block] == ([0, 0, 1, 1] if i % 2 == 0 else [1, 1, 0, 0])
        assert {(a['seed'], a['arm']) for a in block} == {(s, a) for s in (0, 1) for a in N.ARMS}
        for a in block:
            counts[a['arm'], a['execution_position']] += 1
    assert set(counts.values()) == {8}


def test_modified_prompt_appended_inside_user_only():
    class Tokenizer:
        def decode(self, ids, **kw): return ''.join(map(chr, ids))
        def encode(self, text, **kw): return list(map(ord, text))
    text = '<|im_start|>user\nProblem and original instruction.<|im_end|>\n<|im_start|>assistant\n<think>\n'
    ids, modified = N.append_instruction(list(map(ord, text)), Tokenizer())
    assert modified == text.replace('instruction.<|im_end|>', 'instruction.\n\n' + N.INSTRUCTION + '<|im_end|>')
    assert ids == list(map(ord, modified))
    with pytest.raises(ValueError): N.append_instruction([1], Tokenizer())


def test_closure_missing_marker_caps_errors():
    assert frozen.lengths([5, 248069, 8], 'stop')['reasoning_tokens'] == 1
    assert frozen.lengths([5, 248069, 8], 'stop')['answer_tokens'] == 1
    assert frozen.lengths([248069], 'stop')['reasoning_tokens'] == 0
    assert frozen.lengths([5] * N.CAP, 'length')['reasoning_tokens'] == N.CAP
    assert frozen.lengths([5], 'stop')['reasoning_tokens'] is None
    assert frozen.lengths([5], 'error')['reasoning_tokens'] is None
    assert frozen.lengths([5], 'stop', prompt_reasoning_open=False)['boundary_status'] == 'PREFIX_ALREADY_CLOSED'
    assert frozen.lengths([248069, 248069], 'stop')['closing_markers'] == 2
    score = SimpleNamespace(adjudicate_uid=lambda strict, finish, verdict: (None, verdict, verdict == 'EQUIVALENT'))
    assert frozen.adjudicated(False, 'stop', None, score)[0] is None
    assert frozen.adjudicated(True, 'stop', None, score)[0] is True
    assert frozen.adjudicated(False, 'length', None, score)[0] is False
    assert frozen.adjudicated(False, 'stop', 'UNCERTAIN', score)[0] is False


def test_bootstrap_equal_family_seed_coverage_and_unknown_bounds():
    families, rows = endpoint_rows()
    result = O.inference(rows, families, replicates=1000)
    assert result['point_estimates'] == [0., -6.75, -6.75]
    assert len(result['simultaneous_95_intervals']) == 3
    assert result['small_sample'] and result['degenerate_intervals']
    assert result['bootstrap_seed'] == 20261005
    assert result == O.inference(rows, families, replicates=1000)
    rows[0]['operational_correct'] = None
    incomplete = O.inference(rows, families)
    assert incomplete['point_estimates'] is None and incomplete['simultaneous_95_intervals'] is None
    low, high = incomplete['identification_bounds'][0]
    assert high - low == 1 / 16
    with pytest.raises(ValueError): O.paired(rows[:-1], families)
    with pytest.raises(ValueError): O.paired(rows + rows[:1], families)


def test_receipt_reuse_uncommitted_unknown_and_tamper(tmp_path):
    _, rows = factorial(); a = rows[0]['assignment']
    manifest = {'rows': rows[:1], 'family_order': [a['family']], 'sha256': 'manifest'}
    assert N.reconcile(manifest, tmp_path)['records'][0]['status'] == 'MISSING'
    attempt = N.save(tmp_path / 'attempts' / (N.U.digest(a['uid']) + '.json'), {
        'assignment': a, 'manifest_sha256': 'manifest'})
    assert N.reconcile(manifest, tmp_path)['uncommitted_attempts'] == 1
    body = {'assignment': a, 'manifest_sha256': 'manifest', 'status': 'GENERATION_ERROR',
            'attempt_sha256': attempt['sha256'], 'token_ids': [7], 'finish': 'error', 'error': 'test failure'}
    path = N.receipt_path(tmp_path, a)
    receipt = N.save(path, body)
    assert N.save(path, body) == receipt
    index = N.reconcile(manifest, tmp_path)
    assert index['uncommitted_attempts'] == 0
    extracted = O.extract(index)
    assert extracted[0]['operational_correct'] is False and extracted[0]['tokens'] is None
    with pytest.raises(ValueError): N.save(path, {**body, 'error': 'different'})
    changed = copy.deepcopy(manifest); changed['rows'][0]['assignment']['seed'] = 88
    with pytest.raises(ValueError): N.reconcile(changed, tmp_path)
    path.write_text(json.dumps({**receipt, 'error': 'tampered'}))
    with pytest.raises(ValueError): N.U.sealed(path)


def test_blinded_judge_contract():
    from utility_outcomes_v3 import scoring_modules
    score, _ = scoring_modules()
    score.assert_blind([{'uid': 'nf-blind-v1|abc', 'problem': '2+2', 'gold': '4', 'candidate': '4'}], 'test')
    with pytest.raises((ValueError, AssertionError)):
        score.assert_blind([{'uid': 'nf-blind-v1|abc', 'arm': 'finalization'}], 'test')


def test_source_binding_changes_with_adapter_bytes(monkeypatch, tmp_path):
    path = tmp_path / 'source.py'; path.write_text('one')
    original = N.U.file_sha(path)
    path.write_text('two')
    assert N.U.file_sha(path) != original
    current = N.sources()
    assert str(Path(N.__file__)) in current
    assert str(Path(N.__file__).with_name('native_finalization_worker_v1.py')) in current
    manifest = {'schema': 'native-finalization-manifest-v1', 'binding': {'sources': {**current, str(path): original}}}
    with pytest.raises(ValueError, match='source binding changed'): N.validate(manifest)


def test_transport_preserves_module_helper_but_drops_credentials():
    from native_finalization_transport_v2 import safe_environment
    value = safe_environment({'PATH': '/usr/bin', 'BASH_FUNC_module%%': 'module body',
        'BASH_FUNC__module_raw%%': 'module helper', 'HF_TOKEN': 'test-value',
        'OPENAI_API_KEY': 'test-value', 'SLURM_JOB_ID': 'parent'})
    assert value['BASH_FUNC__module_raw%%'] == 'module helper'
    assert 'HF_TOKEN' not in value and 'OPENAI_API_KEY' not in value and 'SLURM_JOB_ID' not in value


def test_compute_readiness_does_not_attempt_login_accounting(monkeypatch):
    import getpass
    import native_finalization_operations_v1 as ops
    import native_finalization_native_dispatch_v3 as dispatch
    commands = []
    def run(args):
        commands.append(args)
        return 'iscrc_miosr normal'
    monkeypatch.setattr(ops, 'command', run)
    monkeypatch.setattr(getpass, 'getuser', lambda: 'lmolfett')
    monkeypatch.setattr(ops.socket, 'gethostname', lambda: 'viz09.leonardo.local')
    monkeypatch.setattr(ops.socket, 'getfqdn', lambda: 'viz09.leonardo.local')
    monkeypatch.setattr(N, 'save', lambda path, value: value)
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    value = ops.live()
    assert all('saldo' not in ' '.join(command) for command in commands)
    assert value['balance_status'].startswith('UNAVAILABLE_ON_COMPUTE')
    with pytest.raises(ValueError, match='native lmolfett login shell'): dispatch.login_guard()
