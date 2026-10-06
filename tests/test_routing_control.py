from types import SimpleNamespace

import numpy as np
import pytest

from moe_exp.routing_control.analysis import (class_summary, dense_window, paired_itt,
    routing_kinematics, trajectory_completion)
from moe_exp.routing_control.design import (Action, Template, balanced_random_assignment,
    edit_logits, families, freeze_pools, price_stage, propose_actions, random_controls)
from moe_exp.routing_control.nll import actual_logprobs, pulse_surprisal, validate_parity
from moe_exp.routing_control.prefix import StreamingAdapter, candidates, numeric_expression
from moe_exp.routing_control.closure import CLOSURE_TEXT, prepare_closure
from moe_exp.routing_control.receipts import ReceiptStore
from moe_exp.routing_control.manifests import build as build_control_manifest
from moe_exp.routing_control.design import digest


@pytest.mark.parametrize('text', ['answer is 42', 'answer is 42 ', 'answer is 42. ',
    'answer is 42 + 1 and so on', 'answer is 1/2 and so on', r'\boxed{1+',
    r'\boxed{\frac{1}{2}', r'\boxed{x^2}', r'answer is $1+2', 'answer is 2,000 with commas',
    'answer is 3 meters long'])
def test_prefix_abstains_on_truncation_and_unsupported_forms(text):
    assert candidates(text) == []


@pytest.mark.parametrize('expr,value', [('2+3*4', '14'), ('1/2', '1/2'), ('0.125', '1/8'),
    (r'\frac{1}{2}+\frac{1}{4}', '3/4'), ('(2+1)^2', '9')])
def test_complete_supported_expressions(expr, value):
    assert numeric_expression(expr) == value
    found = candidates(r'We get \boxed{' + expr + '}')
    assert len(found) == 1 and found[0].value == value


@pytest.mark.parametrize('expr', ['1/0', '__import__("os")', '2^1000000000', 'x+1', '1e309', '1==1'])
def test_expression_grammar_is_bounded(expr):
    assert numeric_expression(expr) is None


def test_prefix_requires_the_entire_lookahead_and_does_not_duplicate_box():
    text = 'answer is 42.' + ' ' * 11
    assert not candidates(text[:-1])
    assert candidates(text)[0].end == len(text)
    assert len(candidates(r'answer is \boxed{42}')) == 1


def test_online_allowlist_future_invariance_and_one_trigger():
    allowed = {'problem': 'compute 2+2', 'emitted_token_ids': [1, 2], 'emitted_text': r'\boxed{4}'}
    a, b = StreamingAdapter(), StreamingAdapter()
    first = a.observe({**allowed, 'future_text': 'correct', 'gold': '4', 'is_correct': True})
    second = b.observe({**allowed, 'future_text': 'wrong', 'gold': '5', 'is_correct': False,
                        'final_length': 10000, 'future_labels': ['Explore']})
    assert first == second and first['can_trigger']
    assert not a.observe(allowed)['can_trigger']
    assert not a.observe({**allowed, 'emitted_text': allowed['emitted_text'] + '</think>',
                          'emitted_token_ids': [1, 2, 3]})['can_trigger']
    with pytest.raises(ValueError, match='history'):
        a.observe({**allowed, 'emitted_token_ids': [9, 2]})


def test_production_adapter_decodes_only_the_emitted_prefix():
    seen = []
    def decode(ids):
        seen.append(ids)
        return r'\boxed{4}'
    assert StreamingAdapter().observe_tokens('question', [1, 2], decode)['can_trigger']
    assert seen == [(1, 2)]


def test_measurement_receipts_resume_without_duplication_and_refuse_stale_binding(tmp_path):
    uid, other = 'a' * 64, 'b' * 64
    out = tmp_path / 'measurement'
    row = {'uid': uid, 'nll': 1.2}
    with ReceiptStore(out, {'manifest': 'v1'}, [uid, other]) as store:
        store.put(row)
        store.put(row)
        # Simulate a crash before atomic publication of another batch result.
        (out / 'records' / ('.' + other + '.json.pending')).write_text('{')
        with pytest.raises(BlockingIOError):
            with ReceiptStore(out, {'manifest': 'v1'}, [uid, other]):
                pass
    with ReceiptStore(out, {'manifest': 'v1'}, [uid, other]) as store:
        assert list(store.read()) == [uid]
        store.put({'uid': other, 'nll': 1.3})
        with pytest.raises(ValueError, match='replace'):
            store.put({'uid': uid, 'nll': 9.})
        assert set(store.read()) == {uid, other}
    with pytest.raises(ValueError, match='different'):
        with ReceiptStore(out, {'manifest': 'v2'}, [uid, other]):
            pass


def test_mechanism_simultaneous_family_includes_frozen_order_and_random_controls():
    assigned, observed = [], []
    for q in range(3):
        for arm in ('native', 'policy', 'random', 'reversed'):
            for seed in (0, 1):
                row = dict(uid=f'{q}-{arm}-{seed}', question=str(q), family=str(q), arm=arm, seed=seed)
                assigned.append(row)
                observed.append({**row, 'status': 'complete', 'semantic_success': int(arm == 'policy'),
                                 'correct': q % 2, 'tokens': 100 + q})
    report = paired_itt(assigned, observed, n_boot=1000,
                       comparison_pairs=[('policy', arm) for arm in ('native', 'random', 'reversed')])
    assert len(report['contrasts']) == 9
    assert all(r['multiplicity_family_size'] == 9 for r in report['contrasts'])
    assert {r['reference'] for r in report['contrasts']} == {'native', 'random', 'reversed'}


def test_new_manifest_requires_complete_enrollment_qualification_and_pricing():
    parent = [str(i) for i in range(48)]
    def seal(value):
        return {**value, 'sha256': digest(value)}
    freeze = seal({'new_parent_pools': {'parent_pools': {'discovery': parent},
        'representative_questions': {f: 'q' + f for f in parent}}})
    receipts = {f: {'eligible': True, 'question': 'q' + f, 'original_prompt_token_ids': [1],
                   'emitted_prefix_token_ids': [2, 3], 'detector_digest': 'd' * 64} for f in parent}
    args = dict(family_freeze=freeze, enrollment=receipts, code_digest='c' * 64,
        qualification=seal({'pass': True, 'worker_code_digest': 'c' * 64}),
        price=seal({'stage': 'discovery', 'status': 'PASS', 'ceiling': 2.}),
        actions=[action()], detector_digest='d' * 64)
    manifest = build_control_manifest('discovery', **args)
    assert len(manifest['requests']) == 48 * 2 * 2
    assert manifest['decode_token_cap'] == 48 * 2 * 2 * 1024
    assert manifest == build_control_manifest('discovery', **args)
    with pytest.raises(ValueError, match='qualification'):
        build_control_manifest('discovery', **{**args, 'qualification': {'pass': False}})
    receipts[parent[0]]['eligible'] = False
    with pytest.raises(ValueError, match='INSUFFICIENT_ELIGIBLE'):
        build_control_manifest('discovery', **args)


def test_semantic_trajectory_excludes_trigger_and_requires_substantive_ordered_events():
    transitions = ('candidate_to_verify', 'failed_check_to_revise')
    rows = [{'sentence_index': i, 'segment': 0, 'token_start': i * 10,
             'token_end': i * 10 + 9, 'label': 'Verify', 'behavioral_transition': event,
             'substantive': substantive} for i, (event, substantive) in enumerate([
                 (transitions[0], True), (transitions[1], True),
                 (transitions[0], False), (transitions[0], True), (transitions[1], True)])]
    assert trajectory_completion(rows, triggering_sentence=0, ordered_transitions=transitions)['success'] == 1
    assert trajectory_completion(rows[:3], triggering_sentence=0, ordered_transitions=transitions)['success'] == 0
    assert trajectory_completion(rows, triggering_sentence=0, ordered_transitions=transitions,
                                 horizon=40)['success'] == 0


def action(name='a', experts=((20, (1, 2)),), bias=.5):
    return Action(name, 'candidate_to_verify', experts, bias)


def test_sparse_actions_and_no_nondiscovery_fit():
    with pytest.raises(ValueError, match='adjacent'):
        action(experts=((20, (1,)), (22, (2,)))).validate()
    with pytest.raises(ValueError, match='one or two'):
        action(experts=((20, (1, 2, 3)),)).validate()
    with pytest.raises(ValueError, match='bias'):
        action(bias=2.).validate()
    contrasts = np.zeros((3, 40, 256))
    contrasts[:, 20, 1] = 1.
    proposed = propose_actions({'candidate_to_verify': contrasts}, source_families=['a', 'b', 'c'],
                               discovery_families=['a', 'b', 'c'])
    assert [a.bias for a in proposed] == [.5, 1.]
    assert all(a.experts == ((20, (1,)),) for a in proposed)
    with pytest.raises(ValueError, match='nondiscovery'):
        propose_actions({}, source_families=['confirm'], discovery_families=['a'])
    assert propose_actions({'candidate_to_verify': -contrasts}, source_families=['a', 'b', 'c'],
                           discovery_families=['a', 'b', 'c']) == []


def test_pulse_boundaries_order_closure_batch_neighbors_and_recompute():
    a, b = action(), action('b', ((20, (3, 4)),), 1.)
    template = Template('candidate_to_verify', (a, b), (0, 512)).validate()
    positions = [0, 255, 256, 511, 512, 767, 768, 1023]
    assert [template.action_at(p) for p in positions] == [a, a, None, None, b, b, None, None]
    assert template.reversed().action_at(0) == b
    logits = np.arange(10 * 256, dtype=np.float32).reshape(10, 256) / 100
    rows = [('edited', 0), ('native', 0), ('edited', 512), ('edited', 256), ('edited', 255),
            ('edited', 511), ('edited', 767), ('edited', 768), ('native', 512), ('edited', 0)]
    out = edit_logits(logits, rows, layer=20, templates={'edited': template},
                      closure_positions={'edited': 600})
    assert np.array_equal(out[[1, 3, 5, 6, 7, 8]], logits[[1, 3, 5, 6, 7, 8]])
    assert np.allclose(out[0, [1, 2]] - logits[0, [1, 2]], .5)
    assert np.allclose(out[2, [3, 4]] - logits[2, [3, 4]], 1.)
    assert np.array_equal(out, edit_logits(logits, rows, layer=20, templates={'edited': template},
                                         closure_positions={'edited': 600}))
    # The reference returns logits only; native top-k=8 selection and the shared expert
    # remain the responsibility of the unchanged native router.
    assert np.argsort(out[0])[-8:].shape == (8,)


def test_random_controls_have_matched_layer_support_and_balanced_assignment():
    controls = random_controls(action(), np.full((40, 256), .03), n_sets=4)
    assert len(controls) == 4
    assert all([l for l, _ in c.experts] == [20] and len(c.experts[0][1]) == 2 for c in controls)
    assert all(set(c.experts[0][1]).isdisjoint({1, 2}) for c in controls)
    assignments = balanced_random_assignment([str(i) for i in range(48)])
    for seed in (0, 1):
        assert [sum(r['seed'] == seed and r['random_set'] == k for r in assignments) for k in range(4)] == [12] * 4


def test_random_controls_fail_when_four_distinct_sets_cannot_be_frozen():
    rates = np.full((40, 256), .5)
    rates[20, [1, 2, 3, 4]] = .03
    with pytest.raises(ValueError, match='insufficient distinct random sets'):
        random_controls(action(), rates, n_sets=4)


def test_transitive_confirm_family_exclusion_and_disjoint_pools():
    groups = families(['a', 'b', 'c', 'd'], [{'a': 'a', 'b': 'b'}, {'a': 'b', 'b': 'c'}])
    pools = freeze_pools(groups, {'a': 'dev', 'b': 'tune', 'c': 'confirm', 'd': 'dev'})
    assert pools['eligible_family_count'] == 1
    assert pools['feasibility'] == 'FAIL_INSUFFICIENT_FAMILIES'
    assert len(pools['confirm_connected_excluded']) == 1


def test_full_stage_cost_rejects_overrun_and_unqualified_timing():
    args = dict(gpu_count=2, decode_tokens=100, prefill_tokens=100, decode_tps=10,
                prefill_tps=10, loads=1, load_seconds=100, measurement_seconds=10,
                retry_seconds=10, overhead_seconds=10)
    assert price_stage('mechanism', **args, throughput_qualified=False)['status'] == 'HOLD_UNQUALIFIED_TIMING'
    assert price_stage('mechanism', **args, throughput_qualified=True)['status'] == 'PASS'
    args['load_seconds'] = 10000
    assert price_stage('mechanism', **args, throughput_qualified=True)['status'] == 'STOP_REVISE_RESOURCE_PROPOSAL'


def test_native_nll_aligns_first_pulse_token_to_its_preceding_prefix():
    prompt = [1, 2, 3, 4]
    lp = [None, {2: SimpleNamespace(logprob=-1.)}, {3: SimpleNamespace(logprob=-2.)},
          {4: SimpleNamespace(logprob=-3.)}]
    assert actual_logprobs(prompt, lp).tolist() == [-1., -2., -3.]
    assert pulse_surprisal(prompt, lp, continuation_start=2, n_pulse_tokens=2)['mean_nll'] == 2.5
    with pytest.raises(ValueError, match='suffix'):
        pulse_surprisal(prompt, lp, continuation_start=1, n_pulse_tokens=2)
    with pytest.raises(ValueError, match='missing'):
        actual_logprobs(prompt, [None, {}, {}, {}])


def test_q3_repeat_rule_has_no_looser_fallback():
    assert validate_parity([[-1., -2.]], [[-1., -2.]], [[-1., -2.]], [[-1., -2.]])['pass']
    assert not validate_parity([[-1.0000001, -2.]], [[-1., -2.]], [[-1., -2.]], [[-1., -2.]])['pass']
    assert validate_parity([[-1.02, -2.02]], [[-1., -2.]], [[-1., -2.]], [[-1.02, -2.02]])['pass']
    assert not validate_parity([[-1.04, -2.04]], [[-1., -2.]], [[-1., -2.]], [[-1.02, -2.02]])['pass']
    with pytest.raises(ValueError, match='positions'):
        validate_parity([[-1.]], [[-1., -2.]], [[-1.]], [[-1.]])


def row(index, label, segment=0):
    return {'sentence_index': index, 'label': label, 'segment': segment,
            'token_start': index * 10, 'token_end': index * 10 + 10}


def test_dense_windows_exclude_trigger_and_never_bridge_gaps():
    rows = [row(0, 'Explore'), row(1, 'Verify'), row(2, 'Explore')]
    assert [r['label'] for r in dense_window(rows, triggering_sentence=0)] == ['Verify', 'Explore']
    assert class_summary(rows)['loop_counts'] == {'Explore|Verify|Explore': 1}
    assert not class_summary([rows[0], rows[2]])['loop_counts']
    with pytest.raises(ValueError, match='sparse gaps'):
        dense_window([rows[0], rows[2]], triggering_sentence=0)
    with pytest.raises(ValueError, match='missing first post-trigger'):
        dense_window([rows[2]], triggering_sentence=0)
    assert dense_window(rows[:2], triggering_sentence=0, horizon=15) == []


def test_fixed_window_velocity_and_acceleration():
    gates = np.array([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[0., 1.]], [[.5, .5]], [[.5, .5]]])
    result = routing_kinematics(gates, window=2)
    assert result['velocity'] == [.5, .25]
    assert result['acceleration'] == [-.125]


def test_itt_includes_nonfires_failures_and_keeps_family_clusters():
    assigned, observed = [], []
    for q in range(6):
        for arm in ('native', 'policy'):
            for seed in (0, 1):
                request = {'uid': f'{q}:{arm}:{seed}', 'question': str(q), 'family': str(q // 2),
                           'arm': arm, 'seed': seed}
                assigned.append(request)
                observed.append({**request, 'status': 'failure' if q == 0 else 'nonfire',
                                 'correct': int(q > 0), 'semantic_success': int(q > 2),
                                 'tokens': 100 if arm == 'native' else 90})
    result = paired_itt(assigned, observed, n_boot=100)
    assert result['statuses']['failure'] == 4
    assert result['contrasts'][2]['estimate'] == -10
    assert result['contrasts'][0]['n_families'] == 3
    assert result['contrasts'][0]['precision_status'] == 'DEGENERATE_BOOTSTRAP_NO_RETENTION_CLAIM'
    with pytest.raises(ValueError, match='receipt'):
        paired_itt(assigned, observed[:-1], n_boot=100)


def test_m9_closure_uses_real_newlines_and_charges_injection_to_the_same_budget():
    seen = []
    def encode(text):
        seen.append(text)
        return [10, 11, 12]
    planned = prepare_closure([1, 2], [3] * 15000, native_natural_stop=True, encode=encode)
    assert seen == ['</think>\n\n'] and CLOSURE_TEXT.endswith('\n\n')
    assert planned['max_new_tokens'] == 2045
    assert planned['prefix_len'] + planned['max_new_tokens'] == 16384
    retained = prepare_closure([1], [3] * 14336, native_natural_stop=True, encode=encode)
    assert retained['status'] == 'retain_native' and len(seen) == 1
