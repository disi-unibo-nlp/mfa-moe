"""Versioned routing/measurement audit and reuse of the 42 reviewed continuations."""
import json
from pathlib import Path
import re

import native_finalization_v1 as N


def architecture(config):
    c = config.get('text_config', config)
    k = c.get('num_experts_per_tok', c.get('num_selected_experts'))
    e = c.get('num_experts', c.get('n_routed_experts', c.get('num_local_experts')))
    shared = {key: c[key] for key in ('n_shared_experts', 'shared_expert_intermediate_size',
                                     'moe_shared_expert_intermediate_size') if key in c}
    return {'routed_experts_per_token': k, 'total_routed_experts': e,
            'configured_count_status': 'KNOWN' if k is not None else 'UNKNOWN',
            'shared_expert_config': shared, 'shared_expert_handling': 'Separate branch, excluded from routed k.'}


def routing_observation(configured_k, expert_count, native_ids=None, executed_weights=None,
                        affinity=None, selection_scores=None, deterministic_topk=False, mask=None):
    """No affinity argmax fallback. Input arrays describe [row, expert/selected]."""
    import numpy as np
    N.require(type(configured_k) is int and 1 <= configured_k <= expert_count, 'invalid configured k')
    result = {'configured_experts_per_token': configured_k,
        'native_selection_status': 'UNKNOWN' if native_ids is None else 'CAPTURED',
        'executed_weights_status': 'UNKNOWN' if executed_weights is None else 'CAPTURED',
        'native_ids': None, 'executed_weights': None, 'top1_top2_affinity_gap': None,
        'affinity_k_gap': None, 'genuine_selection_cutoff_gap': None, 'active_mask': None,
        'selection_semantics': 'deterministic_topk' if deterministic_topk else 'unspecified/non-ranked'}
    ids = None
    if native_ids is not None:
        ids = np.asarray(native_ids)
        N.require(ids.ndim == 2 and ids.shape[1] == configured_k and np.issubdtype(ids.dtype, np.integer)
                  and (ids >= 0).all() and (ids < expert_count).all() and
                  not (np.diff(np.sort(ids, axis=1), axis=1) == 0).any(), 'invalid native selections')
        result['native_ids'] = ids.tolist()
    if executed_weights is not None:
        w = np.asarray(executed_weights)
        N.require(ids is not None and w.shape == ids.shape and np.isfinite(w).all() and
                  (w >= 0).all() and (w.sum(-1) > 0).all(), 'weights without matching captured IDs')
        result['executed_weights'] = w.tolist()
    if mask is not None:
        m = np.asarray(mask)
        N.require(ids is not None and m.dtype == bool and m.shape == (len(ids),), 'invalid row mask')
        result['active_mask'] = m.tolist()
    if affinity is not None:
        a = np.asarray(affinity)
        N.require(a.ndim == 2 and a.shape[1] == expert_count and np.isfinite(a).all() and
                  (ids is None or len(a) == len(ids)), 'invalid affinities')
        ordered = np.sort(a, axis=-1)[:, ::-1]
        if expert_count > 1:
            result['top1_top2_affinity_gap'] = (ordered[:, 0] - ordered[:, 1]).tolist()
        if configured_k < expert_count:
            result['affinity_k_gap'] = (ordered[:, configured_k - 1] - ordered[:, configured_k]).tolist()
    if selection_scores is not None and deterministic_topk and ids is not None:
        s = np.asarray(selection_scores, float)
        N.require(s.shape == (len(ids), expert_count) and np.isfinite(s).all(), 'invalid selection scores')
        if configured_k < expert_count:
            selected_min = np.take_along_axis(s, ids, 1).min(1)
            other = s.copy(); np.put_along_axis(other, ids, -np.inf, 1)
            gap = selected_min - other.max(1)
            N.require((gap >= -1e-6).all(), 'selection scores disagree with native selections')
            result['genuine_selection_cutoff_gap'] = np.maximum(gap, 0).tolist()
    return result


def prediction_rows(prompt_tokens, absolute_input_positions):
    return [p - prompt_tokens + 1 if p >= prompt_tokens - 1 else None for p in absolute_input_positions]


def presence_restored(prefix_ids, emitted_ids, penalty=1.5):
    """Reference union: one penalty for each previously present token, not twice."""
    return {token: penalty for token in set(prefix_ids) | set(emitted_ids)}


def first_candidate(case, completion):
    # Reviewed, independently checkable expressions only. No generative judge.
    patterns = {1: r'x\s*=\s*4', 4: r'x\s*=\s*6',
                5: r'x\s*=\s*\\frac\{3\^2\s*\+\s*3\^4\}\{2\}', 6: r'w\s*=\s*6\s*-\s*5i'}
    pattern = patterns.get(case['case'])
    if pattern is None:
        return {'status': 'NOT_ESTABLISHED', 'region': None, 'character_offset': None}
    for region, text in [('prefix', case['decoded_saved_prefix']), ('continuation', completion['text'].split('</think>')[0])]:
        match = re.search(pattern, text)
        if match:
            return {'status': 'FIRST_REVIEWED_CORRECT_COMPLETE_EXPRESSION', 'region': region,
                'character_offset': match.start(), 'character_end': match.end(), 'expression': match.group(),
                'subsequent_reasoning_characters': len(text) - match.end() +
                    (len(completion['text'].split('</think>')[0]) if region == 'prefix' else 0),
                'precision': 'Character offset in saved decoded text; no invented token alignment.',
                'is_final_answer_grade': False}
    return {'status': 'NOT_LOCATED', 'region': None, 'character_offset': None}


def saved_audit():
    from analyze_completion_audit_v1 import reference_checks
    folder = N.U.REPO / 'report/experimental-resume-v1/completion-audit-v1'
    sample = N.U.sealed(folder / 'SAMPLE.json'); findings = N.U.sealed(folder / 'FINDINGS.json')
    N.require(findings['sample_sha256'] == sample['sha256'], 'review source changed')
    reviewed = {r['case']: r for r in findings['findings']}
    trigger = {1: ('setup_or_redraft', 'already_checked_answer'),
               2: ('setup', 'no_candidate'), 3: ('setup', 'no_candidate'),
               4: ('candidate_set_in_redraft', 'already_checked_answer'),
               5: ('complete_candidate_expression', 'complete_candidate_unchecked'),
               6: ('intermediate_result', 'intermediate_result'),
               7: ('intermediate_result', 'intermediate_result')}
    batches = {}; rows = []; dose_rows = []
    for case in sample['cases']:
        outputs = {}
        for receipt in case['raw_batch_receipts']:
            path = receipt['path']
            if path not in batches:
                N.require(N.U.file_sha(Path(path)) == receipt['file_sha256'], 'saved batch bytes changed')
                batches[path] = N.U.sealed(Path(path))
                N.require(batches[path]['sha256'] == receipt['seal'], 'batch seal changed')
            outputs.update({r['uid']: r for r in batches[path]['outputs']})
        for completion in case['completions']:
            saved = outputs[completion['uid']]
            N.require(N.U.digest(saved['tokens']) == completion['token_ids_sha256'], 'saved completion changed')
            rows.append({'case': case['case'], 'uid': completion['uid'], 'arm': completion['arm'],
                'trigger_text': case['triggering_sentence'], 'trigger_meaning': trigger[case['case']][0],
                'preceding_prefix_state': trigger[case['case']][1],
                'case_level_patterns': reviewed[case['case']]['patterns_observed_in_case'],
                'case_level_observation': reviewed[case['case']]['manual_observation'],
                'first_complete_correct_candidate': first_candidate(case, completion),
                'reasoning_tokens': completion['reasoning_tokens'], 'generated_tokens': completion['generated_tokens'],
                'scope': 'Case-level reviewed categories are not newly assigned per-completion labels.'})
            ranks = saved.get('action_dose', {})
            rank_status = 'UNKNOWN'
            if set(ranks) == {'0', '1'}:
                def leaves(x):
                    if isinstance(x, dict):
                        return {k: leaves(v) for k, v in x.items()}
                    return x
                import numpy as np
                def agree(a, b):
                    if isinstance(a, dict):
                        return isinstance(b, dict) and set(a) == set(b) and all(agree(a[k], b[k]) for k in a)
                    if isinstance(a, list):
                        return isinstance(b, list) and len(a) == len(b) and all(agree(x, y) for x, y in zip(a, b))
                    return bool(np.isclose(a, b, rtol=1e-5, atol=1e-3)) if isinstance(a, (float, int)) else a == b
                rank_status = 'AGREE' if agree(ranks['0'], ranks['1']) else 'DISAGREE'
            dose_rows.append({'uid': completion['uid'], 'arm': completion['arm'], 'ranks': ranks,
                'rank_agreement': rank_status, 'statistical_observations': 1,
                'weight_trajectory_status': 'UNAVAILABLE; saved aggregate target mass and L1 displacement only'})
    configs = {}
    for path in (N.U.REPO.parent / 'cache/hf/hub').glob('*/snapshots/*/config.json'):
        a = architecture(json.loads(path.read_text()))
        if a['routed_experts_per_token'] is not None:
            configs[str(path)] = {'file_sha256': N.U.file_sha(path), **a}
    sources = [Path(__file__), N.BASE / 'moe_steer/logits.py', N.BASE / 'moe_steer/engine.py',
        Path(__file__).with_name('utility_routing_worker_v2.py'), Path(__file__).with_name('overnight_routing_dose_audit_v3.py'),
        Path(__file__).with_name('overnight_routing_runner_v2.py')]
    return N.save(N.DOC / 'SAVED_AUDIT.json', {'schema': 'native-finalization-saved-audit-v1',
        'sample_sha256': sample['sha256'], 'review_sha256': findings['sha256'], 'rows': rows,
        'reference_checks': reference_checks(), 'models': configs, 'saved_dose': dose_rows,
        'source_files': {str(p): N.U.file_sha(p) for p in sources},
        'measurement_audit': {'prediction_row': 'input_position - original_prompt_tokens + 1; last prompt row predicts output zero',
            'prefill': 'Earlier input rows excluded from output-token routing; last prefill row included.',
            'pulse': '256 prediction rows; closure-producing row included, all following answer rows inactive.',
            'presence': 'Original prompt uses built-in 1.5. Continuation uses built-in 0 plus PrefixPresencePenalty over prefix/generated union.',
            'trigger': 'Frozen rubric permits intermediate values; case 6 is rotated vector before adding center, case 7 is radius before area.',
            'gap_semantics': 'Top1/top2 preference and affinity-k gaps are not native selection-cutoff gaps without native IDs and exact selection scores.'},
        'demonstrated_execution_defects': [], 'affected_results_requiring_correction': [],
        'limitations': ['No demonstrated execution defect in this bounded audit; this is not exhaustive proof.',
            'No unavailable trajectories reconstructed; TP ranks retained separately.',
            '42 reviewed 1024-token continuations cannot estimate population incidence or full 16384-token lengths.',
            'Correct candidate annotations never replace final-answer grading.']})
