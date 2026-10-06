from scripts.experimental_resume.analyze_transition_ratings import reconcile, summarize_transition


def _row(uid, fired, start_a, start_b, target=False):
    original = {'uid': uid, 'transition': 'candidate_to_verify',
                'reader_input': {'problem': 'P', 'previous_sentence': '',
                                 'triggering_sentence': 'The answer is 2.',
                                 'later_sentence': 'Substitute 2.'},
                'analysis_meta': {'family': uid, 'detector_fired': fired,
                                  'prefix_tokens': 100, 'event_end': 16}}
    labeled = {**original, 'analysis_meta': {**original['analysis_meta'],
                                             'source_class_audit': 'Analyze',
                                             'later_class_audit': 'Verify'}}
    rated = {'uid': uid, 'transition': original['transition'],
             'readers': [{'finish_reason': 'stop', 'rating': {'start': s, 'target': target}}
                         for s in (start_a, start_b)]}
    return original, labeled, rated


def test_reconcile_rejects_changed_transition_or_visible_text():
    original, labeled, rated = _row('a', True, True, True)
    frame = {'schema': 'transition-audit-candidates-unlabeled-v1', 'sha256': 'frame',
             'families': 1, 'contiguous_pairs': 1, 'frame_counts': {'candidate_to_verify': {'fired': 1}},
             'detector_sha256': 'detector', 'records': [original]}
    enriched = {'schema': 'transition-audit-fixtures-v1', 'families': 1,
                'contiguous_pairs': 1, 'frame_counts': frame['frame_counts'],
                'detector_sha256': 'detector', 'records': [labeled]}
    binding = {'schema': 'transition-LLM-audit-binding-v2', 'fixtures_sha256': 'frame',
               'sha256': 'binding'}
    summary = {'schema': 'transition-LLM-audit-summary-v2', 'binding_sha256': 'binding',
               'rows': 1}
    parts = [{'records': [rated]}]
    assert len(reconcile(frame, enriched, binding, summary, parts)) == 1
    bad = {**rated, 'transition': 'approach_to_commit'}
    import pytest
    with pytest.raises(ValueError):
        reconcile(frame, enriched, binding, summary, [{'records': [bad]}])
    changed = {**labeled, 'reader_input': {**labeled['reader_input'], 'problem': 'different'}}
    with pytest.raises(ValueError):
        reconcile(frame, {**enriched, 'records': [changed]}, binding, summary, parts)


def test_full_fire_counts_and_sample_weight():
    a = _row('a', True, True, True)
    b = _row('b', True, True, False)
    c = _row('c', False, True, True)
    d = _row('d', False, False, False)
    result = summarize_transition([a, b, c, d], {'fired': 2, 'nonfire': 8})
    assert result['families_all_fires'] == 2
    assert result['families_with_both_reader_valid_fires'] == 1
    assert result['fired_start_precision_both_readers'] == .5
    assert result['sampled_nonfire_inverse_probability_weight'] == 4
    assert result['estimated_missed_starts_from_nonfire_sample'] == 4
    assert result['estimated_start_recall_two_reader_audit'] == .2
    assert result['fired_class_pair_counts'] == {'Analyze→Verify': 2}
