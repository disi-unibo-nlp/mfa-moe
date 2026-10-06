"""Preoutcome selection and complete-stage pricing invariants."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'
sys.path.insert(0, str(SCRIPTS))
import prepare_mechanism_start_frame_v1 as prepare
import price_mechanism_start_readers_v1 as pricing


def fixtures():
    families = [f'{i:064x}' for i in range(128)]
    units = []
    for index, family in enumerate(families):
        for sentence in range(5 if index == 0 else 1):
            units.append({'family': family, 'attempt_id': f'attempt-{index}',
                          'sentence_index': sentence, 'segment': 0,
                          'token_end': 10 + sentence,
                          'inputs': {'problem_statement': 'Question'}})
    events = []
    for sentence in range(4):
        events.append({'family': families[0], 'attempt_id': 'attempt-0',
                       'sentence_index': sentence, 'segment': 0,
                       'prefix_tokens': 10 + sentence, 'prefix_end_char': 50,
                       'event_end_char': 49, 'tokenizer_sha256': 'tok'})
    audit = {'schema': 'transition-detector-mechanism-audit-v1',
             'families': 128, 'units_sha256': 'units', 'detector_sha256': 'detector',
             'sha256': 'audit',
             'eligible_events': {'candidate_to_verify': events,
                                 'approach_to_commit': [events[0]],
                                 'failed_check_to_revise': [events[1]]}}
    return {'schema': 'dense-mechanism-units-v1', 'families': 128,
            'records': units, 'sha256': 'units'}, audit


def test_hash_shortlist_is_order_independent_and_keeps_empty_families(monkeypatch):
    monkeypatch.setattr(prepare, 'file_sha', lambda _: 'detector')
    units, audit = fixtures()
    one = prepare.select_events(units, audit)
    reordered = copy.deepcopy(audit)
    reordered['eligible_events']['candidate_to_verify'].reverse()
    two = prepare.select_events(units, reordered)
    assert one['records'] == two['records']
    assert len(one['family_pool']) == 128
    assert len(one['family_coverage']) == 128
    assert one['selected_counts'] == {'candidate_to_verify': 3,
                                      'approach_to_commit': 1}
    assert all(r['transition'] != 'failed_check_to_revise' for r in one['records'])
    assert all(sum(one['family_coverage'][f].values()) == 0
               for f in one['family_pool'][1:])


def test_selection_rejects_event_rebound_to_other_native_sentence(monkeypatch):
    monkeypatch.setattr(prepare, 'file_sha', lambda _: 'detector')
    units, audit = fixtures()
    audit['eligible_events']['candidate_to_verify'][0]['prefix_tokens'] = 999
    with pytest.raises(ValueError, match='differs from sealed native unit'):
        prepare.select_events(units, audit)


def test_complete_price_counts_two_exact_prompts_and_worst_case_decoding():
    frame = {'schema': 'mechanism-start-frame-v1', 'selection_sha256': 'selection',
             'families_in_frozen_pool': 128, 'rows': 2, 'sha256': 'frame'}
    selection = {'sha256': 'selection', 'records': [{}, {}]}
    prior_price = {'schema': 'transition-v22-fullprefix-start-rating-price-v2',
                   'sha256': 'old-price', 'frame_sha256': 'old-frame',
                   'bounded_decode_tokens_per_s': 40,
                   'bounded_prefill_tokens_per_s': 1000,
                   'components_seconds': {'two_cold_loads': 100,
                                          'two_shutdowns': 50}}
    prior_binding = {'sha256': 'old-binding', 'frame_sha256': 'old-frame'}
    prior_summary = {'sha256': 'old-summary', 'binding_sha256': 'old-binding',
                     'counts': {'generated_tokens': 400},
                     'timings': [{'reader_timings': [{'generated_tokens': 400,
                                                     'wall_seconds': 10}]}]}
    value = pricing.price(frame, selection, prior_price, prior_summary,
                          prior_binding, [100, 200])
    assert value['prompt_tokens_exact_twice'] == 600
    assert value['max_decode_tokens'] == 4 * 1024
    assert value['bounded_decode_tps'] == 32
    assert value['complete_stage_projected_wall_seconds'] > 150
    with pytest.raises(ValueError, match='context mismatch'):
        pricing.price(frame, selection, prior_price, prior_summary,
                      prior_binding, [49152, 200])
