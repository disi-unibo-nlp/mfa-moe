"""Blind semantic reader must accept partial gradeability without arm leakage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/'
          'rate_eligible_immediate_semantics_v1.py')
SPEC = importlib.util.spec_from_file_location('eligible_rating_v1', SCRIPT)
rating = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rating)


def fixture(monkeypatch):
    monkeypatch.setattr(rating, 'file_sha', lambda path: {
        str(rating.RUBRIC): 'rubric', str(SCRIPT): 'driver', 'frame.json': 'file',
    }[str(path)])
    row = {'blind_id': 'blind-1', 'reader_input': {
        'transition': 'candidate_to_verify', 'problem': 'Find x.',
        'full_emitted_prefix': 'Suppose x=3.',
        'triggering_sentence': 'Suppose x=3.',
        'continuation': 'Substituting 3 gives the required value.'}}
    frame = {'schema': 'eligible-immediate-blind-frame-v1',
             'records': [row], 'sha256': 'frame-seal',
             'reader_input_allowlist': sorted(rating.ALLOWLIST),
             'rubric_sha256': 'rubric', 'continuation_max_tokens': 256}
    price = {'schema': 'eligible-immediate-blind-rating-price-v1',
             'status': 'PASS_COMPLETE_STAGE', 'frame_sha256': 'frame-seal',
             'frame_file_sha256': 'file', 'driver_sha256': 'driver',
             'rubric_sha256': 'rubric', 'assigned_requests': 156,
             'gradeable_requests': 1, 'ratings': 2, 'max_decode_tokens': 2048}
    return frame, price


def test_partial_gradeability_is_priced_and_forbidden_fields_are_rejected(monkeypatch):
    frame, price = fixture(monkeypatch)
    assert rating.validate(frame, price, Path('frame.json')) == frame['records']
    frame['records'][0]['reader_input']['arm'] = 'target_bias1'
    with pytest.raises(ValueError, match='allowlist'):
        rating.validate(frame, price, Path('frame.json'))


def test_rating_parser_requires_exact_boolean_json():
    assert rating.parse_rating('<think>check</think>{"target": true}') is True
    assert rating.parse_rating('{"target": false}') is False
    assert rating.parse_rating('{"target": true, "arm": "target"}') is None
    assert rating.parse_rating('verification happened') is None


def test_failed_and_interrupted_rating_attempts_remain_cost_liabilities(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    attempts = tmp_path / 'attempts'
    attempts.mkdir()
    receipts = []
    for index, count in enumerate((1, 2, 3)):
        receipts.append(rating.write_once(attempts / f'target-{index:03d}-reader0-attempt000.json', {
            'binding_sha256': 'binding', 'blind_ids': [f'id-{index}-{i}' for i in range(count)]}))
    rating.write_once(attempts / 'target-001-reader0-attempt000-failure.json', {
        'binding_sha256': 'binding', 'attempt_receipt_sha256': receipts[1]['sha256'],
        'elapsed_seconds': 7.5, 'max_decode_tokens_at_risk': 2 * rating.MAX_TOKENS})
    got = rating.attempt_accounting(attempts, 'binding', [
        {'attempt_receipt_sha256': receipts[0]['sha256']}])
    assert (got['attempts'], got['successful_attempts'], got['failed_attempts'],
            got['interrupted_attempts']) == (3, 1, 1, 1)
    assert got['known_failed_attempt_seconds'] == 7.5
    assert got['unknown_decode_token_upper_bound'] == 5 * rating.MAX_TOKENS
