"""Crash-recovery evidence for the versioned mechanism reader driver."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'
sys.path.insert(0, str(SCRIPTS))
import rate_mechanism_start_readers_v2 as reader


def setup(out):
    for name in ('attempts', 'assignments', 'batches'):
        (out / name).mkdir(parents=True, exist_ok=True)
    return [{'uid': 'a'}, {'uid': 'b'}]


def attempt(out, block, index, binding='binding'):
    previous = reader.attempt_ledger(out, 0, 0, block, binding)
    return reader.save(out / 'attempts' / f'000000-reader0-attempt{index:03d}.json',
                       {'schema': 'mechanism-start-reader-attempt-v2',
                        'binding_sha256': binding, 'start': 0, 'reader': 0,
                        'attempt_index': index,
                        'recovery_of_uncommitted_attempts': [a['sha256'] for a in previous],
                        'uids': [r['uid'] for r in block],
                        'seeds': [reader.rating_seed(r['uid'], 0) for r in block],
                        'job_id': 'job', 'state': 'started_before_model_chat'})


def test_recovery_preserves_incomplete_attempt_and_commits_one_result(tmp_path):
    block = setup(tmp_path)
    original = attempt(tmp_path, block, 0)
    # Crash after only one of the pre-call per-UID receipts was written.
    reader.save(tmp_path / 'assignments/a-reader0-attempt000.json',
                {'schema': 'mechanism-start-rating-assignment-v2',
                 'binding_sha256': 'binding', 'uid': 'a', 'reader': 0,
                 'start': 0, 'attempt_index': 0,
                 'attempt_sha256': original['sha256'], 'state': 'attempted_before_model_chat',
                 'job_id': 'job'})
    assert len(reader.attempt_ledger(tmp_path, 0, 0, block, 'binding')) == 1
    recovered = attempt(tmp_path, block, 1)
    for row in block:
        reader.save(tmp_path / 'assignments' / f"{row['uid']}-reader0-attempt001.json",
                    {'schema': 'mechanism-start-rating-assignment-v2',
                     'binding_sha256': 'binding', 'uid': row['uid'], 'reader': 0,
                     'start': 0, 'attempt_index': 1,
                     'attempt_sha256': recovered['sha256'],
                     'state': 'attempted_before_model_chat', 'job_id': 'job'})
    committed = reader.save(tmp_path / 'batches/000000-reader0.json',
                            {'schema': 'mechanism-start-reader-batch-v2',
                             'binding_sha256': 'binding', 'start': 0, 'reader': 0,
                             'attempt_index': 1, 'attempt_sha256': recovered['sha256'],
                             'timing': {}, 'records': [
                                 {'uid': r['uid'], 'result': {'rating': {'start': True}}}
                                 for r in block]})
    assert reader.committed_reader(tmp_path, 0, 0, block, 'binding') == committed
    assert original['sha256'] != recovered['sha256']
    assert len(reader.attempt_ledger(tmp_path, 0, 0, block, 'binding')) == 2
    with pytest.raises(FileExistsError):
        attempt(tmp_path, block, 0)


def test_recovery_rejects_missing_attempt_and_changed_uid(tmp_path):
    block = setup(tmp_path)
    attempt(tmp_path, block, 0)
    # A torn or corrupted attempt index cannot be accepted as a valid chain.
    reader.save(tmp_path / 'attempts/000000-reader0-attempt002.json',
                {'schema': 'mechanism-start-reader-attempt-v2',
                 'binding_sha256': 'binding', 'start': 0, 'reader': 0,
                 'attempt_index': 2,
                 'recovery_of_uncommitted_attempts': [],
                 'uids': [r['uid'] for r in block],
                 'seeds': [reader.rating_seed(r['uid'], 0) for r in block],
                 'job_id': 'job', 'state': 'started_before_model_chat'})
    with pytest.raises(ValueError, match='foreign|not contiguous'):
        reader.attempt_ledger(tmp_path, 0, 0, block, 'binding')
    with pytest.raises(ValueError, match='foreign'):
        reader.attempt_ledger(tmp_path, 0, 0, [{'uid': 'a'}, {'uid': 'other'}], 'binding')
