"""Recovery selects failures without arms and preserves cap/seed/source contracts."""
from pathlib import Path
import copy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))
import mechanism_semantic_cap_recovery_v1 as recovery


def source_row(uid='a'):
    return {'blind_id': uid, 'reader_input': {'transition': 'candidate_to_verify',
            'problem': 'Original problem', 'full_emitted_prefix': 'A candidate is 4.',
            'triggering_sentence': 'A candidate is 4.', 'continuation': 'Now substitute.'}}


def original_record(raw, finish='stop', tokens=10):
    return {'rating': recovery.original.parse_rating(raw), 'raw_completion': raw,
            'finish_reason': finish, 'generated_tokens': tokens}


def test_selects_every_capped_and_malformed_rating_without_target_or_arm_selection():
    frame = {'records': [source_row('b'), source_row('a')]}
    votes = {'b': {0: original_record('still thinking', 'length', 1024),
                   1: original_record('{"target":true}')},
             'a': {0: original_record('broken JSON'), 1: original_record('{"target":false}')}}
    selected = recovery.select_failures(frame, votes)
    assert [r['uid'] for r in selected] == ['a|reader0', 'b|reader0']
    assert [r['selection_reason'] for r in selected] == ['malformed_stop', 'length_cap']
    assert all(r['seed'] == recovery.original.rating_seed(r['blind_id'], 0) for r in selected)
    votes['b'][1] = original_record('{"target":false}')
    votes['a'][1] = original_record('{"target":true}')
    assert recovery.select_failures(frame, votes) == selected


def test_selection_rejects_forged_parser_results_and_invalid_original_caps():
    frame = {'records': [source_row()]}
    votes = {'a': {0: original_record('thinking', 'length', 1023),
                   1: original_record('{"target":false}')}}
    with pytest.raises(ValueError, match='cap'):
        recovery.select_failures(frame, votes)
    votes['a'][0]['generated_tokens'] = 1024
    votes['a'][1]['rating'] = {'target': True}
    with pytest.raises(ValueError, match='parser'):
        recovery.select_failures(frame, votes)


def test_forbidden_fields_stay_out_of_reader_messages():
    row = source_row(); row['reader_input']['arm'] = 'target'
    with pytest.raises(ValueError, match='allowlist'):
        recovery.original.messages(row)


def old_price():
    return {'bounded_prefill_tokens_per_second': 4914., 'bounded_decode_tokens_per_second': 43.71,
            'repeat_work_factor': 1.25, 'cold_load_seconds': 599.25, 'shutdown_seconds': 196.}


def test_complete_pricing_covers_every_uid_and_counts_loads_recovery_and_long_cap():
    rows = [{'prompt_tokens': 3900}] * 180
    price = recovery.projected(rows, old_price())
    assert len(price['shards']) == 5
    assert price['max_decode_tokens'] == 180 * 4096
    assert [i for s in price['shards'] for i in range(s['start'], s['end'])] == list(range(180))
    assert all(s['work_seconds'] + 599.25 + 196 + 900 <= 7200 for s in price['shards'])
    assert 14 < price['estimated_complete_gpu_hours'] < 16
    qual = recovery.projected(rows[:4], old_price(), wall=3600)
    assert len(qual['shards']) == 1 and qual['recovery_loads'] == 1
    assert qual['estimated_complete_gpu_hours'] + price['estimated_complete_gpu_hours'] < 20


def test_one_unpriceable_batch_fails_closed():
    with pytest.raises(ValueError, match='batch exceeds'):
        recovery.projected([{'prompt_tokens': 49000}] * 8, old_price(), wall=1800)


def committed_fixture(workdir):
    req = {'uid': 'a|reader0', 'reader': 0, 'blind_id': 'a', 'seed': 11,
           'original_record_sha256': 'old', 'prompt_ids_sha256': 'prompt'}
    plan = {'sha256': 'manifest', 'requests': [req], 'qualification_indices': [0]}
    dest = recovery.directory(workdir, 'qualification', 0); dest.mkdir()
    binding = recovery.save(dest/'BINDING.json', recovery.binding_for(plan, 'qualification', 0))
    attempt = recovery.save(dest/'attempt-0000-000.json', {'indices': [0], 'seeds': [11],
                           'binding_sha256': binding['sha256'], 'attempt_index': 0})
    record = {'uid': 'a|reader0', 'reader': 0, 'blind_id': 'a', 'source_record_sha256': 'old',
              'prompt_ids_sha256': 'prompt', 'raw_completion': 'reason</think>{"target":true}',
              'rating': {'target': True}, 'generated_tokens': 2, 'token_ids': [1, 2],
              'finish_reason': 'stop'}
    batch = {'binding_sha256': binding['sha256'], 'indices': [0], 'attempt_index': 0,
             'attempt_sha256': attempt['sha256'], 'records': [record]}
    recovery.save(dest/'batch-0000.json', batch)
    return plan, dest, batch


def test_committed_receipts_enforce_prompt_seed_cap_and_parser(workdir):
    plan, dest, batch = committed_fixture(workdir)
    assert recovery.completed(plan, workdir, 'qualification', 0)[0][0]['rating'] == {'target': True}
    for key, value in [('prompt_ids_sha256', 'wrong'), ('generated_tokens', 4097),
                       ('finish_reason', 'length'), ('rating', {'target': False})]:
        wrong = copy.deepcopy(batch); wrong['records'][0][key] = value
        (dest/'batch-0000.json').unlink(); recovery.save(dest/'batch-0000.json', wrong)
        with pytest.raises(ValueError, match='cap/parser/prompt'):
            recovery.completed(plan, workdir, 'qualification', 0)


def test_preserves_original_boolean_parser_contract():
    assert recovery.original.parse_rating('reasoning</think>{"target":true}') == {'target': True}
    assert recovery.original.parse_rating('{"target":"true"}') is None
    assert recovery.original.parse_rating('{"target":true,"explanation":"x"}') is None
    assert recovery.original.parse_rating('unfinished reasoning') is None
