from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from moe_exp.routing_control.capture_v2 import CaptureStore, array_binding, begin_attempt, commit_attempt
from moe_exp.routing_control.study_v1 import digest, sealed
from moe_exp.routing_control.study_v2 import seal, RECOGNIZES
from moe_exp.routing_control.prefix import PrefixInput
from routing_local_worker_v2 import prediction_coordinates, validate_command
from routing_local_semantics_v2 import CheapRecognizer, proposals, messages, parse_rating, audit
from test_routing_study_v2 import boundary
from routing_local_engineering_v2 import save, actions


def test_engine_row_and_reasoning_coordinates_do_not_mix_prompt_prefix_and_offset():
    assert prediction_coordinates(140, 141, 632) == (0, -632)
    assert prediction_coordinates(772, 141, 632) == (632, 0)
    assert prediction_coordinates(800, 141, 632) == (660, 28)


def command(offset=0):
    return {'uid': 'a', 'offset': offset, 'position': 632 + offset, 'prefix_tokens': 632,
            'original_prompt_tokens': 141, 'arm': 'native', 'policy': 'native', 'action': None, 'input_sha256': 'x'}


def test_worker_rejects_native_interventions_stale_requests_and_skipped_rows():
    table = SimpleNamespace()
    validate_command(command(), None, table)
    validate_command(command(1), command(), table)
    for bad in ({**command(1), 'uid': 'b'}, command(2), {**command(1), 'prefix_tokens': 633}):
        with pytest.raises(ValueError):
            validate_command(bad, command(), table)
    with pytest.raises(ValueError, match='native assignment'):
        validate_command({**command(), 'policy': 'force', 'action': {}}, None, table)


def test_binary_capture_retains_both_ranks_both_paths_and_float32_bytes(tmp_path):
    store = CaptureStore(tmp_path / 'capture', 4)
    ids = np.tile(np.arange(8, dtype=np.uint16), (2, 40, 1))
    weights = np.full((2, 40, 8), .125, np.float32)
    for i in range(3):
        weights[1, 0, 0] = np.nextafter(np.float32(.125), np.float32(1.))
        store.append({'token_id': i + 1, 'row_position': 632 + i, 'policy_name': 'force',
            'rank_captures': [{'rank': r, 'ids': ids, 'weights': weights} for r in (0, 1)]}, np.zeros(40), np.zeros(40))
    binding = store.close()
    read = np.load(tmp_path / 'capture/weights.npy')[:, :, :3]
    assert read.dtype == np.float32 and binding['arrays']['weights'] == array_binding(read)
    assert binding['rows'] == 3 and read[0, 1, 0, 0, 0] != .125
    assert sealed(tmp_path / 'capture/BINDING.json') == binding


def test_process_death_requires_one_whole_assignment_retry_and_rejects_duplicate_success(tmp_path):
    assignment = {'uid': 'a', 'prefix_ids_sha256': digest([1, 2]), 'seed': 0}
    first = begin_attempt(tmp_path / 'a', assignment, 'manifest')
    second = begin_attempt(tmp_path / 'a', assignment, 'manifest')
    assert sealed(first / 'FAILURE.json')['continuation'] is False
    assert sealed(second / 'ASSIGNMENT.json')['retry_scope'] == 'WHOLE_ASSIGNMENT_FROM_ORIGINAL_PREFIX'
    with pytest.raises(ValueError, match='retry exhausted'):
        begin_attempt(tmp_path / 'a', assignment, 'manifest')
    commit_attempt(tmp_path / 'a', second, seal({'assignment_uid': 'a'}))
    with pytest.raises(ValueError, match='duplicate'):
        commit_attempt(tmp_path / 'a', first, seal({'assignment_uid': 'a'}))
    with pytest.raises(ValueError, match='successful'):
        begin_attempt(tmp_path / 'a', assignment, 'manifest')


def test_frozen_action_manifest_roundtrips_tuple_expert_ids_without_readback_failure(tmp_path):
    value = save(tmp_path / 'manifest.json', {'actions': [asdict(a) for a in actions()]})
    assert sealed(tmp_path / 'manifest.json') == value
    assert value['actions'][0]['experts'] == [9]


def test_symbolic_proposals_and_reader_parser_abstain_on_ambiguous_or_partial_inputs():
    assert proposals('Therefore $x = f(y)$.\n')['symbolic_candidate']
    assert not proposals('Therefore $x = f(y)\n')['symbolic_candidate']
    assert parse_rating('bad') is None
    value = {d: None for d in RECOGNIZES} | {'source_state': None}
    assert parse_rating(__import__('json').dumps(value)) == value
    assert parse_rating(__import__('json').dumps({**value, 'complete_checked': 1})) is None
    visible = {'reader_input': {'problem': 'q', 'emitted_prefix': 'x', 'triggering_sentence': 'x'}, 'gold': 'SECRET', 'future_text': 'FUTURE'}
    payload = str(messages(visible))
    assert 'SECRET' not in payload and 'FUTURE' not in payload
    prefix = PrefixInput.from_record({'problem': 'q', 'emitted_token_ids': [1], 'emitted_text': 'Therefore $x=f(y)$.\n'})
    assert CheapRecognizer()(prefix)['prefix_sha256'] == digest(asdict(prefix))


def audit_fixture():
    records = [{'uid': str(i), 'family': f'f{i}', 'split': 'apparent' if i < 10 else 'held_out',
                'proposals': {d: True for d in RECOGNIZES} | {'symbolic_candidate': True}}
               for i in range(12)]
    frame = seal({'records': records, 'splits_frozen_before_ratings': True, 'family_boundary': boundary()})
    ratings = [{'uid': str(i), 'reader': r, 'frame_sha256': frame['sha256'], 'finish_reason': 'stop',
                'parsed': {d: i < 7 or i >= 10 for d in RECOGNIZES}} for i in range(12) for r in (0, 1)]
    return frame, ratings


def test_apparent_seventy_percent_rule_with_held_out_reporting_and_missing_reader_abstention():
    frame, ratings = audit_fixture()
    audits, qualification = audit(frame, ratings)
    assert qualification['status'] == 'PASS' and all(a['confirmed_positive_count'] == 7 for a in audits)
    assert all(a['held_out_coverage'] == 1 for a in audits)
    incomplete, failed = audit(frame, ratings[1:])
    assert failed['status'] == 'FAIL' and all(a['missing'] == ['0'] for a in incomplete)
    with pytest.raises(ValueError, match='duplicate'):
        audit(frame, ratings + [ratings[0]])
    changed = deepcopy(ratings)
    changed[0]['frame_sha256'] = 'changed'
    with pytest.raises(ValueError, match='unbound'):
        audit(frame, changed)
