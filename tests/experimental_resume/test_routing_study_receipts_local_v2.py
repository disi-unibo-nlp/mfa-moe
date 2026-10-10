from pathlib import Path
import pytest

from moe_exp.routing_control.study_v1 import digest, save
from moe_exp.routing_control.study_v2 import seal
from routing_study_receipts_local_v2 import blind_grade_rows, mechanism_receipt


def fixture(finish='stop', engineering=False):
    prefix = [1, 2]
    assignment = {'uid': 'a', 'question': 'olympiad|q', 'prefix_ids_sha256': digest(prefix)}
    receipt = seal({'assignment_uid': 'a', 'continuation_cap': 16384, 'engineering_only': engineering,
        'emitted_token_ids': [3, 248046], 'generated_tokens': 2, 'finish_reason': finish,
        'controller': {'controller_tokens': 7}})
    return [assignment], {'a': seal({'receipt': receipt})}, {digest(prefix): prefix}, {
        'olympiad|q': {'dataset': 'olympiad', 'problem': 'q', 'gold': 'GOLD_ONLY_OFFLINE'}}


def test_new_receipts_enter_existing_blind_grading_format_with_bound_prefix_and_cost():
    bundle, mapping, missing = blind_grade_rows(*fixture(), decode=lambda ids: str(ids))
    assert len(bundle) == len(mapping) == 1 and not missing
    assert bundle[0]['gold'] == 'GOLD_ONLY_OFFLINE' and bundle[0]['content_text'] == '[1, 2, 3, 248046]'
    assert not set(bundle[0]) & {'arm', 'policy', 'seed', 'prefix'}
    assert mapping[0]['generated_tokens'] == 2 and mapping[0]['controller_tokens'] == 7


def test_engineering_caps_and_missing_results_cannot_become_scientific_grades():
    with pytest.raises(ValueError, match='unqualified'):
        blind_grade_rows(*fixture(engineering=True), decode=str)
    args = list(fixture(finish='length'))
    bundle, _, missing = blind_grade_rows(*args, decode=str)
    assert not bundle and missing[0]['correct'] is False
    args[1] = {}
    bundle, mapping, missing = blind_grade_rows(*args, decode=str)
    assert not bundle and not mapping and missing[0]['status'] == 'MISSING_EXECUTION'


def test_offline_grading_rejects_changed_prefix_and_cross_assignment_receipt():
    args = list(fixture())
    args[2] = {args[0][0]['prefix_ids_sha256']: [2, 1]}
    with pytest.raises(ValueError, match='prefix'):
        blind_grade_rows(*args, decode=str)
    args = list(fixture())
    args[0][0]['uid'] = 'b'
    args[1]['b'] = args[1].pop('a')
    with pytest.raises(ValueError, match='receipt'):
        blind_grade_rows(*args, decode=str)


def test_engineering_capture_is_rejected_before_semantic_mechanism_measurement(tmp_path, monkeypatch):
    import routing_study_receipts_local_v2 as adapter
    monkeypatch.setattr(adapter, 'compute', lambda: None)
    path = tmp_path / 'SUCCESS.json'
    save(path, {k: v for k, v in fixture(engineering=True)[1]['a'].items() if k != 'sha256'})
    with pytest.raises(ValueError, match='engineering fixtures'):
        mechanism_receipt(path, [], 'candidate_to_verify', tmp_path / 'endpoint.json')


def test_incomplete_pilot_preserves_unavailable_component_prices(tmp_path):
    from routing_study_prices_local_v2 import measured_components
    pilot = tmp_path / 'pilot'
    pilot.mkdir()
    save(pilot / 'PILOT.json', {'status': 'EXPLICIT_INCOMPLETE_OR_FAILED', 'pricing_cells': []})
    status = measured_components(pilot, tmp_path / 'prices')
    assert status['status'] == 'UNAVAILABLE_MEASURED_PRICE'
    assert not status['measurements'] and not status['scientific_dispatch_authorized']
    assert {'generation', 'replay', 'grading', 'recovery'} <= set(status['unavailable_roles'])


def test_accounting_retains_steps_without_confusing_gpu_and_cpu_units(monkeypatch):
    import routing_study_receipts_local_v2 as adapter
    from types import SimpleNamespace
    monkeypatch.setattr(adapter.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=
        '123|COMPLETED|0:0|3600|cpu=16,gres/gpu=2,mem=120G|\n'
        '123.0|COMPLETED|0:0|3500|cpu=16,gres/gpu=2|\n'
        '124|FAILED|1:0|18|cpu=2,mem=32G|\n'))
    result = adapter.accounting(['123', '124'])
    parents = [r for r in result['rows'] if '.' not in r['job_id']]
    assert len(parents) == 2 and len(result['rows']) == 3
    assert sum(r['GPU_hours'] for r in parents) == 2
    assert parents[1]['CPU_core_hours'] == .01


def price_fixture(tmp_path):
    pilot = tmp_path / 'pilot'
    (pilot / 'generator').mkdir(parents=True)
    execution = save(pilot / 'EXECUTION.json', {'status': 'PASS'})
    save(pilot / 'generator/RUNTIME.json', {'code_files': {'worker': 'frozen'}})
    save(pilot / 'generator/COST.json', {'gpus': 2, 'load_seconds': 10., 'metrics': {
        'prefill_seconds': 8., 'decode_seconds': 12., 'command_rpc_seconds': 2., 'capture_rpc_seconds': 2.,
        'engine_step_seconds': 20., 'shutdown_seconds': 3., 'allocated_driver_seconds': 80.}})
    save(pilot / 'READER_COST.json', {'input_tokens': 100, 'output_tokens': 50, 'load_seconds': 8.,
        'reader_seconds': 7., 'shutdown_seconds': 2., 'actual_readers': 8, 'assigned_readers': 8})
    cells = []
    for index, (arm, seed) in enumerate((('native', 0), ('native', 1), ('incumbent', 0), ('incumbent', 1))):
        assignment = {'arm': arm, 'seed': seed, 'original_prompt_ids': [1, 2],
            'execution_contract_sha256': execution['sha256']}
        assignment['uid'] = digest(['routing-outcome-blind-price-v2', 'parent', assignment])
        receipt = seal({'assignment_uid': assignment['uid'], 'execution_contract_sha256': None, 'engineering_only': True,
            'continuation_cap': 16384, 'generated_tokens': 4, 'processed_input_tokens': 8,
            'controller': {'controller_calls': [], 'controller_seconds': 0., 'controller_tokens': 0},
            'generation_wall_seconds': 10., 'capture_binding': {'allocated_storage_bytes': 131072}})
        path = pilot / f'RESULT_{index}.json'
        save(path, {'receipt': receipt})
        cells.append({'status': 'MEASURED', 'cap': 16384, 'receipt_path': str(path), 'receipt_sha256': receipt['sha256'],
            'assignment': assignment})
    return pilot, cells


def test_matching_four_cell_measurements_do_not_authorize_science_or_invent_missing_roles(tmp_path):
    from routing_study_prices_local_v2 import measured_components
    pilot, cells = price_fixture(tmp_path)
    save(pilot / 'PILOT.json', {'status': 'MEASURED', 'manifest_sha256': 'parent', 'pricing_cells': cells, 'code_files': {'worker': 'frozen'}})
    status = measured_components(pilot, tmp_path / 'prices')
    assert status['status'] == 'UNAVAILABLE_COMPLETE_STAGE_PRICE'
    assert not status['scientific_dispatch_authorized']
    assert {'replay', 'grading', 'observed_16k_death_recovery_price'} == set(status['unavailable_roles'])
    assert {r['role'] for r in status['measurements']} == {'generation', 'controller', 'capture', 'runtime'}


def test_measured_prices_reject_receipts_swapped_between_bound_assignments(tmp_path):
    from routing_study_prices_local_v2 import measured_components
    pilot, cells = price_fixture(tmp_path)
    cells[0]['receipt_path'] = cells[1]['receipt_path']
    save(pilot / 'PILOT.json', {'status': 'MEASURED', 'manifest_sha256': 'parent', 'pricing_cells': cells, 'code_files': {'worker': 'frozen'}})
    with pytest.raises(ValueError, match='source mismatch'):
        measured_components(pilot, tmp_path / 'prices')


def test_proposal_coverage_and_reader_availability_do_not_hide_false_negatives(tmp_path):
    from moe_exp.routing_control.study_v2 import RECOGNIZES
    from routing_study_receipts_local_v2 import semantic_coverage
    (tmp_path / 'audit').mkdir()
    (tmp_path / 'prep').mkdir()
    frame = save(tmp_path / 'prep/AUDIT_FRAME.json', {'records': [
        {'uid': 'missed', 'split': 'held_out', 'proposals': {d: False for d in RECOGNIZES}},
        {'uid': 'unknown', 'split': 'held_out', 'proposals': {d: True for d in RECOGNIZES}}]})
    save(tmp_path / 'audit/PILOT.json', {'status': 'EXPLICIT_INCOMPLETE_OR_FAILED'})
    for uid, readers in (('missed', (0, 1)), ('unknown', (0,))):
        for reader in readers:
            save(tmp_path / f'audit/READER_{uid}_{reader}.json', {'uid': uid, 'reader': reader,
                'frame_sha256': frame['sha256'], 'finish_reason': 'stop', 'parsed': {d: True for d in RECOGNIZES}})
    coverage = semantic_coverage(tmp_path, tmp_path / 'prep/MANIFEST.json', tmp_path / 'COVERAGE.json')
    held = [r for r in coverage['rows'] if r['split'] == 'held_out']
    assert all(r['proposal_coverage'] == r['reader_agreement_coverage'] == .5 for r in held)
    assert all(r['recall_on_known_reader_positives'] == 0 and r['unknown_reader_prefixes'] == 1 for r in held)


def test_reader_batch_encoding_and_invalid_batch():
    from routing_study_reader_prompt_local_v3 import normalize_prompt_ids, reader_prompt_ids
    from collections import UserDict
    import pytest

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['return_dict'] is False
            return UserDict({'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]})

    assert reader_prompt_ids(Tokenizer(), [{'role': 'user', 'content': 'prefix'}]) == [1, 2, 3]
    assert normalize_prompt_ids(UserDict({'input_ids': [[1, 2, 3]]})) == [1, 2, 3]
    for invalid in (UserDict({'attention_mask': [1]}), [], [[1], [2]], [True], [-1], '1,2'):
        with pytest.raises(ValueError):
            normalize_prompt_ids(invalid)
