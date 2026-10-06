"""Pre-treatment eligibility must not see assigned future text or arm fields."""
import importlib.util
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[2] /
          'scripts/experimental_resume/rate_micro_blind_semantics_v2.py')
spec = importlib.util.spec_from_file_location('rate_micro_blind_semantics_v2', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(continuation='A distinct continuation'):
    return {'blind_id': 'opaque', 'reader_input': {
        'problem': 'Find x with x+2=4.', 'previous_sentence': 'We seek x.',
        'triggering_sentence': 'Candidate x=2.',
        'full_emitted_prefix': 'We seek x. Candidate x=2.',
        'continuation': continuation}}


def test_start_prompt_is_invariant_to_future():
    first = module.messages(row('Verified: 2+2=4'), 'start')
    second = module.messages(row('Final answer without a check'), 'start')
    assert first == second
    assert 'Verified:' not in repr(first)
    assert 'Final answer' not in repr(first)
    assert module.messages(row('Verified: 2+2=4'), 'target') != module.messages(
        row('Final answer without a check'), 'target')


def test_rating_json_and_grouping():
    assert module.parse_rating('{"start":true}', 'start') == {'start': True}
    assert module.parse_rating('thinking</think>{"target":false}', 'target') == {'target': False}
    assert module.parse_rating('{"start":true,"target":false}', 'start') is None
    assert module.parse_rating('{"target":1}', 'target') is None
    rows = []
    for index in range(4):
        for arm in range(12):
            item = row(str(arm))
            item['blind_id'] = f'{index}-{arm}'
            item['reader_input']['problem'] += str(index)
            rows.append(item)
    assert len(module.start_groups(rows)) == 4
