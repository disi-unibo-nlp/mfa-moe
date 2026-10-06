"""Check that semantic judge inputs stay arm-blind and parse strictly."""
import importlib.util
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[2] /
          'scripts/experimental_resume/rate_micro_blind_semantics.py')
spec = importlib.util.spec_from_file_location('rate_micro_blind_semantics', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_messages_allowlist_and_full_prefix():
    row = {'blind_id': 'opaque', 'reader_input': {
        'problem': 'Find x with x+2=4.', 'previous_sentence': 'We seek x.',
        'triggering_sentence': 'Candidate x=2.',
        'full_emitted_prefix': 'We seek x. Candidate x=2.',
        'continuation': 'Substitution gives 2+2=4.'}}
    messages = module.messages(row)
    assert len(messages) == 2
    assert 'Candidate x=2.' in messages[1]['content']
    assert 'Substitution gives 2+2=4.' in messages[1]['content']
    for forbidden in ('arm', 'seed', 'family', 'gold_answer', 'native_future'):
        assert forbidden not in messages[1]['content']
    row['role'] = 'target'
    try:
        module.messages(row)
    except ValueError:
        pass
    else:
        raise AssertionError('extra assignment field reached judge')


def test_parse_strict_json_and_reasoning_suffix():
    assert module.parse_rating('{"start":true,"target":false}') == {'start': True, 'target': False}
    assert module.parse_rating('scratch</think>{"start":false,"target":true}') == {'start': False, 'target': True}
    assert module.parse_rating('{"start":1,"target":false}') is None
    assert module.parse_rating('{"start":true,"target":false,"arm":"native"}') is None
    assert module.parse_rating('yes') is None
