"""Price prompt token IDs, not container keys or text characters."""
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/price_transition_ratings_v3.py'
SPEC = importlib.util.spec_from_file_location('transition_price_tokens_test', SOURCE)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_batch_encoding_mapping_counts_input_ids():
    assert P.count_prompt_tokens({'input_ids': list(range(50)), 'attention_mask': [1] * 50}) == 50
    assert P.count_prompt_tokens([list(range(25))]) == 25
    assert P.count_prompt_tokens(tuple(range(30))) == 30


def test_reject_container_key_count_as_prompt_length():
    for bad in ({'input_ids': [1, 2]}, {'foo': list(range(50))}, 'a full prompt'):
        try:
            P.count_prompt_tokens(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid prompt token return was accepted')
