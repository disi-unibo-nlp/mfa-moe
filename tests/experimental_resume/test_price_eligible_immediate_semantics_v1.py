"""Complete-stage semantic-reader pricing must include both readers and full cap."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    'price_eligible_immediate_semantics_v1',
    SCRIPTS / 'price_eligible_immediate_semantics_v1.py')
price = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(price)


def test_full_stage_price_counts_two_readers_and_1024_decode_cap():
    scout = {'timings': [{'load_seconds': 400.,
                          'reader_timings': [{'wall_seconds': 50.}]}],
             'counts': {'generated_tokens': 3000}}
    result = price.estimate([100, 200, 300], scout)
    assert result['prompt_tokens_exact_twice'] == 1200
    assert result['max_decode_tokens'] == 6 * price.rating.MAX_TOKENS
    assert result['max_model_context_tokens'] == 300 + price.rating.MAX_TOKENS
    assert result['components_seconds']['cold_load'] == 500.
    assert result['projected_complete_seconds'] == sum(result['components_seconds'].values())
