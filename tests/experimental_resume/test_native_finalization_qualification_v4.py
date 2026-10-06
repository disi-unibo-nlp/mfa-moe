import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
from native_finalization_qualification_v4 import qualify


def evidence():
    replay = {name: {'tokens_equal': True, 'routes_equal': False, 'route_sets_equal': False,
        'routes_status': 'CAPTURED_NATIVE'} for name in ('native_repeat', 'instrumented_repeat')}
    agreement = {name: True for name in ('ids_match_returned_routes', 'rank_ids_agree', 'rank_weights_agree')}
    return replay, agreement


def test_native_route_variation_does_not_reject_faithful_same_execution_capture():
    replay, agreement = evidence()
    assert qualify(replay, agreement)['status'] == 'PASS_TOKEN_REPLAY_AND_SAME_EXECUTION_ROUTING'


@pytest.mark.parametrize('section,key', [('native_repeat','tokens_equal'), ('instrumented_repeat','tokens_equal'),
    ('native_repeat','routes_status'), ('instrumented_repeat','routes_status'),
    ('agreement','ids_match_returned_routes'), ('agreement','rank_ids_agree'), ('agreement','rank_weights_agree')])
def test_unknown_native_routes_token_drift_and_rank_or_alignment_errors_reject(section,key):
    replay, agreement = evidence()
    target = agreement if section == 'agreement' else replay[section]
    target[key] = 'UNKNOWN' if key == 'routes_status' else False
    with pytest.raises(ValueError): qualify(replay, agreement)
