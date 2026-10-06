"""Token replay and same-execution routing checks; no cross-run route identity."""


def qualify(replay, agreement):
    for name in ('native_repeat', 'instrumented_repeat'):
        if replay[name]['tokens_equal'] is not True:
            raise ValueError('engineering token replay differs: ' + name)
        if replay[name]['routes_status'] != 'CAPTURED_NATIVE':
            raise ValueError('native routing is unknown: ' + name)
    for name in ('ids_match_returned_routes', 'rank_ids_agree', 'rank_weights_agree'):
        if agreement.get(name) is not True:
            raise ValueError('same-execution routing check failed: ' + name)
    return {'status': 'PASS_TOKEN_REPLAY_AND_SAME_EXECUTION_ROUTING',
        'cross_run_expert_ID_equality_required': False,
        'reason': 'Native repeated executions showed different expert selections with identical token outputs; per-execution IDs and weights must still be captured faithfully on both ranks.'}
