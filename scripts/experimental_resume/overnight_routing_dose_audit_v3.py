"""Independent CPU correction for the dimension of sparse top-k turnover counts.

The frozen v1 checker bounded inserted expert slots by active token rows. Two
edited experts can insert two slots on one row. This verifier preserves every
other cap, pulse, closure, layer-isolation and rank check and tightens target-hit
bounds. Top-k ties or rounding may reselect untouched experts, so the universal
hard bound is eight inserted slots per row. Target-count times rows is reported
as a tighter diagnostic, not assumed. Reweight and untouched layers require zero.
It does not alter the engine or overwrite any original qualification.
"""
import math
from overnight_routing_runner_v1 import require

def audit_output_dose(result, manifest):
    """Check exact per-action pulse rows, closure clipping and TP dose agreement."""
    from moe_steer import engine
    require(len(result['tokens']) <= manifest['horizon'], 'sampler exceeded sealed cap')
    if result['error']:
        return {'status': 'assigned_error_retained_in_ITT'}
    require(result['routed_present'], 'successful output has no routed array')
    require(result['finish'] in ('length', 'stop'), 'unknown successful sampler finish')
    if result['finish'] == 'length':
        require(len(result['tokens']) == manifest['horizon'], 'length finish differs from cap')
    off = result['inactive_native_checks']
    require(set(off) == {'0', '1'} and all(checks and all(
        value['expert_identity_mismatches'] == value['weight_mismatches'] == 0
        for value in checks.values()) for checks in off.values()), 'inactive routing mismatch')
    if result['role'] == 'native':
        require(not result['policies'] and not result['slots'], 'native request has an intervention')
        return {'status': 'native_inactive_parity_pass'}
    stop = next((i + 1 for i, token in enumerate(result['tokens']) if token == engine.THINK_END_ID),
                len(result['tokens']))
    rows = {name: 0 for name in result['policies']}
    segments = {name: [] for name in result['policies']}
    for name, slot in zip(result['policies'], result['slots']):
        end = min(slot + 256, stop)
        if end > slot:
            rows[name] += end - slot
            segments[name].append([slot, end])
    dose = result['action_dose']
    require(set(dose) == {'0', '1'}, 'missing TP action dose')
    actions = {action['name']: action for action in manifest['actions']}
    diagnostics = []
    for rank in ('0', '1'):
        require(dose[rank]['rows'] == rows and dose[rank]['segments'] == segments and
                set(dose[rank]['dose']) == set(rows), 'pulse boundary/order/closure differs')
        for policy, expected_rows in rows.items():
            layers = dose[rank]['dose'][policy]
            targets = {str(layer): set(ids) for layer, ids in actions[policy]['experts']}
            require(layers and set(targets) <= set(layers), 'missing targeted layer dose')
            for layer, values in layers.items():
                require(all(math.isfinite(float(v)) and v >= 0 for v in values.values()) and
                        values['active_rows'] == expected_rows,
                        'nonfinite, negative or incorrect active dose')
                if layer not in targets:
                    require(all(v == 0 for k, v in values.items() if k != 'active_rows'),
                            'intervention leaked to untargeted layer')
                # Sealed vllm_ext._accumulate counts inserted expert slots:
                # (~actual_in_native).sum(dim=1), not changed token rows.
                action = actions[policy]
                kind = 'bias' if 'bias' in action else action['kind']
                slot_bound = 0 if kind == 'reweight' or layer not in targets else 8 * expected_rows
                require(values['membership_changes'] <= slot_bound and
                        values['weight_l1'] <= 2. * expected_rows + .001,
                        'invalid expert-slot turnover or gate displacement dose')
                require(values['native_target_hits'] <= len(targets.get(layer, ())) * expected_rows and
                        values['actual_target_hits'] <= len(targets.get(layer, ())) * expected_rows,
                        'target-hit dose exceeds targeted expert slots')
                diagnostics.append({'rank': rank, 'policy': policy, 'layer': layer,
                    'inserted_expert_slots': values['membership_changes'],
                    'universal_top8_slot_bound': slot_bound,
                    'edited_target_slot_bound': len(targets.get(layer, ())) * expected_rows,
                    'within_tighter_target_bound': values['membership_changes'] <=
                                                   len(targets.get(layer, ())) * expected_rows})
                peer = dose['1' if rank == '0' else '0']['dose'][policy][layer]
                for key, value in values.items():
                    require(math.isclose(value, peer[key], rel_tol=1e-5, abs_tol=1e-3),
                            'TP ranks disagree on dose: ' + key)
    return {'status': 'pulse_closure_TP_dose_pass', 'active_rows': sum(rows.values()),
            'reasoning_horizon': stop, 'slot_turnover_diagnostics': diagnostics}
