"""Discovery-only, source-bound parallel routing comparison using the qualified worker.

Each shard keeps whole prefix/seed/arm blocks. Completion sealing runs in a
separate invocation because the inherited GPU driver exits with os._exit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
SOURCE = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v4.json'
REFERENCE = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v4.json'
QUALIFICATION = DOC / 'MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json'
PROFILE = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit')
ACTIVE_SHARD = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_once(path, body):
    path = Path(path)
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        require(base.sealed(path) == value, 'existing sealed file differs: ' + str(path))
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental cross-controller overwrite.
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    return value


def validate_design(design):
    from moe_exp.routing_control.design import Action
    from moe_steer.spec import Operator, TargetSet
    require(design['schema'] == 'overnight-routing-design-v1', 'unknown design schema')
    transitions = design.get('transitions', list(TRANSITIONS))
    require(transitions and len(set(transitions)) == len(transitions) and
            set(transitions) <= set(TRANSITIONS), 'unsupported transition filter')
    mode = design.get('mode', 'ordered_positive')
    require(mode in ('ordered_positive', 'native_operator') and
            design.get('horizon', 1024 if mode == 'ordered_positive' else 256) ==
            (1024 if mode == 'ordered_positive' else 256), 'unqualified mode/horizon')
    actions, arms = design['actions'], design['arms']
    by_name = {a['name']: a for a in actions}
    require(actions and len(actions) == len(by_name), 'empty or duplicate actions')
    for action in actions:
        if mode == 'ordered_positive':
            require(set(action) == {'name', 'transition', 'experts', 'bias'}, 'unexpected action fields')
            Action(action['name'], action['transition'],
                   tuple((layer, tuple(ids)) for layer, ids in action['experts']), action['bias']).validate()
        else:
            require(set(action) == {'name', 'transition', 'experts', 'kind', 'sign', 'magnitude'},
                    'unexpected native operator fields')
            # Keep the same sparse geometry as the qualified positive actions.
            Action(action['name'], action['transition'],
                   tuple((layer, tuple(ids)) for layer, ids in action['experts']), 1.).validate()
            Operator(action['kind'], action['sign'], action['magnitude'])
            require(action['kind'] in ('bias', 'force', 'reweight'), 'unsupported operator')
    require(4 <= len(arms) <= 12 and len({a['name'] for a in arms}) == len(arms),
            'four to twelve distinct arms required')
    require(sum(a['role'] == 'native' for a in arms) == 2 and
            {'target', 'random'} <= {a['role'] for a in arms},
            'two native arms, target and matched-random control required')
    names = {a['name'] for a in arms}
    require(design.get('planned_contrasts') and all(
        len(pair) == 2 and pair[0] != pair[1] and set(pair) <= names
        for pair in design['planned_contrasts']), 'planned contrasts must be fixed before generation')
    require(design.get('analysis_scopes') == ['all', *transitions] and
            design.get('analysis_seed') == 20261004, 'freeze analysis scopes and seed')
    for arm in arms:
        require(set(arm) == {'name', 'role', 'policies', 'slots'} and
                arm['role'] in ('native', 'target', 'random'), 'unexpected arm fields')
        if arm['role'] == 'native':
            require(arm['policies'] == {} and arm['slots'] == [], 'native arm cannot edit routing')
            continue
        require(arm['slots'] in (([0], [0, 512]) if mode == 'ordered_positive' else ([],)) and
                set(arm['policies']) == set(transitions), 'unqualified pulse slots or transitions')
        for transition, policies in arm['policies'].items():
            require(len(policies) == (len(arm['slots']) if mode == 'ordered_positive' else 1),
                    'policy/slot count mismatch')
            for policy in policies:
                require(('{random_set}' in policy) == (arm['role'] == 'random'),
                        'random selectors must use frozen balanced set assignment')
                for index in range(4):
                    name = policy.replace('{random_set}', str(index))
                    require(name in by_name and by_name[name]['transition'] == transition,
                            'unknown policy or wrong transition: ' + name)
    return actions, arms


def schedules(rows, arms):
    source = base.sealed(SOURCE)
    sets = source['random_set_by_family_seed']
    names = [a['name'] for a in arms]
    orders, offset = {}, 0
    for transition in TRANSITIONS:
        blocks = [(row['uid'], seed) for row in rows if row['transition'] == transition
                  for seed in (0, 1)]
        blocks.sort(key=lambda key: base.digest(['overnight-routing-order-v1', transition, *key]))
        for i, (uid, seed) in enumerate(blocks):
            shift = (offset + i) % len(names)
            orders[f'{uid}|{seed}'] = names[shift:] + names[:shift]
        offset += len(blocks)
    return sets, orders


def workload(rows, arms, horizon=1024):
    n = len(rows) * 2 * len(arms)
    return {'expected_requests': n,
            'expected_prefill_tokens': sum(len(r['prompt_ids']) + len(r['prefix_ids'])
                                           for r in rows) * 2 * len(arms),
            'maximum_decode_tokens': n * horizon,
            'maximum_context_tokens': max(len(r['prompt_ids']) + len(r['prefix_ids']) + horizon
                                          for r in rows)}


def projection(rows, arms, rows_per_shard, max_wall_seconds, horizon=1024):
    ref = base.sealed(REFERENCE)
    require(type(rows_per_shard) is int and 1 <= rows_per_shard <= len(rows), 'invalid shard size')
    require(type(max_wall_seconds) is int and max_wall_seconds >= 3600, 'invalid job wall')
    shards = []
    for start in range(0, len(rows), rows_per_shard):
        end = min(start + rows_per_shard, len(rows))
        count = workload(rows[start:end], arms, horizon)
        seconds = ref['repeat_factor'] * (
            count['expected_prefill_tokens'] / ref['serial_prefill_stress_tokens_per_second'] +
            count['maximum_decode_tokens'] / ref['serial_decode_stress_tokens_per_second'])
        wall = seconds + ref['cold_load_seconds'] + ref['shutdown_seconds'] + 900
        require(wall <= max_wall_seconds, 'complete shard exceeds specified wall; use fewer rows per shard')
        shards.append({'start_row': start, 'end_row': end,
                       'estimated_work_seconds': seconds, 'estimated_wall_seconds': wall})
    loads = len(shards) + max(1, math.ceil(len(shards) * (ref['repeat_factor'] - 1)))
    seconds = sum(s['estimated_work_seconds'] for s in shards) + loads * (
        ref['cold_load_seconds'] + ref['shutdown_seconds']) + len(shards) * 900
    return {'shards': shards, 'gpus_per_job': 2, 'max_wall_seconds': max_wall_seconds,
            'reference_price_sha256': ref['sha256'], 'repeat_factor': ref['repeat_factor'],
            'cold_loads_including_reserve': loads,
            'estimated_complete_gpu_hours': seconds * 2 / 3600,
            'status': 'PASS_COMPLETE_STAGE_GENERATION_ONLY',
            'scope': 'All assigned prefill, maximum decode, loads, shutdown, 25% repeat work and recovery loads. Blind rating priced separately.'}


def prepare(design_path, manifest_path, price_path, rows_per_shard=4, max_wall_seconds=14400):
    import prepare_mechanism_validation_v1 as qualification
    design = base.sealed(design_path)
    actions, arms = validate_design(design)
    mode = design.get('mode', 'ordered_positive')
    require(mode == 'ordered_positive',
            'v1 native-operator execution is held; use separately qualified common-horizon v2')
    horizon = 1024 if mode == 'ordered_positive' else 256
    source = base.sealed(SOURCE)
    family = base.sealed(base.FAMILY_FREEZE)
    qual = qualification.validate_qualification(base.sealed(QUALIFICATION))
    rows = [r for r in source['rows'] if r['transition'] in design.get('transitions', TRANSITIONS)]
    sets, orders = schedules(rows, arms)
    price = projection(rows, arms, rows_per_shard, max_wall_seconds, horizon)
    code_files = {str(Path(__file__).resolve()): base.file_sha(__file__),
                  str(Path(base.__file__).resolve()): base.file_sha(base.__file__)}
    body = {'schema': 'overnight-routing-manifest-v1', 'design_path': str(Path(design_path).resolve()),
            'design_sha256': design['sha256'], 'source_manifest_sha256': source['sha256'],
            'family_freeze_sha256': family['sha256'], 'qualification_sha256': qual['sha256'],
            'qualified_worker_sha256': qual['qualified_worker_sha256'],
            'base_tree_sha256': base.REQUIRED_BASE_TREE, 'engine_profile': PROFILE,
            'code_files': code_files, 'rows': rows, 'seeds': [0, 1], 'horizon': horizon, 'mode': mode,
            'pulse_width': 256, 'actions': actions, 'arms': arms,
            'planned_contrasts': design['planned_contrasts'],
            'analysis_scopes': design['analysis_scopes'], 'analysis_seed': design['analysis_seed'],
            'bootstrap_replicates': 50000,
            'random_set_by_family_seed': sets, 'arm_order_by_uid_seed': orders,
            'shards': price['shards'], 'rows_per_shard': rows_per_shard,
            'max_wall_seconds': max_wall_seconds, **workload(rows, arms, horizon),
            'claim_limit': 'Exploratory discovery-only screen on previously analyzed families; no independent confirmation, accuracy or utility claim. Repeated pulses compare exposure, not action ordering. Native operators act until reasoning closure or the 256-token cap.'}
    manifest = write_once(manifest_path, body)
    price = write_once(price_path, {'schema': 'overnight-routing-price-v1',
                                   'manifest_sha256': manifest['sha256'], **price})
    return manifest, price


def validate(manifest, driver_path):
    import prepare_mechanism_validation_v1 as qualification
    require(manifest['schema'] == 'overnight-routing-manifest-v1' and
            manifest['mode'] == 'ordered_positive' and
            manifest['horizon'] == (1024 if manifest['mode'] == 'ordered_positive' else 256) and
            manifest['pulse_width'] == 256 and
            manifest['seeds'] == [0, 1] and manifest['engine_profile'] == PROFILE and
            os.environ.get('VLLM_BATCH_INVARIANT', '0') == '0', 'unqualified schema/profile/horizon')
    source, family = base.sealed(SOURCE), base.sealed(base.FAMILY_FREEZE)
    design = base.sealed(manifest['design_path'])
    expected_rows = [r for r in source['rows'] if r['transition'] in design.get('transitions', TRANSITIONS)]
    require(manifest['rows'] == expected_rows and expected_rows and
            manifest['source_manifest_sha256'] == source['sha256'] and
            manifest['family_freeze_sha256'] == family['sha256'] and
            {r['family'] for r in manifest['rows']} <=
            set(family['new_parent_pools']['parent_pools']['discovery']),
            'enrollment differs from frozen discovery starts')
    actions, arms = validate_design(design)
    require(design['sha256'] == manifest['design_sha256'] and
            manifest['actions'] == actions and manifest['arms'] == arms and
            manifest['mode'] == design.get('mode', 'ordered_positive') and
            all(manifest[k] == design[k] for k in ('planned_contrasts', 'analysis_scopes', 'analysis_seed')) and
            manifest['bootstrap_replicates'] == 50000, 'design changed')
    codes = manifest['code_files']
    require(codes.get(str(Path(__file__).resolve())) == base.file_sha(__file__) and
            codes.get(str(Path(driver_path).resolve())) == base.file_sha(driver_path) and
            all(base.file_sha(path) == sha for path, sha in codes.items()), 'runner source changed')
    qual = qualification.validate_qualification(base.sealed(QUALIFICATION))
    require(manifest['qualification_sha256'] == qual['sha256'] and
            manifest['qualified_worker_sha256'] == qual['qualified_worker_sha256'] and
            manifest['base_tree_sha256'] == base.REQUIRED_BASE_TREE,
            'qualification binding changed')
    sets, orders = schedules(manifest['rows'], arms)
    require(manifest['random_set_by_family_seed'] == sets and
            manifest['arm_order_by_uid_seed'] == orders, 'control assignment changed')
    require(all(manifest[k] == v for k, v in workload(manifest['rows'], arms, manifest['horizon']).items()),
            'complete workload counts changed')
    price = projection(manifest['rows'], arms, manifest['rows_per_shard'], manifest['max_wall_seconds'], manifest['horizon'])
    require(manifest['shards'] == price['shards'], 'shard price changed')
    return manifest['rows'], actions, arms


def request_metadata(manifest, rows, arms):
    by_arm = {a['name']: a for a in arms}
    for row in rows:
        prompt = row['prompt_ids'] + row['prefix_ids']
        for seed in manifest['seeds']:
            for position, name in enumerate(manifest['arm_order_by_uid_seed'][f"{row['uid']}|{seed}"]):
                arm = by_arm[name]
                random_set = manifest['random_set_by_family_seed'][row['family']][seed]
                policies = [n.replace('{random_set}', str(random_set))
                            for n in arm['policies'].get(row['transition'], [])]
                policy = policies[0] if policies else 'zero'
                uid = 'overnight-v1|' + base.digest([manifest['sha256'], row['uid'], seed, name])[:24]
                meta = {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                        'transition': row['transition'], 'question': row['canonical_question'],
                        'seed': seed, 'arm': name, 'role': arm['role'], 'policy': policy,
                        'policies': policies, 'slots': arm['slots'],
                        'random_set': random_set if arm['role'] == 'random' else None,
                        'execution_position': position, 'prompt_len': len(prompt),
                        'prompt_sha256': base.digest(prompt)}
                yield row, meta


def build_requests(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    for row, meta in request_metadata(manifest, rows, arms):
        prompt = row['prompt_ids'] + row['prefix_ids']
        info = world.infos.get(meta['question'])
        require(info is not None and info['prompt_token_ids'] == row['prompt_ids'] and
                engine.THINK_END_ID not in row['prefix_ids'], 'frozen native prefix changed')
        extra = Q.steer_extra(table, meta['uid'], meta['policy'], len(row['prompt_ids']),
                              prefix_len=len(row['prefix_ids']), restore_presence=True)
        if meta['role'] != 'native' and manifest['mode'] == 'ordered_positive':
            template = {'action_policy_names': meta['policies'], 'slots': meta['slots'], 'horizon': 1024}
            extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
        sampling = Q.card_params(manifest['horizon'], Q.crn(info, meta['seed']), extra, presence=0., routed_start=len(prompt)-1)
        cases.append((Q.QReq(meta['uid'], prompt, sampling), meta))
    require(len(cases) == manifest['expected_requests'] and
            len({m['uid'] for _, m in cases}) == len(cases), 'duplicate or missing request')
    for start in range(0, len(cases), len(arms)):
        block = cases[start:start+len(arms)]
        require(len({tuple(r.prompt) for r, _ in block}) == 1 and
                len({r.sampling['seed'] for r, _ in block}) == 1 and
                {m['arm'] for _, m in block} == {a['name'] for a in arms}, 'incomplete paired block')
    if ACTIVE_SHARD is not None:
        start, end = ACTIVE_SHARD
        return cases[start * 2 * len(arms):end * 2 * len(arms)]
    return cases


def build_policy_table(actions):
    """Use the same sealed base operators as the positive and deactivation runners."""
    from moe_steer import policies as P
    from moe_steer.spec import Operator, Schedule, TargetSet
    policies = []
    for action in actions:
        targets = TargetSet(action['name'], 'CUSTOM',
                            tuple((layer, tuple(sorted(ids))) for layer, ids in action['experts']),
                            'frozen discovery correlation and causal action comparison')
        operator = (Operator('bias', 1, action['bias']) if 'bias' in action else
                    Operator(action['kind'], action['sign'], action['magnitude']))
        policies.append(P.make_policy(targets, operator, Schedule('always'), name=action['name']))
    table = P.build_table(policies)
    validate_compilation(table, actions)
    return table


def validate_compilation(table, actions):
    """Assert that each sparse edit compiles exactly, including untouched cells."""
    import numpy as np
    compiled = table.compile()
    columns = {int(layer): index for index, layer in enumerate(compiled['layers'])}
    for action in actions:
        index = table.index_of(action['name'])
        expected = {name: np.zeros_like(value[index]) for name, value in compiled.items()
                    if name != 'layers'}
        kind = 'bias' if 'bias' in action else action['kind']
        sign = 1 if 'bias' in action else action['sign']
        magnitude = action.get('bias', action.get('magnitude'))
        for layer, ids in action['experts']:
            column = columns[layer]
            if kind == 'bias':
                expected['bias'][column, ids] = sign * magnitude
            elif kind == 'force':
                expected['force_hi' if sign > 0 else 'force_lo'][column, ids] = True
            else:
                expected['member'][column, ids] = 1.
                expected['beta'][column] = sign * magnitude
        require(all(np.array_equal(compiled[name][index], value)
                    for name, value in expected.items()), 'compiled policy differs: ' + action['name'])


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
                require(values['membership_changes'] <= expected_rows and
                        values['weight_l1'] <= 2. * expected_rows + .001,
                        'invalid membership or gate displacement dose')
                peer = dose['1' if rank == '0' else '0']['dose'][policy][layer]
                for key, value in values.items():
                    require(math.isclose(value, peer[key], rel_tol=1e-5, abs_tol=1e-3),
                            'TP ranks disagree on dose: ' + key)
    return {'status': 'pulse_closure_TP_dose_pass', 'active_rows': sum(rows.values()),
            'reasoning_horizon': stop}


def seal_shard(out, manifest, index):
    shard = manifest['shards'][index]
    directory = out / f'shard-{index:03d}'
    binding, summary = base.sealed(directory/'BINDING.json'), base.sealed(directory/'SUMMARY.json')
    expected = [meta for _, meta in request_metadata(manifest,
        manifest['rows'][shard['start_row']:shard['end_row']], manifest['arms'])]
    require(binding['manifest_sha256'] == manifest['sha256'] and
            summary['binding_sha256'] == binding['sha256'] and
            summary['requests'] == len(expected), 'summary/assignment mismatch')
    results, counts = [], Counter()
    import numpy as np
    for i in range(summary['batches']):
        batch = base.sealed(directory/f'batch-{i:03d}.json')
        require(batch['manifest_sha256'] == manifest['sha256'] and
                batch['binding_sha256'] == binding['sha256'] and
                base.file_sha(directory/f'batch-{i:03d}.npz') == batch['array_sha256'],
                'batch or route arrays changed')
        with np.load(directory/f'batch-{i:03d}.npz', allow_pickle=False) as arrays:
            for result in batch['outputs']:
                if not result['error']:
                    routed = arrays[result['uid']]
                    require(routed.shape == (len(result['tokens']), 40, 8) and
                            ((routed >= 0) & (routed < 256)).all() and
                            (np.diff(np.sort(routed, axis=-1), axis=-1) > 0).all(),
                            'invalid executed native-top8 expert IDs or token alignment')
        results.extend(batch['outputs'])
    require([r['uid'] for r in results] == [r['uid'] for r in expected] and
            len({r['uid'] for r in results}) == len(results), 'incomplete or duplicate assigned UID')
    for result, assigned in zip(results, expected):
        require(all(result[k] == v for k, v in assigned.items()), 'result metadata differs')
        audit_output_dose(result, manifest)
        counts['assigned'] += 1
        counts['error'] += bool(result['error'])
        counts['routed_present'] += bool(result['routed_present'])
        counts['emitted_tokens'] += len(result['tokens'])
        counts['capped'] += len(result['tokens']) == manifest['horizon']
        counts['natural_stop'] += len(result['tokens']) < manifest['horizon'] and not result['error']
    return write_once(directory/'OVERNIGHT_COMPLETION.json',
                      {'schema': 'overnight-routing-completion-v1', 'manifest_sha256': manifest['sha256'],
                       'source_summary_sha256': summary['sha256'], 'shard': index,
                       'counts': dict(counts), 'dose_audit': 'pulse_boundaries_closure_TP_dose_and_top8_checked',
                       'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--price', type=Path)
    p.add_argument('--design', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--rows-per-shard', type=int, default=4)
    p.add_argument('--max-wall-seconds', type=int, default=14400)
    p.add_argument('--cpu-preflight', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--overlay', type=Path)
    p.add_argument('--shard-index', type=int)
    p.add_argument('--seal-shard', action='store_true')
    p.add_argument('--seal-stage', action='store_true')
    args = p.parse_args()
    if args.prepare:
        require(args.design is not None and args.price is not None, 'prepare requires design and price paths')
        manifest, price = prepare(args.design, args.manifest, args.price, args.rows_per_shard, args.max_wall_seconds)
        print(json.dumps({'manifest_sha256': manifest['sha256'], 'requests': manifest['expected_requests'],
                          'shards': len(price['shards']), 'gpu_hours': price['estimated_complete_gpu_hours']}))
        return
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate(manifest, base.__file__)
    if args.seal_stage or args.seal_shard:
        require(args.out is not None, 'sealing requires output root')
        indices = range(len(manifest['shards'])) if args.seal_stage else [args.shard_index]
        receipts = [seal_shard(args.out, manifest, i) for i in indices]
        if args.seal_stage:
            counts = Counter()
            for r in receipts:
                counts.update(r['counts'])
            require(counts['assigned'] == manifest['expected_requests'], 'incomplete full stage')
            write_once(args.out/'STAGE_COMPLETION.json',
                       {'schema': 'overnight-routing-stage-completion-v1', 'manifest_sha256': manifest['sha256'],
                        'counts': dict(counts), 'shard_completion_sha256s': [r['sha256'] for r in receipts],
                        'status': 'COMPLETE_UNGRADED_DISCOVERY_GENERATION'})
        print(json.dumps({'status': 'PASS_COMPLETION', 'shards': len(receipts)}))
        return
    from moe_steer import engine, manifests as M
    if args.cpu_preflight:
        table = build_policy_table(actions)
        cases = build_requests(manifest, rows, arms, M.load_world(), table)
        from moe_exp.routing_control.worker_adapter import OrderedPulse
        for request, meta in cases:
            if meta['role'] != 'native' and manifest['mode'] == 'ordered_positive':
                OrderedPulse.load(request.sampling['extra_args']['steer']['meta']['routing_control'], table, meta['policy'])
            elif manifest['mode'] == 'native_operator':
                require('routing_control' not in request.sampling['extra_args']['steer'].get('meta', {}),
                        'native operator must bypass ordered metadata')
        print(json.dumps({'status': 'PASS_CPU_PREFLIGHT', 'requests': len(cases), 'shards': len(manifest['shards'])}))
        return
    require(args.out is not None and args.overlay is not None and args.shard_index is not None and
            0 <= args.shard_index < len(manifest['shards']), 'generation requires output, overlay and shard')
    args.batch_size = len(arms)
    global ACTIVE_SHARD
    shard = manifest['shards'][args.shard_index]
    ACTIVE_SHARD = shard['start_row'], shard['end_row']
    args.out = args.out / f'shard-{args.shard_index:03d}'
    original_kwargs = engine.engine_kwargs
    # Changes are process-local, never writes to inherited sealed source files.
    base.validate_manifest, base.build_requests, base.build_policy_table = validate, build_requests, build_policy_table
    engine.engine_kwargs = lambda *a, **kw: original_kwargs(*a, **{**kw, 'max_num_seqs': 1, 'enforce_eager': True})
    base.run(args)


if __name__ == '__main__':
    main()
