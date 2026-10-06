"""Two-transition causal feasibility generation with paired native sentinels.

This is a discovery-family 256-token local endpoint. It neither completes the
registered 48-family discovery maximum nor runs 1,024-token mechanism validation
or original-prompt utility. The four-row stage first qualifies the new layer-24
hook/dose; the 21-family stage requires that separately sealed result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
AMENDMENT = REPORT / 'CAUSAL_DISCOVERY_FEASIBILITY_MICRO_AMENDMENT_v1.json'
DICTIONARY = REPORT / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
PILOT = REPORT / 'CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
LAYER24_GATE = REPORT / 'CAUSAL_DISCOVERY_LAYER24_QUAL_AUDIT_v1.json'
OVERLAY_FILES = ('moe_exp/routing_control/ordered_vllm.py',
                 'moe_exp/routing_control/worker_adapter.py',
                 'moe_exp/routing_control/design.py')
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit')
ORIGINAL_VALIDATE = base.validate_manifest
ORIGINAL_BUILD = base.build_requests


def action_spec(dictionary):
    templates = {x['transition']: x for x in dictionary['target_templates']}
    controls = dictionary['matched_random_control_sets']
    if set(templates) != set(TRANSITIONS) or set(controls) != set(TRANSITIONS):
        raise ValueError('supported transition dictionary changed')
    result = []
    for transition in TRANSITIONS:
        template = templates[transition]
        if template['biases'] != [0.5, 1.0]:
            raise ValueError('positive dose grid changed')
        for bias in template['biases']:
            result.append({'name': f'{transition}_target_bias{bias:g}',
                           'transition': transition,
                           'experts': template['experts'], 'bias': bias})
        if [c['set_index'] for c in controls[transition]] != list(range(4)):
            raise ValueError('four frozen matched-random sets changed')
        for control in controls[transition]:
            for bias in template['biases']:
                result.append({'name': f'{transition}_random{control["set_index"]}_bias{bias:g}',
                               'transition': transition,
                               'experts': control['experts'], 'bias': bias})
    if len(result) != 20:
        raise ValueError('two transition target/random table incomplete')
    return result


def conditions_for(transition):
    return [{'name': 'zero', 'policy': 'zero'},
            {'name': 'target05', 'policy': f'{transition}_target_bias0.5'},
            {'name': 'target1', 'policy': f'{transition}_target_bias1'},
            {'name': 'random05', 'policy': 'random_selector_bias0.5'},
            {'name': 'random1', 'policy': 'random_selector_bias1'}]


def group_rows(rows):
    """Four fixed families per eight-request batch; mark padded repeats technical."""
    groups = []
    for index in range(0, len(rows), 4):
        block = [(row, False) for row in rows[index:index + 4]]
        for filler in rows[:4 - len(block)]:
            block.append((filler, True))
        if len(block) != 4 or len({row['family'] for row, _ in block}) != 4:
            raise ValueError('filler would duplicate a family inside a batch')
        groups.append(block)
    return groups


def stage_layout(manifest):
    rows = manifest['rows']
    if manifest['stage'] == 'layer24_engineering_qual':
        transitions = [('approach_to_commit', rows)]
        allowed = {'zero', 'target05', 'target1', 'random1'}
    elif manifest['stage'] == 'discovery_feasibility21':
        transitions = [(transition, [r for r in rows if r['transition'] == transition])
                       for transition in TRANSITIONS]
        allowed = {'zero', 'target05', 'target1', 'random05', 'random1'}
    else:
        raise ValueError('unknown discovery feasibility stage')
    result = []
    for transition, transition_rows in transitions:
        groups = group_rows(transition_rows)
        for seed in (0, 1):
            conditions = [c for c in conditions_for(transition) if c['name'] in allowed]
            if seed == 1:
                conditions.reverse()
            for group_index, group in enumerate(groups):
                for condition in conditions:
                    result.append((transition, seed, group_index, condition, group))
    return result


def workload_counts(manifest):
    batches = stage_layout(manifest)
    prefill = sum((len(row['prompt_ids']) + len(row['prefix_ids'])) * 2
                  for _, _, _, _, group in batches for row, _ in group)
    return {'expected_requests': len(batches) * 8,
            'expected_batches': len(batches),
            'expected_prefill_tokens': prefill,
            'maximum_decode_tokens': len(batches) * 8 * 256,
            'maximum_context_tokens': max(len(r['prompt_ids']) +
                                          len(r['prefix_ids']) + 256 for r in manifest['rows'])}


def validate_stage(manifest, driver_path):
    from moe_exp.routing_control.design import Action
    if manifest.get('schema') != 'routing-discovery-feasibility-generation-v1':
        raise ValueError('unknown discovery feasibility generation schema')
    if manifest.get('driver_sha256') != base.file_sha(driver_path) or \
            manifest.get('entry_driver_sha256') != base.file_sha(__file__):
        raise ValueError('frozen driver source differs')
    amendment, dictionary, pilot = (base.sealed(p) for p in
                                    (AMENDMENT, DICTIONARY, PILOT))
    if (manifest.get('amendment_sha256') != amendment['sha256'] or
            manifest.get('action_dictionary_sha256') != dictionary['sha256'] or
            manifest.get('base_tree_sha256') != pilot['base_tree_sha256'] or
            manifest.get('family_freeze_sha256') != amendment['family_freeze_sha256']):
        raise ValueError('frozen parent population or source differs')
    if manifest.get('actions') != action_spec(dictionary):
        raise ValueError('policy table differs from frozen causal action dictionary')
    for action in manifest['actions']:
        Action(action['name'], action['transition'],
               tuple((layer, tuple(experts)) for layer, experts in action['experts']),
               action['bias']).validate()
    if (manifest.get('profile') != {'max_num_seqs': 8, 'enforce_eager': False,
                                    'VLLM_BATCH_INVARIANT': 0, 'batch_size': 8} or
            os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0'):
        raise ValueError('batched non-BI execution profile differs')
    if manifest.get('arms') != [
        {'name': 'active', 'policy': 'condition', 'role': 'conditional'},
        {'name': 'sentinel', 'policy': 'zero', 'role': 'native'}]:
        raise ValueError('active/sentinel arm shape differs')
    if manifest.get('seeds') != [0, 1] or manifest.get('max_tokens') != 256:
        raise ValueError('local feasibility endpoint changed')
    if manifest['stage'] == 'layer24_engineering_qual':
        if manifest.get('rows') != pilot['rows'] or \
                manifest.get('source_pilot_sha256') != pilot['sha256']:
            raise ValueError('layer24 engineering fixture differs from old pilot')
        expected_schedule = {r['family']: [i % 4, (i + 1) % 4]
                             for i, r in enumerate(pilot['rows'])}
        if manifest.get('random_set_by_transition_family_seed') != {
                'approach_to_commit': expected_schedule}:
            raise ValueError('engineering matched random schedule differs')
    elif manifest['stage'] == 'discovery_feasibility21':
        if (manifest.get('rows') != amendment['rows'] or
                manifest.get('random_set_by_transition_family_seed') !=
                amendment['random_set_by_transition_family_seed']):
            raise ValueError('21-family enrolled prefixes or random schedule changed')
        gate = base.sealed(LAYER24_GATE)
        if (manifest.get('layer24_qualification_sha256') != gate['sha256'] or
                not gate.get('pass')):
            raise ValueError('new layer24 hook/dose lacks accepted qualification')
        neighbor = base.sealed(REPORT / 'CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v1.json')
        if (manifest.get('neighbor_qualification_sha256') != neighbor['sha256'] or
                not neighbor['accepted_for_limited_batched_behavioral_discovery']):
            raise ValueError('matched batched neighbor profile lacks accepted qualification')
    else:
        raise ValueError('unsupported discovery stage')
    if manifest.get('code_files', {}).get(str(Path(__file__))) != base.file_sha(__file__):
        raise ValueError('entry driver not pinned in manifest')
    if any(base.file_sha(path) != sha for path, sha in manifest['code_files'].items()):
        raise ValueError('qualified code file differs')
    worker_prep, worker_qual = base.sealed(base.WORKER_PREP), base.sealed(base.WORKER_QUAL)
    if (not worker_qual['pass'] or
            manifest.get('qualified_worker_sha256') != worker_qual['sha256'] or
            worker_qual['worker_code_digest'] != worker_prep['sha256']):
        raise ValueError('qualified ordered worker differs')
    overlay = Path(worker_prep['overlay'])
    required = [Path(__file__), Path(driver_path)] + [overlay / x for x in OVERLAY_FILES]
    if any(manifest['code_files'].get(str(p)) != base.file_sha(p) for p in required):
        raise ValueError('entry/base/overlay hashes incomplete')
    freeze = base.sealed(base.FAMILY_FREEZE)
    discovery = set(freeze['new_parent_pools']['parent_pools']['discovery'])
    if (len({r['family'] for r in manifest['rows']}) != len(manifest['rows']) or
            not {r['family'] for r in manifest['rows']} <= discovery or
            any(r['question'] not in freeze['new_parent_pools']['families'][r['family']]
                for r in manifest['rows'])):
        raise ValueError('families overlap or leave frozen discovery parent pool')
    counts = workload_counts(manifest)
    if any(manifest.get(k) != v for k, v in counts.items()):
        raise ValueError('manifest workload accounting differs')
    table = base.build_policy_table(manifest['actions'])
    if table.hooked_layers() != (24, 28):
        raise ValueError('layer24 and layer28 hooks not both present')
    return manifest['rows'], manifest['actions'], manifest['arms']


def build_cases(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    for transition, seed, group_index, condition, group in stage_layout(manifest):
        for slot_index, (row, technical_filler) in enumerate(group):
            prompt = row['prompt_ids'] + row['prefix_ids']
            if engine.THINK_END_ID in row['prefix_ids']:
                raise ValueError('reasoning already closed in frozen prefix')
            if row['question'] not in world.infos:
                raise ValueError('question missing from frozen model world')
            common_seed = Q.crn(world.infos[row['question']], seed)
            for slot in ('active', 'sentinel'):
                logical = condition['policy'] if slot == 'active' else 'zero'
                policy = logical
                if logical.startswith('random_selector_bias'):
                    dose = logical.removeprefix('random_selector_bias')
                    k = manifest['random_set_by_transition_family_seed'][transition][row['family']][seed]
                    policy = f'{transition}_random{k}_bias{dose}'
                uid = 'discovery-micro-v1|' + base.digest([
                    manifest['sha256'], transition, row['uid'], seed,
                    group_index, slot_index, condition['name'], slot])[:24]
                extra = Q.steer_extra(table, uid, policy, len(row['prompt_ids']),
                                      prefix_len=len(row['prefix_ids']), restore_presence=True)
                if logical != 'zero':
                    template = {'action_policy_names': [policy], 'slots': [0], 'horizon': 1024}
                    extra['steer']['meta'] = {'routing_control': {
                        **template, 'sha256': worker_digest(template)}}
                sampling = Q.card_params(256, common_seed, extra, presence=0.,
                                         routed_start=len(prompt) - 1)
                role = ('native' if policy == 'zero' else
                        'random' if logical.startswith('random_') else 'target')
                meta = {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                        'question': row['question'], 'transition': transition, 'seed': seed,
                        'group_index': group_index, 'slot_index': slot_index,
                        'technical_filler': technical_filler,
                        'analysis_enrolled': not technical_filler,
                        'condition': condition['name'], 'slot': slot,
                        'arm': condition['name'] + '|' + slot, 'role': role,
                        'policy': policy, 'prompt_len': len(prompt),
                        'prompt_sha256': base.digest(prompt)}
                cases.append((Q.QReq(uid, prompt, sampling), meta))
    if (len(cases) != manifest['expected_requests'] or
            len({meta['uid'] for _, meta in cases}) != len(cases)):
        raise ValueError('request count or unique identifiers differ')
    for start in range(0, len(cases), 8):
        batch = [meta for _, meta in cases[start:start + 8]]
        if (len(batch) != 8 or len({r['transition'] for r in batch}) != 1 or
                len({r['condition'] for r in batch}) != 1 or
                len({r['seed'] for r in batch}) != 1 or
                [r['slot'] for r in batch] != ['active', 'sentinel'] * 4 or
                len({r['family'] for r in batch}) != 4):
            raise ValueError('nonhomogeneous or unmatched eight-request batch')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError('active/sentinel batch size must be eight')
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate_stage(manifest, base.__file__)
    from moe_steer import engine, manifests as M
    original = engine.engine_kwargs
    profile_kwargs = lambda *a, **kw: original(*a, **{**kw, 'max_num_seqs': 8,
                                                     'enforce_eager': False})
    kwargs = profile_kwargs(plugin=True, max_num_seqs=48, return_routed_experts=True)
    if kwargs['max_num_seqs'] != 8 or kwargs['enforce_eager']:
        raise ValueError('engine profile did not apply')
    if args.cpu_preflight:
        cases = build_cases(manifest, rows, arms, M.load_world(),
                            base.build_policy_table(actions))
        print(json.dumps({'status': 'PASS_TWO_TRANSITION_CPU_PREFLIGHT',
                          'stage': manifest['stage'], 'manifest_sha256': manifest['sha256'],
                          'hooks': [24, 28], 'requests': len(cases),
                          'batches': len(cases) // 8,
                          'analysis_enrolled_request_slots': sum(
                              meta['analysis_enrolled'] for _, meta in cases)}))
        return
    base.validate_manifest = validate_stage
    base.build_requests = build_cases
    engine.engine_kwargs = profile_kwargs
    try:
        base.run(args)
    finally:
        engine.engine_kwargs = original
        base.build_requests = ORIGINAL_BUILD
        base.validate_manifest = ORIGINAL_VALIDATE


if __name__ == '__main__':
    main()
