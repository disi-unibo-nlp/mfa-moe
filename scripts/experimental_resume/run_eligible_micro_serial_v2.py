"""Serial same-prefix causal screen on frozen, independently rated starts.

This is a discovery screen.  The serial/eager engine profile was exercised in
the separate four-family engineering qualification; outcomes need blind ratings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
MANIFEST = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v2.json'
POOL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
            'dense-discovery/JOINT_QWEN_NATIVE_EXACT_POOL_v1.json')
DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
PROFILE = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit')
ORIGINAL_VALIDATE = base.validate_manifest
ORIGINAL_BUILD = base.build_requests


def serial_kwargs(original, *args, **kwargs):
    return original(*args, **{**kwargs, 'max_num_seqs': 1, 'enforce_eager': True})


def expected_rows(pool):
    by_transition = {name: [] for name in TRANSITIONS}
    for record in pool['records']:
        if record['transition'] in by_transition:
            by_transition[record['transition']].append(record)
    representatives = base.sealed(base.FAMILY_FREEZE)['new_parent_pools']['representative_questions']
    chosen, used = [], set()
    for transition in TRANSITIONS:
        for row in sorted(by_transition[transition], key=lambda r: base.digest(
                ['eligible-micro-serial-v1', transition, r['family'], r['uid']])):
            if row['family'] not in used:
                chosen.append({**row, 'canonical_question': representatives[row['family']]})
                used.add(row['family'])
    if len(chosen) != 13:
        raise ValueError('high-agreement, globally disjoint support is not 13 families')
    return chosen


def validate(manifest, driver_path):
    if manifest.get('schema') != 'routing-eligible-micro-serial-v2':
        raise ValueError('unknown causal screen schema')
    if manifest.get('engine_profile') != PROFILE or os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0':
        raise ValueError('serial/eager execution profile differs')
    if manifest.get('entry_driver_sha256') != base.file_sha(__file__) or manifest.get('driver_sha256') != base.file_sha(driver_path):
        raise ValueError('screen or base driver source differs')
    if manifest.get('base_tree_sha256') != base.REQUIRED_BASE_TREE:
        raise ValueError('qualified sampler tree differs')
    pool, dictionary = base.sealed(POOL), base.sealed(DICTIONARY)
    if manifest.get('source_pool_sha256') != pool['sha256'] or manifest.get('action_dictionary_sha256') != dictionary['sha256']:
        raise ValueError('pretreatment pool or discovery action dictionary differs')
    rows = expected_rows(pool)
    if manifest.get('rows') != rows:
        raise ValueError('exact globally disjoint starts differ')
    family_freeze = base.sealed(base.FAMILY_FREEZE)
    discovery = set(family_freeze['new_parent_pools']['parent_pools']['discovery'])
    if (manifest.get('family_freeze_sha256') != family_freeze['sha256'] or
            not {row['family'] for row in rows} <= discovery):
        raise ValueError('non-discovery family entered screen')
    for row in rows:
        for field in ('prompt_ids', 'prefix_ids'):
            ids = row[field]
            if (not ids or any(type(token) is not int or token < 0 for token in ids)
                    or base.digest(ids) != row[field + '_sha256']):
                raise ValueError('native exact token IDs differ')
        if len(row['prefix_ids']) > 8192:
            raise ValueError('unqualified late-context start')
    if manifest.get('seeds') != [0, 1] or manifest.get('max_tokens') != 256:
        raise ValueError('assigned horizon or seeds differ')
    arms = manifest.get('arms')
    if arms != [
        {'name': 'native', 'role': 'native'},
        {'name': 'native_duplicate', 'role': 'native'},
        {'name': 'target_bias0.5', 'role': 'target'},
        {'name': 'target_bias1', 'role': 'target'},
        {'name': 'random_bias0.5', 'role': 'random'},
        {'name': 'random_bias1', 'role': 'random'},
    ]:
        raise ValueError('six paired causal arms differ')
    action_by_transition = {t['transition']: t for t in dictionary['target_templates']}
    if set(action_by_transition) != set(TRANSITIONS):
        raise ValueError('frozen target templates differ')
    actions = manifest.get('actions', [])
    names = {a['name'] for a in actions}
    if len(actions) != len(names) or len(actions) != 20:
        raise ValueError('target and four matched random policy sets incomplete')
    for transition, stem in (('candidate_to_verify', 'verify'), ('approach_to_commit', 'commit')):
        template = action_by_transition[transition]
        controls = dictionary['matched_random_control_sets'][transition]
        if len(controls) != 4:
            raise ValueError('four frozen random expert sets required')
        for bias, suffix in ((0.5, '0.5'), (1.0, '1')):
            expect = [(f'{stem}_target_bias{suffix}', template['experts'])]
            expect += [(f'{stem}_random{i}_bias{suffix}', item['experts']) for i, item in enumerate(controls)]
            for name, experts in expect:
                if {'name': name, 'transition': transition, 'experts': experts, 'bias': bias} not in actions:
                    raise ValueError('policy differs from observational dictionary: ' + name)
    schedule = manifest.get('random_set_by_family_seed', {})
    if set(schedule) != {r['family'] for r in rows} or any(
            type(v) is not list or len(v) != 2 or any(type(x) is not int or x not in range(4) for x in v)
            for v in schedule.values()):
        raise ValueError('random control assignment differs')
    for transition in TRANSITIONS:
        members = [r['family'] for r in rows if r['transition'] == transition]
        counts = [sum(schedule[family].count(i) for family in members) for i in range(4)]
        if max(counts) - min(counts) > 1:
            raise ValueError('matched random sets unbalanced within transition')
    expected_requests = len(rows) * 2 * len(arms)
    expected_prefill = sum(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 2 * len(arms)
    if (manifest.get('expected_requests') != expected_requests or
            manifest.get('expected_prefill_tokens') != expected_prefill or
            manifest.get('maximum_decode_tokens') != expected_requests * 256 or
            manifest.get('maximum_context_tokens') != max(len(r['prompt_ids']) + len(r['prefix_ids']) + 256 for r in rows)):
        raise ValueError('exact workload price differs')
    codes = manifest.get('code_files', {})
    if (codes.get(str(Path(__file__))) != base.file_sha(__file__) or
            codes.get(str(Path(driver_path))) != base.file_sha(driver_path) or
            any(base.file_sha(path) != expected for path, expected in codes.items())):
        raise ValueError('bound driver/worker source changed')
    return rows, actions, arms


def build_requests(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    for row in rows:
        prompt = row['prompt_ids'] + row['prefix_ids']
        canonical = row['canonical_question']
        if (engine.THINK_END_ID in row['prefix_ids'] or canonical not in world.infos or
                row['prompt_ids'] != world.infos[canonical]['prompt_token_ids']):
            raise ValueError('closed reasoning or missing frozen question')
        stem = 'verify' if row['transition'] == 'candidate_to_verify' else 'commit'
        for seed in manifest['seeds']:
            common_seed = Q.crn(world.infos[canonical], seed)
            for arm in arms:
                uid = 'eligible-v2|' + base.digest([manifest['sha256'], row['uid'], seed, arm['name']])[:24]
                name = arm['name']
                if arm['role'] == 'native':
                    policy = 'zero'
                elif arm['role'] == 'target':
                    policy = f'{stem}_{name}'
                else:
                    dose = name.removeprefix('random_bias')
                    index = manifest['random_set_by_family_seed'][row['family']][seed]
                    policy = f'{stem}_random{index}_bias{dose}'
                extra = Q.steer_extra(table, uid, policy, len(row['prompt_ids']),
                                      prefix_len=len(row['prefix_ids']), restore_presence=True)
                if arm['role'] != 'native':
                    template = {'action_policy_names': [policy], 'slots': [0], 'horizon': 1024}
                    extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
                sampling = Q.card_params(256, common_seed, extra, presence=0.,
                                         routed_start=len(prompt) - 1)
                meta = {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                        'transition': row['transition'], 'question': canonical,
                        'seed': seed, 'arm': name, 'role': arm['role'], 'policy': policy,
                        'prompt_len': len(prompt), 'prompt_sha256': base.digest(prompt)}
                cases.append((Q.QReq(uid, prompt, sampling), meta))
    if len(cases) != manifest['expected_requests'] or len({m['uid'] for _, m in cases}) != len(cases):
        raise ValueError('missing or duplicate assigned request')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    if args.batch_size != 6 or args.manifest.resolve() != MANIFEST.resolve():
        raise ValueError('screen requires frozen manifest and one matched six-arm batch')
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate(manifest, base.__file__)
    from moe_steer import engine, manifests as M
    kwargs = serial_kwargs(engine.engine_kwargs, plugin=True, max_num_seqs=48,
                           return_routed_experts=True)
    if kwargs['max_num_seqs'] != 1 or not kwargs['enforce_eager']:
        raise ValueError('serial/eager kwargs did not apply')
    if args.cpu_preflight:
        cases = build_requests(manifest, rows, arms, M.load_world(), base.build_policy_table(actions))
        print(json.dumps({'status': 'PASS_ELIGIBLE_MICRO_SERIAL_PREFLIGHT',
                          'manifest_sha256': manifest['sha256'], 'requests': len(cases),
                          'batches': len(cases) // 6, 'profile': PROFILE}))
        return
    original = engine.engine_kwargs
    base.validate_manifest, base.build_requests = validate, build_requests
    engine.engine_kwargs = lambda *a, **kw: serial_kwargs(original, *a, **kw)
    try:
        base.run(args)
    finally:
        engine.engine_kwargs = original
        base.validate_manifest, base.build_requests = ORIGINAL_VALIDATE, ORIGINAL_BUILD


if __name__ == '__main__':
    main()
