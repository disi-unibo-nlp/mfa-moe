"""Batched native-sentinel isolation qualification on frozen, invalid pilot starts.

Each eight-request batch has four active requests and four zero-policy sentinels
at identical prefix/seed positions. This is an engine test, not semantic efficacy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
PILOT = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json'
SOURCE_RUN = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
                  'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
                  'micro-screen-qual4-ac4c9651e71fe067')
ORIGINAL_VALIDATE = base.validate_manifest
ORIGINAL_BUILD = base.build_requests
CONDITIONS = [
    {'name': 'zero', 'policy': 'zero'},
    {'name': 'target05', 'policy': 'target_bias0.5'},
    {'name': 'target1', 'policy': 'target_bias1'},
    {'name': 'random05', 'policy': 'random_selector_bias0.5'},
    {'name': 'random1', 'policy': 'random_selector_bias1'},
]


def profile_kwargs(original, *args, **kwargs):
    return original(*args, **{**kwargs, 'max_num_seqs': 8, 'enforce_eager': False})


def validate_neighbor(manifest, driver_path):
    if manifest.get('schema') != 'routing-boundary-neighbor-qualification-v1':
        raise ValueError('unknown native-sentinel qualification schema')
    pilot = base.sealed(PILOT)
    if manifest.get('source_pilot_sha256') != pilot['sha256']:
        raise ValueError('frozen pilot source differs')
    if manifest.get('entry_driver_sha256') != base.file_sha(__file__):
        raise ValueError('native-sentinel entry source differs')
    if manifest.get('driver_sha256') != base.file_sha(driver_path):
        raise ValueError('base runner source differs')
    if manifest.get('profile') != {'max_num_seqs': 8, 'enforce_eager': False,
                                   'VLLM_BATCH_INVARIANT': 0, 'batch_size': 8}:
        raise ValueError('native-sentinel engine profile differs')
    if os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0':
        raise ValueError('unsupported batch-invariant backend must remain disabled')
    if manifest.get('conditions') != CONDITIONS:
        raise ValueError('conditions or prospective order changed')
    for key in ('rows', 'actions', 'seeds', 'max_tokens', 'stage',
                'random_set_by_family_seed', 'family_freeze_sha256',
                'base_tree_sha256', 'prefix_scout_sha256', 'prepared_sha256',
                'qualified_worker_sha256'):
        if manifest.get(key) != pilot[key]:
            raise ValueError('frozen pilot field differs: ' + key)
    if manifest.get('arms') != [
        {'name': 'active', 'policy': 'condition', 'role': 'conditional'},
        {'name': 'sentinel', 'policy': 'zero', 'role': 'native'}]:
        raise ValueError('active/sentinel pairing changed')
    codes = manifest.get('code_files', {})
    if (codes.get(str(Path(__file__))) != base.file_sha(__file__) or
            any(codes.get(path) != value for path, value in pilot['code_files'].items()) or
            any(base.file_sha(path) != value for path, value in codes.items())):
        raise ValueError('entry, base, or qualified worker code differs')
    rows, actions, _ = ORIGINAL_VALIDATE(pilot, driver_path)
    table = base.build_policy_table(actions)
    if table.hooked_layers() != (28,) or table.sealed() != json.loads(
            (SOURCE_RUN / 'policy-table.json').read_text()):
        raise ValueError('native-sentinel action table differs from frozen pilot')
    n = len(rows) * len(pilot['seeds']) * len(CONDITIONS) * 2
    prefill = sum((len(r['prompt_ids']) + len(r['prefix_ids'])) *
                  len(pilot['seeds']) * len(CONDITIONS) * 2 for r in rows)
    if (n != 80 or manifest.get('expected_requests') != n or
            manifest.get('expected_prefill_tokens') != prefill or
            manifest.get('maximum_decode_tokens') != n * 256 or
            manifest.get('maximum_context_tokens') != pilot['maximum_context_tokens']):
        raise ValueError('exact matched workload price differs')
    return rows, actions, manifest['arms']


def build_neighbor_requests(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    by_seed = {seed: list(CONDITIONS if seed == 0 else reversed(CONDITIONS))
               for seed in manifest['seeds']}
    for seed in manifest['seeds']:
        for condition in by_seed[seed]:
            for row in rows:
                prompt = row['prompt_ids'] + row['prefix_ids']
                if engine.THINK_END_ID in row['prefix_ids']:
                    raise ValueError('reasoning closure already in native prefix')
                common_seed = Q.crn(world.infos[row['question']], seed)
                for slot in ('active', 'sentinel'):
                    logical = condition['policy'] if slot == 'active' else 'zero'
                    policy = logical
                    if logical.startswith('random_selector_bias'):
                        dose = logical.removeprefix('random_selector_bias')
                        index = manifest['random_set_by_family_seed'][row['family']][seed]
                        policy = f'random{index}_bias{dose}'
                    uid = ('neighbor-v1|' + base.digest([
                        manifest['sha256'], row['uid'], seed, condition['name'], slot])[:24])
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
                            'question': row['question'], 'seed': seed,
                            'condition': condition['name'], 'slot': slot,
                            'arm': condition['name'] + '|' + slot, 'role': role,
                            'policy': policy, 'prompt_len': len(prompt),
                            'prompt_sha256': base.digest(prompt)}
                    cases.append((Q.QReq(uid, prompt, sampling), meta))
    if len(cases) != 80 or len({meta['uid'] for _, meta in cases}) != 80:
        raise ValueError('missing or duplicate matched requests')
    for index in range(0, 80, 8):
        batch = [meta for _, meta in cases[index:index + 8]]
        if (len({m['seed'] for m in batch}) != 1 or
                len({m['condition'] for m in batch}) != 1 or
                [m['slot'] for m in batch] != ['active', 'sentinel'] * 4 or
                len({m['family'] for m in batch}) != 4):
            raise ValueError('native sentinel batch layout differs')
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
        raise ValueError('native-sentinel matched batch size must be eight')
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate_neighbor(manifest, base.__file__)
    from moe_steer import engine, manifests as M
    kwargs = profile_kwargs(engine.engine_kwargs, plugin=True, max_num_seqs=48,
                            return_routed_experts=True)
    if kwargs['max_num_seqs'] != 8 or kwargs['enforce_eager']:
        raise ValueError('engine profile did not apply')
    if args.cpu_preflight:
        cases = build_neighbor_requests(manifest, rows, arms, M.load_world(),
                                        base.build_policy_table(actions))
        print(json.dumps({'status': 'PASS_MATCHED_SENTINEL_CPU_PREFLIGHT',
                          'manifest_sha256': manifest['sha256'], 'requests': len(cases),
                          'batches': len(cases) // 8, 'profile': manifest['profile']}))
        return
    original = engine.engine_kwargs
    base.validate_manifest = validate_neighbor
    base.build_requests = build_neighbor_requests
    engine.engine_kwargs = lambda *a, **kw: profile_kwargs(original, *a, **kw)
    try:
        base.run(args)
    finally:
        engine.engine_kwargs = original
        base.build_requests = ORIGINAL_BUILD
        base.validate_manifest = ORIGINAL_VALIDATE


if __name__ == '__main__':
    main()
