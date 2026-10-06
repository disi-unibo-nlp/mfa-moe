"""Four-arm serial 1024-token mechanism generation on sealed accepted starts.

The stage remains fail-closed while its full-stage price is held. It does not
construct an unsupported second action or a reversed-order comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
MANIFEST = DOC / 'MECHANISM_VALIDATION_MANIFEST_v1.json'
PRICE = DOC / 'MECHANISM_VALIDATION_PRICE_v1.json'
DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
FAMILY = DOC / 'family-freeze.json'
PROFILE = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}
ARMS = ['native', 'native_duplicate', 'target_bias1', 'random_bias1']
TRANSITIONS = {'candidate_to_verify': 'verify', 'approach_to_commit': 'commit'}
ORIGINAL_VALIDATE = base.validate_manifest
ORIGINAL_BUILD = base.build_requests
ACTIVE_SHARD = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(manifest, driver_path, *, require_price=True):
    import prepare_mechanism_validation_v1 as prep

    require(manifest['schema'] == 'mechanism-validation-manifest-v1' and
            manifest['horizon'] == 1024 and manifest['pulse_slots'] == [0] and
            manifest['pulse_width'] == 256 and manifest['seeds'] == [0, 1] and
            manifest['arms'] == ARMS and
            manifest['template_status'] == 'one_action_only_no_frozen_two_action_template',
            'unexpected mechanism template, horizon or arms')
    require(manifest['generation_driver_sha256'] == base.file_sha(__file__) and
            manifest['preparation_driver_sha256'] == base.file_sha(prep.__file__) and
            manifest['base_driver_sha256'] == base.file_sha(driver_path) and
            manifest['base_tree_sha256'] == base.REQUIRED_BASE_TREE,
            'generation source differs from sealed manifest')
    selection = base.sealed(prep.start.SELECTION)
    frame = base.sealed(prep.start.FRAME)
    family = base.sealed(FAMILY)
    dictionary = base.sealed(DICTIONARY)
    require(manifest['selection_sha256'] == selection['sha256'] and
            manifest['frame_sha256'] == frame['sha256'] and
            manifest['family_freeze_sha256'] == family['sha256'] and
            manifest['action_dictionary_sha256'] == dictionary['sha256'],
            'source seal differs from generation assignment')
    frozen = family['new_parent_pools']['parent_pools']
    require(len(frozen['mechanism']) == 128 and
            set(frozen['mechanism']) == set(selection['family_pool']) and
            not set(frozen['mechanism']) & set(frozen['discovery']) and
            not set(frozen['mechanism']) & set(frozen['utility']),
            'mechanism family pool overlaps discovery or utility')
    rows = manifest['rows']
    selected_by_uid = {r['uid']: r for r in selection['records']}
    frame_by_uid = {r['uid']: r for r in frame['records']}
    require(rows and len({r['uid'] for r in rows}) == len(rows) and
            {r['uid'] for r in rows} <= {r['uid'] for r in selection['records']} and
            {r['family'] for r in rows} <= set(frozen['mechanism']) and
            len({(r['family'], r['transition']) for r in rows}) == len(rows),
            'duplicate, unselected or overlapping mechanism start')
    accepted_families = len({r['family'] for r in rows})
    registered = accepted_families == 128
    require(manifest['frozen_family_pool_size'] == 128 and
            manifest['accepted_family_count'] == accepted_families and
            manifest['accepted_start_count'] == len(rows) and
            manifest['registered_128_family_feasibility'] ==
            ('PASS' if registered else 'FAIL') and
            manifest['stage_label'] == (
                'registered_128_family_mechanism_validation' if registered else
                'exploratory_accepted_family_subset'),
            'registered family feasibility differs from accepted starts')
    require(all(r['transition'] in TRANSITIONS and
                r['question'] == family['new_parent_pools']['representative_questions'][r['family']] and
                1 <= len(r['prefix_ids']) <= 8192 and r['prompt_ids'] and
                all(type(x) is int and x >= 0 for x in r['prompt_ids'] + r['prefix_ids']) and
                base.digest(r['prompt_ids']) == r['prompt_ids_sha256'] ==
                frame_by_uid[r['uid']]['analysis_meta']['prompt_ids_sha256'] and
                base.digest(r['prefix_ids']) == r['prefix_ids_sha256'] ==
                frame_by_uid[r['uid']]['analysis_meta']['prefix_ids_sha256'] and
                all(r[field] == selected_by_uid[r['uid']][field]
                    for field in ('family', 'transition', 'attempt_id', 'prefix_tokens'))
                for r in rows), 'native exact prefix or canonical question differs')
    require(manifest['rating_binding_sha256'] and manifest['rating_summary_sha256'],
            'reader acceptance provenance missing')
    # Revalidate the complete audit and its deterministic first-accepted rule.
    ratings_dir = Path(manifest['ratings_dir'])
    require(ratings_dir.is_absolute(), 'ratings source path must be absolute')
    chosen, binding, summary = prep.accepted_starts(selection, frame, ratings_dir)
    require(manifest['rating_binding_sha256'] == binding['sha256'] and
            manifest['rating_summary_sha256'] == summary['sha256'] and
            [r['uid'] for r in rows] == [r['uid'] for r in chosen],
            'generation enrolled a reader-rejected or substituted start')
    random_sets, orders = prep.schedule(rows)
    require(manifest['random_set_by_family'] == random_sets and
            manifest['arm_order_by_uid_seed'] == orders,
            'control identity or arm-position schedule differs')
    requests = len(rows) * 8
    require(manifest['expected_requests'] == requests and
            manifest['expected_prefill_tokens'] == sum(
                len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 8 and
            manifest['maximum_decode_tokens'] == requests * 1024 and
            manifest['maximum_context_tokens'] == max(
                len(r['prompt_ids']) + len(r['prefix_ids']) + 1024 for r in rows),
            'complete-stage token counts differ')
    require(os.environ.get('VLLM_BATCH_INVARIANT', '0') == '0',
            'serial profile environment differs')
    if require_price:
        qualification = prep.validate_qualification(base.sealed(prep.QUAL_RESULT))
        price = base.sealed(PRICE)
        expected_price = prep.price_stage(
            manifest, base.sealed(prep.REFERENCE_PRICE),
            price['max_wall_seconds_per_job'], qualification)
        require(price == {**expected_price, 'sha256': base.digest(expected_price)} and
                price['status'] == 'PASS_COMPLETE_STAGE',
                'complete-stage price or full-horizon qualification gate is held')
    templates = {t['transition']: t for t in dictionary['target_templates']}
    controls = dictionary['matched_random_control_sets']
    require(set(templates) == set(TRANSITIONS) and set(controls) == set(TRANSITIONS) and
            all(len(controls[t]) == 4 for t in TRANSITIONS),
            'frozen target or four matched random expert sets missing')
    actions = []
    for transition, stem in TRANSITIONS.items():
        actions.append({'name': f'{stem}_target_bias1', 'transition': transition,
                        'experts': templates[transition]['experts'], 'bias': 1.0})
        for i, control in enumerate(controls[transition]):
            actions.append({'name': f'{stem}_random{i}_bias1', 'transition': transition,
                            'experts': control['experts'], 'bias': 1.0})
    return rows, actions, [{'name': name} for name in ARMS]


def build_requests(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest

    cases = []
    for row in rows:
        prompt = row['prompt_ids'] + row['prefix_ids']
        info = world.infos.get(row['question'])
        require(info is not None and info['prompt_token_ids'] == row['prompt_ids'] and
                engine.THINK_END_ID not in row['prefix_ids'],
                'model world or reasoning closure differs from native prefix')
        stem = TRANSITIONS[row['transition']]
        for seed in manifest['seeds']:
            common_seed = Q.crn(info, seed)
            key = f"{row['uid']}|{seed}"
            for position, name in enumerate(manifest['arm_order_by_uid_seed'][key]):
                role = ('native' if name.startswith('native') else
                        'target' if name.startswith('target') else 'random')
                policy = ('zero' if role == 'native' else
                          f'{stem}_target_bias1' if role == 'target' else
                          f"{stem}_random{manifest['random_set_by_family'][row['family']]}_bias1")
                uid = 'mechanism-v1|' + base.digest([
                    manifest['sha256'], row['uid'], seed, name])[:24]
                extra = Q.steer_extra(table, uid, policy, len(row['prompt_ids']),
                                      prefix_len=len(row['prefix_ids']), restore_presence=True)
                if role != 'native':
                    template = {'action_policy_names': [policy], 'slots': [0],
                                'horizon': 1024}
                    extra['steer']['meta'] = {'routing_control': {
                        **template, 'sha256': worker_digest(template)}}
                sampling = Q.card_params(1024, common_seed, extra, presence=0.,
                                         routed_start=len(prompt) - 1)
                meta = {'uid': uid, 'prefix_uid': row['uid'],
                        'family': row['family'], 'transition': row['transition'],
                        'question': row['question'], 'seed': seed, 'arm': name,
                        'role': role, 'policy': policy,
                        'random_set': (manifest['random_set_by_family'][row['family']]
                                       if role == 'random' else None),
                        'execution_position': position, 'prompt_len': len(prompt),
                        'prompt_sha256': base.digest(prompt)}
                cases.append((Q.QReq(uid, prompt, sampling), meta))
    require(len(cases) == manifest['expected_requests'] and
            len({meta['uid'] for _, meta in cases}) == len(cases),
            'missing or duplicate mechanism request')
    for start in range(0, len(cases), 4):
        block = cases[start:start + 4]
        require(len(block) == 4 and
                len({tuple(req.prompt) for req, _ in block}) == 1 and
                len({req.sampling['seed'] for req, _ in block}) == 1 and
                len({meta['prefix_uid'] for _, meta in block}) == 1 and
                [meta['execution_position'] for _, meta in block] == [0, 1, 2, 3] and
                {meta['arm'] for _, meta in block} == set(ARMS),
                'same-prefix four-arm block differs')
    if ACTIVE_SHARD is not None:
        first, last = ACTIVE_SHARD
        require(0 <= first < last <= len(rows), 'shard row range differs')
        return cases[first * 8:last * 8]
    return cases


def completion(out, manifest, expected_requests):
    """Create a mechanism-specific receipt after base batch/array validation."""
    binding = base.sealed(out / 'BINDING.json')
    source = base.sealed(out / 'SUMMARY.json')
    require(binding['manifest_sha256'] == manifest['sha256'] and
            source['binding_sha256'] == binding['sha256'] and
            source['requests'] == expected_requests,
            'generation summary belongs to another manifest')
    counts = Counter()
    uids = []
    for i in range(source['batches']):
        batch = base.sealed(out / f'batch-{i:03d}.json')
        require(batch['manifest_sha256'] == manifest['sha256'] and
                batch['binding_sha256'] == binding['sha256'] and
                base.file_sha(out / f'batch-{i:03d}.npz') == batch['array_sha256'],
                'batch receipt or routing array differs')
        for row in batch['outputs']:
            uids.append(row['uid'])
            counts['assigned'] += 1
            counts['error'] += bool(row['error'])
            counts['routed_present'] += bool(row['routed_present'])
            counts['emitted_tokens'] += len(row['tokens'])
            counts['cap_1024'] += len(row['tokens']) == 1024
            counts['natural_stop'] += len(row['tokens']) < 1024 and not row['error']
    require(len(uids) == expected_requests and len(set(uids)) == len(uids),
            'completion omits or duplicates assigned requests')
    for failure_path in out.glob('batch-*-failure-*.json'):
        failed = base.sealed(failure_path)
        require(failed['manifest_sha256'] == manifest['sha256'] and
                failed['binding_sha256'] == binding['sha256'] and
                failed['status'] == 'batch incomplete; same-manifest resume required',
                'failure receipt belongs to another assignment')
        counts['incomplete_batch_attempts_recorded'] += 1
    body = {'schema': 'mechanism-validation-completion-v1',
            'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'],
            'source_summary_sha256': source['sha256'], 'counts': dict(counts),
            'status': 'COMPLETE_UNGRADED_GENERATION',
            'registered_128_family_feasibility':
            manifest['registered_128_family_feasibility'],
            'claim_limit': manifest['claim_limit'] +
            ' Logical assignments are unique; interrupted incomplete batches may have repeated physical inference.'}
    path = out / 'MECHANISM_COMPLETION.json'
    sealed = {**body, 'sha256': base.digest(body)}
    if path.exists():
        require(base.sealed(path) == sealed, 'mechanism completion differs')
    else:
        base.atomic_json(path, sealed)


def seal_stage(out, manifest, price):
    """Require every priced shard and every assigned UID before stage completion."""
    all_uids = []
    counts = Counter()
    for index, shard in enumerate(price['shards']):
        directory = out / f'shard-{index:03d}'
        receipt = base.sealed(directory / 'MECHANISM_COMPLETION.json')
        expected = (shard['end_row'] - shard['start_row']) * 8
        require(receipt['manifest_sha256'] == manifest['sha256'] and
                receipt['counts']['assigned'] == expected,
                'missing or rebound complete shard')
        source = base.sealed(directory / 'SUMMARY.json')
        require(source['sha256'] == receipt['source_summary_sha256'] and
                source['requests'] == expected,
                'shard summary differs')
        shard_uids = []
        for batch_index in range(source['batches']):
            batch = base.sealed(directory / f'batch-{batch_index:03d}.json')
            require(batch['manifest_sha256'] == manifest['sha256'] and
                    batch['binding_sha256'] == receipt['binding_sha256'] and
                    base.file_sha(directory / f'batch-{batch_index:03d}.npz') ==
                    batch['array_sha256'], 'shard receipt or routes differ')
            shard_uids.extend(row['uid'] for row in batch['outputs'])
        expected_uids = []
        for row in manifest['rows'][shard['start_row']:shard['end_row']]:
            for seed in manifest['seeds']:
                for arm in manifest['arm_order_by_uid_seed'][f"{row['uid']}|{seed}"]:
                    expected_uids.append('mechanism-v1|' + base.digest([
                        manifest['sha256'], row['uid'], seed, arm])[:24])
        require(shard_uids == expected_uids,
                'shard outputs differ from sealed row, seed or arm assignments')
        all_uids.extend(shard_uids)
        counts.update(receipt['counts'])
    require(len(all_uids) == manifest['expected_requests'] and
            len(set(all_uids)) == len(all_uids) and
            counts['assigned'] == manifest['expected_requests'],
            'stage has missing or duplicate assigned UID')
    body = {'schema': 'mechanism-validation-stage-completion-v1',
            'manifest_sha256': manifest['sha256'], 'price_sha256': price['sha256'],
            'shards': len(price['shards']), 'counts': dict(counts),
            'status': 'COMPLETE_UNGRADED_GENERATION',
            'accepted_family_count': manifest['accepted_family_count'],
            'accepted_start_count': manifest['accepted_start_count'],
            'registered_128_family_feasibility':
            manifest['registered_128_family_feasibility'],
            'claim_limit': manifest['claim_limit']}
    value = {**body, 'sha256': base.digest(body)}
    path = out / 'STAGE_COMPLETION.json'
    if path.exists():
        require(base.sealed(path) == value, 'stage completion differs')
    else:
        from prepare_mechanism_start_frame_v1 import write_once
        write_once(path, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--seal-stage', action='store_true')
    args = parser.parse_args()
    require(args.manifest.resolve() == MANIFEST.resolve() and args.batch_size == 4,
            'mechanism stage requires its frozen manifest and four-arm blocks')
    manifest = base.sealed(args.manifest)
    rows, actions, arms = validate(manifest, base.__file__,
                                   require_price=not args.cpu_preflight)
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_CPU_STRUCTURE_PREFLIGHT',
                          'manifest_sha256': manifest['sha256'],
                          'requests': manifest['expected_requests'],
                          'rows': len(rows), 'profile': PROFILE}))
        return
    price = base.sealed(PRICE)
    if args.seal_stage:
        seal_stage(args.out, manifest, price)
        print(json.dumps({'status': 'COMPLETE_UNGRADED_GENERATION',
                          'requests': manifest['expected_requests']}))
        return
    require(args.overlay is not None and args.shard_index is not None and
            0 <= args.shard_index < len(price['shards']),
            'GPU run requires qualified overlay and priced shard index')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('mechanism inference requires an authorized GPU Slurm step')
    shard = price['shards'][args.shard_index]
    global ACTIVE_SHARD
    ACTIVE_SHARD = (shard['start_row'], shard['end_row'])
    args.out = args.out / f'shard-{args.shard_index:03d}'
    from moe_steer import engine
    original = engine.engine_kwargs
    base.validate_manifest = validate
    base.build_requests = build_requests
    engine.engine_kwargs = lambda *a, **kw: original(
        *a, **{**kw, 'max_num_seqs': 1, 'enforce_eager': True})
    try:
        base.run(args)
        completion(args.out, manifest, (ACTIVE_SHARD[1] - ACTIVE_SHARD[0]) * 8)
    finally:
        ACTIVE_SHARD = None
        base.validate_manifest = ORIGINAL_VALIDATE
        base.build_requests = ORIGINAL_BUILD
        engine.engine_kwargs = original


if __name__ == '__main__':
    main()
