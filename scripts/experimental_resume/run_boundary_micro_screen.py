"""Exploratory same-prefix routing-action micro-screen on a frozen discovery manifest.

This driver measures actual first-stage routing and saves raw, arm-identified
continuations. A separate, bound blind-rating frame is required. It does not train
or execute an online detector, and cannot be used as mechanism validation.
GPU execution is restricted to Slurm compute nodes and a separately frozen
ordered-worker overlay that passed engineering qualification.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


FAMILY_FREEZE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/family-freeze.json')
REQUIRED_BASE_TREE = '9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430'
SCOUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/CANDIDATE_PREFIX_SCOUT_v1.json')
PREPARED = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/CAUSAL_MICROSCREEN_PREPARED_v0.2.json')
WORKER_PREP = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/ordered-qualification-v1/PREPARED.cpu-recovery-v2.json')
WORKER_QUAL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/ordered-qualification-v1/results-cpu-recovery-v2/QUALIFICATION.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed JSON seal: ' + str(path))
    return value


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def validate_manifest(manifest, driver_path):
    from moe_exp.routing_control.design import Action
    if manifest.get('schema') != 'routing-boundary-micro-screen-v1':
        raise ValueError('unknown micro-screen manifest schema')
    if manifest.get('base_tree_sha256') != REQUIRED_BASE_TREE:
        raise ValueError('base tree differs from qualified ordered worker')
    if manifest.get('driver_sha256') != file_sha(driver_path):
        raise ValueError('frozen micro-screen driver differs')
    scout, prepared = sealed(SCOUT), sealed(PREPARED)
    if (manifest.get('prefix_scout_sha256') != scout['sha256'] or
        manifest.get('prepared_sha256') != prepared['sha256']):
        raise ValueError('frozen native prefix scout or action preparation differs')
    worker_prep, worker_qual = sealed(WORKER_PREP), sealed(WORKER_QUAL)
    if (not worker_qual['pass'] or worker_qual['base_tree'] != REQUIRED_BASE_TREE or
        worker_qual['worker_code_digest'] != worker_prep['sha256'] or
        manifest.get('qualified_worker_sha256') != worker_qual['sha256']):
        raise ValueError('ordered worker does not match passing GPU qualification')
    if any(file_sha(path) != expected for path, expected in worker_prep['files'].items()):
        raise ValueError('qualified overlay source changed')
    if manifest.get('max_tokens') != 256 or manifest.get('seeds') != [0, 1]:
        raise ValueError('this exploratory screen requires 256 tokens and seeds 0,1')
    if len(manifest.get('code_files', {})) < 3:
        raise ValueError('worker and driver source hashes are required')
    if any(file_sha(path) != expected for path, expected in manifest['code_files'].items()):
        raise ValueError('frozen worker or source file differs')
    overlay = Path(worker_prep['overlay'])
    required_files = (Path(driver_path), overlay / 'moe_exp/routing_control/worker_adapter.py',
                      overlay / 'moe_exp/routing_control/ordered_vllm.py')
    if any(manifest['code_files'].get(str(path)) != file_sha(path) for path in required_files):
        raise ValueError('manifest does not bind the exact qualified worker overlay')
    family_freeze = sealed(FAMILY_FREEZE)
    if manifest.get('family_freeze_sha256') != family_freeze['sha256']:
        raise ValueError('family freeze differs')
    discovery = set(family_freeze['new_parent_pools']['parent_pools']['discovery'])
    rows = manifest.get('rows', [])
    stage = manifest.get('stage')
    if stage not in ('pilot', 'full') or len(rows) != (4 if stage == 'pilot' else 12):
        raise ValueError('pilot needs four and full stage twelve fixed prefixes')
    if len({r['family'] for r in rows}) != len(rows):
        raise ValueError('one prefix per discovery family')
    if not {r['family'] for r in rows} <= discovery:
        raise ValueError('validation or utility family entered discovery micro-screen')
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('duplicate prefix UID')
    scout_rows = {r['uid']: r for r in scout['records']}
    if stage == 'full' and {r['uid'] for r in rows} != set(scout_rows):
        raise ValueError('full stage must enroll the complete frozen scout')
    for row in rows:
        if set(row) != {'uid', 'question', 'family', 'prompt_ids', 'prefix_ids'}:
            raise ValueError('unknown prefix fields')
        if row['question'] not in family_freeze['new_parent_pools']['families'][row['family']]:
            raise ValueError('question is outside its frozen family')
        if not row['prompt_ids'] or not 1 <= len(row['prefix_ids']) <= 8192:
            raise ValueError('empty or oversized eligible native prefix')
        if any(type(x) is not int or x < 0 for x in row['prompt_ids'] + row['prefix_ids']):
            raise ValueError('invalid token ID in prefix')
        source = scout_rows.get(row['uid'])
        if source is None or any(row[key] != source[key] for key in row):
            raise ValueError('prompt or prefix differs from the sealed native scout')
    actions = manifest.get('actions', [])
    if not 1 <= len(actions) <= 10 or len({a['name'] for a in actions}) != len(actions):
        raise ValueError('one to ten uniquely named exploratory actions required')
    for action in actions:
        if set(action) != {'name', 'transition', 'experts', 'bias'}:
            raise ValueError('unknown action field')
        Action(action['name'], action['transition'],
               tuple((int(layer), tuple(experts)) for layer, experts in action['experts']),
               action['bias']).validate()
    names = {a['name'] for a in actions}
    arms = manifest.get('arms', [])
    if not 3 <= len(arms) <= 6 or len({a['name'] for a in arms}) != len(arms):
        raise ValueError('three to six uniquely named arms required')
    if sum(a['role'] == 'native' for a in arms) < 2:
        raise ValueError('duplicate-native isolation arms required')
    if not {'target', 'random'} <= {a['role'] for a in arms}:
        raise ValueError('target and matched-random roles required')
    for arm in arms:
        if set(arm) != {'name', 'policy', 'role'} or arm['role'] not in ('native', 'target', 'random'):
            raise ValueError('unknown arm fields or role')
        if arm['role'] == 'native' and arm['policy'] != 'zero':
            raise ValueError('native arm must use zero policy')
        if arm['role'] == 'target' and arm['policy'] not in names:
            raise ValueError('edited arm references unknown action')
        if arm['role'] == 'random' and arm['policy'] not in (
            'random_selector_bias0.5', 'random_selector_bias1'
        ):
            raise ValueError('random arm must use a frozen set selector and dose')
    by_name = {a['name']: a for a in actions}
    schedule = manifest.get('random_set_by_family_seed')
    if set(schedule or {}) != {row['family'] for row in rows}:
        raise ValueError('random control assignment must cover exact enrolled families')
    assignments = []
    for row in rows:
        pair = schedule[row['family']]
        if type(pair) is not list or len(pair) != 2 or any(type(k) is not int or k not in range(4) for k in pair):
            raise ValueError('each family needs two valid random-set assignments')
        assignments.extend(pair)
    if max(assignments.count(k) for k in range(4)) - min(assignments.count(k) for k in range(4)) > 1:
        raise ValueError('four random sets must be balanced across families and seeds')
    for index in range(4):
        for dose in ('0.5', '1'):
            name = f'random{index}_bias{dose}'
            if name not in by_name or by_name[name]['bias'] != float(dose):
                raise ValueError('missing frozen matched-random set and dose')
    target_sets = {tuple((layer, tuple(ids)) for layer, ids in by_name[a['policy']]['experts'])
                   for a in arms if a['role'] == 'target'}
    random_sets = {tuple((layer, tuple(ids)) for layer, ids in a['experts'])
                   for a in actions if a['name'].startswith('random')}
    if target_sets & random_sets:
        raise ValueError('matched-random control reuses target expert identities')
    expected_requests = len(rows) * len(arms) * len(manifest['seeds'])
    expected_prefill = sum((len(r['prompt_ids']) + len(r['prefix_ids'])) *
                           len(arms) * len(manifest['seeds']) for r in rows)
    expected_decode = expected_requests * manifest['max_tokens']
    expected_context = max(len(r['prompt_ids']) + len(r['prefix_ids']) +
                           manifest['max_tokens'] for r in rows)
    if any(manifest.get(key) != value for key, value in (
        ('expected_requests', expected_requests),
        ('expected_prefill_tokens', expected_prefill),
        ('maximum_decode_tokens', expected_decode),
        ('maximum_context_tokens', expected_context),
    )):
        raise ValueError('manifest resource quantities disagree with exact enrollment')
    return rows, actions, arms


def build_requests(manifest, rows, arms, world, table):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    for row in rows:
        prompt = row['prompt_ids'] + row['prefix_ids']
        if engine.THINK_END_ID in row['prefix_ids']:
            raise ValueError('reasoning closure is already inside a prefix')
        if row['question'] not in world.infos:
            raise ValueError('question not in frozen model world')
        for seed in manifest['seeds']:
            common_seed = Q.crn(world.infos[row['question']], seed)
            for arm in arms:
                uid = 'micro-v1|' + digest([manifest['sha256'], row['uid'], seed, arm['name']])[:24]
                policy_name = arm['policy']
                if arm['role'] == 'random':
                    dose = arm['policy'].removeprefix('random_selector_bias')
                    set_index = manifest['random_set_by_family_seed'][row['family']][seed]
                    policy_name = f'random{set_index}_bias{dose}'
                extra = Q.steer_extra(table, uid, policy_name, len(row['prompt_ids']),
                                      prefix_len=len(row['prefix_ids']), restore_presence=True)
                if arm['role'] != 'native':
                    template = {'action_policy_names': [policy_name], 'slots': [0], 'horizon': 1024}
                    extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
                sampling = Q.card_params(256, common_seed, extra, presence=0.,
                                         routed_start=len(prompt) - 1)
                req = Q.QReq(uid, prompt, sampling)
                meta = {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                        'question': row['question'], 'seed': seed, 'arm': arm['name'],
                        'role': arm['role'], 'policy': policy_name,
                        'prompt_len': len(prompt), 'prompt_sha256': digest(prompt)}
                cases.append((req, meta))
    if len({m['uid'] for _, m in cases}) != len(cases):
        raise ValueError('request UID collision')
    return cases


def build_policy_table(actions):
    """Validate proposed expert sets against the real qualified policy spec."""
    from moe_steer import policies as P
    from moe_steer.spec import Schedule, TargetSet
    targets = [TargetSet(a['name'], 'CUSTOM',
                         tuple((int(layer), tuple(sorted(experts))) for layer, experts in a['experts']),
                         'exploratory matched native contrast; causal validation pending')
               for a in actions]
    policies = [P.make_policy(target, P.Operator('bias', 1, a['bias']),
                              Schedule('always'), name=a['name'])
                for target, a in zip(targets, actions, strict=True)]
    return P.build_table(policies)


def prepare_worker_import_path(engine, env, overlay):
    """Export the qualified worker path in the parent and vLLM spawn children."""
    engine.apply_env(env)
    overlay_root = str(Path(overlay).resolve())
    sys.path[:] = [overlay_root, *(path for path in sys.path if path != overlay_root)]
    import moe_exp
    expected_package = (Path(overlay) / 'moe_exp').resolve()
    expected_worker = (expected_package / 'routing_control/ordered_vllm.py').resolve()
    worker_spec = importlib.util.find_spec('moe_exp.routing_control.ordered_vllm')
    if (Path(moe_exp.__file__).resolve().parent != expected_package or
        worker_spec is None or worker_spec.origin is None or
        Path(worker_spec.origin).resolve() != expected_worker):
        raise ValueError('worker import resolves outside the qualified overlay')


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('micro-screen generation requires GPU Slurm')
    from moe_steer import engine, manifests as M, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest

    manifest = sealed(args.manifest)
    rows, actions, arms = validate_manifest(manifest, __file__)
    worker_prep = sealed(WORKER_PREP)
    if args.overlay.resolve() != Path(worker_prep['overlay']).resolve():
        raise ValueError('runtime overlay differs from qualified ordered worker')
    if args.batch_size % len(arms):
        raise ValueError('batch size splits same-prefix arms')
    if engine.code_tree_sha256() != REQUIRED_BASE_TREE:
        raise ValueError('runtime sampler tree differs')
    binding = {'schema': 'routing-boundary-micro-screen-binding-v1',
               'manifest_sha256': manifest['sha256'], 'driver_sha256': file_sha(__file__),
               'ordered_worker_overlay': str(args.overlay), 'base_tree_sha256': REQUIRED_BASE_TREE}
    binding = {**binding, 'sha256': digest(binding)}
    args.out.mkdir(parents=True, exist_ok=True)
    lock_stream = (args.out / '.writer.lock').open('a+')
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_path = args.out / 'BINDING.json'
    if binding_path.exists():
        if sealed(binding_path) != binding:
            raise ValueError('output directory already belongs to another manifest or code')
    elif any(path.name != '.writer.lock' for path in args.out.iterdir()):
        raise ValueError('nonempty unbound output directory')
    else:
        atomic_json(binding_path, binding)
    world = M.load_world()
    table = build_policy_table(actions)
    table_path = args.out / 'policy-table.json'
    table_json = table.sealed()
    if table_path.exists() and json.loads(table_path.read_text()) != table_json:
        raise ValueError('same-bound policy table changed')
    if not table_path.exists():
        atomic_json(table_path, table_json)
    cases = build_requests(manifest, rows, arms, world, table)
    batches = [cases[i:i + args.batch_size] for i in range(0, len(cases), args.batch_size)]
    if not len(arms) <= args.batch_size <= 48:
        raise ValueError('batch size must hold all arms and fit the qualified engine')
    remaining = []
    for i, batch in enumerate(batches):
        receipt = args.out / f'batch-{i:03d}.json'
        if not receipt.exists():
            remaining.append(i)
            continue
        saved = sealed(receipt)
        array_path = args.out / f'batch-{i:03d}.npz'
        if (saved['manifest_sha256'] != manifest['sha256'] or
            saved['binding_sha256'] != binding['sha256'] or
            [r['uid'] for r in saved['outputs']] != [meta['uid'] for _, meta in batch] or
            not array_path.is_file() or file_sha(array_path) != saved['array_sha256']):
            raise ValueError('completed batch cannot be reused under this manifest')
    if not remaining:
        if not (args.out / 'SUMMARY.json').exists():
            result = {'schema': 'routing-boundary-micro-screen-summary-v1',
                      'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'],
                      'job_id': os.environ['SLURM_JOB_ID'], 'batches': len(batches),
                      'requests': len(cases), 'elapsed_driver_seconds': 0.,
                      'status': 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION',
                      'interpretation': 'all batches recovered from same-bound sealed receipts; semantic ratings and first-stage audit are separate'}
            atomic_json(args.out / 'SUMMARY.json', {**result, 'sha256': digest(result)})
        else:
            completed = sealed(args.out / 'SUMMARY.json')
            if (completed['manifest_sha256'] != manifest['sha256'] or
                completed['binding_sha256'] != binding['sha256'] or
                completed['requests'] != len(cases)):
                raise ValueError('same-bound summary differs')
        return

    deadline = Q.Deadline.from_env()
    deadline.require(950, 'cold import, load and first context-matched batch')
    fingerprint = engine.fingerprint()
    telemetry = args.out / 'telemetry'
    telemetry.mkdir(exist_ok=True)
    env = engine.engine_env(table_path, telemetry, expect_fingerprint=fingerprint['combined'])
    env['PYTHONPATH'] = os.pathsep.join((str(args.overlay), env['PYTHONPATH']))
    kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=48, return_routed_experts=True)
    kwargs['worker_extension_cls'] = 'moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension'
    kwargs.update(gpu_memory_utilization=.80, long_prefill_token_threshold=1024)
    driver = None
    started = time.time()
    current_batch = None
    try:
        # vLLM spawn copies the parent's sys.path. engine.apply_env prepends
        # PYTHONPATH entries in reverse order, so restore the qualified overlay
        # to first place before any child is launched.
        prepare_worker_import_path(engine, env, args.overlay)
        model = engine.build_llm(kwargs)
        driver = Q.QDriver(model, plugin=True)
        harness = Q.Harness(driver, telemetry, deadline=deadline, abort_margin=90.)
        for i in remaining:
            deadline.require(120, f'micro-screen batch {i}')
            cases_in_batch = batches[i]
            current_batch = i
            assignment = {'schema': 'routing-boundary-micro-screen-assignment-v1',
                          'manifest_sha256': manifest['sha256'],
                          'binding_sha256': binding['sha256'], 'batch_index': i,
                          'requests': [item for _, item in cases_in_batch]}
            assignment_path = args.out / f'batch-{i:03d}-assignment.json'
            assignment = {**assignment, 'sha256': digest(assignment)}
            if assignment_path.exists():
                if sealed(assignment_path) != assignment:
                    raise ValueError('same-bound batch assignment changed')
            else:
                atomic_json(assignment_path, assignment)
            batch = harness.run([req for req, _ in cases_in_batch], cap_in_flight=48)
            harness.flush()
            records, counts, info = harness.telemetry()
            arrays = {}
            outputs = []
            for _, item in cases_in_batch:
                out = batch.outcomes.get(item['uid'])
                if out is None:
                    raise ValueError('missing assigned request outcome')
                routed = np.asarray(out.routed) if out.routed is not None else None
                valid_routed = routed is not None and routed.shape == (len(out.tokens), 40, 8)
                if not out.error and not valid_routed:
                    raise ValueError('routed arrays and emitted tokens disagree')
                if valid_routed:
                    arrays[item['uid']] = routed
                native_checks = {}
                action_dose = {}
                missing_telemetry = []
                for rank in (0, 1):
                    record = records.get(rank, {}).get(item['uid'])
                    if record is None:
                        if not out.error:
                            raise ValueError('missing worker-rank telemetry for ' + item['uid'])
                        missing_telemetry.append(rank)
                        continue
                    native_checks[str(rank)] = record.get('inactive_native_checks')
                    if not out.error and (not native_checks[str(rank)] or any(
                        row.get('expert_identity_mismatches') or row.get('weight_mismatches')
                        for row in native_checks[str(rank)].values()
                    )):
                        raise ValueError('inactive routing differs from native')
                    if not out.error and item['role'] == 'native' and record.get('cpu_active_rows') != 0:
                        raise ValueError('native request had active edited rows')
                    if item['role'] != 'native':
                        action_dose[str(rank)] = {'rows': record.get('ordered_action_rows'),
                                                  'segments': record.get('ordered_segments'),
                                                  'dose': record.get('ordered_action_dose')}
                outputs.append({**item, 'tokens': out.tokens, 'finish': out.finish,
                                'stop_reason': out.stop_reason, 'error': out.error,
                                'routed_present': bool(valid_routed),
                                'missing_telemetry_ranks': missing_telemetry,
                                'inactive_native_checks': native_checks,
                                'action_dose': action_dose})
            array_path = args.out / f'batch-{i:03d}.npz'
            array_tmp = array_path.with_suffix('.npz.part-' + os.environ['SLURM_JOB_ID'])
            with array_tmp.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(array_tmp, array_path)
            body = {'schema': 'routing-boundary-micro-screen-batch-v1',
                    'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'],
                    'batch_index': i, 'job_id': os.environ['SLURM_JOB_ID'],
                    'elapsed_seconds': batch.seconds, 'steps': batch.steps,
                    'array_sha256': file_sha(array_path), 'telemetry_parse_info': info,
                    'telemetry_line_counts': {str(rank): {m['uid']: counts.get(rank, {}).get(m['uid'], 0)
                                                           for _, m in cases_in_batch}
                                              for rank in (0, 1)},
                    'outputs': outputs}
            atomic_json(args.out / f'batch-{i:03d}.json', {**body, 'sha256': digest(body)})
            print(json.dumps({'batch': i, 'requests': len(cases_in_batch),
                              'seconds': batch.seconds, 'completed_batches': i + 1,
                              'total_batches': len(batches)}), flush=True)
            current_batch = None
        result = {'schema': 'routing-boundary-micro-screen-summary-v1',
                  'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'],
                  'job_id': os.environ['SLURM_JOB_ID'], 'batches': len(batches),
                  'requests': len(cases), 'elapsed_driver_seconds': time.time() - started,
                  'status': 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION',
                  'interpretation': 'same-prefix causal assignment; semantic ratings, first-stage audit, and intention-to-treat analysis are separate'}
        atomic_json(args.out / 'SUMMARY.json', {**result, 'sha256': digest(result)})
        Q.finish_child(0, driver)
    except Exception as error:
        if current_batch is not None:
            failure_path = args.out / f'batch-{current_batch:03d}-failure-{os.environ["SLURM_JOB_ID"]}.json'
            if not failure_path.exists():
                failure = {'schema': 'routing-boundary-micro-screen-failure-v1',
                           'manifest_sha256': manifest['sha256'],
                           'binding_sha256': binding['sha256'],
                           'batch_index': current_batch, 'job_id': os.environ['SLURM_JOB_ID'],
                           'assigned_uids': [item['uid'] for _, item in batches[current_batch]],
                           'exception_type': type(error).__name__,
                           'exception_message': str(error)[:500],
                           'status': 'batch incomplete; same-manifest resume required'}
                atomic_json(failure_path, {**failure, 'sha256': digest(failure)})
        Q.finish_child(3, driver)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    run(parser.parse_args())
