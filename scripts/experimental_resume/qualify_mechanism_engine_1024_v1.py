"""Serial/eager 1,024-token routing qualification on frozen engineering prefixes.

This is an engineering check, not a semantic-control result or engine-equivalence
claim. H1-H4 and the prior batched ordered-worker tests are provenance, not
substitutes for this exact serial execution profile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
STAGE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
MANIFEST = REPO / 'report/experimental-resume-v1/MECHANISM_ENGINE_1024_QUAL_MANIFEST_v1.json'
RESULT = REPO / 'report/experimental-resume-v1/MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json'
PROFILE = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}


class PreemptingDriver:
    def __init__(self, inner, calls=(256, 600)):
        self.inner = inner
        self.calls = set(calls)
        self.count = 0
        self.events = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self):
        self.count += 1
        if self.count in self.calls:
            flying = self.inner.in_flight()
            ok = bool(self.inner.llm.reset_prefix_cache(reset_running_requests=True)) if flying else False
            self.events.append({'call': self.count, 'in_flight': flying, 'reset_ok': ok})
        return self.inner.step()


def validate_manifest(manifest: dict) -> None:
    if (manifest.get('schema') != 'routing-mechanism-serial-eager-1024-qual-manifest-v1' or
            manifest.get('engine_profile') != PROFILE or
            os.environ.get('VLLM_BATCH_INVARIANT', '0') != '0' or
            manifest.get('qualification_driver_sha256') != base.file_sha(__file__) or
            manifest.get('base_tree_sha256') != base.REQUIRED_BASE_TREE or
            manifest.get('max_tokens') != 1024 or
            manifest.get('pulse_slots') != [0, 512] or
            manifest.get('pulse_length') != 256 or
            manifest.get('seeds') != [0]):
        raise ValueError('serial/eager 1,024-token qualification contract differs')
    for path, expected in manifest['code_files'].items():
        if base.file_sha(path) != expected:
            raise ValueError('qualified worker/driver code differs: ' + path)
    for name in ('source_fixture', 'source_family_freeze', 'source_action_dictionary',
                 'source_ordered_qualification', 'source_serial_qualification'):
        source = manifest[name]
        loaded = base.sealed(source['path'])
        if loaded['sha256'] != source['sha256']:
            raise ValueError('sealed qualification source differs: ' + name)
    h14 = manifest['source_h14']
    if base.file_sha(h14['path']) != h14['file_sha256']:
        raise ValueError('H1-H4 source bytes changed')
    h14_body = json.loads(Path(h14['path']).read_text())
    if (not h14_body['pass'] or not all(h14_body['evidence']['holes'][x]['pass']
                                       for x in ('H1', 'H2', 'H3', 'H4'))):
        raise ValueError('H1-H4 inherited force/prefix/neighbour/recompute source did not pass')
    ordered = base.sealed(manifest['source_ordered_qualification']['path'])
    if not ordered.get('pass') or ordered.get('base_tree') != base.REQUIRED_BASE_TREE:
        raise ValueError('ordered-worker source did not pass')
    if (manifest['qualified_worker_sha256'] != ordered['sha256'] or
            manifest['worker_code_digest'] != ordered['worker_code_digest']):
        raise ValueError('qualified worker differs from ordered source')
    fixture = base.sealed(manifest['source_fixture']['path'])
    freeze = base.sealed(manifest['source_family_freeze']['path'])
    dictionary = base.sealed(manifest['source_action_dictionary']['path'])
    if (fixture['family_freeze_sha256'] != freeze['sha256'] or
            len(fixture['rows']) != 4 or
            manifest['fixtures'] != [
                {'family': fixture['rows'][0]['family'], 'question': fixture['rows'][0]['question'],
                 'prefix_tokens': 128},
                {'family': fixture['rows'][1]['family'], 'question': fixture['rows'][1]['question'],
                 'prefix_tokens': 2048}] or
            len({r['family'] for r in manifest['fixtures']}) != 2 or
            {t['transition'] for t in dictionary['target_templates']} !=
            {'candidate_to_verify', 'approach_to_commit'}):
        raise ValueError('two fixed short/long engineering fixtures or templates differ')
    if manifest['expected_requests'] != 12 or manifest['maximum_decode_tokens'] != 11 * 1024 + 128:
        raise ValueError('complete qualification request/cap price differs')
    expected_prefill = (4 * (len(fixture['rows'][0]['prompt_ids']) + 128) +
                        7 * (len(fixture['rows'][1]['prompt_ids']) + 2048) +
                        len(fixture['rows'][0]['prompt_ids']) + 129)
    if manifest['expected_prefill_tokens'] != expected_prefill:
        raise ValueError('qualification prefill price differs')


def build_policy_table(dictionary):
    from moe_steer import policies as P
    from moe_steer.spec import Schedule, TargetSet
    actions = []
    mapping = {}
    for transition, stem in (('candidate_to_verify', 'verify'), ('approach_to_commit', 'commit')):
        target = next(t for t in dictionary['target_templates'] if t['transition'] == transition)
        random = dictionary['matched_random_control_sets'][transition][0]
        for role, item in (('target', target), ('random', random)):
            named = TargetSet(f'qual-{stem}-{role}', 'CUSTOM',
                              tuple((int(layer), tuple(sorted(experts)))
                                    for layer, experts in item['experts']),
                              'fixed engineering set; no semantic discovery')
            policy = P.make_policy(named, P.Operator('bias', 1, 1.0), Schedule('always'),
                                   name=f'qual-{stem}-{role}-bias1')
            actions.append(policy)
            mapping[(transition, role)] = policy.name
    return P.build_table(actions), mapping


def cases_for_manifest(manifest, fixture, world, table, mapping):
    from moe_steer import engine, qualify as Q
    from moe_exp.routing_control.design import digest as worker_digest
    cases = []
    for index, transition in ((0, 'candidate_to_verify'), (1, 'approach_to_commit')):
        row = fixture['rows'][index]
        count = manifest['fixtures'][index]['prefix_tokens']
        prefix = row['completion_ids'][:count]
        if engine.THINK_END_ID in prefix or row['prompt_ids'] != world.infos[row['question']]['prompt_token_ids']:
            raise ValueError('engineering prefix already closed or original prompt differs')
        prompt = row['prompt_ids'] + prefix
        for role in ('native', 'target', 'random', 'native_duplicate'):
            name = mapping.get((transition, role), 'zero')
            uid = 'mechanism-qual-v1|' + base.digest([manifest['sha256'], index, role])[:24]
            extra = Q.steer_extra(table, uid, name, len(row['prompt_ids']),
                                  prefix_len=len(prefix), restore_presence=True)
            if role in ('target', 'random'):
                template = {'action_policy_names': [name], 'slots': [0], 'horizon': 1024}
                extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
            sampling = Q.card_params(1024, Q.crn(world.infos[row['question']], 0), extra,
                                     presence=0., routed_start=len(prompt) - 1)
            cases.append((Q.QReq(uid, prompt, sampling), {'uid': uid, 'role': role,
                          'transition': transition, 'family': row['family'], 'question': row['question'],
                          'prompt_sha256': base.digest(prompt), 'prompt_len': len(prompt),
                          'action_names': [] if name == 'zero' else [name],
                          'closed_prefix': False, 'cap': 1024}))
    row = fixture['rows'][1]
    prompt = row['prompt_ids'] + row['completion_ids'][:2048]
    verify = mapping[('candidate_to_verify', 'target')]
    commit = mapping[('approach_to_commit', 'target')]
    for role, names in (('ordered_ab', [verify, commit]),
                        ('ordered_ba', [commit, verify]),
                        ('preempted_ab', [verify, commit])):
        uid = 'mechanism-qual-v1|' + base.digest([manifest['sha256'], role])[:24]
        extra = Q.steer_extra(table, uid, names[0], len(row['prompt_ids']),
                              prefix_len=2048, restore_presence=True)
        template = {'action_policy_names': names, 'slots': [0, 512], 'horizon': 1024}
        extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
        sampling = Q.card_params(1024, Q.crn(world.infos[row['question']], 0), extra,
                                 presence=0., routed_start=len(prompt) - 1)
        cases.append((Q.QReq(uid, prompt, sampling), {'uid': uid, 'role': role,
                      'transition': 'ordered', 'family': row['family'], 'question': row['question'],
                      'prompt_sha256': base.digest(prompt), 'prompt_len': len(prompt),
                      'action_names': names, 'closed_prefix': False, 'cap': 1024}))
    row = fixture['rows'][0]
    prefix = row['completion_ids'][:128] + [engine.THINK_END_ID]
    prompt = row['prompt_ids'] + prefix
    uid = 'mechanism-qual-v1|' + base.digest([manifest['sha256'], 'closed'])[:24]
    names = [verify, commit]
    extra = Q.steer_extra(table, uid, names[0], len(row['prompt_ids']),
                          prefix_len=len(prefix), restore_presence=True)
    template = {'action_policy_names': names, 'slots': [0, 512], 'horizon': 1024}
    extra['steer']['meta'] = {'routing_control': {**template, 'sha256': worker_digest(template)}}
    cases.append((Q.QReq(uid, prompt, Q.card_params(128,
                  Q.crn(world.infos[row['question']], 0), extra,
                  presence=0., routed_start=len(prompt) - 1)),
                  {'uid': uid, 'role': 'closed', 'transition': 'closed',
                   'family': row['family'], 'question': row['question'],
                   'prompt_sha256': base.digest(prompt), 'prompt_len': len(prompt),
                   'action_names': names, 'closed_prefix': True, 'cap': 128}))
    if len(cases) != 12 or len({meta['uid'] for _, meta in cases}) != 12 or sum(
            len(req.prompt) for req, _ in cases) != manifest['expected_prefill_tokens']:
        raise ValueError('qualification request inventory differs from sealed price')
    return cases


def check_case(meta, outcome, records, engine, preempt_events):
    reasons = []
    routed = np.asarray(outcome.routed) if outcome.routed is not None else None
    if outcome.error or routed is None or routed.shape != (len(outcome.tokens), 40, 8):
        reasons.append('generation error or routed-array shape')
    if len(outcome.tokens) > meta['cap'] or not outcome.tokens:
        reasons.append('empty or over-cap continuation')
    stop = next((i + 1 for i, token in enumerate(outcome.tokens)
                 if token == engine.THINK_END_ID), len(outcome.tokens))
    expected = {name: 0 for name in meta['action_names']}
    segments = {name: [] for name in meta['action_names']}
    if not meta['closed_prefix']:
        for slot, name in zip((0, 512), meta['action_names']):
            end = min(slot + 256, stop)
            if end > slot:
                expected[name] += end - slot
                segments[name].append([slot, end])
    for rank in (0, 1):
        record = records.get(rank, {}).get(meta['uid'])
        if record is None:
            reasons.append(f'missing rank {rank} telemetry')
            continue
        inactive = record.get('inactive_native_checks', {})
        if not inactive or any(v.get('expert_identity_mismatches') or v.get('weight_mismatches')
                               for v in inactive.values()):
            reasons.append(f'inactive routing mismatch rank {rank}')
        if not expected:
            if record.get('cpu_active_rows') != 0:
                reasons.append(f'native request edited rank {rank}')
        else:
            if (record.get('ordered_action_rows') != expected or
                    record.get('ordered_segments') != segments):
                reasons.append(f'pulse order/boundary rank {rank}')
            dose = record.get('ordered_action_dose', {})
            for name, count in expected.items():
                if name not in dose or any(value.get('active_rows') != count
                                           for value in dose[name].values()):
                    reasons.append(f'device dose differs rank {rank}')
        if meta['role'] == 'preempted_ab' and (record.get('preemptions', 0) < 1 or
                                               record.get('recompute_rows', 0) < 1):
            reasons.append(f'preemption/recompute absent rank {rank}')
    if meta['role'] == 'preempted_ab' and not any(e['reset_ok'] for e in preempt_events):
        reasons.append('no successful deliberate reset')
    if meta['role'] in ('ordered_ab', 'ordered_ba', 'preempted_ab') and stop < 768:
        reasons.append('both 256-token pulse windows not evaluable')
    if meta['closed_prefix'] and any(expected.values()):
        reasons.append('closed prefix should have zero action rows')
    return {'uid': meta['uid'], 'role': meta['role'], 'transition': meta['transition'],
            'pass': not reasons, 'reasons': reasons, 'tokens': len(outcome.tokens),
            'reasoning_horizon': stop, 'expected_action_rows': expected,
            'expected_segments': segments, 'preempt_events': preempt_events}, routed


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('1,024-token routed qualification requires GPU Slurm')
    from moe_steer import engine, manifests as M, qualify as Q
    manifest = base.sealed(args.manifest)
    validate_manifest(manifest)
    if args.overlay.resolve() != Path(manifest['overlay']).resolve():
        raise ValueError('qualified worker overlay path differs')
    if engine.code_tree_sha256() != base.REQUIRED_BASE_TREE:
        raise ValueError('base sampler tree differs')
    if args.out.exists():
        raise FileExistsError('qualification output must be a fresh manifest-digest directory')
    args.out.mkdir(parents=True)
    binding = base.sealed(args.manifest)
    Q.atomic_json(args.out / 'BINDING.json', {'qualification_manifest_sha256': binding['sha256'],
                                               'qualification_driver_sha256': base.file_sha(__file__),
                                               'job_id': os.environ['SLURM_JOB_ID']})
    fixture = base.sealed(manifest['source_fixture']['path'])
    dictionary = base.sealed(manifest['source_action_dictionary']['path'])
    table, mapping = build_policy_table(dictionary)
    world = M.load_world()
    cases = cases_for_manifest(manifest, fixture, world, table, mapping)
    deadline = Q.Deadline.from_env()
    deadline.require(950, 'cold load and serial qualification')
    started = time.time()
    driver = None
    checks, raw, arrays = [], [], {}
    try:
        table_path = args.out / 'policy-table.json'
        Q.atomic_json(table_path, table.sealed())
        telemetry = args.out / 'telemetry'
        telemetry.mkdir()
        fingerprint = engine.fingerprint()
        env = engine.engine_env(table_path, telemetry, expect_fingerprint=fingerprint['combined'])
        env['PYTHONPATH'] = os.pathsep.join((str(args.overlay), env['PYTHONPATH']))
        kwargs = engine.engine_kwargs(plugin=True, max_num_seqs=1, return_routed_experts=True)
        kwargs.update(max_num_seqs=1, enforce_eager=True, gpu_memory_utilization=.80,
                      long_prefill_token_threshold=1024,
                      worker_extension_cls='moe_exp.routing_control.ordered_vllm.OrderedWorkerExtension')
        base.prepare_worker_import_path(engine, env, args.overlay)
        model = engine.build_llm(kwargs)
        driver = Q.QDriver(model, plugin=True)
        harness = Q.Harness(driver, telemetry, deadline=deadline, abort_margin=90.)
        for request, meta in cases:
            deadline.require(120, 'serial qualification request ' + meta['role'])
            preempt = PreemptingDriver(driver) if meta['role'] == 'preempted_ab' else None
            harness.driver = preempt or driver
            batch = harness.run([request], cap_in_flight=1)
            harness.flush()
            rank_records, _, _ = harness.telemetry()
            outcome = batch.outcomes.get(meta['uid'])
            if outcome is None:
                raise ValueError('missing request outcome: ' + meta['uid'])
            checked, routed = check_case(meta, outcome, rank_records, engine,
                                         [] if preempt is None else preempt.events)
            checks.append(checked)
            raw.append({**meta, 'tokens': outcome.tokens, 'finish': outcome.finish,
                        'stop_reason': outcome.stop_reason, 'error': outcome.error,
                        'batch_seconds': batch.seconds})
            if routed is not None and routed.shape == (len(outcome.tokens), 40, 8):
                arrays[meta['uid']] = routed
            print(json.dumps({'role': meta['role'], 'tokens': len(outcome.tokens),
                              'pass': checked['pass'], 'seconds': batch.seconds}), flush=True)
        with (args.out / 'routed.npz').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        Q.atomic_json(args.out / 'raw-results.json', raw)
        four = [c for c in checks if c['transition'] in
                ('candidate_to_verify', 'approach_to_commit')]
        short = [c for c in four if c['transition'] == 'candidate_to_verify']
        long = [c for c in four if c['transition'] == 'approach_to_commit']
        order = [c for c in checks if c['role'] in ('ordered_ab', 'ordered_ba')]
        preempt = next(c for c in checks if c['role'] == 'preempted_ab')
        closed = next(c for c in checks if c['role'] == 'closed')
        flags = {'same_prefix_four_arm_pass': len(short) == len(long) == 4 and all(c['pass'] for c in four),
                 'native_isolation_pass': all(c['pass'] for c in four if c['role'].startswith('native')),
                 'ordered_pulse_pass': len(order) == 2 and all(c['pass'] for c in order),
                 'preemption_recompute_pass': preempt['pass'],
                 'closure_pass': closed['pass'],
                 'inherited_h14_force_pass': True}
        body = {'schema': 'routing-mechanism-serial-eager-1024-qualification-v1',
                'qualification_manifest_sha256': manifest['sha256'],
                'qualification_driver_sha256': base.file_sha(__file__),
                'qualified_worker_sha256': manifest['qualified_worker_sha256'],
                'worker_code_digest': manifest['worker_code_digest'],
                'base_tree_sha256': base.REQUIRED_BASE_TREE,
                'engine_profile': PROFILE, 'max_tokens': 1024,
                'pulse_slots': [0, 512], 'pulse_length': 256,
                'source_h14_file_sha256': manifest['source_h14']['file_sha256'],
                'source_ordered_qualification_sha256': manifest['source_ordered_qualification']['sha256'],
                'source_serial_qualification_sha256': manifest['source_serial_qualification']['sha256'],
                'code_files': manifest['code_files'],
                'job_id': os.environ['SLURM_JOB_ID'],
                'same_prefix_four_arm_pass': flags['same_prefix_four_arm_pass'],
                'native_isolation_pass': flags['native_isolation_pass'],
                'ordered_pulse_pass': flags['ordered_pulse_pass'],
                'preemption_recompute_pass': flags['preemption_recompute_pass'],
                'closure_pass': flags['closure_pass'],
                'inherited_h14_force_pass': flags['inherited_h14_force_pass'],
                'checks': checks, 'requests': len(raw),
                'actual_decoded_tokens': sum(len(r['tokens']) for r in raw),
                'actual_prefill_tokens': sum(r['prompt_len'] for r in raw),
                'routed_file_sha256': base.file_sha(args.out / 'routed.npz'),
                'raw_file_sha256': base.file_sha(args.out / 'raw-results.json'),
                'elapsed_driver_seconds': time.time() - started,
                'interpretation': 'engineering check on exact serial/eager profile; native repeats need not be identical; no engine equivalence, semantic steering, or accuracy claim',
                'pass': all(flags.values()) and len(raw) == 12}
        value = {**body, 'sha256': base.digest(body)}
        Q.atomic_json(args.out / 'QUALIFICATION.json', value)
        if RESULT.exists():
            raise FileExistsError('public qualification result already exists')
        Q.atomic_json(RESULT, value)
        Q.finish_child(0 if value['pass'] else 3, driver)
    except Exception as error:
        failure = {'schema': 'routing-mechanism-serial-eager-1024-qualification-failure-v1',
                   'qualification_manifest_sha256': manifest['sha256'],
                   'job_id': os.environ['SLURM_JOB_ID'],
                   'error': f'{type(error).__name__}: {error}',
                   'traceback_tail': traceback.format_exc()[-3000:],
                   'completed_cases': len(raw), 'checks': checks}
        Q.atomic_json(args.out / 'FAILURE.json', {**failure, 'sha256': base.digest(failure)})
        Q.finish_child(3, driver)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    manifest = base.sealed(args.manifest)
    validate_manifest(manifest)
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_CPU_PREFLIGHT', 'sha256': manifest['sha256'],
                          'requests': manifest['expected_requests'],
                          'maximum_decode_tokens': manifest['maximum_decode_tokens']}))
        return
    if args.out is None or args.overlay is None:
        raise SystemExit('--out and --overlay required for GPU qualification')
    run(args)


if __name__ == '__main__':
    main()
