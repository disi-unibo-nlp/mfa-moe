"""Finite CPU preparation, GPU bootstrap qualification and receipt inspection.

No scientific qualification is fabricated to enter this engineering path.
The two approved GPU allocations are submitted separately, after observed
passing prerequisites. All writes are additive and all code/source hashes bind.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import time
import numpy as np

from moe_exp.routing_control.study_v1 import require, digest, sealed, save as historical_save, file_sha
from moe_exp.routing_control.study_v2 import Action, CAP, CHECKS, seal, verify
from moe_exp.routing_control.families_v2 import family_boundary, require_discovery
from moe_exp.routing_control.capture_v2 import begin_attempt, commit_attempt
from moe_exp.routing_control.policy_stages_v2 import PrefixPolicyController
from moe_exp.routing_control.generation_v2 import execute_engineering_request
from routing_local_backend_v2 import LocalBackend, source_binding
from routing_local_semantics_v2 import proposals

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / 'report/experimental-resume-v1'
FULL_FRAME = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/TRANSITION_V22_FULL_PREFIX_START_FRAME.json')


def save(path, body):
    # Immutable readback compares Python objects as well as their JSON digest.
    return historical_save(path, json.loads(json.dumps(body, allow_nan=False)))


def compute(gpu=False):
    import getpass
    require(getpass.getuser() == 'lmolfett' and os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            not socket.gethostname().startswith('login'), 'requires allocated LEONARDO step')
    require(os.environ.get('SLURM_CLUSTER_NAME') == 'leonardo', 'wrong Slurm cluster')
    require(os.environ.get('SLURM_JOB_PARTITION') == ('boost_usr_prod' if gpu else 'lrd_all_viz'), 'wrong resource partition')


def validate_sources(manifest):
    verify(manifest)
    require(manifest['code_files'] == source_binding(), 'engineering code/source hash mismatch')
    require(all(file_sha(p) == sha for p, sha in manifest['source_files'].items()), 'frozen source input hash mismatch')


def actions():
    return [Action('positive', 'candidate_to_verify', 28, (9,), 'bias', 1.),
            Action('negative', 'candidate_to_verify', 28, (9,), 'bias', -1.),
            Action('force', 'candidate_to_verify', 28, (9, 189), 'force', 0.),
            Action('reweight', 'candidate_to_verify', 28, (9,), 'reweight', 1.),
            Action('matched', 'candidate_to_verify', 28, (13,), 'bias', 1.),
            Action('second', 'approach_to_commit', 24, (159,), 'bias', 1.)]


def prepare(output):
    compute()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    boundary = family_boundary(sealed(DOC / 'family-freeze.json'), sealed(REPO / 'report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json'))
    source_path = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v4.json'
    source, frame = sealed(source_path), sealed(FULL_FRAME)
    native = sorted(source['rows'], key=lambda r: digest(['engineering-fixture-v2', r['uid']]))[0]
    require_discovery([native['family'], *[r['family'] for r in frame['records']]], boundary)
    families = sorted({r['family'] for r in frame['records']}, key=lambda f: digest(['six-decision-split-v2', f]))
    require(len(families) >= 12, 'too few discovery families for a frozen audit split')
    held = set(families[::4])
    # Round-robin by family, with a fixed two-prefix ceiling per family. Source
    # sampling precedes all new ratings and contains no gold/utility outcomes.
    pools = {f: sorted([r for r in frame['records'] if r['family'] == f],
                       key=lambda r: digest(['six-decision-prefix-v2', r['uid']])) for f in families}
    selected = [pools[f][i] for i in range(2) for f in families if len(pools[f]) > i][:64]
    records = []
    for row in selected:
        records.append({'uid': row['uid'], 'family': row['family'], 'split': 'held_out' if row['family'] in held else 'apparent',
            'prefix_tokens': row['prefix_tokens'], 'reader_input': row['reader_input'],
            'source_prefix_binding': row['analysis_meta'], 'proposals': proposals(row['reader_input']['emitted_prefix'])})
    audit = save(output / 'AUDIT_FRAME.json', {'schema': 'routing-six-decision-audit-frame-v2', 'records': records,
        'family_boundary': boundary, 'source_frame_sha256': frame['sha256'], 'maximum_prefixes': 64,
        'readers': [0, 1], 'reader_cap': 1024, 'splits_frozen_before_ratings': True,
        'apparent_families': [f for f in families if f not in held], 'held_out_families': sorted(held),
        'reader_input_allowlist': ['problem', 'emitted_prefix', 'triggering_sentence'],
        'interpretation': 'Frozen limited discovery audit; missing decisions/support remain explicit.'})
    available = {a.action_id: a for a in actions()}
    cases = [('native-before', [], 384, 'length', False), *[(name, [name], 384, 'length', False)
        for name in ('positive', 'negative', 'force', 'reweight', 'matched')],
        ('delayed-no-fire', ['positive'], 96, 'stop', False),
        ('composition', ['force', 'second'], 704, 'length', False),
        ('closure', ['force'], 128, 'stop', False),
        ('chunked-prefill', [], 32, 'length', True), ('native-after', [], 384, 'length', False),
        ('recovery', ['force'], 96, 'length', False), ('16k-overhead', ['positive'], CAP, 'length', False)]
    assignments = []
    for name, names, cap, finish, long_prefix in cases:
        prefix = native['prefix_ids']
        if long_prefix:
            prefix = (prefix * (9000 // len(prefix) + 1))[:9000]
        chosen = [asdict(available[n]) for n in names]
        if name == 'delayed-no-fire':
            chosen[0]['delay'] = 64
        body = {'case': name, 'arm': 'fixture' if names else 'native', 'actions': chosen,
            'problem': native['question'], 'family': native['family'], 'seed': 0,
            'original_prompt_ids': native['prompt_ids'], 'original_prompt_ids_sha256': digest(native['prompt_ids']),
            'prefix_ids': prefix, 'prefix_ids_sha256': digest(prefix), 'continuation_cap': cap,
            'engineering_only': True, 'expected_finish_reason': finish,
            'source_uid': native['uid'], 'early_endpoint_window': 1024}
        assignments.append({**body, 'uid': digest(['routing-local-engineering-v2', body])})
    sources = {str(p): file_sha(p) for p in (source_path, FULL_FRAME, DOC / 'family-freeze.json',
        REPO / 'report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json')}
    manifest = save(output / 'MANIFEST.json', {'schema': 'routing-engineering-manifest-v2', 'assignments': assignments,
        'code_files': source_binding(), 'source_files': sources, 'audit_frame_sha256': audit['sha256'],
        'pricing_family': native, 'family_boundary': boundary,
        'resource_ceiling': {'gpu_jobs': 2, 'gpus_per_job': 2, 'seconds_per_gpu_job': 14400, 'GPU_hours': 16,
                             'CPU_jobs': 4, 'cpus_per_cpu_job': 2, 'seconds_per_cpu_job': 3600, 'CPU_core_hours': 8},
        'maximum_assignment_attempts': 2, 'scientific_dispatch_authorized': False,
        'bounded_16k_pricing_cells': [{'arm': a, 'seed': s, 'continuation_cap': CAP} for a in ('native', 'incumbent') for s in (0, 1)]})
    save(output / 'AUTHORIZATION.json', {'schema': 'routing-engineering-authorization-v2',
        'status': 'AUTHORIZED_ENGINEERING_ONLY', 'manifest_sha256': manifest['sha256'],
        'resource_ceiling': manifest['resource_ceiling'],
        'authority': 'User supplied implementation plan authorizes <=16 GPU-hours and <=8 CPU-core-hours; two sequential two-GPU four-hour allocations.'})
    save(output / 'PREPARATION.json', {'schema': 'routing-engineering-preparation-v2', 'status': 'PASS',
        'manifest_sha256': manifest['sha256'], 'audit_frame_sha256': audit['sha256'], 'code_files': source_binding(),
        'records': len(records), 'families': len({r['family'] for r in records}), 'job_id': os.environ['SLURM_JOB_ID']})
    print(json.dumps({'status': 'PASS', 'manifest': str(output / 'MANIFEST.json'), 'audit_prefixes': len(records)}), flush=True)


def fixture_recognizer(origin, name):
    def recognize(prefix):
        position = len(prefix.emitted_token_ids) - origin
        return {'prefix_sha256': digest(asdict(prefix)), 'boundary': position in (32, 400),
            'source_state': 'Analyze', 'candidate_status': 'complete_unchecked', 'approach_identified': position == 400,
            'substantive_acts': ['candidate_to_verify'] if name == 'delayed-no-fire' and position >= 48 else [],
            'controller_tokens': 0, 'controller_input_tokens': len(prefix.emitted_token_ids), 'controller_seconds': 0.}
    return recognize


def qualify(manifest_path, output, deadline):
    compute(True)
    manifest, authorization = sealed(manifest_path), sealed(manifest_path.with_name('AUTHORIZATION.json'))
    validate_sources(manifest)
    prerequisite = sealed(manifest_path.with_name('CPU_CHECKS.json'))
    require(prerequisite['status'] == 'PASS' and prerequisite['code_files'] == source_binding(), 'passing exact CPU prerequisite required')
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'QUALIFICATION.json').exists(), 'qualification already committed; no duplicate job work')
    available = {a.action_id: a for a in actions()}
    backend_output = output / ('backend-' + str(os.getpid()))
    backend = LocalBackend([a.worker_action() for a in actions()], backend_output, deadline_epoch=deadline)
    results, snapshots, failures = [], [], []
    try:
        safe = backend.tokenizer.encode(' x', add_special_tokens=False)[0]
        for assignment in manifest['assignments']:
            name = assignment['case']
            attempts = output / assignment['uid']
            if (attempts / 'SUCCESS.json').exists():
                accepted = sealed(attempts / 'SUCCESS.json')
                result = sealed(Path(accepted['result_path']))
                receipt = result['receipt']
                results.append({'case': name, 'receipt_path': accepted['result_path'], 'receipt_sha256': receipt['sha256'],
                                'capture_sha256': receipt['capture_binding']['sha256'], 'status': 'PASS', 'attempt': accepted['attempt']})
                snapshots.extend(sealed(p) for p in Path(accepted['result_path']).parent.glob('PAUSE_*.json'))
                continue
            used = len(list(attempts.glob('attempt-*/ASSIGNMENT.json')))
            for attempt_number in range(used, 2):
                attempt = begin_attempt(attempts, assignment, manifest['sha256'])
                ctl = None
                if assignment['actions']:
                    ctl = PrefixPolicyController([Action(**{**a, 'experts': tuple(a['experts'])}) for a in assignment['actions']],
                        fixture_recognizer(len(assignment['prefix_ids']), name), seal({'status': 'PASS', 'prefix_only': True,
                            'recognizes_substantive_acts': True, 'supported_transitions': ['candidate_to_verify', 'approach_to_commit'],
                            'scope': 'SCRIPTED_ENGINEERING_ONLY_NO_SEMANTIC_QUALIFICATION'}), uid=assignment['uid'], seed=0,
                        origin_position=len(assignment['prefix_ids']))
                sequence = [safe] * assignment['continuation_cap']
                if name == 'closure':
                    sequence[64] = 248069
                if assignment['expected_finish_reason'] == 'stop':
                    sequence[-1] = 248046
                # Bind fixture tokens in a separate request manifest before any
                # engine input; the original enrolled assignment remains intact.
                request = {**assignment, 'engineering_token_sequence': sequence} if len(sequence) <= 1024 else {
                    **assignment, 'engineering_repeat': {'token_id': safe, 'cap': len(sequence), 'overrides': {}}}
                request_manifest = seal({**{k: v for k, v in manifest.items() if k != 'sha256'}, 'assignments': [request]})
                request_auth = seal({**{k: v for k, v in authorization.items() if k != 'sha256'}, 'manifest_sha256': request_manifest['sha256']})
                save(attempt / 'REQUEST_MANIFEST.json', {k: v for k, v in request_manifest.items() if k != 'sha256'})
                pause_positions = (0, 40, 310) if name in ('force', 'native-before') else ()
                def observer(engine, controller, offset, policy):
                    if offset in pause_positions:
                        before = engine.pause()
                        saved = save(attempt / f'PAUSE_{offset}.json', {k: v for k, v in before.items() if k != 'sha256'})
                        engine.resume(saved)
                        snapshots.append(saved)
                    if name == 'recovery' and attempt_number == 0 and offset == 48:
                        raise RuntimeError('CONTROLLED_WHOLE_ASSIGNMENT_INTERRUPTION')
                from utility_controller_v2 import NativeDecodeStream
                try:
                    receipt, ids, weights = execute_engineering_request(request, backend, NativeDecodeStream(backend.tokenizer),
                        manifest=request_manifest, authorization=request_auth, controller=ctl, prefix_ids=assignment['prefix_ids'],
                        capture_dir=attempt / 'capture', observer=observer)
                    require(receipt['emitted_token_ids'] == sequence and receipt['finish_reason'] == assignment['expected_finish_reason'],
                            'deterministic sequence or natural/cap finish differs')
                    binding = receipt['capture_binding']
                    dose = np.load(attempt / 'capture/dose.npy', mmap_mode='r')[:binding['rows']]
                    if ctl:
                        decisions = receipt['controller']['decisions']
                        if name == 'delayed-no-fire':
                            require(all(d['status'] == 'NO_FIRE_DESIRED_ACT_OCCURRED' for d in decisions.values()) and not np.any(dose), 'delayed no-fire differs')
                        else:
                            require(all(d['status'] == 'FIRED' for d in decisions.values()), 'fixture pulse did not fire')
                            for action in ctl.actions:
                                d = decisions[action.action_id]
                                left = d['position'] - len(assignment['prefix_ids'])
                                right = min(left + 256, len(ids), 65 if name == 'closure' else len(ids))
                                require(np.any(dose[left:right, action.layer, 1] > 0), 'zero achieved action dose')
                                if action.operator == 'force':
                                    require(all(np.all(np.any(ids[left:right, action.layer] == expert, axis=1)) for expert in action.experts), 'forced expert absent')
                    commit_attempt(attempts, attempt, receipt)
                    results.append({'case': name, 'receipt_path': str(attempt / 'SUCCESS.json'), 'receipt_sha256': receipt['sha256'],
                                    'capture_sha256': binding['sha256'], 'status': 'PASS', 'attempt': attempt_number})
                    break
                except BaseException as error:
                    save(attempt / 'FAILURE.json', {'schema': 'routing-engineering-attempt-failure-v2', 'error': repr(error),
                        'partial': getattr(error, 'routing_partial', None), 'continued_execution': False,
                        'backend_metrics': dict(backend.metrics)})
                    if name == 'recovery' and attempt_number == 0 and 'CONTROLLED_WHOLE_ASSIGNMENT_INTERRUPTION' in str(error):
                        continue
                    failures.append({'case': name, 'error': repr(error)})
                    raise
        # Cross-run variability is separate from exact inactive-path fidelity.
        before, after = [r for r in results if r['case'] in ('native-before', 'native-after')]
        arrays = [np.load(Path(r['receipt_path']).parent / 'capture/ids.npy', mmap_mode='r')[0, 1, :384] for r in (before, after)]
        variation = {'native_ID_cell_mismatches': int(np.count_nonzero(arrays[0] != arrays[1])),
                     'scope': 'Different isolated executions; no cross-run exactness requirement.'}
    except BaseException as error:
        failures.append({'status': 'EXPLICIT_QUALIFICATION_FAILURE', 'error': repr(error)})
        variation = None
    finally:
        backend.close()
        complete = len(results) == len(manifest['assignments']) and not failures
        checks = {k: bool(complete) for k in CHECKS}
        # Death recovery is verified by the durable CPU attempt-journal test;
        # the GPU case observes one interrupted attempt and one full rerun.
        body = {'schema': 'routing-local-backend-qualification-v2', 'status': 'PASS' if complete else 'FAIL',
            'manifest_sha256': manifest['sha256'], 'code_files': source_binding(), 'runtime': backend.runtime,
            'checks': checks, 'synchronous_one_token_steps': complete, 'preserves_live_sampler': complete,
            'cross_run_variability_audited': complete, 'cross_run_variability': variation,
            'cases': results, 'failures': failures, 'pause_receipt_sha256': [s['sha256'] for s in snapshots],
            'recovery_scope': 'One whole-assignment retry from original prefix; retained partial attempt; no seed restart described as continuation.',
            'cost': sealed(backend_output / 'COST.json'),
            'all_backend_costs': [sealed(p) for p in output.glob('backend-*/COST.json')],
            'job_id': os.environ['SLURM_JOB_ID'],
            'semantic_qualification': 'SEPARATE_SIX_DECISION_AUDIT_REQUIRED',
            '16k_overhead': next((r for r in results if r['case'] == '16k-overhead'), None)}
        save(output / 'QUALIFICATION.json', body)
        print(json.dumps({'status': body['status'], 'cases': len(results), 'failures': failures}), flush=True)
    require(complete, 'qualification failed; do not submit dependent GPU audit')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['prepare', 'qualify'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--deadline-epoch', type=float)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    else:
        try:
            qualify(args.manifest, args.output, args.deadline_epoch)
        except BaseException as error:
            if not (args.output / 'QUALIFICATION.json').exists():
                args.output.mkdir(parents=True, exist_ok=True)
                save(args.output / 'QUALIFICATION.json', {'schema': 'routing-local-backend-qualification-v2',
                    'status': 'FAIL', 'phase': 'INITIALIZATION_OR_PREREQUISITE', 'error': repr(error),
                    'code_files': source_binding(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                    'manifest_sha256': sealed(args.manifest)['sha256']})
            raise


if __name__ == '__main__':
    main()
