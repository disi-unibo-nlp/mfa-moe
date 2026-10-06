"""Prepare/run an outcome-blind 16k utility price pilot; never submit jobs.

Enrollment is the first explicitly requested N families of the frozen 96-family
utility order, with all four arm/seed cells intact. Production is intentionally
outside this interface until measured complete-stage pricing is reviewed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from qualify_utility_pair_v2 import bootstrap

if __name__ in ('__main__', '__mp_main__'):
    bootstrap()


def code_files():
    from utility_source_contract_v2 import code_files as sources
    return sources()


def prepare(args):
    import utility_scout_v1 as utility
    import rate_overnight_semantics_v2 as storage
    from utility_controller_v2 import hard_side_work_bound
    plan, policy, qual = map(utility.sealed, (args.plan, args.policy, args.qualification))
    utility.validate_plan(plan)
    storage.require(policy['schema'] == 'routing-frozen-utility-policy-v2' and
        policy['status'] == 'FROZEN_LOCAL_ACTIONS_CONTROLLER_AND_ENGINE_REQUIRED' and policy['selections'] and
        qual['schema'] == 'utility-pair-engine-qualification-v2' and qual['status'] == 'PASS_ENGINEERING' and
        type(args.pilot_families) is int and 1 <= args.pilot_families <= 96,
        'exact selected policy, GPU qualification and explicit family count required')
    binding = utility.sealed(args.qualification.parent / 'BINDING.json')
    storage.require(qual['binding_sha256'] == binding['sha256'] and
        all(utility.file_sha(Path(p)) == sha for p, sha in binding['code_files'].items()),
        'qualified utility engine code changed')
    prompts, questions = utility.sealed(utility.PROMPTS), utility.sealed(utility.QUESTIONS)
    storage.require(prompts['sha256'] == plan['sources']['prompt_table_sha256'] and
        utility.file_sha(utility.PROMPTS) == plan['sources']['prompt_table_file_sha256'], 'original prompts changed')
    families = plan['family_order'][:args.pilot_families]
    rows = []
    for assignment in plan['assignments']:
        if assignment['family'] not in families:
            continue
        prompt = prompts['questions'][assignment['question']]['prompt_token_ids']
        storage.require(utility.digest(prompt) == assignment['prompt_token_ids_sha256'], 'pilot prompt hash differs')
        # Gold and reference answers are never materialized into the generation
        # manifest, original-prompt backend, online observer, or side model.
        rows.append({'assignment': assignment, 'original_prompt_ids': prompt,
                     'problem': questions['questions'][assignment['question']]['problem']})
    manifest = storage.save(args.out, {'schema': 'utility-price-pilot-manifest-v2',
        'plan_sha256': plan['sha256'], 'policy_sha256': policy['sha256'], 'qualification_sha256': qual['sha256'],
        'qualification_binding_sha256': binding['sha256'], 'policy': policy,
        'family_order': families, 'rows': rows, 'expected_assignments': len(rows),
        'all_utility_assignment_count': 384, 'utility_family_count': 96,
        'family_selection': 'First N frozen utility families, chosen before pilot outcomes; all two-arm/two-seed cells retained.',
        'original_question_table_sha256': questions['sha256'], 'maximum_tokens_per_assignment': 16384,
        'episode_rule': 'One first two-reader-accepted episode; no re-entry; no side-query count cap.',
        'allocated_gpus_per_job': 4, 'code_files': code_files(),
        'side_work_hard_bound': hard_side_work_bound(2 * len(families)),
        'status': 'PILOT_ONLY_PRODUCTION_REQUIRES_REVIEWED_COMPLETE_STAGE_PROPOSAL',
        'reuse_rule': 'Pilot assignments retain frozen utility UIDs and may enter final ITT only when policy, sampler, source, controller, and execution profile remain exactly unchanged.'})
    print('PREPARED', manifest['sha256'], len(rows), '16k assigned cells', flush=True)


def run(args):
    import numpy as np
    import utility_scout_v1 as utility
    import rate_overnight_semantics_v2 as storage
    from utility_pair_backend_v2 import FourGPUBackend
    manifest = utility.sealed(args.manifest)
    storage.require(manifest['schema'] == 'utility-price-pilot-manifest-v2' and
        manifest['code_files'] == code_files() and
        manifest['maximum_tokens_per_assignment'] == 16384, 'pilot manifest or frozen code changed')
    storage.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'), 'pilot requires Slurm step')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('assignments', 'attempts', 'receipts', 'routes', 'allocations'):
        (args.out / name).mkdir(exist_ok=True)
    binding = storage.save(args.out / 'BINDING.json', {'schema': 'utility-price-pilot-binding-v2',
        'manifest_sha256': manifest['sha256'], 'policy_sha256': manifest['policy_sha256'],
        'assigned_uids': [r['assignment']['uid'] for r in manifest['rows']]}, existing_ok=True)
    receipts, pending = [], []
    for row in manifest['rows']:
        uid = row['assignment']['uid']
        name = utility.digest(uid)
        storage.save(args.out / 'assignments' / (name + '.json'), {'schema': 'utility-pilot-assignment-v2',
            'binding_sha256': binding['sha256'], 'assignment': row['assignment']}, existing_ok=True)
        path = args.out / 'receipts' / (name + '.json')
        if path.exists():
            receipt = utility.sealed(path)
            storage.require(receipt['binding_sha256'] == binding['sha256'] and receipt['assignment'] == row['assignment'],
                            'completed pilot UID rebound')
            receipts.append(receipt)
        else:
            pending.append(row)
    failure = None
    if pending:
        allocation = args.out / 'allocations' / (os.environ['SLURM_JOB_ID'] + '-' + str(time.time_ns()))
        with FourGPUBackend(manifest['policy']['selections'], allocation, deadline_epoch=args.deadline_epoch) as backend:
            for row in pending:
                if time.time() >= args.deadline_epoch - 60:
                    failure = 'Allocation deadline; remaining assignments not attempted.'
                    break
                uid = row['assignment']['uid']
                name = utility.digest(uid)
                old_attempts = [utility.sealed(p) for p in sorted((args.out / 'attempts').glob(name + '-*.json'))]
                if old_attempts and not args.resume_uncommitted:
                    raise ValueError('uncommitted pilot attempt requires explicit --resume-uncommitted')
                attempt = storage.save(args.out / 'attempts' / f'{name}-{len(old_attempts):03d}.json', {
                    'schema': 'utility-pilot-attempt-v2', 'binding_sha256': binding['sha256'],
                    'assignment': row['assignment'], 'attempt_index': len(old_attempts),
                    'prior_attempt_sha256s': [r['sha256'] for r in old_attempts],
                    'allocation_path': str(allocation), 'job_id': os.environ['SLURM_JOB_ID']})
                try:
                    result = backend.generate_one(row['assignment'], row['original_prompt_ids'], row['problem'])
                    route_path = args.out / 'routes' / (name + '.npz')
                    with route_path.open('xb') as stream:
                        np.savez_compressed(stream, routed=result.pop('routed'))
                    receipt = {'status': 'COMMITTED_GENERATION', 'result': result,
                               'routed_array_sha256': utility.file_sha(route_path), 'error': None}
                except Exception as error:
                    receipt = {'status': 'GENERATION_ERROR', 'result': getattr(error, 'utility_partial', None),
                               'routed_array_sha256': None, 'error': repr(error)}
                    failure = 'Generator/side failure; remaining assignments retain missing ITT status.'
                receipt = storage.save(args.out / 'receipts' / (name + '.json'), {
                    'schema': 'utility-price-pilot-receipt-v2', 'binding_sha256': binding['sha256'],
                    'attempt_sha256': attempt['sha256'], 'assignment': row['assignment'], **receipt})
                receipts.append(receipt)
                if failure:
                    break
    assigned = {row['assignment']['uid']: row['assignment'] for row in manifest['rows']}
    observed = {r['assignment']['uid']: r for r in receipts}
    complete = set(observed) == set(assigned)
    body = {'schema': 'utility-price-pilot-summary-v2', 'binding_sha256': binding['sha256'],
        'manifest_sha256': manifest['sha256'], 'assigned': len(assigned), 'receipts': len(receipts),
        'missing': len(assigned) - len(observed), 'generation_errors': sum(r['error'] is not None for r in receipts),
        'receipt_sha256s': [r['sha256'] for r in receipts],
        'all_assigned_itt': [{**assignment, 'receipt_status': observed[uid]['status'] if uid in observed else 'MISSING'}
                            for uid, assignment in assigned.items()],
        'status': 'COMPLETE_UNGRADED_PRICE_PILOT' if complete else 'INCOMPLETE_PRICE_PILOT',
        'failure': failure, 'production_status': 'HOLD_MEASURED_COMPLETE_STAGE_PROPOSAL',
        'cost_rule': 'Use all four GPUs for every allocation wall second, including failed attempts and waiting. Slurm sacct overhead supplements PAIR_COST.json; side costs are descriptive components, not added twice.'}
    summary = storage.save(args.out / ('SUMMARY-' + os.environ['SLURM_JOB_ID'] + '.json'), body)
    print(summary['status'], summary['receipts'], '/', summary['assigned'], flush=True)
    if not complete:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('plan', 'policy', 'qualification', 'out'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--pilot-families', type=int, required=True)
    exe = sub.add_parser('run')
    for name in ('manifest', 'out'):
        exe.add_argument('--' + name, type=Path, required=True)
    exe.add_argument('--deadline-epoch', type=float, required=True)
    exe.add_argument('--resume-uncommitted', action='store_true')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
