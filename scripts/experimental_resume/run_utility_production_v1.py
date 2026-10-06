"""Run one immutable utility shard using the unchanged qualified v2 backend."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time

from qualify_utility_pair_v2 import bootstrap

if __name__ in ('__main__', '__mp_main__'):
    bootstrap()

import utility_production_v1 as P


def accounting_job_id():
    if os.environ.get('SLURM_ARRAY_JOB_ID'):
        return os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
    return os.environ['SLURM_JOB_ID']


def shard_directory(output, index):
    return Path(output) / f'shard-{index:03d}'


def pending_rows(manifest, shard, directory, binding, recovery=None):
    by_uid = {r['assignment']['uid']: r for r in manifest['rows']}
    allowed = set(shard['assigned_uids'])
    if recovery is not None:
        P.require(recovery['schema'] == 'utility-production-recovery-v1' and
                  recovery['manifest_sha256'] == manifest['sha256'] and recovery['wave'] == 1,
                  'unregistered infrastructure recovery')
        allowed = set(recovery['shards'][str(shard['index'])]['assigned_uids'])
        P.require(allowed <= set(shard['assigned_uids']), 'recovery contains foreign assignment')
    pending, committed = [], []
    for uid in shard['assigned_uids']:
        row = by_uid[uid]
        key = P.U.digest(uid)
        P.save(directory / 'assignments' / (key + '.json'), {
            'schema': 'utility-production-assignment-v1', 'binding_sha256': binding['sha256'],
            'assignment': row['assignment']})
        receipt_path = directory / 'receipts' / (key + '.json')
        attempts = [P.U.sealed(p) for p in sorted((directory / 'attempts').glob(key + '-*.json'))]
        if receipt_path.exists():
            receipt = P.U.sealed(receipt_path)
            P.require(any(a['sha256'] == receipt['attempt_sha256'] and
                          a['assignment'] == row['assignment'] and a['binding_sha256'] == binding['sha256']
                          for a in attempts), 'completed receipt lacks its matching attempt journal')
            P.validate_receipt(receipt, row['assignment'], binding['sha256'],
                Path(receipt['routed_path']) if receipt['status'] == 'COMMITTED_GENERATION' else None)
            committed.append(receipt)
            continue
        if uid not in allowed:
            continue
        if recovery is None:
            P.require(not attempts, 'uncommitted prior attempt requires sealed infrastructure recovery')
        else:
            expected = recovery['prior_attempt_sha256s'][uid]
            P.require([a['sha256'] for a in attempts] == expected and len(attempts) <= 1,
                      'recovery attempt journal changed or retry allowance exhausted')
        pending.append((row, attempts))
    return pending, committed


def run(args, backend_factory=None):
    import numpy as np
    if backend_factory is None:
        from utility_pair_backend_v2 import FourGPUBackend
        backend_factory = FourGPUBackend
    manifest = P.U.sealed(args.manifest)
    P.validate_manifest(manifest)
    config = P.U.sealed(args.config)
    P.validate_config(config)
    price = P.U.sealed(args.manifest.parent / 'PRICE.json')
    P.require(config['sha256'] == manifest['config_sha256'] == price['config_sha256'] and
              price['sha256'] == manifest['price_sha256'] and
              price['status'] == 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION' and
              price['shards'] == manifest['shards'], 'production run lacks its exact complete-stage price')
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'),
              'production requires a GPU Slurm step')
    P.require(0 <= args.shard_index < len(manifest['shards']), 'invalid exact shard index')
    shard = manifest['shards'][args.shard_index]
    recovery = P.U.sealed(args.recovery_manifest) if args.recovery_manifest else None
    directory = shard_directory(args.out, args.shard_index)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in ('assignments', 'attempts', 'receipts', 'routes', 'allocations'):
            (directory / folder).mkdir(exist_ok=True)
        binding = P.save(directory / 'BINDING.json', {'schema': 'utility-production-shard-binding-v1',
            'manifest_sha256': manifest['sha256'], 'policy_sha256': manifest['policy_sha256'],
            'shard_index': args.shard_index, 'assigned_uids': shard['assigned_uids']})
        pending, receipts = pending_rows(manifest, shard, directory, binding, recovery)
        P.require(not pending or recovery is not None or not any((directory / 'allocations').glob('*/ALLOCATION.json')),
                  'a prior allocation requires sealed infrastructure recovery even for assignments not yet attempted')
        account_id = accounting_job_id()
        allocation = directory / 'allocations' / account_id
        P.save(allocation / 'ALLOCATION.json', {'schema': 'utility-production-allocation-v1',
            'manifest_sha256': manifest['sha256'], 'binding_sha256': binding['sha256'],
            'job_id': os.environ['SLURM_JOB_ID'], 'accounting_job_id': account_id,
            'recovery_sha256': recovery['sha256'] if recovery else None,
            'allocated_gpus': 4, 'assigned_uids_this_attempt': [row['assignment']['uid'] for row, _ in pending],
            'deadline_epoch': args.deadline_epoch})
        failure = None
        try:
            if pending:
                P.require(time.time() < args.deadline_epoch - 60, 'insufficient allocation time before model load')
                with backend_factory(manifest['policy']['selections'], allocation / 'pair',
                                     deadline_epoch=args.deadline_epoch) as backend:
                    for row, old_attempts in pending:
                        if time.time() >= args.deadline_epoch - 60:
                            failure = 'Allocation deadline before next assignment; remaining cells stay missing.'
                            break
                        assignment = row['assignment']
                        key = P.U.digest(assignment['uid'])
                        attempt = P.save(directory / 'attempts' / f'{key}-{len(old_attempts):03d}.json', {
                            'schema': 'utility-production-attempt-v1', 'binding_sha256': binding['sha256'],
                            'assignment': assignment, 'attempt_index': len(old_attempts),
                            'prior_attempt_sha256s': [a['sha256'] for a in old_attempts],
                            'allocation_path': str(allocation), 'accounting_job_id': account_id,
                            'job_id': os.environ['SLURM_JOB_ID'],
                            'recovery_sha256': recovery['sha256'] if recovery else None})
                        try:
                            result = backend.generate_one(assignment, row['original_prompt_ids'], row['problem'])
                            routes = result.pop('routed')
                            route_path = directory / 'routes' / (key + '-' + attempt['sha256'][:16] + '.npz')
                            temporary = route_path.with_name('.' + route_path.name + '.' + str(os.getpid()))
                            with temporary.open('xb') as stream:
                                np.savez_compressed(stream, routed=routes)
                                stream.flush()
                                os.fsync(stream.fileno())
                            try:
                                os.link(temporary, route_path)
                            finally:
                                temporary.unlink()
                            body = {'status': 'COMMITTED_GENERATION', 'result': result,
                                    'routed_path': str(route_path), 'routed_array_sha256': P.U.file_sha(route_path),
                                    'error': None}
                        except Exception as error:
                            body = {'status': 'GENERATION_ERROR', 'result': getattr(error, 'utility_partial', None),
                                    'routed_path': None, 'routed_array_sha256': None, 'error': repr(error)}
                            failure = 'Committed generator/reader error; do not retry this outcome.'
                        receipt = P.U.seal({'schema': 'utility-production-generation-receipt-v1',
                            'binding_sha256': binding['sha256'], 'manifest_sha256': manifest['sha256'],
                            'attempt_sha256': attempt['sha256'], 'assignment': assignment, **body})
                        P.validate_receipt(receipt, assignment, binding['sha256'])
                        receipt = P.save(directory / 'receipts' / (key + '.json'),
                                         {k: v for k, v in receipt.items() if k != 'sha256'})
                        receipts.append(receipt)
                        if failure:
                            break
        except BaseException as error:
            failure = 'Allocation/backend failure: ' + repr(error)
        observed = {r['assignment']['uid']: r for r in receipts}
        P.require(len(observed) == len(receipts), 'duplicate committed UID within shard')
        missing = [uid for uid in shard['assigned_uids'] if uid not in observed]
        result = P.save(directory / ('SUMMARY-' + account_id + '.json'), {
            'schema': 'utility-production-shard-summary-v1', 'manifest_sha256': manifest['sha256'],
            'binding_sha256': binding['sha256'], 'accounting_job_id': account_id,
            'assigned': len(shard['assigned_uids']), 'receipts': len(receipts), 'missing_uids': missing,
            'generation_errors': sum(r['status'] == 'GENERATION_ERROR' for r in receipts),
            'receipt_sha256s': [r['sha256'] for r in receipts], 'failure': failure,
            'status': 'COMPLETE_SHARD_WITH_COMMITTED_OUTCOMES' if not missing else 'INCOMPLETE_SHARD',
            'automatic_retry_committed_errors': False})
        print(result['status'], result['receipts'], '/', result['assigned'], flush=True)
        if missing:
            raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'out', 'config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--recovery-manifest', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
