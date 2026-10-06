"""Finite one-shot post-pilot pricing, production and infrastructure recovery.

No submission occurs on import. Every live submission uses the reviewed shared
dispatcher with sanitized parent placement and durable verified receipts.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess

import utility_production_v1 as P
import dispatch_overnight_readers_v1 as shared


def duration_seconds(text):
    days, clock = (text.split('-', 1) if '-' in text else ('0', text))
    values = clock.split(':')
    P.require(days.isdigit() and len(values) in (2, 3) and all(v.isdigit() for v in values),
              'unknown scheduler duration: ' + text)
    numbers = [int(v) for v in values]
    seconds = numbers[-1] + numbers[-2] * 60 + (numbers[0] * 3600 if len(numbers) == 3 else 0)
    return int(days) * 86400 + seconds


def array_multiplicity(identifier):
    if '_[' not in identifier:
        return 1
    specification = identifier.split('_[', 1)[1].split(']', 1)[0].split('%', 1)[0]
    total = 0
    for part in specification.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+)(?::(\d+))?)?', part)
        P.require(match is not None, 'unknown compressed array task set: ' + identifier)
        start, end, step = int(match[1]), int(match[2] or match[1]), int(match[3] or 1)
        P.require(step > 0 and end >= start, 'invalid array range')
        total += (end - start) // step + 1
    return total


def remaining_commitments(saldo, queue):
    rows = [line.split() for line in saldo.splitlines() if line.strip().startswith('IscrC_MIOSR ')]
    P.require(len(rows) == 1 and len(rows[0]) >= 9, 'unrecognized Booster account balance')
    remaining = float(rows[0][3]) - float(rows[0][5])
    commitments = []
    for line in queue.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split('|')]
        P.require(len(fields) == 7, 'unrecognized active commitment row')
        identifier, state, limit, left, cpus, gres, nodes = fields
        count = array_multiplicity(identifier)
        seconds = duration_seconds(left)
        gpu_match = re.search(r'gres/gpu(?::[^:,]+)?:(\d+)', gres)
        gpus = int(gpu_match[1]) * int(nodes) if gpu_match else 0
        billing_cores = max(int(cpus), gpus * 8)
        commitments.append({'job_or_array': identifier, 'state': state, 'tasks': count,
            'remaining_seconds': seconds, 'billing_cores_each': billing_cores,
            'remaining_billing_core_hours': count * billing_cores * seconds / 3600})
    reserved = sum(row['remaining_billing_core_hours'] for row in commitments)
    return {'reported_remaining_billing_core_hours': remaining,
            'active_remaining_commitment_billing_core_hours': reserved,
            'uncommitted_reported_billing_core_hours': remaining - reserved,
            'commitments': commitments,
            'limit': 'Conservative active-resource estimate; saldo can lag usage and future jobs are not yet queue commitments. Preserve the separately recorded generation allowance and project headroom.'}


def live_preflight(directory, phase):
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
              not socket.gethostname().startswith('login'), 'production dispatcher requires CPU Slurm')
    P.require(subprocess.run(['id', '-un'], check=True, capture_output=True, text=True).stdout.strip() == 'lmolfett',
              'wrong cluster user')
    outputs = {}
    for label, command in (
        ('cpu_partition', ['scontrol', 'show', 'partition', 'lrd_all_viz']),
        ('gpu_partition', ['scontrol', 'show', 'partition', 'boost_usr_prod']),
        ('balance', ['saldo', '-b', 'lmolfett']),
        ('queue_commitments', ['squeue', '-h', '-A', 'iscrc_miosr', '-o', '%100i|%T|%l|%L|%C|%b|%D']),
        ('normal_qos', ['sacctmgr', '--noheader', '--parsable2', 'show', 'qos', 'normal',
                        'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxTRESPerJob']),
        ('associations', ['sacctmgr', '--noheader', '--parsable2', 'show', 'assoc', 'where', 'user=lmolfett',
                          'format=User,Account,Partition,QOS'])):
        outputs[label] = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    P.require('iscrc_miosr' in outputs['associations'], 'authorized project association missing')
    budget = remaining_commitments(outputs['balance'], outputs['queue_commitments'])
    return P.save(directory / (phase + '-live-' + os.environ['SLURM_JOB_ID'] + '.json'), {
        'schema': 'utility-production-live-preflight-v1', 'hostname': socket.gethostname(), 'outputs': outputs,
        'budget': budget})


def dependency(identifier, artifact, *, proposal_sha=None):
    """Completed jobs may be purged from the controller; use durable evidence."""
    P.require(re.fullmatch(r'[0-9]+', str(identifier)), 'invalid predecessor job ID')
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', identifier,
        '--format=JobIDRaw,State%40,ExitCode'], check=True, capture_output=True, text=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == identifier]
    P.require(len(rows) == 1, 'predecessor accounting missing or ambiguous')
    state, exit_code = rows[0][1:3]
    if state == 'COMPLETED':
        P.require(exit_code == '0:0', 'completed predecessor has a nonzero exit')
        receipt = P.U.sealed(artifact)
        if proposal_sha is not None:
            P.require(receipt['proposal_sha256'] == proposal_sha, 'completed predecessor artifact rebound')
        return [], {'accounting': result.stdout, 'artifact_path': str(artifact), 'artifact_sha256': receipt['sha256']}
    P.require(state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'),
              'predecessor failed or unknown: ' + state)
    return ['--dependency=afterok:' + identifier], {'accounting': result.stdout, 'active_predecessor': identifier}


def cpu_submit(directory, label, env, binding, dependencies=()):
    return shared.submit(directory, label, [*dependencies, '--job-name=utility-prod-' + label,
        str(P.SCRIPTS / 'dispatch_utility_production_v1.sbatch')], env, binding)


def gpu_submit(directory, label, manifest, config, env, binding, indices):
    seconds = max(manifest['shards'][index]['wall_seconds'] for index in indices)
    wall = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
    array = ','.join(str(i) for i in indices) + '%' + str(config['maximum_concurrent_shards'])
    return shared.submit(directory, label, ['--array=' + array, '--time=' + wall,
        '--job-name=utility-prod-' + label, str(P.SCRIPTS / 'run_utility_production_v1.sbatch')], env, binding)


def run(args):
    config = P.U.sealed(args.config)
    P.validate_config(config)
    attachment = Path(args.pilot_attachment)
    directory = P.DOC / ('utility-production-submissions-v1-' + config['sha256'][:16])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = live_preflight(directory, args.phase)
        env = os.environ.copy()
        env.update(UTILITY_PRODUCTION_CONFIG=str(args.config.resolve()), UTILITY_PILOT_ATTACHMENT=str(attachment.resolve()))
        binding = {'config_sha256': config['sha256'], 'pilot_proposal_sha256': config['pilot_proposal_sha256'],
                   'pilot_attachment': str(attachment.resolve())}
        if args.phase in ('follow-pilot-generation', 'follow-pilot-analysis', 'after-pilot'):
            specifications = {
                'follow-pilot-generation': ('GENERATION_ATTACHMENT.json', 'next_attachment_job',
                    'ANALYSIS_ATTACHMENT.json', 'follow-pilot-analysis'),
                'follow-pilot-analysis': ('ANALYSIS_ATTACHMENT.json', 'selector_job',
                    'PILOT_CHAIN.json', 'after-pilot'),
                'after-pilot': ('PILOT_CHAIN.json', 'accounting_job', 'PILOT_ACCOUNTING.json', 'price')}
            prior_name, job_key, artifact_name, next_phase = specifications[args.phase]
            prior = P.U.sealed(attachment / prior_name)
            P.require(prior['proposal_sha256'] == config['pilot_proposal_sha256'], 'pilot attachment chain changed')
            dependencies, evidence = dependency(prior[job_key], attachment / artifact_name,
                                                 proposal_sha=config['pilot_proposal_sha256'])
            env['UTILITY_PRODUCTION_PHASE'] = next_phase
            job = cpu_submit(directory, next_phase, env, {**binding, 'predecessor_sha256': prior['sha256']}, dependencies)
            P.save(directory / (args.phase + '-CHAIN.json'), {'schema': 'utility-production-follow-pilot-v1',
                **binding, 'next_job': job, 'next_phase': next_phase, 'predecessor': evidence})
            return
        prepared = P.DOC / ('utility-production-prepared-v1-' + config['sha256'][:16])
        if args.phase == 'price':
            try:
                price, manifest = P.prepare(config, attachment, prepared)
            except (ValueError, FileNotFoundError, KeyError) as error:
                P.save(prepared / 'HOLD.json', {'schema': 'utility-production-input-hold-v1', **binding,
                    'status': 'HOLD_PILOT_QUALIFICATION_OR_COMPLETE_PRICE', 'reason': repr(error),
                    'action': 'Reconcile the missing or failed pilot evidence before preparing a new exact resource proposal. No production job submitted.'})
                print('HOLD_PILOT_QUALIFICATION_OR_COMPLETE_PRICE', repr(error), flush=True)
                return
            if manifest is None or not config['dispatch_on_pass']:
                print('PREPARED_FOR_REVIEW_NO_SUBMISSION', price['status'], flush=True)
                return
            required = price['generation_billing_core_hour_ceiling'] + config['offline_grading_reserve_gpu_hours'] * 8
            if required > live['budget']['uncommitted_reported_billing_core_hours']:
                P.save(prepared / 'LIVE_BUDGET_HOLD.json', {'schema': 'utility-production-live-budget-hold-v1',
                    **binding, 'live_preflight_sha256': live['sha256'], 'required_billing_core_hours': required,
                    'available_after_queued_commitments': live['budget']['uncommitted_reported_billing_core_hours'],
                    'status': 'HOLD_RECONCILE_ACCOUNT_AND_COMMITMENTS',
                    'action': 'Reconcile current account capacity and priced commitments; existing user authorization remains valid. No production GPU job submitted.'})
                print('HOLD_RECONCILE_ACCOUNT_AND_COMMITMENTS', flush=True)
                return
            output = P.RUNS / ('utility-production-v1-' + manifest['sha256'][:16])
            env.update(UTILITY_PRODUCTION_MANIFEST=str(prepared / 'MANIFEST.json'),
                       UTILITY_PRODUCTION_OUT=str(output))
            env.pop('UTILITY_PRODUCTION_RECOVERY', None)
            array = gpu_submit(directory, 'initial', manifest, config, env,
                {**binding, 'manifest_sha256': manifest['sha256'], 'price_sha256': price['sha256']},
                list(range(len(manifest['shards']))))
            env['UTILITY_PRODUCTION_PHASE'] = 'account-initial'
            account = cpu_submit(directory, 'account-initial', env,
                {**binding, 'manifest_sha256': manifest['sha256'], 'array_job': array}, ['--dependency=afterany:' + array])
            P.save(directory / 'INITIAL_CHAIN.json', {'schema': 'utility-production-initial-chain-v1',
                **binding, 'manifest_sha256': manifest['sha256'], 'manifest_path': str(prepared / 'MANIFEST.json'),
                'price_sha256': price['sha256'], 'output': str(output), 'array_job': array, 'account_job': account,
                'generation_only': True, 'offline_measurement': price['offline_measurement']})
            return
        initial = P.U.sealed(directory / 'INITIAL_CHAIN.json')
        P.require(initial['config_sha256'] == config['sha256'], 'production submission configuration changed')
        manifest = P.U.sealed(Path(initial['manifest_path']))
        P.validate_manifest(manifest)
        output = Path(initial['output'])
        arrays = [{'array_job': initial['array_job'], 'wave': 0}]
        wave_shards = {0: [s['index'] for s in manifest['shards']]}
        if args.phase == 'account-final':
            recovered = P.U.sealed(directory / 'RECOVERY_CHAIN.json')
            recovery = P.U.sealed(output / 'RECOVERY.json')
            P.require(recovered['recovery_sha256'] == recovery['sha256'] and
                      recovered['manifest_sha256'] == manifest['sha256'], 'recovery chain changed')
            arrays.append({'array_job': recovered['array_job'], 'wave': 1})
            wave_shards[1] = sorted(map(int, recovery['shards']))
        accounting = P.allocation_accounting(arrays, wave_shards)
        if args.phase == 'account-final':
            result = P.reconcile(initial['manifest_path'], output, accounting, final=True)
        elif args.phase == 'account-initial':
            result = P.reconcile(initial['manifest_path'], output, accounting, final=False)
            if result['missing']:
                recovery = P.save(output / 'RECOVERY.json', P.recovery_plan(manifest, result))
                price = P.U.sealed(Path(initial['manifest_path']).parent / 'PRICE.json')
                exceeds_reserve = (recovery['allocation_gpu_hour_ceiling'] >
                                   price['infrastructure_recovery_reserve_gpu_hour_ceiling'])
                exceeds_account = ((recovery['allocation_gpu_hour_ceiling'] + config['offline_grading_reserve_gpu_hours']) * 8 >
                                   live['budget']['uncommitted_reported_billing_core_hours'])
                if exceeds_reserve or exceeds_account:
                    held = P.save(directory / 'RECOVERY_HOLD.json', {'schema': 'utility-production-recovery-hold-v1',
                        **binding, 'manifest_sha256': manifest['sha256'], 'initial_index_sha256': result['sha256'],
                        'recovery_manifest_sha256': recovery['sha256'], 'missing_assignments': recovery['assigned'],
                        'required_shards': len(recovery['shards']),
                        'reserved_shards': price['infrastructure_recovery_reserved_shards'],
                        'required_gpu_hour_ceiling': recovery['allocation_gpu_hour_ceiling'],
                        'reserved_gpu_hour_ceiling': price['infrastructure_recovery_reserve_gpu_hour_ceiling'],
                        'additional_gpu_hour_ceiling': max(0., recovery['allocation_gpu_hour_ceiling'] -
                                                          price['infrastructure_recovery_reserve_gpu_hour_ceiling']),
                        'live_preflight_sha256': live['sha256'],
                        'status': 'HOLD_RECONCILE_ADDITIONAL_INFRASTRUCTURE_WORK',
                        'reason': 'missing work exceeds aggregate reserve' if exceeds_reserve else 'live account capacity needs reconciliation',
                        'action': 'Reconcile the exact additional-work proposal against existing user authorization and current project capacity. No partial recovery subset selected or extra GPU job submitted.'})
                    P.save(directory / 'FINAL_CHAIN.json', {'schema': 'utility-production-final-chain-v1', **binding,
                        'manifest_sha256': manifest['sha256'],
                        'reconciled_index': str(output / 'RECONCILED_INDEX_INITIAL.json'),
                        'reconciled_index_sha256': result['sha256'], 'status': held['status'],
                        'missing': result['missing'], 'generation_errors': result['generation_errors'],
                        'total_generation_allocated_gpu_hours': result['total_generation_allocated_gpu_hours'],
                        'offline_measurement_status': 'INCOMPLETE_ITT_WITH_EXPLICIT_UNKNOWN_ENDPOINTS',
                        'resource_proposal_sha256': held['sha256'], 'further_automatic_recovery': False})
                    print(held['status'], 'missing', recovery['assigned'], flush=True)
                    return
                env.update(UTILITY_PRODUCTION_MANIFEST=initial['manifest_path'], UTILITY_PRODUCTION_OUT=str(output),
                           UTILITY_PRODUCTION_RECOVERY=str(output / 'RECOVERY.json'))
                array = gpu_submit(directory, 'recovery', manifest, config, env,
                    {**binding, 'manifest_sha256': manifest['sha256'], 'recovery_sha256': recovery['sha256']},
                    sorted(map(int, recovery['shards'])))
                env['UTILITY_PRODUCTION_PHASE'] = 'account-final'
                account = cpu_submit(directory, 'account-final', env,
                    {**binding, 'manifest_sha256': manifest['sha256'], 'array_job': array}, ['--dependency=afterany:' + array])
                P.save(directory / 'RECOVERY_CHAIN.json', {'schema': 'utility-production-recovery-chain-v1',
                    **binding, 'manifest_sha256': manifest['sha256'], 'recovery_sha256': recovery['sha256'],
                    'array_job': array, 'account_job': account, 'further_automatic_recovery': False})
                return
            result = P.reconcile(initial['manifest_path'], output, accounting, final=True)
        else:
            raise ValueError('unknown production phase')
        P.save(directory / 'FINAL_CHAIN.json', {'schema': 'utility-production-final-chain-v1', **binding,
            'manifest_sha256': manifest['sha256'], 'reconciled_index': str(output / 'RECONCILED_INDEX_FINAL.json'),
            'reconciled_index_sha256': result['sha256'], 'status': result['status'],
            'missing': result['missing'], 'generation_errors': result['generation_errors'],
            'total_generation_allocated_gpu_hours': result['total_generation_allocated_gpu_hours'],
            'offline_measurement_status': 'READY_FOR_SEPARATELY_PRICED_FROZEN_STRICT_J1_PIPELINE',
            'further_automatic_recovery': False})
        print(result['status'], 'missing', result['missing'], 'errors', result['generation_errors'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pilot-attachment', type=Path, required=True)
    parser.add_argument('--phase', required=True, choices=('follow-pilot-generation', 'follow-pilot-analysis',
        'after-pilot', 'price', 'account-initial', 'account-final'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
