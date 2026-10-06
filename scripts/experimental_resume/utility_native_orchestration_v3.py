"""Finite CPU readiness and true-login utility dispatch with native accounting.

No loop, daemon, cached production balance or fabricated Slurm environment.
CPU jobs retain frozen pricing/accounting science and stop at sealed readiness.
Actual login-node invocations validate readiness and live account commitments
before submitting the separately priced generation/recovery/J1 GPU children.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time

# Bootstrap the immutable scoring package before shared helpers cache moe_exp.
import utility_j1_entry_v2 as j1_entry
J1 = j1_entry.install()
import utility_production_v1 as P
import dispatch_utility_production_v1 as original
import dispatch_overnight_readers_v1 as shared

WRAPPER = P.SCRIPTS / 'utility_native_cpu_v3.sbatch'


def source_files():
    return {str(path): P.U.file_sha(path) for path in (Path(__file__).resolve(), WRAPPER)}


def login_guard():
    P.require(socket.gethostname().startswith('login') and socket.gethostname().endswith('.leonardo.local') and
        pwd.getpwuid(os.getuid()).pw_name == 'lmolfett' and not os.environ.get('SLURM_JOB_ID'),
        'native dispatch requires the actual lmolfett LEONARDO login shell, without a Slurm environment')


def cpu_guard():
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
        os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
        pwd.getpwuid(os.getuid()).pw_name == 'lmolfett' and not socket.gethostname().startswith('login'),
        'readiness production requires an actual authorized CPU Slurm step')


def terminal_job(identifier, expected=None):
    result = subprocess.run(['sacct', '-X', '-nP', '-j', str(identifier),
        '--format=JobIDRaw,State%40,ExitCode,ElapsedRaw'], check=True, capture_output=True, text=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == str(identifier)]
    P.require(len(rows) == 1, 'exact native readiness/failure accounting missing or ambiguous')
    row = rows[0]
    if expected is not None:
        P.require(row[1].split(' by ', 1)[0] == expected, 'preserved failure state differs')
    else:
        P.require(row[1:3] == ['COMPLETED', '0:0'], 'readiness producer did not complete successfully')
    return {'job_id': str(identifier), 'accounting': result.stdout, 'elapsed_seconds': int(row[3])}


def validate_operations(path):
    operations = P.U.sealed(Path(path))
    P.require(operations['schema'] == 'utility-native-orchestration-plan-v3' and
        operations['operational_code_files'] == source_files(), 'native operational source closure changed')
    config = P.U.sealed(Path(operations['config_path']))
    P.require(config['sha256'] == operations['config_sha256'], 'native utility configuration changed')
    P.validate_config(config)
    previous = P.U.sealed(Path(operations['prior_recovery_path']))
    plan = P.U.sealed(Path(previous['pilot_recovery_plan_path']))
    P.require(previous['sha256'] == operations['prior_recovery_sha256'] and
        plan['sha256'] == operations['pilot_recovery_plan_sha256'] and
        previous['pilot_attachment_dir'] == operations['pilot_attachment_dir'] and
        all(P.U.file_sha(Path(name)) == digest for name, digest in previous['operational_code_files'].items()),
        'preserved v2/v5 operational closure or valid pilot continuation changed')
    probe = P.U.sealed(Path(operations['failed_transport_probe_path']))
    P.require(probe['sha256'] == operations['failed_transport_probe_sha256'] and
        probe['status'] == 'FAIL_NATIVE_ACCOUNT_TRANSPORT', 'failed actual-node transport evidence changed')
    P.require(J1.validate_plan()['sha256'] == operations['grading_plan_sha256'], 'frozen J1 plan changed')
    return operations, config


def live_preflight(directory, label, required):
    login_guard()
    outputs = {}
    for name, command in (
        ('balance', ['/cineca/bin/saldo', '-b', 'lmolfett']),
        ('queue_commitments', ['squeue', '-h', '-A', 'iscrc_miosr', '-o', '%100i|%T|%l|%L|%C|%b|%D']),
        ('cpu_partition', ['scontrol', 'show', 'partition', 'lrd_all_viz']),
        ('gpu_partition', ['scontrol', 'show', 'partition', 'boost_usr_prod']),
        ('associations', ['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett',
                          'format=User,Account,Partition,QOS']),
        ('normal_qos', ['sacctmgr', '-nP', 'show', 'qos', 'normal',
                        'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxTRESPerJob'])):
        outputs[name] = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    P.require('iscrc_miosr' in outputs['associations'] and
        'PartitionName=lrd_all_viz' in outputs['cpu_partition'] and
        'PartitionName=boost_usr_prod' in outputs['gpu_partition'], 'live native account or partitions differ')
    budget = original.remaining_commitments(outputs['balance'], outputs['queue_commitments'])
    value = P.save(directory / (label + '-LIVE-' + str(time.time_ns()) + '.json'), {
        'schema': 'utility-native-live-account-v3', 'hostname': socket.gethostname(), 'outputs': outputs,
        'budget': budget, 'required_billing_core_hours': required,
        'status': 'PASS_LIVE_ACCOUNT' if required <= budget['uncommitted_reported_billing_core_hours'] else 'HOLD_LIVE_ACCOUNT',
        'scope': 'Current native saldo and complete queued project commitments. Reported account consumption can lag; this snapshot is observed evidence, not cached capacity for a later submission.'})
    P.require(value['status'] == 'PASS_LIVE_ACCOUNT', 'native live account cannot fit the complete priced stage and reserves')
    return value


def root_path(operations):
    return Path(operations['native_root'])


def binding(operations):
    return {'native_operations_sha256': operations['sha256'], 'config_sha256': operations['config_sha256'],
        'pilot_attachment': operations['pilot_attachment_dir']}


def child_environment(operations_path, phase, **extra):
    env = os.environ.copy()
    env.update(UTILITY_NATIVE_OPERATIONS=str(operations_path), UTILITY_NATIVE_PHASE=phase,
               **{key: str(value) for key, value in extra.items()})
    return env


def cpu_submit(operations, operations_path, name, phase, dependencies=(), extra=None, identity=None):
    directory = root_path(operations)
    args = [*dependencies, '--time=' + ('00:05:00' if phase == 'after-pilot' else
                                      '02:00:00' if phase == 'grade-prepare' else '01:00:00'),
            '--job-name=utility-native-' + name + '-v3', str(WRAPPER)]
    saved = directory / (name + '.json')
    if saved.exists():
        args = P.U.sealed(saved)['arguments']
    return shared.submit(directory, name, args,
        child_environment(operations_path, phase, **(extra or {})), {**binding(operations), **(identity or {})})


def ready(operations, kind, artifacts, **extra):
    return P.save(root_path(operations) / (kind.upper() + '_READY.json'), {
        'schema': 'utility-native-readiness-v3', **binding(operations), 'kind': kind,
        'status': 'READY_FOR_NATIVE_LOGIN_DISPATCH', 'producer_job_id': os.environ['SLURM_JOB_ID'],
        'producer_hostname': socket.gethostname(), 'artifacts': artifacts, **extra})


def artifact(path):
    value = P.U.sealed(Path(path))
    return {'path': str(path), 'sha256': value['sha256']}


def validate_ready(operations, path):
    receipt = P.U.sealed(Path(path))
    P.require(receipt['schema'] == 'utility-native-readiness-v3' and
        receipt['status'] == 'READY_FOR_NATIVE_LOGIN_DISPATCH' and
        all(receipt[key] == value for key, value in binding(operations).items()) and
        Path(path).resolve() == (root_path(operations) / (receipt['kind'].upper() + '_READY.json')).resolve() and
        not receipt['producer_hostname'].startswith('login'), 'native readiness identity or path differs')
    proof = terminal_job(receipt['producer_job_id'])
    for reference in receipt['artifacts'].values():
        P.require(P.U.sealed(Path(reference['path']))['sha256'] == reference['sha256'],
                  'sealed readiness artifact changed')
    return receipt, proof


def final_chain(operations, initial, manifest, result, output):
    chain_dir = Path(operations['production_chain_dir'])
    result_path = output / 'RECONCILED_INDEX_FINAL.json'
    P.save(chain_dir / 'FINAL_CHAIN.json', {'schema': 'utility-production-final-chain-v1', **binding(operations),
        'manifest_sha256': manifest['sha256'], 'reconciled_index': str(result_path),
        'reconciled_index_sha256': result['sha256'], 'status': result['status'], 'missing': result['missing'],
        'generation_errors': result['generation_errors'],
        'total_generation_allocated_gpu_hours': result['total_generation_allocated_gpu_hours'],
        'offline_measurement_status': 'READY_FOR_SEPARATELY_PRICED_FROZEN_STRICT_J1_PIPELINE',
        'further_automatic_recovery': False})
    return ready(operations, 'final', {'initial': artifact(chain_dir / 'INITIAL_CHAIN.json'),
        'final': artifact(chain_dir / 'FINAL_CHAIN.json'), 'index': artifact(result_path)})


def run_cpu(operations, config, operations_path, phase):
    cpu_guard()
    directory = root_path(operations)
    attachment = Path(operations['pilot_attachment_dir'])
    if phase == 'after-pilot':
        chain = P.U.sealed(attachment / 'PILOT_CHAIN.json')
        P.require(chain['proposal_sha256'] == config['pilot_proposal_sha256'], 'pilot continuation proposal changed')
        dependencies, proof = original.dependency(chain['accounting_job'], attachment / 'PILOT_ACCOUNTING.json',
                                                   proposal_sha=config['pilot_proposal_sha256'])
        job = cpu_submit(operations, operations_path, 'price', 'price', dependencies,
                         identity={'pilot_chain_sha256': chain['sha256']})
        return P.save(directory / 'PRICE_CPU_CHAIN.json', {'schema': 'utility-native-price-cpu-chain-v3',
            **binding(operations), 'prepare_job': job, 'pilot_predecessor': proof})
    if phase == 'price':
        prepared = P.DOC / ('utility-production-prepared-v1-' + config['sha256'][:16])
        try:
            price, manifest = P.prepare(config, attachment, prepared)
        except (ValueError, KeyError, FileNotFoundError) as error:
            return P.save(directory / 'PREPARATION_HOLD.json', {'schema': 'utility-native-preparation-hold-v3',
                **binding(operations), 'status': 'HOLD_PILOT_QUALIFICATION_OR_COMPLETE_PRICE', 'reason': repr(error),
                'producer_job_id': os.environ['SLURM_JOB_ID'], 'gpu_submitted': False})
        if manifest is None or not config['dispatch_on_pass']:
            return P.save(directory / 'PREPARATION_HOLD.json', {'schema': 'utility-native-preparation-hold-v3',
                **binding(operations), 'status': price['status'], 'price_sha256': price['sha256'],
                'producer_job_id': os.environ['SLURM_JOB_ID'], 'gpu_submitted': False})
        return ready(operations, 'initial', {'price': artifact(prepared / 'PRICE.json'),
            'manifest': artifact(prepared / 'MANIFEST.json'),
            'pilot_accounting': artifact(attachment / 'PILOT_ACCOUNTING.json')})
    if phase == 'grade-prepare':
        chain_dir = Path(operations['production_chain_dir'])
        final = P.U.sealed(chain_dir / 'FINAL_CHAIN.json')
        initial = P.U.sealed(chain_dir / 'INITIAL_CHAIN.json')
        index_path, parent = Path(final['reconciled_index']), Path(initial['output'])
        J1.prepare(index_path, parent)
        _, grade_root = J1.stage_paths(index_path, parent)
        return ready(operations, 'j1', {'price': artifact(grade_root / 'J1_PRICE.json'),
            'grade_binding': artifact(grade_root / 'BINDING.json'),
            'grade_preparation': artifact(grade_root / 'preparation/GRADE_PREP.json'),
            'final': artifact(chain_dir / 'FINAL_CHAIN.json')}, grade_root=str(grade_root))
    P.require(phase in ('account-initial', 'account-final'), 'unknown CPU readiness phase')
    chain_dir = Path(operations['production_chain_dir'])
    initial = P.U.sealed(chain_dir / 'INITIAL_CHAIN.json')
    P.require(initial['native_operations_sha256'] == operations['sha256'] and
        initial['config_sha256'] == config['sha256'], 'native production initial chain differs')
    manifest = P.U.sealed(Path(initial['manifest_path']))
    P.validate_manifest(manifest)
    output = Path(initial['output'])
    arrays = [{'array_job': initial['array_job'], 'wave': 0}]
    shards = {0: [s['index'] for s in manifest['shards']]}
    if phase == 'account-final':
        recovered = P.U.sealed(chain_dir / 'RECOVERY_CHAIN.json')
        recovery = P.U.sealed(output / 'RECOVERY.json')
        P.require(recovered['native_operations_sha256'] == operations['sha256'] and
            recovered['recovery_sha256'] == recovery['sha256'], 'native bounded recovery chain differs')
        arrays.append({'array_job': recovered['array_job'], 'wave': 1})
        shards[1] = sorted(map(int, recovery['shards']))
    costs = P.allocation_accounting(arrays, shards)
    result = P.reconcile(initial['manifest_path'], output, costs, final=phase == 'account-final')
    if phase == 'account-initial' and result['missing']:
        recovery = P.save(output / 'RECOVERY.json', P.recovery_plan(manifest, result))
        price = P.U.sealed(Path(initial['manifest_path']).parent / 'PRICE.json')
        if recovery['allocation_gpu_hour_ceiling'] > price['infrastructure_recovery_reserve_gpu_hour_ceiling']:
            held = P.save(chain_dir / 'RECOVERY_HOLD.json', {'schema': 'utility-production-recovery-hold-v1',
                **binding(operations), 'manifest_sha256': manifest['sha256'],
                'status': 'HOLD_RECONCILE_ADDITIONAL_INFRASTRUCTURE_WORK',
                'recovery_manifest_sha256': recovery['sha256'], 'missing_assignments': recovery['assigned'],
                'required_gpu_hour_ceiling': recovery['allocation_gpu_hour_ceiling'],
                'reserved_gpu_hour_ceiling': price['infrastructure_recovery_reserve_gpu_hour_ceiling'],
                'further_automatic_recovery': False, 'gpu_submitted': False})
            P.save(directory / 'RECOVERY_RESERVE_HOLD.json', {k: v for k, v in held.items() if k != 'sha256'})
            index_path = output / 'RECONCILED_INDEX_INITIAL.json'
            P.save(chain_dir / 'FINAL_CHAIN.json', {'schema': 'utility-production-final-chain-v1',
                **binding(operations), 'manifest_sha256': manifest['sha256'],
                'reconciled_index': str(index_path), 'reconciled_index_sha256': result['sha256'],
                'status': held['status'], 'missing': result['missing'],
                'generation_errors': result['generation_errors'],
                'total_generation_allocated_gpu_hours': result['total_generation_allocated_gpu_hours'],
                'offline_measurement_status': 'INCOMPLETE_ITT_WITH_EXPLICIT_UNKNOWN_ENDPOINTS',
                'resource_proposal_sha256': held['sha256'], 'further_automatic_recovery': False})
            return ready(operations, 'final', {'initial': artifact(chain_dir / 'INITIAL_CHAIN.json'),
                'final': artifact(chain_dir / 'FINAL_CHAIN.json'), 'index': artifact(index_path),
                'recovery_hold': artifact(chain_dir / 'RECOVERY_HOLD.json')})
        return ready(operations, 'recovery', {'initial': artifact(chain_dir / 'INITIAL_CHAIN.json'),
            'initial_index': artifact(output / 'RECONCILED_INDEX_INITIAL.json'),
            'manifest': artifact(initial['manifest_path']), 'recovery': artifact(output / 'RECOVERY.json'),
            'price': artifact(Path(initial['manifest_path']).parent / 'PRICE.json')})
    if phase == 'account-initial':
        result = P.reconcile(initial['manifest_path'], output, costs, final=True)
    return final_chain(operations, initial, manifest, result, output)


def dispatch_native(operations, config, operations_path, receipt, live):
    """Submit only from the real login; preparation/accounting stay on Slurm."""
    login_guard()
    directory, chain_dir = root_path(operations), Path(operations['production_chain_dir'])
    refs = receipt['artifacts']
    kind = receipt['kind']
    # Job identity is stable on interrupted resume. Fresh live snapshots are
    # recorded separately; a changed timestamp cannot silently duplicate work.
    identity = {**binding(operations), 'readiness_sha256': receipt['sha256']}
    env = child_environment(operations_path, 'account-initial')
    if kind in ('initial', 'recovery'):
        existing_chain = chain_dir / ('INITIAL_CHAIN.json' if kind == 'initial' else 'RECOVERY_CHAIN.json')
        if existing_chain.exists():
            previous = P.U.sealed(existing_chain)
            P.require(previous['native_operations_sha256'] == operations['sha256'] and
                previous['readiness_sha256'] == receipt['sha256'], 'existing native generation chain differs')
            return previous
        manifest = P.U.sealed(Path(refs['manifest']['path']))
        price = P.U.sealed(Path(refs['price']['path']))
        P.validate_manifest(manifest)
        P.require(manifest['config_sha256'] == config['sha256'] and manifest['price_sha256'] == price['sha256'] and
            manifest['shards'] == price['shards'] and
            price['status'] == 'PASS_COMPLETE_GENERATION_PROJECTION_BOUNDED_ALLOCATION',
            'complete generation price or prepared manifest changed')
        output = P.RUNS / ('utility-production-v1-' + manifest['sha256'][:16])
        env.update(UTILITY_PRODUCTION_CONFIG=operations['config_path'],
            UTILITY_PILOT_ATTACHMENT=operations['pilot_attachment_dir'],
            UTILITY_PRODUCTION_MANIFEST=refs['manifest']['path'], UTILITY_PRODUCTION_OUT=str(output))
        if kind == 'initial':
            env.pop('UTILITY_PRODUCTION_RECOVERY', None)
            indices = list(range(len(manifest['shards'])))
        else:
            recovery = P.U.sealed(Path(refs['recovery']['path']))
            P.require(recovery['wave'] == 1 and recovery['manifest_sha256'] == manifest['sha256'] and
                recovery['allocation_gpu_hour_ceiling'] <= price['infrastructure_recovery_reserve_gpu_hour_ceiling'],
                'recovery is not the single fully reserved infrastructure wave')
            env['UTILITY_PRODUCTION_RECOVERY'] = refs['recovery']['path']
            indices = sorted(map(int, recovery['shards']))
        array = original.gpu_submit(chain_dir, kind, manifest, config, env,
            {**identity, 'manifest_sha256': manifest['sha256'], 'price_sha256': price['sha256']}, indices)
        account_phase = 'account-initial' if kind == 'initial' else 'account-final'
        account = cpu_submit(operations, operations_path, account_phase, account_phase,
            ['--dependency=afterany:' + array], identity={'manifest_sha256': manifest['sha256'], 'array_job': array})
        name = 'INITIAL_CHAIN.json' if kind == 'initial' else 'RECOVERY_CHAIN.json'
        body = {'schema': 'utility-production-' + kind + '-chain-v1', **binding(operations),
            'pilot_proposal_sha256': config['pilot_proposal_sha256'], 'manifest_sha256': manifest['sha256'],
            'manifest_path': refs['manifest']['path'], 'price_sha256': price['sha256'],
            'output': str(output), 'array_job': array, 'account_job': account,
            'readiness_sha256': receipt['sha256'], 'live_preflight_sha256': live['sha256']}
        if kind == 'initial':
            body.update(generation_only=True, offline_measurement=price['offline_measurement'])
        else:
            body.update(recovery_sha256=recovery['sha256'], further_automatic_recovery=False)
        return P.save(chain_dir / name, body)
    if kind == 'final':
        job = cpu_submit(operations, operations_path, 'grade-prepare', 'grade-prepare',
                         identity={'final_chain_sha256': refs['final']['sha256']})
        return P.save(directory / 'GRADE_PREPARE_CHAIN.json', {'schema': 'utility-native-grade-prepare-chain-v3',
            **identity, 'grade_prepare_job': job})
    P.require(kind == 'j1', 'unsupported native readiness kind')
    grade_root = Path(receipt['grade_root'])
    plan, prep, price, items = J1.priced(grade_root)
    dispatch_dir = grade_root / 'dispatch'
    dispatch_dir.mkdir(exist_ok=True)
    common = {'schema': 'utility-j1-dispatch-binding-v1', 'plan_sha256': plan['sha256'],
        'grade_preparation_sha256': prep['sha256'], 'price_sha256': price['sha256'], **identity}
    with (dispatch_dir / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        child = J1.safe_environment(os.environ, {'STEER_CODE': str(J1.BASE),
            'STEER_EXPECT_TREE': J1.sealed(J1.BASE / 'MANIFEST.json')['tree_sha256'],
            'STEER_PROVENANCE_DIR': str(grade_root / 'j1/provenance')})
        for shard in price['shards']:
            name = 'utility-j1-' + str(shard['index']).zfill(3)
            job = shared.submit(dispatch_dir, name, ['--time=06:00:00', '--job-name=' + name,
                str(J1.LEGACY), shard['items_path'], shard['verdicts_path']], child,
                {**common, 'shard': shard['index'], 'items_file_sha256': shard['items_file_sha256'],
                 'wrapper_sha256': J1.sha(J1.LEGACY)})
            J1.verify.ensure_verified(dispatch_dir, name, job)
            jobs.append(job)
        active, proofs = [], []
        for shard, job in zip(price['shards'], jobs, strict=True):
            deps, proof = J1.dependencies.dependency(job, lambda s=shard: {
                'validated_items': len(J1.valid_verdicts(Path(s['verdicts_path']), s['item_ids']))})
            if deps:
                active.append(job)
            proofs.append(proof)
        args = [*(['--dependency=afterok:' + ':'.join(active)] if active else []),
                str(J1.WRAPPERS['finalize']), '--grade-root', str(grade_root)]
        finalizer_path = dispatch_dir / 'utility-j1-finalize.json'
        if finalizer_path.exists():
            previous = J1.sealed(finalizer_path)
            P.require(all(previous['binding'].get(key) == value for key, value in common.items()),
                      'prior J1 finalizer binding differs')
            finalizer = previous['job_id']
        else:
            finalizer = shared.submit(dispatch_dir, 'utility-j1-finalize', args,
                J1.safe_environment(os.environ), {**common, 'parent_jobs': jobs, 'dependency_proofs': proofs})
        J1.verify.ensure_verified(dispatch_dir, 'utility-j1-finalize', finalizer)
        return P.save(dispatch_dir / 'CHAIN.json', {'schema': 'utility-j1-dispatch-chain-v1', **common,
            'J1_jobs': jobs, 'finalize_job': finalizer, 'grade_root': str(grade_root)})


def required_budget(operations, config, receipt):
    kind, refs = receipt['kind'], receipt['artifacts']
    if kind == 'initial':
        price = P.U.sealed(Path(refs['price']['path']))
        previous = Path(operations['production_chain_dir']) / 'initial.json'
        remaining = price['generation_billing_core_hour_ceiling']
        if previous.exists():
            submitted = P.U.sealed(previous)
            P.require(submitted['binding']['native_operations_sha256'] == operations['sha256'] and
                submitted['binding']['readiness_sha256'] == receipt['sha256'], 'existing initial submission differs')
            remaining = price['infrastructure_recovery_reserve_gpu_hour_ceiling'] * 8
        return remaining + config['offline_grading_reserve_gpu_hours'] * 8
    if kind == 'recovery':
        previous = Path(operations['production_chain_dir']) / 'recovery.json'
        remaining = P.U.sealed(Path(refs['recovery']['path']))['allocation_gpu_hour_ceiling']
        if previous.exists():
            submitted = P.U.sealed(previous)
            P.require(submitted['binding']['native_operations_sha256'] == operations['sha256'] and
                submitted['binding']['readiness_sha256'] == receipt['sha256'], 'existing recovery submission differs')
            remaining = 0
        return (remaining + config['offline_grading_reserve_gpu_hours']) * 8
    if kind == 'j1':
        price = P.U.sealed(Path(refs['price']['path']))
        remaining = price['requested_allocation_GPU_hour_ceiling']
        dispatch_dir = Path(receipt['grade_root']) / 'dispatch'
        for shard in price['shards']:
            path = dispatch_dir / ('utility-j1-' + str(shard['index']).zfill(3) + '.json')
            if path.exists():
                submitted = P.U.sealed(path)
                P.require(submitted['binding']['native_operations_sha256'] == operations['sha256'] and
                    submitted['binding']['readiness_sha256'] == receipt['sha256'] and
                    submitted['binding']['price_sha256'] == price['sha256'], 'existing J1 shard submission differs')
                remaining -= shard['requested_wall_seconds'] * 2 / 3600
        P.require(remaining >= -1e-9, 'J1 committed reservation exceeds its exact stage price')
        return max(0., remaining) * 8
    P.require(kind == 'final', 'unrecognized native readiness kind')
    return config['offline_grading_reserve_gpu_hours'] * 8


def freeze(prior_root, probe_path):
    login_guard()
    previous = P.U.sealed(prior_root / 'RECOVERY.json')
    old = P.U.sealed(prior_root / 'CHAIN.json')
    config = P.U.sealed(Path(previous['config_path']))
    P.validate_config(config)
    P.require(old['operational_recovery_sha256'] == previous['sha256'], 'v2 root recovery differs')
    failures = {'production': terminal_job(old['production_follow_job'], 'FAILED'),
                'grading': terminal_job(old['grading_follow_job'], 'CANCELLED')}
    P.require(failures['grading']['elapsed_seconds'] == 0, 'cancelled grading already consumed an allocation')
    probe = P.U.sealed(probe_path)
    P.require(probe['status'] == 'FAIL_NATIVE_ACCOUNT_TRANSPORT' and
        not probe['compute_hostname'].startswith('login'), 'actual compute transport failure is missing')
    failures['transport'] = terminal_job(probe['probe_job_id'], 'FAILED')
    directory = P.DOC / ('utility-production-root-submissions-v3-' + config['sha256'][:16])
    directory.mkdir(exist_ok=True)
    plan_path = directory / 'OPERATIONS.json'
    plan = P.save(plan_path, {'schema': 'utility-native-orchestration-plan-v3',
        'config_path': previous['config_path'], 'config_sha256': config['sha256'],
        'prior_recovery_path': str(prior_root / 'RECOVERY.json'), 'prior_recovery_sha256': previous['sha256'],
        'prior_root_chain_sha256': old['sha256'], 'preserved_failures': failures,
        'pilot_attachment_dir': previous['pilot_attachment_dir'],
        'pilot_recovery_plan_sha256': previous['pilot_recovery_plan_sha256'],
        'production_chain_dir': previous['production_chain_dir'], 'native_root': str(directory),
        'failed_transport_probe_path': str(probe_path), 'failed_transport_probe_sha256': probe['sha256'],
        'grading_plan_sha256': J1.validate_plan()['sha256'], 'operational_code_files': source_files(),
        'scope': 'Finite actual CPU preparation/accounting -> sealed readiness -> actual native login balance/queue/dispatch. No daemon, polling loop, credential changes, cached production budget or fabricated Slurm environment. Frozen v1 generation/pricing/reconciliation and v2 J1 scientific adapters remain unchanged.'})
    validate_operations(plan_path)
    return plan_path, plan, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('attach', 'cpu', 'dispatch'))
    parser.add_argument('--operations', type=Path)
    parser.add_argument('--prior-root', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--phase', choices=('after-pilot', 'price', 'account-initial', 'account-final', 'grade-prepare'))
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.mode == 'attach':
        P.require(args.prior_root and args.probe, 'attach requires exact prior root and failed actual-node probe')
        operations_path, operations, config = freeze(args.prior_root.resolve(), args.probe.resolve())
        directory = root_path(operations)
        with (directory / 'WRITER.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / 'ATTACH_CHAIN.json').exists():
                value = P.U.sealed(directory / 'ATTACH_CHAIN.json')
                P.require(value['native_operations_sha256'] == operations['sha256'], 'existing native attachment differs')
                print(json.dumps({'status': 'ALREADY_SUBMITTED', 'chain': str(directory / 'ATTACH_CHAIN.json')})); return
            pilot = P.U.sealed(Path(operations['pilot_attachment_dir']) / 'ANALYSIS_ATTACHMENT.json')
            P.require(pilot['proposal_sha256'] == config['pilot_proposal_sha256'], 'valid pilot analysis attachment changed')
            deps, proof = original.dependency(pilot['selector_job'], Path(operations['pilot_attachment_dir']) / 'PILOT_CHAIN.json',
                proposal_sha=config['pilot_proposal_sha256'])
            allowance = config['available_generation_billing_core_hours'] + config['offline_grading_reserve_gpu_hours'] * 8 + 32 * 8
            live = live_preflight(directory, 'attach', allowance)
            if not args.submit:
                print(json.dumps({'status': 'PREPARED_NO_SUBMISSION', 'operations': str(operations_path),
                    'sha256': operations['sha256'], 'selector_job': pilot['selector_job'],
                    'dependency_arguments': deps, 'live_preflight_sha256': live['sha256']})); return
            job = cpu_submit(operations, operations_path, 'after-pilot', 'after-pilot', deps,
                identity={'pilot_analysis_attachment_sha256': pilot['sha256']})
            value = P.save(directory / 'ATTACH_CHAIN.json', {'schema': 'utility-native-attachment-chain-v3',
                **binding(operations), 'after_pilot_job': job, 'selector_predecessor': proof,
                'live_preflight_sha256': live['sha256'], 'initial_readiness_path': str(directory / 'INITIAL_READY.json')})
            print(json.dumps(value)); return
    P.require(args.operations, 'sealed native operations plan required')
    operations_path = args.operations.resolve()
    operations, config = validate_operations(operations_path)
    if args.mode == 'cpu':
        P.require(args.phase, 'CPU readiness phase required')
        value = run_cpu(operations, config, operations_path, args.phase)
        print(json.dumps({'schema': value['schema'], 'sha256': value['sha256'], 'status': value.get('status', 'CPU_SUCCESSOR_SUBMITTED')})); return
    login_guard()
    P.require(args.ready, 'dispatch requires an exact sealed readiness artifact')
    receipt, proof = validate_ready(operations, args.ready.resolve())
    directory = root_path(operations)
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed_path = directory / (receipt['kind'].upper() + '_NATIVE_DISPATCH.json')
        if completed_path.exists():
            done = P.U.sealed(completed_path)
            P.require(done['readiness_sha256'] == receipt['sha256'], 'prior native dispatch readiness changed')
            print(json.dumps({'status': 'ALREADY_SUBMITTED', 'receipt': str(completed_path)})); return
        live = live_preflight(directory, receipt['kind'], required_budget(operations, config, receipt))
        if not args.submit:
            print(json.dumps({'status': 'READY_NO_SUBMISSION', 'kind': receipt['kind'],
                'readiness_sha256': receipt['sha256'], 'live_preflight_sha256': live['sha256']})); return
        result = dispatch_native(operations, config, operations_path, receipt, live)
        done = P.save(completed_path, {'schema': 'utility-native-dispatch-receipt-v3', **binding(operations),
            'readiness_sha256': receipt['sha256'], 'producer_proof': proof,
            'live_preflight_sha256': live['sha256'], 'result_sha256': result['sha256']})
        print(json.dumps(done))


if __name__ == '__main__':
    main()
