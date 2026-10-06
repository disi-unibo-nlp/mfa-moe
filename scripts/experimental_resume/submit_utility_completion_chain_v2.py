"""Restore finite utility followers from an exact sealed fresh-reader recovery.

Default invocation validates and prepares the recovery; --submit attaches the
previously authorized workload. Failed v1 roots and pilot attachments are kept.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import utility_production_v1 as P
import dispatch_utility_production_v1 as dispatcher
import dispatch_overnight_readers_v1 as shared
import attach_utility_price_pilot_v4 as pilot

FILES = ('submit_utility_completion_chain_v2.py', 'dispatch_utility_production_v2.py',
    'dispatch_utility_production_v2.sbatch', 'attach_utility_price_pilot_v5.py',
    'attach_utility_price_pilot_v5.sbatch', 'run_utility_price_pilot_v5.sbatch',
    'fresh_frame_recovery_v1.py', 'select_utility_policy_fresh_adapter_v1.py')


def code_files():
    return {str(P.SCRIPTS / name): P.U.file_sha(P.SCRIPTS / name) for name in FILES}


def accounting(identifier):
    P.require(re.fullmatch(r'[0-9]+', str(identifier)), 'invalid exact job ID')
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(identifier),
        '--format=JobIDRaw,State%40,ExitCode,ElapsedRaw'], check=True, capture_output=True, text=True)
    rows = [row.split('|') for row in result.stdout.splitlines() if row.strip()]
    rows = [row for row in rows if row[0] == str(identifier)]
    P.require(len(rows) == 1, 'exact predecessor accounting missing or ambiguous')
    return rows[0], result.stdout


def failed_root(config):
    root = P.DOC / ('utility-production-root-submissions-v1-' + config['sha256'][:16])
    chain = P.U.sealed(root / 'CHAIN.json')
    P.require(chain['config_sha256'] == config['sha256'], 'preserved failed root configuration differs')
    evidence = {}
    for name, key in (('production-follow', 'production_follow_job'), ('grading-follow', 'grading_follow_job')):
        submitted = P.U.sealed(root / (name + '.json'))
        verified = P.U.sealed(root / (name + '.verified.json'))
        P.require(submitted['job_id'] == chain[key] == verified['job_id'] and
            submitted['binding']['config_sha256'] == config['sha256'] and
            'UserId=lmolfett(' in verified['scontrol'], 'preserved exact failure receipt differs')
        row, raw = accounting(chain[key])
        allowed = ('FAILED',) if name == 'production-follow' else ('CANCELLED',)
        P.require(row[1].split(' by ', 1)[0] in allowed and
            (row[2] != '0:0' if name == 'production-follow' else int(row[3]) == 0),
            'prior follower was not the preserved pre-child failure or unused cancelled grading job')
        evidence[name] = {'receipt_sha256': submitted['sha256'], 'verification_sha256': verified['sha256'],
            'job_id': chain[key], 'accounting': raw}
    return chain, evidence


def live_preflight(root, config):
    records = {}
    for label, command in (
        ('balance', ['/cineca/bin/saldo', '-b', 'lmolfett']),
        ('queue_commitments', ['squeue', '-h', '-A', 'iscrc_miosr', '-o', '%100i|%T|%l|%L|%C|%b|%D']),
        ('associations', ['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett',
                          'format=User,Account,Partition,QOS']),
        ('cpu_partition', ['scontrol', 'show', 'partition', 'lrd_all_viz']),
        ('gpu_partition', ['scontrol', 'show', 'partition', 'boost_usr_prod']),
        ('normal_qos', ['sacctmgr', '-nP', 'show', 'qos', 'normal',
                        'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxTRESPerJob'])):
        records[label] = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    P.require('iscrc_miosr' in records['associations'] and
        'PartitionName=lrd_all_viz' in records['cpu_partition'] and
        'PartitionName=boost_usr_prod' in records['gpu_partition'], 'live account or partitions differ')
    budget = dispatcher.remaining_commitments(records['balance'], records['queue_commitments'])
    allowance = config['available_generation_billing_core_hours']
    P.require(allowance is not None, 'complete generation allowance must be explicit')
    # Preserve headroom for the complete generation allowance, all grading and
    # the still-unrun four-GPU/eight-hour fixed pilot before attaching its chain.
    required = allowance + config['offline_grading_reserve_gpu_hours'] * 8 + 32 * 8
    P.require(required <= budget['uncommitted_reported_billing_core_hours'],
        'live account cannot fit fixed pilot, full generation allowance and separately reserved grading')
    return P.save(root / ('PREFLIGHT-' + str(time.time_ns()) + '.json'), {
        'schema': 'utility-completion-recovery-preflight-v2', 'hostname': socket.gethostname(),
        'records': records, 'budget': budget, 'reserved_billing_core_hours': required})


def prepare(config_path, reader_recovery_chain_path, reader_dispatch_key, amendment_path):
    config = P.U.sealed(config_path)
    P.validate_config(config)
    P.require(config['dispatch_on_pass'] is True, 'authorized production dispatch is disabled')
    old, failures = failed_root(config)
    original_attachment = Path(old['pilot_attachment_dir'])
    prior_path = original_attachment / 'GENERATION_ATTACHMENT.json'
    prior = P.U.sealed(prior_path)
    P.require(prior['proposal_sha256'] == config['pilot_proposal_sha256'] and
        not any((original_attachment / name).exists() for name in
                ('ANALYSIS_ATTACHMENT.json', 'PILOT_MANIFEST.json', 'PILOT_CHAIN.json')),
        'prior pilot has already selected policy or submitted runtime work; reconcile before recovery')
    row, raw = accounting(prior['next_attachment_job'])
    P.require(row[1].split(' by ', 1)[0] in ('FAILED', 'CANCELLED'),
              'old pilot attachment is still active or successfully attached; reconcile before recovery')
    failures['pilot-attach-analysis'] = {'job_id': prior['next_attachment_job'], 'accounting': raw}
    repair = P.U.sealed(reader_recovery_chain_path)
    amendment = P.U.sealed(amendment_path)
    reader_job = str(repair[reader_dispatch_key])
    P.require(re.fullmatch(r'[0-9]+', reader_job), 'replacement reader dispatcher ID is invalid')
    manifest_path = P.DOC / 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'
    manifest = P.U.sealed(manifest_path)
    P.require(prior['manifest_sha256'] == manifest['sha256'], 'fresh manifest differs from original pilot parent')
    import fresh_frame_recovery_v1 as fresh
    fresh.validate_amendment(amendment, manifest)
    P.require(repair['schema'] == 'overnight-fresh-frame-recovery-chain-v1' and
        repair['amendment_sha256'] == amendment['sha256'],
        'reader recovery receipt is not bound to the verified fresh amendment')
    # A recovery receipt must identify the same immutable generation manifest.
    identities = [repair.get('manifest_sha256'), repair.get('generation_manifest_sha256'),
                  repair.get('binding', {}).get('manifest_sha256')]
    P.require(manifest['sha256'] in identities, 'reader recovery receipt lacks the exact fresh manifest binding')
    source = P.DOC / ('overnight-submissions-v2-' + manifest['sha256'][:16])
    root = P.DOC / ('utility-production-root-submissions-v2-' + config['sha256'][:16] + '-' + reader_job)
    root.mkdir(exist_ok=True)
    own = P.DOC / ('utility-price-pilot-attachments-v5-' + config['pilot_proposal_sha256'][:16] + '-' + reader_job)
    own.mkdir(exist_ok=True)
    chain_dir = P.DOC / ('utility-production-submissions-v1-' + config['sha256'][:16])
    # The failed root never reached a successor. Immutable production receipts
    # would indicate a competing recovery and must not be overwritten.
    if not (root / 'CHAIN.json').exists() and not (root / 'RECOVERY.json').exists():
        P.require(not any((chain_dir / name).exists() for name in
            ('follow-pilot-generation-CHAIN.json', 'follow-pilot-analysis-CHAIN.json',
             'after-pilot-CHAIN.json', 'INITIAL_CHAIN.json')),
            'production already has a successor; reconcile exact recovery instead of duplicating')
    plan_path = root / 'PILOT_RECOVERY_PLAN.json'
    plan = P.save(plan_path, {'schema': 'utility-pilot-operational-recovery-v5',
        'proposal_path': config['pilot_proposal_path'], 'proposal_sha256': config['pilot_proposal_sha256'],
        'manifest_sha256': manifest['sha256'], 'measurement_source_dir': str(source),
        'attachment_dir': str(own), 'prior_generation_attachment_path': str(prior_path),
        'prior_generation_attachment_sha256': prior['sha256'],
        'fresh_generation_dispatch_job': prior['fresh_generation_dispatch_job'],
        'reader_recovery_chain_path': str(reader_recovery_chain_path),
        'reader_recovery_chain_sha256': repair['sha256'], 'reader_dispatch_key': reader_dispatch_key,
        'reader_dispatch_job': reader_job, 'reader_amendment_path': str(amendment_path),
        'reader_amendment_sha256': amendment['sha256'],
        'guarded_selector_path': str(P.SCRIPTS / 'select_utility_policy_fresh_adapter_v1.py'),
        'guarded_selector_sha256': P.U.file_sha(P.SCRIPTS / 'select_utility_policy_fresh_adapter_v1.py'),
        'operational_code_files': code_files(),
        'scope': 'Only exact fresh-reader predecessor and separate attachment/entry paths change; v4 proposal, deterministic selection, qualification, first two families and eight canonical assignments remain frozen.'})
    j1 = P.U.sealed(P.DOC / 'UTILITY_J1_CHAIN_PLAN_v2.json')
    P.require(config['offline_grading_envelope']['chain_plan_sha256'] == j1['sha256'] and
        all(P.U.file_sha(Path(path)) == digest for path, digest in j1['binding']['code_files'].items()),
        'frozen grading plan or source closure changed')
    recovery_path = root / 'RECOVERY.json'
    recovery = P.save(recovery_path, {'schema': 'utility-completion-operational-recovery-v2',
        'config_path': str(config_path), 'config_sha256': config['sha256'],
        'prior_root_sha256': old['sha256'], 'preserved_failures': failures,
        'pilot_attachment_dir': str(own), 'pilot_recovery_plan_path': str(plan_path),
        'pilot_recovery_plan_sha256': plan['sha256'], 'production_chain_dir': str(chain_dir),
        'native_saldo_path': '/cineca/bin/saldo', 'operational_code_files': code_files(),
        'grading_plan_sha256': j1['sha256'],
        'scope': 'Finite operational recovery, native PATH repair and restored exact predecessors. Every production phase retains live saldo, project queued commitments, frozen full pricing and bounded recovery checks. No cached account balance substitutes for a live check.'})
    return root, config, plan_path, plan, recovery_path, recovery


def dispatch(root, config, plan_path, plan, recovery_path, recovery):
    reader = plan['reader_dispatch_job']
    dependencies, proof = dispatcher.dependency(reader, Path(plan['measurement_source_dir']) / 'MEASUREMENT_CHAIN.json')
    if not dependencies:
        # Completed/purged reader dispatchers must also have their exact saved
        # analysis child, attempted receipt, canonical manifest and output.
        manifest = P.U.sealed(P.DOC / 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json')
        proposal = P.U.sealed(Path(plan['proposal_path']))
        proof['sealed_reader_artifacts'] = pilot.completed_evidence('reader_dispatch', reader,
            proposal, manifest, Path(plan['measurement_source_dir']))
    env = os.environ.copy()
    env['PATH'] = '/cineca/bin:' + env.get('PATH', '')
    env.update(UTILITY_PILOT_PROPOSAL=plan['proposal_path'], UTILITY_ATTACH_PHASE='generation',
        UTILITY_FRESH_DISPATCH_JOB=plan['fresh_generation_dispatch_job'],
        UTILITY_PILOT_OPERATIONAL_RECOVERY=str(plan_path),
        UTILITY_PRODUCTION_CONFIG=str(recovery['config_path']),
        UTILITY_PILOT_ATTACHMENT=plan['attachment_dir'],
        UTILITY_PRODUCTION_PHASE='follow-pilot-generation',
        UTILITY_PRODUCTION_OPERATIONAL_RECOVERY=str(recovery_path))
    binding = {'operational_recovery_sha256': recovery['sha256'], 'config_sha256': config['sha256'],
        'pilot_operational_recovery_sha256': plan['sha256'], 'pilot_proposal_sha256': plan['proposal_sha256']}
    def submit(name, arguments, identity):
        previous = root / (name + '.json')
        if previous.exists():
            arguments = P.U.sealed(previous)['arguments']
        return shared.submit(root, name, arguments, env, identity)
    attach = submit('pilot-generation-attach', [*dependencies, '--job-name=utility-pilot-recovery-v5',
        str(P.SCRIPTS / 'attach_utility_price_pilot_v5.sbatch')], binding)
    next_dependencies, attach_proof = dispatcher.dependency(attach,
        Path(plan['attachment_dir']) / 'GENERATION_ATTACHMENT.json', proposal_sha=plan['proposal_sha256'])
    follow = submit('production-follow', [*next_dependencies, '--job-name=utility-production-follow-v2',
        str(P.SCRIPTS / 'dispatch_utility_production_v2.sbatch')], {**binding, 'pilot_attach_job': attach})
    grade_dependencies, grade_proof = dispatcher.dependency(follow,
        Path(recovery['production_chain_dir']) / 'follow-pilot-generation-CHAIN.json')
    grading = submit('grading-follow', [*grade_dependencies, '--job-name=utility-grading-follow-v2-recovery',
        str(P.SCRIPTS / 'utility_j1_attach_v2.sbatch'), '--chain-dir', recovery['production_chain_dir']],
        {**binding, 'production_follow_job': follow, 'grading_plan_sha256': recovery['grading_plan_sha256']})
    return P.save(root / 'CHAIN.json', {'schema': 'utility-completion-root-chain-v2', **binding,
        'pilot_generation_attachment_job': attach, 'production_follow_job': follow,
        'grading_follow_job': grading, 'pilot_attachment_dir': plan['attachment_dir'],
        'production_chain_dir': recovery['production_chain_dir'], 'reader_predecessor': proof,
        'pilot_predecessor': attach_proof, 'production_predecessor': grade_proof,
        'scope': recovery['scope']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reader-recovery-chain', type=Path, required=True)
    parser.add_argument('--reader-dispatch-key', default='reader_dispatch_job')
    parser.add_argument('--reader-amendment', type=Path,
        default=P.DOC / 'OVERNIGHT_FRESH_FRAME_RECOVERY_AMENDMENT_v1.json')
    parser.add_argument('--submit', action='store_true', help='Submit the already authorized finite recovery chain')
    args = parser.parse_args()
    P.require(socket.gethostname().endswith('.leonardo.local') and
        subprocess.run(['id', '-un'], check=True, capture_output=True, text=True).stdout.strip() == 'lmolfett',
        'wrong cluster or account')
    root, config, plan_path, plan, recovery_path, recovery = prepare(args.config.resolve(),
        args.reader_recovery_chain.resolve(), args.reader_dispatch_key, args.reader_amendment.resolve())
    with (root / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'CHAIN.json').exists():
            result = P.U.sealed(root / 'CHAIN.json')
            P.require(result['operational_recovery_sha256'] == recovery['sha256'], 'existing recovery chain differs')
            print(json.dumps({'chain': str(root / 'CHAIN.json'), 'sha256': result['sha256'],
                'status': 'ALREADY_SUBMITTED'}), flush=True)
            return
        snapshot = live_preflight(root, config)
        for name in FILES:
            if name.endswith('.sbatch'):
                subprocess.run(['bash', '-n', str(P.SCRIPTS / name)], check=True)
        if not args.submit:
            print(json.dumps({'recovery': str(recovery_path), 'sha256': recovery['sha256'],
                'pilot_attachment_dir': plan['attachment_dir'], 'live_preflight_sha256': snapshot['sha256'],
                'status': 'PREPARED_NO_SUBMISSION'}), flush=True)
            return
        result = dispatch(root, config, plan_path, plan, recovery_path, recovery)
        print(json.dumps({'chain': str(root / 'CHAIN.json'), 'sha256': result['sha256'],
            'pilot_generation_attachment_job': result['pilot_generation_attachment_job'],
            'production_follow_job': result['production_follow_job'],
            'grading_follow_job': result['grading_follow_job']}), flush=True)


if __name__ == '__main__':
    main()
