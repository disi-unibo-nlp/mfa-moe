"""Attach the authorized, fully priced utility production and grading chain."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time

import utility_production_v1 as production
import dispatch_utility_production_v1 as dispatcher
import dispatch_overnight_readers_v1 as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    production.require(socket.gethostname().endswith('.leonardo.local') and
        subprocess.run(['id', '-un'], check=True, capture_output=True, text=True).stdout.strip() == 'lmolfett',
        'wrong cluster or account')
    config = production.U.sealed(args.config)
    production.validate_config(config)
    production.require(config['dispatch_on_pass'] is True, 'production dispatch is not enabled')
    proposal = production.U.sealed(Path(config['pilot_proposal_path']))
    attachment = shared.DOC / ('utility-price-pilot-attachments-v4-' + proposal['sha256'][:16])
    root = shared.DOC / ('utility-production-root-submissions-v1-' + config['sha256'][:16])
    chain_dir = shared.DOC / ('utility-production-submissions-v1-' + config['sha256'][:16])
    j1 = production.U.sealed(shared.DOC / 'UTILITY_J1_CHAIN_PLAN_v2.json')
    envelope = production.U.sealed(Path(config['offline_grading_envelope_path']))
    production.require(envelope['chain_plan_sha256'] == j1['sha256'] and
        all(production.U.file_sha(Path(path)) == digest for path, digest in j1['binding']['code_files'].items()),
        'grading source closure differs')
    source_chain = production.U.sealed(shared.DOC / 'overnight-additive-coverage-submissions-v4/CHAIN.json')
    fresh = [row for row in source_chain['records'] if row['stage'] == 'FRESH']
    production.require(len(fresh) == 1, 'fresh pilot attachment is ambiguous')
    pilot_parent = fresh[0]['attachments']['utility_price_pilot']
    dependencies, proof = dispatcher.dependency(pilot_parent, attachment / 'GENERATION_ATTACHMENT.json',
                                                proposal_sha=proposal['sha256'])
    wrappers = {name: shared.REPO / 'scripts/experimental_resume' / filename for name, filename in (
        ('production', 'dispatch_utility_production_v1.sbatch'), ('grading', 'utility_j1_attach_v2.sbatch'))}
    for wrapper in wrappers.values():
        subprocess.run(['bash', '-n', str(wrapper)], check=True)
    root.mkdir(exist_ok=True)
    binding = {'config_sha256': config['sha256'], 'pilot_proposal_sha256': proposal['sha256'],
        'grading_plan_sha256': j1['sha256'], 'grading_envelope_sha256': envelope['sha256'],
        'submission_source_sha256': production.U.file_sha(Path(__file__)),
        'production_chain_dir': str(chain_dir), 'pilot_attachment_dir': str(attachment)}
    with (root / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'CHAIN.json').exists():
            previous = production.U.sealed(root / 'CHAIN.json')
            production.require(all(previous[key] == value for key, value in binding.items()),
                               'existing utility completion chain differs')
            print(str(root / 'CHAIN.json'), previous['sha256'], 'ALREADY_SUBMITTED', flush=True)
            return
        records = {}
        for label, command in (
            ('balance', ['saldo', '-b', 'lmolfett']),
            ('association', ['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett',
                             'format=Account,Partition,QOS']),
            ('cpu_partition', ['scontrol', 'show', 'partition', 'lrd_all_viz']),
            ('gpu_partition', ['scontrol', 'show', 'partition', 'boost_usr_prod']),
            ('qos', ['sacctmgr', '-nP', 'show', 'qos', 'normal',
                     'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU'])):
            records[label] = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        production.require('iscrc_miosr' in records['association'], 'project association absent')
        shared.save(root / ('PREFLIGHT-' + str(time.time_ns()) + '.json'), {'schema': 'utility-completion-root-preflight-v1',
            **binding, 'records': records, 'pilot_predecessor': proof})
        def submit(name, arguments, environment, identity):
            prior = root / (name + '.json')
            if prior.exists():
                # A completed parent can change the needed dependency syntax.
                # Preserve the already submitted exact command on crash resume.
                arguments = production.U.sealed(prior)['arguments']
            return shared.submit(root, name, arguments, environment, identity)
        env = os.environ.copy()
        env.update(UTILITY_PRODUCTION_CONFIG=str(args.config.resolve()),
            UTILITY_PILOT_ATTACHMENT=str(attachment), UTILITY_PRODUCTION_PHASE='follow-pilot-generation')
        follow = submit('production-follow', [*dependencies,
            '--job-name=utility-production-follow-v1', str(wrappers['production'])], env, binding)
        next_dependencies, next_proof = dispatcher.dependency(follow,
            chain_dir / 'follow-pilot-generation-CHAIN.json')
        grading = submit('grading-follow', [*next_dependencies,
            '--job-name=utility-grading-follow-v2', str(wrappers['grading']), '--chain-dir', str(chain_dir)],
            os.environ.copy(), {**binding, 'production_follow_job': follow})
        result = shared.save(root / 'CHAIN.json', {'schema': 'utility-completion-root-chain-v1', **binding,
            'production_follow_job': follow, 'grading_follow_job': grading,
            'production_predecessor_proof': next_proof,
            'scope': 'Finite CPU followers. Production GPU submission requires successful fixed pilot, complete measured price and live account reconciliation. Grading requires the reconciled assigned index and exact prompt pricing.'})
        print(str(root / 'CHAIN.json'), result['sha256'], follow, grading, flush=True)


if __name__ == '__main__':
    main()
