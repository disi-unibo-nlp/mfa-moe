"""Operational PATH recovery for the immutable utility production v1 chain.

The frozen configuration, pricing, science and GPU worker stay in v1. Every
CPU successor uses this entry and records the sealed operational amendment.
Native saldo must remain callable; no cached balance replaces the live check.
"""
from __future__ import annotations

import os
from pathlib import Path

import dispatch_utility_production_v1 as original

P = original.P
SCRIPTS = P.SCRIPTS
WRAPPER = SCRIPTS / 'dispatch_utility_production_v2.sbatch'
SALDO = Path('/cineca/bin/saldo')
INSTALLED = None


def validate_recovery(path):
    recovery = P.U.sealed(Path(path))
    P.require(recovery['schema'] == 'utility-completion-operational-recovery-v2' and
              recovery['native_saldo_path'] == str(SALDO) and
              all(P.U.file_sha(Path(name)) == digest
                  for name, digest in recovery['operational_code_files'].items()),
              'operational recovery source or native accounting path changed')
    config = P.U.sealed(Path(recovery['config_path']))
    P.require(config['sha256'] == recovery['config_sha256'], 'recovery configuration changed')
    P.validate_config(config)
    pilot = P.U.sealed(Path(recovery['pilot_recovery_plan_path']))
    P.require(pilot['sha256'] == recovery['pilot_recovery_plan_sha256'] and
              pilot['proposal_sha256'] == config['pilot_proposal_sha256'] and
              pilot['attachment_dir'] == recovery['pilot_attachment_dir'],
              'sealed pilot operational recovery differs')
    return recovery


def install(recovery):
    global INSTALLED
    if INSTALLED is not None:
        P.require(INSTALLED == recovery['sha256'], 'another operational recovery is already installed')
        return original
    P.require(SALDO.is_file() and os.access(SALDO, os.X_OK),
              'native saldo unavailable; retain fail-closed live accounting gate')
    os.environ['PATH'] = str(SALDO.parent) + os.pathsep + os.environ.get('PATH', '')
    old_gpu = original.gpu_submit

    def bound(binding):
        return {**binding, 'operational_recovery_sha256': recovery['sha256']}

    def cpu_submit(directory, label, env, binding, dependencies=()):
        return original.shared.submit(directory, label, [*dependencies,
            '--job-name=utility-prod-' + label + '-v2', str(WRAPPER)], env, bound(binding))

    def gpu_submit(directory, label, manifest, config, env, binding, indices):
        return old_gpu(directory, label, manifest, config, env, bound(binding), indices)

    original.cpu_submit = cpu_submit
    original.gpu_submit = gpu_submit
    INSTALLED = recovery['sha256']
    return original


def main():
    path = os.environ.get('UTILITY_PRODUCTION_OPERATIONAL_RECOVERY')
    P.require(path, 'sealed operational recovery record required')
    recovery = validate_recovery(path)
    P.require(os.environ.get('UTILITY_PRODUCTION_CONFIG') == recovery['config_path'] and
              os.environ.get('UTILITY_PILOT_ATTACHMENT') == recovery['pilot_attachment_dir'],
              'runtime configuration or pilot attachment differs from recovery')
    P.require(os.environ.get('SLURM_JOB_ID'), 'operational recovery requires authorized Slurm')
    directory = P.DOC / ('utility-production-submissions-v1-' + recovery['config_sha256'][:16])
    P.save(directory / ('OPERATIONAL_RECOVERY-' + os.environ['SLURM_JOB_ID'] + '.json'), {
        'schema': 'utility-production-operational-entry-v2',
        'operational_recovery_sha256': recovery['sha256'],
        'phase': os.environ.get('UTILITY_PRODUCTION_PHASE'),
        'native_saldo_path': str(SALDO),
        'scope': 'PATH and finite CPU successor entry only; immutable v1 live balance, commitments, pricing, configuration and qualified science are unchanged.'})
    install(recovery).main()


if __name__ == '__main__':
    main()
