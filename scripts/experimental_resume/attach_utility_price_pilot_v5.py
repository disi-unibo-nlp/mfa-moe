"""Restore the frozen v4 pilot to an exact repaired fresh-reader dispatcher.

The original failed attachments remain sealed. A separate v5 directory binds
the recovery receipt; policy selection, eight pilot IDs, engine qualification,
runtime work and accounting reuse the unchanged v4 scientific functions.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

import attach_utility_price_pilot_v4 as original

SCRIPTS = original.SCRIPTS
CPU_WRAPPER = SCRIPTS / 'attach_utility_price_pilot_v5.sbatch'
GPU_WRAPPER = SCRIPTS / 'run_utility_price_pilot_v5.sbatch'
GUARDED_SELECTOR = SCRIPTS / 'select_utility_policy_fresh_adapter_v1.py'
INSTALLED = None


def validate_recovery(path):
    plan = original.utility.sealed(Path(path))
    original.require(plan['schema'] == 'utility-pilot-operational-recovery-v5' and
        all(original.utility.file_sha(Path(name)) == digest
            for name, digest in plan['operational_code_files'].items()),
        'pilot operational recovery code changed')
    proposal = original.utility.sealed(Path(plan['proposal_path']))
    original.require(proposal['sha256'] == plan['proposal_sha256'], 'frozen pilot proposal changed')
    original.validate_sources(proposal)
    prior = original.utility.sealed(Path(plan['prior_generation_attachment_path']))
    repair = original.utility.sealed(Path(plan['reader_recovery_chain_path']))
    amendment = original.utility.sealed(Path(plan['reader_amendment_path']))
    original.require(prior['sha256'] == plan['prior_generation_attachment_sha256'] and
        prior['proposal_sha256'] == plan['proposal_sha256'] and
        prior['manifest_sha256'] == plan['manifest_sha256'] and
        prior['fresh_generation_dispatch_job'] == plan['fresh_generation_dispatch_job'] and
        repair['sha256'] == plan['reader_recovery_chain_sha256'] and
        str(repair[plan['reader_dispatch_key']]) == plan['reader_dispatch_job'] and
        amendment['sha256'] == plan['reader_amendment_sha256'] and
        repair['amendment_sha256'] == amendment['sha256'] and
        plan['guarded_selector_path'] == str(GUARDED_SELECTOR) and
        plan['guarded_selector_sha256'] == original.utility.file_sha(GUARDED_SELECTOR),
        'pilot restored reader predecessor or amendment changed')
    return plan


def validate_selected_policy(plan, manifest, own):
    import fresh_frame_recovery_v1 as fresh
    amendment = original.utility.sealed(Path(plan['reader_amendment_path']))
    analysis = original.utility.sealed(original.DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ANALYSIS.json')
    policy = original.utility.sealed(own / 'SELECTED_POLICY.json')
    provenance = fresh.validate_analysis(manifest, analysis, amendment)
    original.require(policy.get('recovery_provenance') == provenance and
        policy.get('selection_operational_entry_sha256') == plan['guarded_selector_sha256'],
        'selected policy lacks guarded fresh recovery provenance')


def install(plan):
    global INSTALLED
    if INSTALLED is not None:
        original.require(INSTALLED == plan['sha256'], 'another pilot recovery is already installed')
        return original
    old_paths, old_submit = original.paths, original.submit
    old_run, old_inputs = original.subprocess.run, original.pilot_inputs

    def paths(proposal):
        manifest_path, manifest, source, _ = old_paths(proposal)
        original.require(manifest['sha256'] == plan['manifest_sha256'] and
            str(source) == plan['measurement_source_dir'], 'restored fresh manifest or measurement source differs')
        own = Path(plan['attachment_dir'])
        own.mkdir(exist_ok=True)
        return manifest_path, manifest, source, own

    def submit(directory, name, arguments, env, binding):
        replacements = {
            str(SCRIPTS / 'attach_utility_price_pilot_v4.sbatch'): str(CPU_WRAPPER),
            str(SCRIPTS / 'run_utility_price_pilot_v4.sbatch'): str(GPU_WRAPPER)}
        arguments = [replacements.get(argument, argument) for argument in arguments]
        child = dict(env)
        child['UTILITY_PILOT_OPERATIONAL_RECOVERY'] = os.environ['UTILITY_PILOT_OPERATIONAL_RECOVERY']
        return old_submit(directory, name, arguments, child,
            {**binding, 'pilot_operational_recovery_sha256': plan['sha256']})

    def run(arguments, *args, **kwargs):
        # v4 constructs the selector subprocess explicitly. Replace that one
        # program path while retaining every frozen argument and subprocess.
        if isinstance(arguments, (list, tuple)):
            arguments = [str(GUARDED_SELECTOR) if argument == str(SCRIPTS / 'select_utility_policy_v2.py')
                         else argument for argument in arguments]
        return old_run(arguments, *args, **kwargs)

    def pilot_inputs(proposal, own, manifest):
        validate_selected_policy(plan, manifest, own)
        return old_inputs(proposal, own, manifest)

    original.paths, original.submit = paths, submit
    original.subprocess.run, original.pilot_inputs = run, pilot_inputs
    INSTALLED = plan['sha256']
    return original


def generation(plan, proposal, proposal_path):
    _, manifest, source, own = original.paths(proposal)
    original.validate_sources(proposal)
    preserved = original.utility.sealed(source / 'GENERATION_CHAIN.json')
    original.require(preserved['schema'] == 'overnight-generation-chain-v2' and
        preserved['binding']['manifest_sha256'] == manifest['sha256'], 'preserved fresh generation chain differs')
    env = os.environ.copy()
    env.update(UTILITY_PILOT_PROPOSAL=str(proposal_path), UTILITY_ATTACH_PHASE='analysis')
    binding = {'proposal_sha256': proposal['sha256'], 'manifest_sha256': manifest['sha256'],
        'fresh_generation_dispatch_job': plan['fresh_generation_dispatch_job'],
        'generation_chain_sha256': preserved['sha256'],
        'reader_recovery_chain_sha256': plan['reader_recovery_chain_sha256']}
    with (own / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.active_preflight(own, 'generation')
        dependencies = original.dependency_arguments([('reader_dispatch', plan['reader_dispatch_job'])],
            proposal, manifest, source, own, 'generation')
        child = original.submit(own, 'attach-analysis', [*dependencies,
            '--job-name=utility-pilot-attach-analysis-v5', str(CPU_WRAPPER)], env, binding)
        original.shared.save(own / 'GENERATION_ATTACHMENT.json', {
            'schema': 'utility-generation-attachment-v4',
            'proposal_sha256': proposal['sha256'], 'manifest_sha256': manifest['sha256'],
            'fresh_generation_dispatch_job': plan['fresh_generation_dispatch_job'],
            'reader_dispatch_job': plan['reader_dispatch_job'], 'next_attachment_job': child,
            'pilot_operational_recovery_sha256': plan['sha256']})


def main():
    path = os.environ.get('UTILITY_PILOT_OPERATIONAL_RECOVERY')
    original.require(path, 'sealed pilot operational recovery required')
    plan = validate_recovery(path)
    original.require(os.environ.get('UTILITY_PILOT_PROPOSAL') == plan['proposal_path'] and
        os.environ.get('UTILITY_FRESH_DISPATCH_JOB') == plan['fresh_generation_dispatch_job'],
        'pilot runtime proposal or preserved fresh generation ID differs')
    install(plan)
    if os.environ.get('UTILITY_ATTACH_PHASE') == 'generation':
        original.require(os.environ.get('SLURM_JOB_ID') and
            os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz', 'pilot recovery requires CPU Slurm')
        proposal = original.utility.sealed(Path(plan['proposal_path']))
        generation(plan, proposal, Path(plan['proposal_path']))
    else:
        original.main()


if __name__ == '__main__':
    main()
