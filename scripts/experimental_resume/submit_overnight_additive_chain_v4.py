"""Attach the minimal coverage supplement and all authorized downstream stages."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import subprocess
import dispatch_overnight_readers_v1 as d

SCRIPTS = d.REPO / 'scripts/experimental_resume'
DIRECTORY = d.DOC / 'overnight-additive-coverage-submissions-v4'


def wrapper(name):
    path = SCRIPTS / name
    subprocess.run(['bash', '-n', str(path)], check=True)
    return path


def main():
    amendment = d.base.sealed(d.DOC / 'OVERNIGHT_OPERATOR_ADDITIVE_COVERAGE_AMENDMENT_v4.json')
    inventory = amendment.get('source_files', amendment.get('code_files'))
    if not inventory:
        raise ValueError('amendment source inventory missing')
    for path, digest in inventory.items():
        if d.base.file_sha(path) != digest:
            raise ValueError('frozen additive qualification source changed: ' + path)
    enrollment = d.base.sealed(d.DOC / 'MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_v1.json')
    account = subprocess.run(['sacct', '-X', '-nP', '-j', '59344223', '--format=JobID,State,ExitCode'],
                             check=True, capture_output=True, text=True).stdout
    if ['59344223','COMPLETED','0:0'] not in [line.strip().split('|') for line in account.splitlines()] or enrollment['status'] != 'READY_EXACT_NATIVE_PREFIXES':
        raise ValueError('frozen enrollment prerequisite not verified')
    binding = {'amendment_sha256': amendment['sha256'], 'enrollment_sha256': enrollment['sha256'],
               'enrollment_job': '59344223', 'enrollment_sacct': account,
               'submit_code_sha256': d.base.file_sha(__file__), 'shared_submit_sha256': d.base.file_sha(d.__file__)}
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        combined = d.submit(DIRECTORY, 'combined-qualification',
            ['--dependency=afterok:59348871', str(wrapper('combine_overnight_operator_qualification_v4.sbatch'))], os.environ.copy(), binding)
        records = []
        for stage in ('C', 'FRESH'):
            env = os.environ.copy(); env.update(OVERNIGHT_STAGE=stage, OVERNIGHT_DISPATCH_PHASE='generation')
            prep = d.submit(DIRECTORY, stage.lower() + '-prepare', ['--dependency=afterok:' + combined,
                '--job-name=st-overnight-' + stage + '-prepare-v4', str(wrapper('prepare_overnight_operator_stage_v4.sbatch'))],
                env, {**binding, 'stage': stage})
            dispatcher = d.submit(DIRECTORY, stage.lower() + '-generation-dispatch', ['--dependency=afterok:' + prep,
                '--job-name=st-overnight-' + stage + '-launch-v4', str(wrapper('dispatch_overnight_stage_v4.sbatch'))],
                env, {**binding, 'stage': stage, 'preparation_job': prep})
            attachments = {}
            for name, script, arguments in (
                ('dense', 'submit_generated_dense_chain_v1.sbatch', ['--stage', stage]),
                ('paper', 'routing_paper_v3.sbatch', ['attach', stage]),
                ('engagement', 'submit_routing_first_stage_v2.sbatch', ['--stage', stage])):
                path = wrapper(script)
                attachments[name] = d.submit(DIRECTORY, stage.lower() + '-' + name + '-attach',
                    ['--dependency=afterok:' + dispatcher, '--job-name=st-' + name + '-' + stage + '-attach-v4', str(path), *arguments],
                    os.environ.copy(), {**binding, 'stage': stage, 'generation_dispatch_job': dispatcher,
                                       'wrapper_sha256': d.base.file_sha(path)})
            if stage == 'FRESH':
                from attach_utility_price_pilot_v4 import validate_sources
                proposal_path = d.DOC / 'UTILITY_RUNTIME_PILOT_PROPOSAL_v4.json'
                proposal = d.base.sealed(proposal_path); validate_sources(proposal)
                env.update(UTILITY_PILOT_PROPOSAL=str(proposal_path), UTILITY_ATTACH_PHASE='generation',
                           UTILITY_FRESH_DISPATCH_JOB=dispatcher)
                path = wrapper('attach_utility_price_pilot_v4.sbatch')
                attachments['utility_price_pilot'] = d.submit(DIRECTORY, 'utility-pilot-attach',
                    ['--dependency=afterok:' + dispatcher, '--job-name=utility-pilot-attach-generation-v4', str(path)], env,
                    {**binding, 'proposal_sha256': proposal['sha256'], 'fresh_generation_dispatch_job': dispatcher,
                     'wrapper_sha256': d.base.file_sha(path)})
            records.append({'stage': stage, 'combined_qualification_job': combined, 'preparation_job': prep,
                            'generation_dispatch_job': dispatcher, 'attachments': attachments})
        d.save(DIRECTORY / 'CHAIN.json', {'schema': 'overnight-additive-qualified-chain-v4', 'binding': binding,
               'supplement_job': '59348871', 'combined_qualification_job': combined, 'records': records,
               'preservation': 'Original failed qualifications and cancelled zero-runtime descendants are retained; no original outcomes or enrollment replaced.'})


if __name__ == '__main__':
    main()
