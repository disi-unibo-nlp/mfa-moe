"""One-shot dependency attachment and all-cost accounting for the 8-cell pilot.

Each phase reads a completed predecessor receipt, attaches one next phase, and
exits. There is no polling process, resident service or automatic GPU retry.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess

import dispatch_overnight_readers_v1 as shared
import utility_scout_v1 as utility

REPO, DOC = shared.REPO, shared.DOC
SCRIPTS = REPO / 'scripts/experimental_resume'
def require(ok, message):
    if not ok:
        raise ValueError(message)


def job_id(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9]+', value), 'invalid frozen job ID')
    return value


def first_two_assignments(plan):
    utility.validate_plan(plan)
    families = plan['family_order'][:2]
    rows = [row for row in plan['assignments'] if row['family'] in families]
    require(len(rows) == 8 and len({row['uid'] for row in rows}) == 8 and
            all({(r['arm'], r['seed']) for r in rows if r['family'] == family} ==
                {('native', 0), ('native', 1), ('frozen_policy', 0), ('frozen_policy', 1)}
                for family in families), 'pilot does not preserve first-two-family factorial')
    return families, rows


def validate_sources(proposal):
    require(proposal['schema'] == 'utility-runtime-pilot-proposal-v4' and
            proposal['pilot_families'] == 2 and proposal['assigned_cells'] == 8 and
            proposal['gpus'] == 4 and proposal['wall_seconds'] == 28800 and
            proposal['allocation_gpu_hour_ceiling'] == 32 and
            all(utility.file_sha(Path(path)) == sha for path, sha in proposal['attachment_files'].items()),
            'attachment proposal/code/resources changed')
    previous = utility.sealed(Path(proposal['supersedes_proposal_path']))
    require(previous['sha256'] == proposal['supersedes_proposal_sha256'] and
            previous['schema'] == 'utility-runtime-pilot-proposal-v3' and
            utility.file_sha(SCRIPTS / 'dispatch_overnight_readers_v1.py') == proposal['corrected_dispatcher_sha256'],
            'preserved v3 proposal or reviewed dispatcher correction changed')
    plan = utility.sealed(Path(proposal['utility_plan_path']))
    require(plan['sha256'] == proposal['utility_plan_sha256'], 'frozen utility enrollment changed')
    first_two_assignments(plan)
    qual_receipt = utility.sealed(Path(proposal['qualification_submission_path']))
    require(qual_receipt['sha256'] == proposal['qualification_submission_sha256'] and
            qual_receipt['qualification_job'] == proposal['qualification_job'] and
            str(Path(qual_receipt['binding']['output']) / 'QUALIFICATION.json') == proposal['qualification_result'],
            'qualification submission or exact output path changed')
    return plan


def active_preflight(directory, phase):
    require(pwd.getpwuid(os.getuid()).pw_name == 'lmolfett', 'wrong LEONARDO user')
    outputs = {}
    commands = [(['scontrol', 'show', 'partition', 'lrd_all_viz'], 'cpu_partition'),
                (['scontrol', 'show', 'partition', 'boost_usr_prod'], 'gpu_partition'),
                (['sacctmgr', '--noheader', '--parsable2', 'show', 'assoc', 'where', 'user=lmolfett',
                  'format=User,Account,Partition,QOS'], 'associations')]
    for command, name in commands:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        outputs[name] = result.stdout
    require('iscrc_miosr' in outputs['associations'], 'live account association missing')
    require('PartitionName=lrd_all_viz' in outputs['cpu_partition'] and
            'PartitionName=boost_usr_prod' in outputs['gpu_partition'], 'live partition inspection differs')
    shared.save(directory / (phase + '-live-preflight-' + os.environ['SLURM_JOB_ID'] + '.json'), {
        'schema': 'utility-pilot-live-preflight-v4', 'hostname': socket.gethostname(),
        'user': 'lmolfett', 'dispatcher_job': os.environ['SLURM_JOB_ID'], 'outputs': outputs})


def submit(directory, name, arguments, env, binding):
    identifier = shared.submit(directory, name, arguments, env, binding)
    # Existing receipt reuse must also verify a prior interrupted verification.
    verification = directory / (name + '.verified.json')
    if not verification.exists():
        result = subprocess.run(['scontrol', 'show', 'job', identifier], check=True, capture_output=True, text=True)
        require(f'JobId={identifier}' in result.stdout and 'UserId=lmolfett(' in result.stdout,
                'reused submitted job did not verify')
        shared.save(verification, {'schema': 'overnight-job-verification-v1', 'job_id': identifier,
                                  'scontrol': result.stdout})
    require(utility.sealed(verification)['job_id'] == identifier, 'submitted job verification differs')
    return identifier


def paths(proposal):
    manifest_path = DOC / 'OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json'
    manifest = utility.sealed(manifest_path)
    require(manifest['design_sha256'] == proposal['fresh_design_sha256'] and manifest['horizon'] == 1024,
            'fresh generation changed its frozen comparison')
    source_dir = DOC / ('overnight-submissions-v2-' + manifest['sha256'][:16])
    own = DOC / ('utility-price-pilot-attachments-v4-' + proposal['sha256'][:16])
    own.mkdir(exist_ok=True)
    return manifest_path, manifest, source_dir, own


def pilot_inputs(proposal, own, manifest):
    plan = validate_sources(proposal)
    qual = utility.sealed(Path(proposal['qualification_result']))
    require(qual['schema'] == 'utility-pair-engine-qualification-v2' and qual['status'] == 'PASS_ENGINEERING',
            'utility engineering qualification has not passed')
    qual_binding = utility.sealed(Path(proposal['qualification_result']).parent / 'BINDING.json')
    require(qual_binding['sha256'] == qual['binding_sha256'] and
            qual_binding['prepared_plan_sha256'] == proposal['engineering_plan_sha256'],
            'utility qualification is not the assigned engineering probe')
    policy = utility.sealed(own / 'SELECTED_POLICY.json')
    pilot = utility.sealed(own / 'PILOT_MANIFEST.json')
    families, assignments = first_two_assignments(plan)
    require(policy['mechanism_manifest_sha256'] == manifest['sha256'] and
            policy['status'] == 'FROZEN_LOCAL_ACTIONS_CONTROLLER_AND_ENGINE_REQUIRED' and
            pilot['schema'] == 'utility-price-pilot-manifest-v2' and pilot['plan_sha256'] == plan['sha256'] and
            pilot['policy_sha256'] == policy['sha256'] and pilot['qualification_sha256'] == qual['sha256'] and
            pilot['family_order'] == families and pilot['expected_assignments'] == 8 and
            [row['assignment'] for row in pilot['rows']] == assignments and
            pilot['maximum_tokens_per_assignment'] == 16384 and pilot['allocated_gpus_per_job'] == 4,
            'pilot changed its first two families, exact canonical UIDs, policy, or resources')
    return pilot


def completed_evidence(kind, identifier, proposal, manifest, source):
    """Require the assigned producer's sealed result before retiring its edge."""
    evidence = {}
    def read(path):
        value = utility.sealed(path)
        evidence[str(path)] = value['sha256']
        return value
    if kind == 'reader_dispatch':
        chain = read(source / 'MEASUREMENT_CHAIN.json')
        attempted = read(source / 'analysis.attempt.json')
        submitted = read(source / 'analysis.json')
        require(chain['schema'] == 'overnight-measurement-chain-v2' and
                chain['binding']['manifest_sha256'] == manifest['sha256'] and
                attempted['dispatcher_job_id'] == identifier and
                attempted['binding'] == submitted['binding'] == chain['binding'] and
                submitted['job_id'] == chain['analysis_job'] and
                chain['analysis_output'] == str(DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2'),
                'completed reader dispatcher lacks its matching sealed successor')
    elif kind == 'analysis':
        chain = read(source / 'MEASUREMENT_CHAIN.json')
        submitted = read(source / 'analysis.json')
        analysis = read(DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ANALYSIS.json')
        assigned = read(DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ASSIGNED_RESULTS.json')
        require(chain['analysis_job'] == submitted['job_id'] == identifier and
                chain['binding'] == submitted['binding'] and
                chain['binding']['manifest_sha256'] == manifest['sha256'] and
                analysis['schema'] == 'overnight-semantic-itt-v2' and
                analysis['manifest_sha256'] == assigned['manifest_sha256'] == manifest['sha256'] and
                analysis['assigned_results_sha256'] == assigned['sha256'] and
                analysis['frame_sha256'] == chain['binding']['frame_sha256'] and
                analysis['price_sha256'] == chain['binding']['price_sha256'] and
                analysis['assigned'] == len(assigned['records']) == manifest['expected_requests'] and
                analysis['horizon'] == manifest['horizon'],
                'completed analysis lacks its matching complete sealed result')
    elif kind == 'qualification':
        qual = read(Path(proposal['qualification_result']))
        binding = read(Path(proposal['qualification_result']).parent / 'BINDING.json')
        require(qual['schema'] == 'utility-pair-engine-qualification-v2' and
                qual['status'] == 'PASS_ENGINEERING' and
                qual['binding_sha256'] == binding['sha256'] and
                binding['job_id'] == identifier == proposal['qualification_job'] and
                binding['prepared_plan_sha256'] == proposal['engineering_plan_sha256'],
                'completed qualification lacks its matching sealed PASS')
    else:
        raise ValueError('unrecognized completed prerequisite kind')
    return evidence


def dependency_arguments(specifications, proposal, manifest, source, own, phase):
    """Use active Slurm edges; completed/purged jobs need accounting plus artifacts."""
    identifiers = [job_id(identifier) for _, identifier in specifications]
    require(len(set(identifiers)) == len(identifiers), 'duplicate prerequisite job ID')
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(identifiers),
        '--format=JobIDRaw,State%40,ExitCode'], check=True, capture_output=True, text=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    active, decisions = [], []
    for kind, identifier in specifications:
        matches = [row for row in rows if row[0] == identifier]
        require(len(matches) == 1 and len(matches[0]) >= 3,
                'prerequisite accounting missing or ambiguous: ' + identifier)
        _, state, exit_code, *_ = matches[0]
        evidence = {}
        if state == 'COMPLETED':
            require(exit_code == '0:0', 'completed prerequisite exit code failed: ' + identifier)
            evidence = completed_evidence(kind, identifier, proposal, manifest, source)
            decision = 'SATISFIED_ACCOUNTING_AND_SEALED_ARTIFACT_OMIT_SLURM_EDGE'
        elif state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'):
            active.append(identifier)
            decision = 'RETAIN_ACTIVE_AFTEROK_EDGE'
        else:
            raise ValueError('prerequisite failed or unknown state: ' + identifier + ' ' + state)
        decisions.append({'kind': kind, 'job_id': identifier, 'state': state, 'exit_code': exit_code,
                          'decision': decision, 'sealed_artifacts': evidence})
    shared.save(own / (phase + '-prerequisites-' + os.environ['SLURM_JOB_ID'] + '.json'), {
        'schema': 'utility-prerequisite-resolution-v4', 'proposal_sha256': proposal['sha256'],
        'manifest_sha256': manifest['sha256'], 'accounting': result.stdout, 'decisions': decisions})
    return ['--dependency=afterok:' + ':'.join(active)] if active else []


def phase_attach(proposal, proposal_path, phase):
    validate_sources(proposal)
    manifest_path, manifest, source, own = paths(proposal)
    wrapper = SCRIPTS / 'attach_utility_price_pilot_v4.sbatch'
    env = os.environ.copy()
    env['UTILITY_PILOT_PROPOSAL'] = str(proposal_path)
    binding = {'proposal_sha256': proposal['sha256'], 'manifest_sha256': manifest['sha256'],
               'fresh_generation_dispatch_job': job_id(os.environ['UTILITY_FRESH_DISPATCH_JOB'])}
    with (own / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active_preflight(own, phase)
        if phase == 'generation':
            chain = utility.sealed(source / 'GENERATION_CHAIN.json')
            require(chain['schema'] == 'overnight-generation-chain-v2' and
                    chain['binding']['manifest_sha256'] == manifest['sha256'], 'fresh generation chain differs')
            upstream = job_id(chain['reader_dispatch_job'])
            env['UTILITY_ATTACH_PHASE'] = 'analysis'
            dependencies = dependency_arguments([('reader_dispatch', upstream)],
                proposal, manifest, source, own, phase)
            next_job = submit(own, 'attach-analysis', [*dependencies,
                '--job-name=utility-pilot-attach-analysis-v4', str(wrapper)], env,
                {**binding, 'generation_chain_sha256': chain['sha256']})
            shared.save(own / 'GENERATION_ATTACHMENT.json', {'schema': 'utility-generation-attachment-v4',
                **binding, 'reader_dispatch_job': upstream, 'next_attachment_job': next_job})
        elif phase == 'analysis':
            chain = utility.sealed(source / 'MEASUREMENT_CHAIN.json')
            require(chain['schema'] == 'overnight-measurement-chain-v2' and
                    chain['binding']['manifest_sha256'] == manifest['sha256'] and
                    chain['analysis_output'] == str(DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2'),
                    'fresh measurement/analysis chain differs')
            analysis = job_id(chain['analysis_job'])
            env['UTILITY_ATTACH_PHASE'] = 'select'
            dependencies = dependency_arguments([('analysis', analysis),
                ('qualification', job_id(proposal['qualification_job']))],
                proposal, manifest, source, own, phase)
            next_job = submit(own, 'select-policy', [*dependencies,
                '--time=00:30:00', '--cpus-per-task=2', '--mem=16G',
                '--job-name=utility-pilot-select-v4', str(wrapper)], env,
                {**binding, 'measurement_chain_sha256': chain['sha256'], 'qualification_job': proposal['qualification_job']})
            shared.save(own / 'ANALYSIS_ATTACHMENT.json', {'schema': 'utility-analysis-attachment-v4',
                **binding, 'analysis_job': analysis, 'qualification_job': proposal['qualification_job'],
                'selector_job': next_job})
        elif phase == 'select':
            import sys
            analysis_path = DOC / 'OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ANALYSIS.json'
            # Subprocesses provide fresh namespace bootstrap, with no module
            # sharing between selector and the immutable pilot preparer.
            subprocess.run([sys.executable, '-B', str(SCRIPTS / 'select_utility_policy_v2.py'),
                '--manifest', str(manifest_path), '--analysis', str(analysis_path),
                '--out', str(own / 'SELECTED_POLICY.json')], check=True)
            if not (own / 'PILOT_MANIFEST.json').exists():
                subprocess.run([sys.executable, '-B', str(SCRIPTS / 'run_utility_price_pilot_v2.py'), 'prepare',
                    '--plan', proposal['utility_plan_path'], '--policy', str(own / 'SELECTED_POLICY.json'),
                    '--qualification', proposal['qualification_result'], '--pilot-families', '2',
                    '--out', str(own / 'PILOT_MANIFEST.json')], check=True)
            pilot = pilot_inputs(proposal, own, manifest)
            output = shared.RUNS / ('utility-price-pilot-v2-' + pilot['sha256'][:16])
            env.update(UTILITY_PILOT_MANIFEST=str(own / 'PILOT_MANIFEST.json'), UTILITY_PILOT_OUT=str(output))
            gpu = submit(own, 'runtime-pilot', ['--job-name=utility-16k-price-pilot-v4',
                str(SCRIPTS / 'run_utility_price_pilot_v4.sbatch')], env,
                {**binding, 'pilot_manifest_sha256': pilot['sha256'], 'output': str(output)})
            env['UTILITY_ATTACH_PHASE'] = 'account'
            env['UTILITY_PILOT_JOB'] = gpu
            accounting = submit(own, 'pilot-account', ['--dependency=afterany:' + gpu,
                '--job-name=utility-pilot-account-v4', str(wrapper)], env,
                {**binding, 'pilot_job': gpu, 'pilot_manifest_sha256': pilot['sha256']})
            shared.save(own / 'PILOT_CHAIN.json', {'schema': 'utility-runtime-pilot-chain-v4',
                **binding, 'pilot_manifest_sha256': pilot['sha256'], 'pilot_job': gpu,
                'accounting_job': accounting, 'output': str(output), 'production_launched': False})
        else:
            raise ValueError('unknown attachment phase')


def account(proposal):
    validate_sources(proposal)
    _, manifest, _, own = paths(proposal)
    chain = utility.sealed(own / 'PILOT_CHAIN.json')
    identifier = job_id(chain['pilot_job'])
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', identifier,
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],
        check=True, capture_output=True, text=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == identifier]
    require(len(rows) == 1, 'final pilot job accounting missing or ambiguous; no fabricated cost')
    row = rows[0]
    require(row[1] not in ('RUNNING', 'PENDING', 'COMPLETING'), 'pilot accounting is not final')
    tres = dict(item.split('=', 1) for item in row[5].split(',') if '=' in item)
    require(int(tres.get('gres/gpu', -1)) == 4, 'pilot allocated GPU count differs')
    summary_path = Path(chain['output']) / ('SUMMARY-' + identifier + '.json')
    summary = utility.sealed(summary_path) if summary_path.exists() else None
    complete = bool(row[1] == 'COMPLETED' and row[2] == '0:0' and summary and
                    summary['status'] == 'COMPLETE_UNGRADED_PRICE_PILOT' and summary['assigned'] == 8 and
                    summary['receipts'] == 8 and summary['missing'] == 0)
    shared.save(own / 'PILOT_ACCOUNTING.json', {'schema': 'utility-runtime-pilot-accounting-v4',
        'proposal_sha256': proposal['sha256'], 'pilot_chain_sha256': chain['sha256'],
        'pilot_job': identifier, 'state': row[1], 'exit_code': row[2], 'elapsed_seconds': int(row[3]),
        'allocated_gpus': 4, 'actual_allocated_gpu_hours': 4 * int(row[3]) / 3600,
        'slurm_accounting': result.stdout, 'summary_sha256': summary['sha256'] if summary else None,
        'status': 'COMPLETE_UNGRADED_RUNTIME_PILOT' if complete else 'INCOMPLETE_OR_FAILED_RUNTIME_PILOT',
        'generation_errors': summary.get('generation_errors') if summary else None,
        'cost_scope': 'All four allocated GPUs from Slurm elapsed time, including loading, waiting, semantic vetoes/nonfires, failures and shutdown. Do not add PAIR_COST or side-reader GPU-hours again.',
        'production_status': 'HOLD_MEASURED_COMPLETE_STAGE_PROPOSAL',
        'automatic_gpu_retry': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--phase', choices=('generation', 'analysis', 'select', 'account', 'validate-pilot'), required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'attachment runs in an authorized Slurm job')
    proposal = utility.sealed(args.proposal)
    if args.phase == 'validate-pilot':
        _, manifest, _, own = paths(proposal)
        pilot = pilot_inputs(proposal, own, manifest)
        require(utility.sealed(Path(os.environ['UTILITY_PILOT_MANIFEST']))['sha256'] == pilot['sha256'],
                'GPU wrapper received another pilot manifest')
        print('PASS_EXACT_8_CELL_16K_PILOT', pilot['sha256'], flush=True)
        return
    require(os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz', 'attachment requires CPU lrd_all_viz')
    if args.phase == 'account':
        account(proposal)
    else:
        phase_attach(proposal, args.proposal, args.phase)


if __name__ == '__main__':
    main()
