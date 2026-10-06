"""Finite native-login dispatch and Slurm CPU readiness for the independent panel."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import time

# Install the frozen scoring namespace before helpers import moe_exp.
import utility_j1_entry_v2
J1 = utility_j1_entry_v2.install()
import operator_panel_v1 as P
import operator_panel_outcomes_v1 as O
import dispatch_overnight_readers_v1 as shared
from dispatch_utility_production_v1 import duration_seconds, array_multiplicity

OPERATIONS = P.DOC / 'OPERATOR_PANEL_OPERATIONS_v1.json'
MEASUREMENT = P.DOC / 'OPERATOR_PANEL_MEASUREMENT_v1.json'
CPU = P.SCRIPTS / 'operator_panel_cpu_v1.sbatch'
GPU = P.SCRIPTS / 'run_operator_panel_v1.sbatch'
DIRECTORY = P.DOC / 'operator-panel-submissions-v1'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


def login_guard():
    P.require(socket.gethostname().startswith('login') and socket.gethostname().endswith('.leonardo.local') and
              pwd.getpwuid(os.getuid()).pw_name == 'lmolfett' and not os.environ.get('SLURM_JOB_ID'),
              'dispatch requires actual lmolfett native LEONARDO login')


def cpu_guard():
    P.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
              os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and not socket.gethostname().startswith('login') and
              pwd.getpwuid(os.getuid()).pw_name == 'lmolfett', 'requires actual CPU Slurm step')


def measurement_sources():
    import utility_outcomes_v3
    return {**utility_outcomes_v3.source_files(),
            **{str(p): P.U.file_sha(p) for p in (Path(O.__file__), Path(__file__).resolve(), CPU,
                                                P.SCRIPTS / 'operator_panel_legacy_lengths_v1.py')}}


def validate():
    plan = P.U.sealed(P.PLAN); P.validate_plan(plan)
    operations = P.U.sealed(OPERATIONS); measurement = P.U.sealed(MEASUREMENT)
    P.require(operations['plan_sha256'] == measurement['plan_sha256'] == plan['sha256'] and
              operations['measurement_sha256'] == measurement['sha256'] and
              measurement['source_files'] == measurement_sources() and
              operations['generation_sources'] == P.generation_sources(), 'panel frozen sources changed')
    J1.validate_plan()
    return plan, operations, measurement


def freeze():
    login_guard(); DIRECTORY.mkdir(exist_ok=True)
    plan = P.save(P.PLAN, {k: v for k, v in P.build_plan(P.U.sealed(P.DOC / 'UTILITY_SCOUT_PLAN_v1.json'),
                    P.U.sealed(P.DOC / 'OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json')).items() if k != 'sha256'})
    j1 = J1.validate_plan()
    measurement = P.save(MEASUREMENT, {'schema': 'operator-panel-measurement-v1',
        'plan_sha256': plan['sha256'], 'source_files': measurement_sources(),
        'question_table_sha256': P.U.sealed(P.U.QUESTIONS)['sha256'],
        'grading_chain_plan_sha256': j1['sha256'], 'inference': plan['inference'],
        'reasoning': 'Verified tokenizer </think> singleton; count generated IDs before first closing marker, excluding that marker. Report answer-region IDs and boundary status. Unclosed caps count generated reasoning inside the budget; unclosed natural stops have unknown region lengths.',
        'accuracy': 'Frozen strict/J1 scorer; naturally stopped strict-rejected answers get arm-blind J1. Caps and committed errors operationally wrong. Missing execution/unfinished grades unknown. Completed UNCERTAIN retains original operational-wrong convention and is disclosed.',
        'supplement': 'CPU-only A/B/C/fresh generated continuation lengths; 1024/1024/256/1024-token horizons; no pooling with original-prompt 16k endpoints.'})
    proposal = P.U.sealed(P.DOC / 'UTILITY_RUNTIME_PILOT_PROPOSAL_v4.json')
    qual = P.U.sealed(Path(proposal['qualification_result']))
    P.require(qual['status'] == 'PASS_ENGINEERING', 'existing original-prompt engine not qualified')
    value = P.save(OPERATIONS, {'schema': 'operator-panel-operations-v1', 'plan_sha256': plan['sha256'],
        'measurement_sha256': measurement['sha256'], 'generation_sources': P.generation_sources(),
        'qualification_path': proposal['qualification_result'], 'qualification_sha256': qual['sha256'],
        'grading_settings': j1['binding'], 'grading_plan_sha256': j1['sha256'],
        'additional_ceiling_billing_core_hours': P.CEILING, 'submission_directory': str(DIRECTORY),
        'root': str(P.ROOT), 'status': 'AUTHORIZED_PREPARATION_AND_TWO_PILOTS',
        'native_dispatch': 'Finite start/advance calls with fresh saldo, project queue and association checks. CPU jobs produce readiness; they do not submit GPU children.'})
    print(json.dumps({'operations': str(OPERATIONS), 'sha256': value['sha256']}))


def accounting(jobs):
    if not jobs:
        return {'rows': {}, 'raw': '', 'actual_billing_core_hours': 0., 'reserved_billing_core_hours': 0.}
    raw = subprocess.run(['sacct', '-X', '-nP', '-j', ','.join(sorted(set(jobs))),
        '--format=JobID%100,State%40,ExitCode,ElapsedRaw,TimelimitRaw,AllocTRES%200,JobIDRaw'],
        check=True, capture_output=True, text=True).stdout
    rows = {}
    for line in raw.splitlines():
        if not line.strip(): continue
        fields = line.split('|'); identifier, state, exit_code, elapsed, limit, tres = fields[:6]
        if not re.fullmatch(r'\d+(?:_\d+)?', identifier): continue
        state = state.split(' by ', 1)[0]
        counts = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
        billing = float(counts.get('billing', 0)); seconds = int(elapsed)
        gpu = int(counts.get('gres/gpu', 0))
        P.require(identifier not in rows, 'ambiguous accounting job')
        rows[identifier] = {'job_id': identifier, 'job_id_raw': fields[6], 'state': state, 'exit_code': exit_code,
            'elapsed_seconds': seconds, 'allocated_TRES': tres, 'billing_cores': billing,
            'gpus': gpu, 'billing_core_hours': billing * seconds / 3600,
            'gpu_hours': gpu * seconds / 3600,
            'reserved_billing_core_hours': billing * (max(seconds, int(limit) * 60) if state in ACTIVE else seconds) / 3600}
    # Root arrays have no allocation; only exact allocated tasks are charged.
    return {'rows': rows, 'raw': raw,
        'actual_billing_core_hours': sum(r['billing_core_hours'] for r in rows.values()),
        'reserved_billing_core_hours': sum(r['reserved_billing_core_hours'] for r in rows.values())}


def known_jobs():
    return sorted({P.U.sealed(p)['job_id'] for p in DIRECTORY.glob('*.json')
                   if not p.name.endswith(('.attempt.json', '.verified.json')) and
                   P.U.sealed(p).get('schema') == 'overnight-submit-receipt-v1'})


def terminal(jobs):
    evidence = accounting(jobs)
    for job in jobs:
        exact = [r for key, r in evidence['rows'].items() if key == job or key.startswith(job + '_')]
        P.require(exact and all(r['state'] not in ACTIVE for r in exact), 'predecessor not terminal: ' + job)
    return evidence


def live_preflight(plan, stage_required):
    login_guard()
    commands = {
        'balance': ['/cineca/bin/saldo', '-b', 'lmolfett'],
        'queue': ['squeue', '-h', '-A', 'iscrc_miosr', '-o', '%100i|%T|%l|%L|%C|%b|%D|%m'],
        'gpu_partition': ['scontrol', 'show', 'partition', 'boost_usr_prod'],
        'cpu_partition': ['scontrol', 'show', 'partition', 'lrd_all_viz'],
        'associations': ['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=lmolfett', 'format=User,Account,Partition,QOS'],
        'qos': ['sacctmgr', '-nP', 'show', 'qos', 'normal', 'format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxTRESPerJob']}
    outputs = {name: subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
               for name, cmd in commands.items()}
    P.require('iscrc_miosr' in outputs['associations'] and 'normal' in outputs['associations'] and
              'PartitionName=boost_usr_prod' in outputs['gpu_partition'] and
              'PartitionName=lrd_all_viz' in outputs['cpu_partition'], 'live association/partition differs')
    balance = [r.split() for r in outputs['balance'].splitlines() if r.startswith('IscrC_MIOSR ')]
    P.require(len(balance) == 1, 'unknown live saldo format')
    remaining = float(balance[0][3]) - float(balance[0][5]); queue_cost = 0.
    for line in outputs['queue'].splitlines():
        if not line.strip(): continue
        identifier, state, limit, left, cpus, gres, nodes, memory = [x.strip() for x in line.split('|')]
        gpu = re.search(r'gres/gpu(?::[^:,]+)?:(\d+)', gres)
        # Full GPU allocations bill 32 cores; a 120G/2GPU request bills 16.
        # CPU viz memory uses conservative 128/512 GiB. No tempfs is requested.
        mem = re.fullmatch(r'([0-9.]+)([KMGTP])', memory)
        P.require(mem is not None, 'unknown queue memory')
        gib = float(mem[1]) * {'K': 1/1048576, 'M': 1/1024, 'G': 1., 'T': 1024., 'P': 1048576.}[mem[2]]
        billed = max(int(cpus), int(gpu[1]) * 8 * int(nodes) if gpu else gib * 128 / 512 * int(nodes))
        queue_cost += array_multiplicity(identifier) * billed * duration_seconds(left) / 3600
    preserved = plan['budget']['preserved_existing_utility_reserve_billing_core_hours']
    available = remaining - queue_cost - preserved
    value = P.save(DIRECTORY / ('LIVE-' + str(time.time_ns()) + '.json'), {
        'schema': 'operator-panel-live-account-v1', 'hostname': socket.gethostname(), 'outputs': outputs,
        'reported_remaining': remaining, 'queue_commitments': queue_cost, 'preserved_utility_reserve': preserved,
        'available_additional': available, 'required_new_unqueued_stage': stage_required,
        'status': 'PASS' if available >= stage_required else 'HOLD_LIVE_BUDGET'})
    P.require(value['status'] == 'PASS', 'fresh live headroom insufficient for panel and preserved commitments')
    return value


def submit(name, args, env, binding):
    job = shared.submit(DIRECTORY, name, args, env, binding)
    verified = DIRECTORY / (name + '.verified.json')
    if not verified.exists():
        raw = subprocess.run(['scontrol', 'show', 'job', job], check=True, capture_output=True, text=True).stdout
        P.require('JobId=' + job in raw and 'UserId=lmolfett(' in raw, 'reused job verification failed')
        P.save(verified, {'schema': 'overnight-job-verification-v1', 'job_id': job, 'scontrol': raw})
    return job


def cpu_submit(name, phase, dependencies=(), extra=None):
    env = os.environ.copy(); env.update(OPERATOR_PANEL_PHASE=phase, **(extra or {}))
    binding = {'operations_sha256': P.U.sealed(OPERATIONS)['sha256'], 'phase': phase, 'extra': extra or {}}
    args = [*dependencies, '--job-name=operator-panel-' + name, str(CPU)]
    if (DIRECTORY / (name + '.json')).exists(): args = P.U.sealed(DIRECTORY / (name + '.json'))['arguments']
    return submit(name, args, env, binding)


def start():
    plan, operations, measurement = validate(); login_guard()
    with (DIRECTORY / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DIRECTORY / 'START.json').exists():
            print(json.dumps({'status': 'ALREADY_SUBMITTED', 'chain': str(DIRECTORY / 'START.json')})); return
        live = live_preflight(plan, 2 * 256 + plan['budget']['cpu_reserve_billing_core_hours'])
        prepare_job = cpu_submit('prepare', 'prepare')
        legacy_job = cpu_submit('legacy-lengths', 'legacy')
        jobs = []
        for index in (0, 1):
            name = 'pilot-' + str(index)
            env = os.environ.copy(); env.update(OPERATOR_PANEL_MANIFEST=str(P.ROOT / 'PILOT_MANIFEST.json'),
                OPERATOR_PANEL_OUT=str(P.ROOT / name), OPERATOR_PANEL_PILOT_INDEX=str(index))
            args = ['--dependency=afterok:' + prepare_job, '--job-name=operator-panel-' + name, str(GPU)]
            if (DIRECTORY / (name + '.json')).exists(): args = P.U.sealed(DIRECTORY / (name + '.json'))['arguments']
            jobs.append(submit(name, args, env, {'operations_sha256': operations['sha256'], 'pilot_index': index}))
        price = cpu_submit('pilot-price', 'price', ['--dependency=afterany:' + ':'.join(jobs)])
        value = P.save(DIRECTORY / 'START.json', {'schema': 'operator-panel-start-v1',
            'operations_sha256': operations['sha256'], 'prepare_job': prepare_job, 'legacy_job': legacy_job,
            'pilot_jobs': jobs, 'price_job': price, 'live_sha256': live['sha256']})
        print(json.dumps(value))


def reconciliation(manifest, outputs, job_ids, index_path, imports=()):
    """Exact immutable pilot imports, all attempts and terminal allocations."""
    assigned = [r['assignment'] for r in manifest['rows']]
    by_uid = {a['uid']: a for a in assigned}; observed = {r['assignment']['uid']: r for r in imports}
    evidence = terminal(job_ids)
    allowed_job_ids = set(job_ids) | {r['job_id_raw'] for r in evidence['rows'].values()}
    P.require(len(observed) == len(imports), 'duplicate pilot imports')
    for row in imports:
        receipt = P.U.sealed(Path(row['receipt_path']))
        P.require(row['assignment'] == by_uid[row['assignment']['uid']] and
                  receipt['sha256'] == row['receipt_sha256'] and receipt['assignment'] == row['assignment'],
                  'pilot import changed')
    attempt_paths = []
    for output in outputs:
        if not output.exists(): continue
        binding = P.U.sealed(output / 'BINDING.json')
        P.require(binding['manifest_sha256'] == manifest['sha256'] and
                  all(a == by_uid[a['uid']] for a in binding['assigned']), 'shard/pilot execution binding differs')
        for path in sorted((output / 'attempts').glob('*.json')):
            attempt = P.U.sealed(path)
            P.require(attempt['assignment'] == by_uid[attempt['assignment']['uid']] and
                      attempt['binding_sha256'] == binding['sha256'] and attempt['job_id'] in allowed_job_ids,
                      'attempt assignment/job provenance differs')
            attempt_paths.append({'path': str(path), 'sha256': attempt['sha256']})
        for path in sorted((output / 'receipts').glob('*.json')):
            receipt = P.U.sealed(path); a = receipt['assignment']; uid = a['uid']
            P.require(uid not in observed and a == by_uid[uid] and receipt['binding_sha256'] == binding['sha256'],
                      'duplicate or rebound committed cell')
            P.require(any(p['sha256'] == receipt['attempt_sha256'] for p in attempt_paths),
                      'receipt has no immutable attempt provenance')
            route = output / 'routes' / (P.U.digest(uid) + '.npz')
            observed[uid] = {'assignment': a, 'status': receipt['status'], 'receipt_path': str(path),
                'receipt_sha256': receipt['sha256'], 'source_binding_sha256': binding['sha256'],
                'source_manifest_sha256': manifest['sha256'],
                'routed_path': str(route) if receipt['status'] == 'COMMITTED_GENERATION' else None,
                'origin': 'pilot' if 'pilot-' in output.name else 'production'}
    records = [observed.get(a['uid'], {'assignment': a, 'status': 'MISSING', 'receipt_path': None,
               'receipt_sha256': None, 'source_binding_sha256': None, 'routed_path': None, 'origin': 'missing'}) for a in assigned]
    # Validate the scientific information boundary and route hashes before import.
    O.extract({'records': records})
    return P.save(index_path, {'schema': 'panel-reconciled-index-v1', 'plan_sha256': manifest['plan_sha256'],
        'manifest_sha256': manifest['sha256'], 'family_order': manifest['family_order'],
        'records': records, 'attempts': attempt_paths, 'accounting': evidence,
        'assigned': len(records), 'missing': sum(r['status'] == 'MISSING' for r in records),
        'errors': sum(r['status'] == 'GENERATION_ERROR' for r in records),
        'status': 'TERMINAL_ALL_ASSIGNMENTS_ACCOUNTED'})


def prepare_cpu(plan, operations):
    P.ROOT.mkdir(parents=True, exist_ok=True)
    prompts, questions = P.U.sealed(P.U.PROMPTS), P.U.sealed(P.U.QUESTIONS)
    manifest = P.save(P.ROOT / 'PILOT_MANIFEST.json', P.make_manifest(plan, 2,
        Path(operations['qualification_path']), prompts, questions))
    P.validate_manifest(manifest)
    score, engine = O.scoring()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    boundary = O.boundary_contract(tokenizer, plan['tokenizer_sha256'])
    for row in manifest['rows']:
        text = tokenizer.decode(row['original_prompt_ids'], skip_special_tokens=False)
        P.require(text.rfind('<think>') > text.rfind('</think>'), 'original prompt does not open reasoning')
    value = P.save(P.ROOT / 'PREPARED.json', {'schema': 'operator-panel-preparation-v1',
        'manifest_sha256': manifest['sha256'], 'boundary': boundary, 'producer_job_id': os.environ['SLURM_JOB_ID'],
        'qualifier_sha256': operations['qualification_sha256']})
    print('PANEL_PREPARED', value['sha256'], flush=True)


def price_cpu(plan, operations):
    chain = P.U.sealed(DIRECTORY / 'START.json'); manifest = P.U.sealed(P.ROOT / 'PILOT_MANIFEST.json')
    outputs = [P.ROOT / ('pilot-' + str(i)) for i in (0, 1)]
    index = reconciliation(manifest, outputs, chain['pilot_jobs'], P.ROOT / 'PILOT_INDEX.json')
    receipts = [P.U.sealed(Path(r['receipt_path'])) for r in index['records'] if r['receipt_path']]
    loads, costs, allocations = [], [], []
    for out, job in zip(outputs, chain['pilot_jobs'], strict=True):
        allocation = index['accounting']['rows'].get(job)
        P.require(allocation and allocation['gpus'] == 4 and allocation['billing_cores'] == 32,
                  'pilot allocation resource accounting differs')
        allocations.append(allocation)
        loads.extend(P.U.sealed(p) for p in sorted((out / 'allocations').glob('*/PAIR_LOAD.json')))
        costs.extend(P.U.sealed(p) for p in sorted((out / 'allocations').glob('*/PAIR_COST.json')))
    profile = P.measured_profile(receipts, loads, costs, allocations)
    grading = {n: J1.price_rows([{'item_id': format(i, '024x'), 'prompt_tokens': 16384} for i in range(8 * n)],
                               operations['grading_settings']) for n in P.COHORTS}
    if profile['status'] != 'MEASURED_RUNTIME_ONLY':
        price = {'status': 'HOLD_MEASURED_PROFILE', 'profile': profile, 'selected': None}
    else:
        spent = accounting([chain['prepare_job'], chain['legacy_job']])['actual_billing_core_hours']
        price = P.choose_cohort(plan, profile, grading, spent)
    value = P.save(P.ROOT / 'PRICE_READY.json', {'schema': 'operator-panel-price-readiness-v1',
        'operations_sha256': operations['sha256'], 'pilot_index_sha256': index['sha256'],
        'producer_job_id': os.environ['SLURM_JOB_ID'], 'grading_envelopes': {str(n): g for n, g in grading.items()},
        **price})
    print('PANEL_PRICE', value['status'], value['sha256'], flush=True)


def production_manifest(plan, operations, price):
    selected = price['selected']; count = selected['families']
    manifest = P.make_manifest(plan, count, Path(operations['qualification_path']),
                               P.U.sealed(P.U.PROMPTS), P.U.sealed(P.U.QUESTIONS))
    families = plan['family_order'][2:count]; capacity = selected['families_per_shard']
    manifest['shards'] = [families[i:i + capacity] for i in range(0, len(families), capacity)]
    pilot = P.U.sealed(P.ROOT / 'PILOT_INDEX.json')
    P.require(pilot['sha256'] == price['pilot_index_sha256'] and pilot['missing'] == pilot['errors'] == 0,
              'pilot imports incomplete')
    manifest['pilot_imports'] = pilot['records']
    adapter = P.ROOT / 'pilot-0' / 'engineering' / P.U.sealed(DIRECTORY / 'START.json')['pilot_jobs'][0] / 'QUALIFICATION.json'
    qualification = P.U.sealed(adapter)
    P.require(qualification['status'] == 'PASS_ENGINEERING', 'panel adapter unqualified')
    manifest.update(adapter_qualification_path=str(adapter), adapter_qualification_sha256=qualification['sha256'],
                    price_readiness_sha256=price['sha256'])
    return P.save(P.ROOT / 'PRODUCTION_MANIFEST.json', manifest)


def production_dispatch(plan, operations, price):
    terminal([price['producer_job_id']])
    P.require(price['operations_sha256'] == operations['sha256'], 'price readiness rebound')
    if price['selected'] is None:
        update_index(price['status']); print(json.dumps({'status': price['status'], 'price': str(P.ROOT / 'PRICE_READY.json')})); return
    selected = price['selected']; manifest = production_manifest(plan, operations, price)
    # Reuse already submitted receipts after interruption. Existing queued jobs
    # are accounted in the live queue; avoid charging them a second time here.
    if not (DIRECTORY / 'production.json').exists():
        required = selected['generation_with_recovery_billing_core_hours'] + selected['grading_reserve_billing_core_hours'] + 128
        live_preflight(plan, required)
    env = os.environ.copy(); env.update(OPERATOR_PANEL_MANIFEST=str(P.ROOT / 'PRODUCTION_MANIFEST.json'),
                                       OPERATOR_PANEL_OUT=str(P.ROOT / 'production'))
    env.pop('OPERATOR_PANEL_PILOT_INDEX', None)
    args = [f'--array=0-{len(manifest["shards"])-1}%16',
            '--time=' + str(math.ceil(selected['requested_wall_seconds'] / 60)),
            '--job-name=operator-panel-production', str(GPU)]
    job = submit('production', args, env, {'operations_sha256': operations['sha256'], 'manifest_sha256': manifest['sha256']})
    reconcile = cpu_submit('production-reconcile', 'reconcile', ['--dependency=afterany:' + job])
    P.save(DIRECTORY / 'PRODUCTION.json', {'schema': 'operator-panel-production-chain-v1',
        'generation_job': job, 'reconcile_job': reconcile, 'manifest_sha256': manifest['sha256'],
        'price_sha256': price['sha256']})
    update_index('PRODUCTION_SUBMITTED')


def reconcile_cpu(plan, operations, final=False):
    manifest = P.U.sealed(P.ROOT / 'PRODUCTION_MANIFEST.json')
    chain = P.U.sealed(DIRECTORY / 'PRODUCTION.json')
    outputs = [P.ROOT / 'production' / f'shard-{i:03d}' for i in range(len(manifest['shards']))]
    jobs = [chain['generation_job']]
    if final:
        recovery = P.U.sealed(DIRECTORY / 'RECOVERY.json')
        jobs.append(recovery['generation_job'])
    # The second pass reuses the same output bindings and committed cells.
    index = reconciliation(manifest, outputs, jobs,
        P.ROOT / ('FINAL_INDEX.json' if final else 'INITIAL_INDEX.json'), manifest['pilot_imports'])
    value = P.save(P.ROOT / ('FINAL_READY.json' if final else 'GENERATION_READY.json'), {
        'schema': 'operator-panel-generation-readiness-v1', 'operations_sha256': operations['sha256'],
        'index_path': str(P.ROOT / ('FINAL_INDEX.json' if final else 'INITIAL_INDEX.json')),
        'index_sha256': index['sha256'], 'producer_job_id': os.environ['SLURM_JOB_ID'],
        'missing': index['missing'], 'errors': index['errors'], 'recovery_used': final})
    print('PANEL_RECONCILED', value['sha256'], index['missing'], 'missing', flush=True)


def generation_dispatch(plan, operations, readiness):
    terminal([readiness['producer_job_id']])
    index = P.U.sealed(Path(readiness['index_path']))
    P.require(index['sha256'] == readiness['index_sha256'], 'generation readiness changed')
    price = P.U.sealed(P.ROOT / 'PRICE_READY.json'); selected = price['selected']
    # A single infrastructure recovery wave may retry only uncommitted cells.
    # Committed generation errors remain operationally wrong and are never retried.
    if index['missing'] and not readiness['recovery_used']:
        manifest = P.U.sealed(P.ROOT / 'PRODUCTION_MANIFEST.json')
        missing = {r['assignment']['family'] for r in index['records'] if r['status'] == 'MISSING'}
        shards = [i for i, families in enumerate(manifest['shards']) if set(families) & missing]
        if len(shards) <= selected['recovery_shards']:
            required = len(shards) * selected['requested_wall_seconds'] / 3600 * 32 + selected['grading_reserve_billing_core_hours'] + 64
            live_preflight(plan, required)
            env = os.environ.copy(); env.update(OPERATOR_PANEL_MANIFEST=str(P.ROOT / 'PRODUCTION_MANIFEST.json'),
                OPERATOR_PANEL_OUT=str(P.ROOT / 'production'), OPERATOR_PANEL_RESUME='1')
            env.pop('OPERATOR_PANEL_PILOT_INDEX', None)
            job = submit('recovery', ['--array=' + ','.join(map(str, shards)) + '%16',
                '--time=' + str(math.ceil(selected['requested_wall_seconds'] / 60)),
                '--job-name=operator-panel-recovery', str(GPU)], env,
                {'operations_sha256': operations['sha256'], 'index_sha256': index['sha256'], 'shards': shards})
            cpu = cpu_submit('recovery-reconcile', 'reconcile-final', ['--dependency=afterany:' + job])
            P.save(DIRECTORY / 'RECOVERY.json', {'schema': 'operator-panel-recovery-v1',
                'generation_job': job, 'reconcile_job': cpu, 'shards': shards, 'maximum_waves': 1})
            update_index('RECOVERY_SUBMITTED'); return
        P.save(P.ROOT / 'RECOVERY_HOLD.json', {'schema': 'operator-panel-recovery-hold-v1',
            'missing_shards': shards, 'reserved_shards': selected['recovery_shards'],
            'reason': 'Missing work exceeds reserved single wave; retain unknowns and analyze bounds.'})
    cpu = cpu_submit('grade-prepare', 'grade', extra={'OPERATOR_PANEL_INDEX': readiness['index_path']})
    P.save(DIRECTORY / 'GRADE_PREPARATION.json', {'schema': 'operator-panel-grade-prepare-chain-v1',
        'index_sha256': index['sha256'], 'preparation_job': cpu})
    update_index('GRADING_PREPARATION_SUBMITTED')


def grade_cpu(plan, operations, measurement):
    index_path = Path(os.environ['OPERATOR_PANEL_INDEX']); root = P.ROOT / 'grading'
    root.mkdir(exist_ok=True)
    prep_path = root / 'preparation/GRADE_PREP.json'
    prep = P.U.sealed(prep_path) if prep_path.exists() else O.prepare(index_path, root / 'preparation', measurement)
    P.require(prep['index_sha256'] == P.U.sealed(index_path)['sha256'] and
              prep['measurement_sha256'] == measurement['sha256'], 'grade preparation rebound')
    items = O.read_preparation(prep_path.parent, prep)['items']
    score, _ = O.scoring(); score.assert_blind(items, 'panel exact J1 price')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(J1.MODEL, local_files_only=True) if items else None
    rows = []
    for item in items:
        ids = J1.exact_ids(tokenizer.apply_chat_template(J1.messages(item, score.answer_equivalence),
            tokenize=True, add_generation_prompt=True, enable_thinking=True))
        rows.append({'item_id': item['item_id'], 'prompt_tokens': len(ids), 'prompt_ids_sha256': P.U.digest(ids)})
    price = J1.price_rows(rows, operations['grading_settings'])
    selected = P.U.sealed(P.ROOT / 'PRICE_READY.json')['selected']
    P.require(price['requested_allocation_GPU_hour_ceiling'] * 8 <= selected['grading_reserve_billing_core_hours'],
              'exact grading exceeds reserve; keep hold rather than submit')
    by_id = {r['item_id']: r for r in items}
    for shard in price['shards']:
        directory = root / 'j1' / f'shard-{shard["index"]:03d}'; directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'items.jsonl'
        content = ''.join(json.dumps(by_id[i], ensure_ascii=False) + '\n' for i in shard['item_ids'])
        if path.exists(): P.require(path.read_text() == content, 'J1 input changed')
        else: path.write_text(content)
        shard.update(items_path=str(path), items_file_sha256=P.U.file_sha(path),
                     verdicts_path=str(directory / 'verdicts.jsonl'))
    value = P.save(root / 'J1_READY.json', {'schema': 'operator-panel-j1-readiness-v1',
        'operations_sha256': operations['sha256'], 'grade_preparation_sha256': prep['sha256'],
        'producer_job_id': os.environ['SLURM_JOB_ID'], 'prompt_records': rows,
        'tokenizer_files': {str(p): P.U.file_sha(p) for p in (J1.MODEL / 'tokenizer.json', J1.MODEL / 'tokenizer_config.json')},
        **price})
    print('PANEL_J1_READY', value['sha256'], len(items), 'items', flush=True)


def grading_dispatch(plan, operations, ready):
    terminal([ready['producer_job_id']])
    prep = P.U.sealed(P.ROOT / 'grading/preparation/GRADE_PREP.json')
    P.require(ready['operations_sha256'] == operations['sha256'] and
              ready['grade_preparation_sha256'] == prep['sha256'] and
              all(P.U.file_sha(Path(p)) == h for p, h in ready['tokenizer_files'].items()), 'J1 readiness changed')
    expected = J1.price_rows(ready['prompt_records'], operations['grading_settings'])
    P.require(all(ready[k] == v for k, v in expected.items() if k != 'shards') and
              all(all(s[k] == v for k, v in e.items()) and P.U.file_sha(Path(s['items_path'])) == s['items_file_sha256']
                  for s, e in zip(ready['shards'], expected['shards'], strict=True)), 'J1 price changed')
    evidence = accounting(known_jobs())
    required = ready['requested_allocation_GPU_hour_ceiling'] * 8 + 16
    P.require(evidence['reserved_billing_core_hours'] + required <= P.CEILING, 'additional 6000-hour ceiling exceeded')
    live_preflight(plan, required)
    env = J1.safe_environment(os.environ, {'STEER_CODE': str(J1.BASE),
        'STEER_EXPECT_TREE': P.U.sealed(J1.BASE / 'MANIFEST.json')['tree_sha256'],
        'STEER_PROVENANCE_DIR': str(P.ROOT / 'grading/j1/provenance')})
    jobs = []
    for shard in ready['shards']:
        name = 'j1-' + str(shard['index']).zfill(3)
        jobs.append(submit(name, ['--time=06:00:00', '--job-name=operator-panel-' + name,
            str(J1.LEGACY), shard['items_path'], shard['verdicts_path']], env,
            {'operations_sha256': operations['sha256'], 'price_sha256': ready['sha256'],
             'shard': shard['index'], 'items_file_sha256': shard['items_file_sha256']}))
    final = cpu_submit('final-analysis', 'finalize', ['--dependency=afterany:' + ':'.join(jobs)] if jobs else [])
    P.save(DIRECTORY / 'GRADING.json', {'schema': 'operator-panel-grading-chain-v1',
        'price_sha256': ready['sha256'], 'J1_jobs': jobs, 'finalize_job': final})
    update_index('GRADING_SUBMITTED')


def finalize_cpu(plan, operations):
    root = P.ROOT / 'grading'; ready = P.U.sealed(root / 'J1_READY.json')
    chain = P.U.sealed(DIRECTORY / 'GRADING.json'); evidence = terminal(chain['J1_jobs'])
    P.require(chain['price_sha256'] == ready['sha256'], 'grading final price differs')
    verdicts = []
    for job, shard in zip(chain['J1_jobs'], ready['shards'], strict=True):
        allocation = evidence['rows'][job]; path = Path(shard['verdicts_path'])
        if allocation['state'] == 'COMPLETED' and allocation['exit_code'] == '0:0':
            provenance = P.U.sealed(root / 'j1/provenance' / (job + '.json'))
            P.require(provenance['items_sha256'] == shard['items_file_sha256'] and
                      provenance['launcher_sha256'] == P.U.file_sha(J1.LEGACY) and provenance['dry_run'] is False and
                      provenance['tree_sha256'] == P.U.sealed(J1.BASE / 'MANIFEST.json')['tree_sha256'] and
                      provenance['versions'] == {'vllm': '0.29.0+cu129', 'torch': '2.13.0+cu129'},
                      'J1 runtime provenance differs')
            verdicts.extend(J1.valid_verdicts(path, shard['item_ids']))
        # Failed/unfinished shards remain unknown; physical partial files and
        # allocation cost stay intact. Never coerce their missing votes to false.
    P.require(len({r['item_id'] for r in verdicts}) == len(verdicts), 'duplicate final J1 verdict')
    path = root / 'verdicts.jsonl'
    content = ''.join(json.dumps(r) + '\n' for r in verdicts)
    if path.exists(): P.require(path.read_text() == content, 'merged verdicts changed')
    else: path.write_text(content)
    all_cost = accounting(known_jobs())
    O.analysis(root / 'preparation/GRADE_PREP.json', path, root / 'analysis', all_cost)
    update_index('ANALYZED')


def update_index(status):
    """Append a new study-index supplement; preserve every existing sealed index."""
    artifacts = {}
    for path in (P.PLAN, OPERATIONS, MEASUREMENT, P.ROOT / 'PRICE_READY.json',
                 P.ROOT / 'legacy/CONTINUATION_LENGTHS.json', P.ROOT / 'grading/analysis/ANALYSIS.json'):
        if path.exists(): artifacts[str(path)] = P.U.file_sha(path)
    prior = sorted((P.DOC / 'routing-study-closeout-v2').glob('*/STUDY_INDEX.json'))
    evidence = accounting(known_jobs())
    if (P.ROOT / 'grading/analysis/ANALYSIS.json').exists():
        effect = P.U.sealed(P.ROOT / 'grading/analysis/ANALYSIS.json')['effect']
        status = effect['status']
    else:
        effect = None
    value = P.save(DIRECTORY / ('STUDY_INDEX-' + str(time.time_ns()) + '.json'), {
        'schema': 'routing-study-operator-panel-index-v1', 'status': status,
        'existing_study_indexes': {str(p): P.U.file_sha(p) for p in prior},
        'artifacts': artifacts, 'effect': effect, 'accounting': evidence,
        'additional_ceiling_billing_core_hours': P.CEILING,
        'uncertainty': 'Panel results require terminal generation, blind grades and all nine paired endpoints. Short-horizon continuation summaries are separate. No accuracy-retention/equivalence claim.'})
    # Human-readable pointer is intentionally mutable; all evidence JSON remains
    # append-only. This is repository documentation, not a ChatGPT Page.
    pointer = DIRECTORY / 'STUDY_INDEX.md'
    pointer.write_text('# Operator accuracy and reasoning-length panel\n\nStatus: **' + status + '**.\n\n' +
        'Latest sealed index: `' + value['sha256'] + '`.\n\n' +
        f'Observed allocated billing-core-hours: {evidence["actual_billing_core_hours"]:.4f}; ceiling: 6000.\n\n' +
        'See the adjacent STUDY_INDEX JSON snapshots for exact jobs, source hashes, results and missing-data bounds.\n')
    return value


def advance():
    login_guard(); plan, operations, measurement = validate()
    with (DIRECTORY / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (P.ROOT / 'grading/analysis/ANALYSIS.json').exists():
            update_index('ANALYZED'); print(json.dumps({'status': 'ANALYZED'})); return
        if (P.ROOT / 'grading/J1_READY.json').exists() and not (DIRECTORY / 'GRADING.json').exists():
            grading_dispatch(plan, operations, P.U.sealed(P.ROOT / 'grading/J1_READY.json')); return
        for name, consumed in [('FINAL_READY.json', 'GRADE_PREPARATION.json'),
                               ('GENERATION_READY.json', 'RECOVERY.json')]:
            path = P.ROOT / name
            if path.exists() and not (DIRECTORY / 'GRADE_PREPARATION.json').exists() and not (name == 'GENERATION_READY.json' and (DIRECTORY / consumed).exists()):
                generation_dispatch(plan, operations, P.U.sealed(path)); return
        if (P.ROOT / 'PRICE_READY.json').exists() and not (DIRECTORY / 'PRODUCTION.json').exists():
            production_dispatch(plan, operations, P.U.sealed(P.ROOT / 'PRICE_READY.json')); return
        update_index('WAIT_SEALED_READINESS'); print(json.dumps({'status': 'WAIT_SEALED_READINESS'}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'start', 'advance', 'cpu', 'status'))
    parser.add_argument('--phase', choices=('prepare', 'price', 'legacy', 'reconcile', 'reconcile-final', 'grade', 'finalize'))
    args = parser.parse_args()
    if args.command == 'freeze': freeze(); return
    if args.command == 'start': start(); return
    if args.command == 'advance': advance(); return
    plan, operations, measurement = validate()
    if args.command == 'status':
        login_guard(); value = update_index('WAIT_SEALED_READINESS'); print(json.dumps({'status': value['status'], 'accounting': value['accounting']})); return
    cpu_guard()
    if args.phase == 'prepare': prepare_cpu(plan, operations)
    elif args.phase == 'price': price_cpu(plan, operations)
    elif args.phase == 'legacy':
        from operator_panel_legacy_lengths_v1 import run
        run(plan)
    elif args.phase in ('reconcile', 'reconcile-final'): reconcile_cpu(plan, operations, args.phase.endswith('final'))
    elif args.phase == 'grade': grade_cpu(plan, operations, measurement)
    elif args.phase == 'finalize': finalize_cpu(plan, operations)
    else: raise ValueError('CPU phase is required')


if __name__ == '__main__':
    main()
