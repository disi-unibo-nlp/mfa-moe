"""Plan, submit, inspect, resume and export the existing eight-model study."""
from pathlib import Path
import argparse
import getpass
import json
import os
import socket
import time
from .common import MODELS, REPO, check_files, control_root, digest, file_sha, publish, require, verified
from .config import active, initialize, read_config
from .scheduler import Slurm, command, duration_minutes, job_fields


def identity():
    require(getpass.getuser() == 'lmolfett' and 'leonardo' in socket.getfqdn(), 'wrong user or cluster')


def worker(binding_path, card_path):
    from .journal import run_stage
    from .execution import handler
    identity()
    require(os.environ.get('SLURM_JOB_ID'), 'worker needs a real Slurm allocation')
    binding = verified(binding_path)
    root = control_root(binding)
    c = {k: v for k, v in verified(card_path).items() if k != 'sha256'}
    require(Path(card_path).resolve() == root / 'cards' / (digest(c) + '.json'), 'worker card path differs')
    require(c['source_commit'] == binding['source_commit'], 'worker source version differs')
    check_files(binding['code_files'])
    require(file_sha(Path(binding['source_campaign']) / 'CAMPAIGN.json') == binding['source_campaign_sha256'], 'inherited source changed')
    acceptance = verified(root / 'submissions' / digest(c) / 'ACCEPTED.json')
    require(acceptance['job_id'] == os.environ['SLURM_JOB_ID'], 'allocation not accepted for this card')
    fields = Slurm(binding['config']).readback(os.environ['SLURM_JOB_ID'], c)['fields']
    remaining = (c['minutes'] - duration_minutes(fields['RunTime'])) * 60
    return run_stage(binding, c, handler, allocation_deadline=time.time() + remaining - 60)


def transfer(binding, graph, launch):
    from .workflow import native_retirement, unused_cancelled
    root = control_root(binding)
    path = root / 'OWNERSHIP_TRANSFER.json'
    if path.exists():
        return verified(path)
    scheduler = Slurm(binding['config'])
    cards = {digest(c): c for c in graph['cards']}
    owners = graph['guards'] + [graph['controller']]
    observed = {}
    for key in owners:
        jid = launch['jobs'][key]
        fields = scheduler.readback(jid, cards[key])['fields']
        if fields['JobState'] == 'PENDING' and fields.get('Reason') == 'JobHeldUser':
            command(['scontrol', 'release', jid])
        observed[jid] = scheduler.readback(jid, cards[key])
        require(observed[jid]['fields'].get('Reason') != 'JobHeldUser', 'replacement owner remains held')
    held = verified(Path(binding['root']) / 'STUDY_IMPLEMENTATION_HANDOFF_HOLD.json')['after']
    old_ids = sorted(held)
    require(not set(old_ids).intersection(binding['config']['protected_report_jobs']), 'reporting owner entered retirement')
    before = native_retirement(old_ids)
    for jid in old_ids:
        if jid in before and unused_cancelled(before[jid]):
            continue
        fields = job_fields(command(['scontrol', 'show', 'job', '-o', jid]))
        require(fields['JobName'] == held[jid]['JobName'] and
                fields['Account'] == held[jid]['Account'] and fields['UserId'] == held[jid]['UserId'],
                'predecessor ownership identity changed')
        require(fields['JobState'] == 'PENDING' and fields['Reason'] == 'JobHeldUser' and
                fields['RunTime'] == '00:00:00' and fields['StartTime'] == 'Unknown', 'old controller started or hold changed')
        command(['scancel', jid])
    after = {}
    for _ in range(10):
        after = native_retirement(old_ids)
        if set(after) == set(old_ids) and all(unused_cancelled(row) for row in after.values()):
            break
        time.sleep(1)
    require(set(after) == set(old_ids) and all(unused_cancelled(row) for row in after.values()), 'old controller retirement lacks native proof')
    return publish(path, {'new_owners': observed, 'before': before, 'after': after,
            'all_replacements_accepted_before_retirement': True,
            'protected_report_jobs': binding['config']['protected_report_jobs']})


def submit(config):
    from .workflow import initial_graph, payload, submit_wave, release_wave
    identity()
    binding = initialize(config)
    root = control_root(binding)
    path = root / 'LAUNCH.json'
    if path.exists():
        return verified(path)
    frozen_plan = root / 'waves/preflight/PLAN.json'
    graph = payload(verified(frozen_plan)) if frozen_plan.exists() else initial_graph(binding['config'])
    scheduler = Slurm(binding['config'])
    if not (root / 'SCHEDULER_VALIDATION.json').exists():
        publish(root / 'SCHEDULER_VALIDATION.json', {'status': 'complete',
            'checks': [scheduler.submit(c, hold=True, test_only=True) for c in graph['cards'][:2]]})
    launch = submit_wave(binding, graph, release=False)
    transfer(binding, graph, launch)
    release_wave(binding, graph, launch)
    return publish(path, {'status': 'released', 'source_commit': binding['source_commit'],
            'jobs': launch['jobs'], 'control_root': str(root), 'gpu_repair_ceiling': config['budget']['gpu_repair_hours'],
            'automatic_after_verified_gates': True, 'remote_pushes': False})


def status(config):
    from .resources import accounting
    binding = active(config)
    root = Path(config['root'])
    stages = []
    if binding is not None:
        for path in sorted((control_root(binding) / 'stages').glob('*/RESULT.json')):
            value = verified(path)
            stages.append({k: value.get(k) for k in ('card_sha256', 'job_id', 'status', 'error')})
    terminal = control_root(binding) / 'TERMINAL.json' if binding is not None else None
    return {'root': str(root), 'source_commit': binding['source_commit'] if binding else None,
            'jobs': accounting(root), 'stages': stages,
            'qualified_models': [m for m in MODELS if (root / m / 'QUALIFIED.json').exists()],
            'admission': verified(root / 'ADMISSION.json')['status'] if (root / 'ADMISSION.json').exists() else 'pending',
            'terminal': verified(terminal) if terminal and terminal.exists() else None,
            'manuscript_integrated': False}


def resume(config):
    """Reconcile partial waves, or replace a terminal controller once."""
    from .workflow import accepted, payload, protect, submit_wave
    from .resources import accounting
    from .reservations import TERMINAL
    launch = submit(config)
    binding = active(config)
    root = control_root(binding)
    if (root / 'FINALIZATION_LAUNCH.json').exists():
        return verified(root / 'FINALIZATION_LAUNCH.json')
    usage = accounting(binding['root'])
    resumed = []
    for path in sorted((root / 'waves').glob('*/PLAN.json')):
        plan = payload(verified(path))
        if not (path.parent / 'RELEASED.json').exists():
            resumed.append(submit_wave(binding, plan))
            continue
        if (root / 'controllers' / plan['controller'] / 'HANDOFF.json').exists():
            continue
        # No second controller while any existing owner can still run.
        owners = [accepted(root, key) for key in [plan['controller'], *plan['guards']]]
        rows = [usage.get(owner['job_id']) if owner else None for owner in owners]
        if any(row is None or row['state'] not in TERMINAL for row in rows):
            continue
        if str(plan['wave']).endswith('-resume'):
            continue
        candidate = protect(binding['config'], [], plan['wave'], plan['terminal_cards'],
                            'finalization', resume_of=plan['controller'])
        candidate['wave'] = str(plan['wave']) + '-resume'
        resumed.append(submit_wave(binding, candidate))
    return {'launch': launch, 'reconciled_waves': resumed}


def plan(config):
    from .resources import snapshot, availability
    from .reservations import ledger
    root = Path(config['root'])
    base = verified(root / 'BINDING.json')
    assignments = {}
    missing = []
    for model in MODELS:
        assignments[model] = {}
        for mode in ('full', 'fallback'):
            path = root / model / ('ASSIGNMENTS-' + mode + '.json')
            if path.exists():
                value = verified(path)
                assignments[model][mode] = {'path': str(path), 'sha256': value['sha256'],
                    'assigned_cells': len(value['records']), 'available_executions': value['available_executions']}
            else:
                assignments[model][mode] = {'status': 'pending native discovery'}
        if not (root / model / 'QUALIFIED.json').exists():
            missing.append(model + ': qualification')
    snap = snapshot()
    return {'read_only': True, 'assignments': assignments, 'missing_prerequisites': missing,
            'existing_preparation_ledger': ledger(base), 'available': availability(config, snap),
            'proposed': {'cpu_validation': '4 CPU / 24 GiB / 30 min',
                'engineering': '2 GPU / 16 CPU / 120 GiB / 120 min; LFM then GPT',
                'qualification': '8 independent four-hour jobs after engineering passes',
                'shared_recovery': 'one four-hour native interruption recovery',
                'inherited_cpu_pricing': 'one 80-minute CPU task per unfinished model; five measured models reused',
                'margin_inclusive_gpu_hours': config['budget']['gpu_repair_hours'],
                'remaining': 'reader, complete admission, LFM, interventions, analyses and export'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'submit', 'status', 'resume', 'export', 'worker'))
    parser.add_argument('--config', type=Path, default=REPO / 'configs/final_routing.yaml')
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--card', type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'worker':
        require(args.binding and args.card, 'worker needs binding and card')
        result = worker(args.binding, args.card)
        print(json.dumps({k: result.get(k) for k in ('status', 'job_id', 'error', 'elapsed_seconds')}))
        return int(result['status'] != 'complete')
    config = read_config(args.config)
    if args.command == 'resume':
        value = resume(config)
    elif args.command == 'submit':
        value = submit(config)
    elif args.command == 'plan':
        value = plan(config)
    elif args.command == 'status':
        value = status(config)
    else:
        from .reservations import reserve
        from .workflow import node
        binding = initialize(config)
        candidate = node(binding['config'], 'export', phase='finalization', minutes=30,
                         export_state=digest(status(config)['stages']))
        reserve(binding, candidate)
        value = Slurm(binding['config']).submit(candidate)
    if args.json:
        print(json.dumps(value, sort_keys=True))
    elif args.command == 'status':
        print('Source:', value['source_commit'] or 'not released')
        print('Qualified:', ', '.join(value['qualified_models']) or 'none')
        print('Production admission:', value['admission'])
        print('Collection terminal:', bool(value['terminal']))
        print('Required results complete:', bool(value['terminal'] and value['terminal']['required_results_complete']))
        for row in value['stages']:
            print(row['job_id'], row['status'], row.get('error') or '')
    elif args.command == 'plan':
        print(json.dumps(value, indent=2))
    else:
        print(json.dumps(value, sort_keys=True))
    return 0
