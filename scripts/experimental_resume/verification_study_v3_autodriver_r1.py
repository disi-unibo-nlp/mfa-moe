"""Finite event-driven Slurm continuation of the sealed all-seven campaign.

One allocated CPU pass arms a failure rescue, reconciles exact journals, consumes
terminal science, advances eligible stages, and submits an OR-dependent successor.
No daemon, timer sweep, scientific gate change, or speculative GPU retry exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from moe_exp.verification_study import campaign as native
from moe_exp.verification_study.contracts import check_files, file_sha, load, private_path, require, save, seal
from moe_exp.verification_study.registry import MODEL_KEYS, REPO
import verification_study_v3_discovery_dispatch_r1 as dispatch
import verification_study_v3_health_r4 as health

SCRIPT = Path(__file__).resolve()
SHELL = SCRIPT.with_suffix('.sbatch')
TERMINAL = health.TERMINAL
MAX_CONSECUTIVE_RESCUES = 2


def command(argv, timeout=30):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def program_files():
    directory = SCRIPT.parent
    names = [SCRIPT.name, SHELL.name, 'verification_study_v3_autostages_r1.py',
             'verification_study_v3_discovery_continue_r2.py', 'verification_study_v3_discovery_dispatch_r1.py',
             'verification_study_v3_health_r2.py', 'verification_study_v3_health_r4.py', 'verification_study_v3_health_r4.sbatch',
             'verification_study_v3_reader_r1.py', 'verification_study_v3_reader_r1.sbatch',
             'verification_study_v3_j1_r1.py', 'verification_study_v3_extension_r1.py', 'verification_study_v3_codex_review_r1.py',
             'verification_study_v3_extension_r1_gpu.sbatch', 'verification_study_v3_extension_r1_cpu.sbatch']
    return {str(directory / name): file_sha(directory / name) for name in names}


def scientific_submissions(root, campaign):
    rows = []
    for path in sorted((root / 'submissions').glob('*/SUBMISSION.json')):
        card_path = path.parent / 'JOB_CARD.json'
        if not card_path.exists():
            continue
        card = load(card_path)
        if card.get('campaign_sha256') != campaign['sha256']:
            continue
        require(card['code_files'] == campaign['code_files'], 'scientific source closure changed')
        receipt = load(path, 'verification-submission-v3')
        require(receipt['job_card_sha256'] == card['sha256'], 'foreign scientific submission')
        rows.append({'job_id': str(receipt['job_id']), 'card': card, 'submission': receipt,
                     'card_path': card_path, 'submission_path': path})
    require(len({r['job_id'] for r in rows}) == len(rows), 'duplicate scientific job identity')
    return rows


def observations(jobs):
    result = {}
    jobs = sorted(set(map(str, jobs)), key=int)
    if jobs:
        raw = command(['sacct', '-nP', '-j', ','.join(jobs),
                       '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
        for line in raw.splitlines():
            f = line.split('|')
            if f[0] in jobs:
                result[f[0]] = {'state': f[1].split(' ', 1)[0].rstrip('+'), 'exit_code': f[2],
                                'elapsed_seconds': int(f[3]), 'allocated_tres': f[4]}
    raw = command(['squeue', '-h', '-u', 'lmolfett', '-o', '%i|%T|%P|%k|%r'])
    for line in raw.splitlines():
        f = line.split('|', 4)
        if f[0] in jobs:
            result.setdefault(f[0], {}).update(state=f[1], partition=f[2], comment=f[3], reason=f[4], live=True)
    return result


def comment_matches(comment):
    result = {}
    raw = command(['squeue', '-h', '-u', 'lmolfett', '-o', '%i|%k|%a|%P'])
    for line in raw.splitlines():
        f = line.split('|')
        if len(f) == 4 and f[1] == comment:
            require(f[2] == 'iscrc_miosr', 'foreign comment account')
            result[f[0]] = {'job_id': f[0], 'account': f[2], 'partition': f[3], 'queue': True}
    # Exact comment history also covers a crash after sbatch followed by a short job.
    raw = command(['sacct', '-nP', '-S', '2026-10-07', '-u', 'lmolfett',
                   '--format=JobIDRaw,Comment%128,Account,Partition,State,ExitCode'])
    for line in raw.splitlines():
        f = line.split('|')
        if len(f) >= 6 and re.fullmatch(r'\d+', f[0]) and f[1].strip() == comment:
            require(f[2] == 'iscrc_miosr', 'foreign historical comment account')
            result.setdefault(f[0], {}).update(job_id=f[0], account=f[2], partition=f[3], state=f[4], exit_code=f[5])
    require(len(result) == 1, 'submission intent needs exact accounting reconciliation; never blindly retry: ' + comment)
    return next(iter(result.values()))


def reconcile_scientific_intents(root, campaign):
    repaired = []
    for path in sorted((root / 'submissions').glob('*/INTENT.json')):
        if (path.parent / 'SUBMISSION.json').exists():
            continue
        card = load(path.parent / 'JOB_CARD.json')
        require(card['campaign_sha256'] == campaign['sha256'] and card['code_files'] == campaign['code_files'],
                'unresolved foreign scientific intent')
        intent = load(path)
        require(intent['job_card_sha256'] == card['sha256'], 'intent/card mismatch')
        prefix = 'cv3x:' if 'extension' in card else 'cv3r:' if 'reader_executor' in card else 'cv3:'
        matched = comment_matches(intent.get('comment', prefix + card['sha256']))
        require(matched['partition'] == card['resources']['partition'], 'reconciled scientific partition differs')
        proof = save(path.parent / 'INTENT_ACCOUNTING_RECONCILIATION.json', {
            'schema': 'verification-intent-accounting-reconciliation-v3', 'job_card_sha256': card['sha256'],
            'intent_sha256': intent['sha256'], 'matching': matched, 'new_allocations': 0})
        receipt = save(path.parent / 'SUBMISSION.json', {'schema': 'verification-submission-v3',
            'job_card_sha256': card['sha256'], 'job_id': matched['job_id'], 'reconciled': True,
            'accounting_reconciliation_sha256': proof['sha256'], 'preflight_sha256': intent['preflight_sha256']})
        repaired.append(receipt['job_id'])
    return repaired


def controller_card(campaign_path, mode, *, parent=None, wait_ids=(), consecutive_rescues=0, files=None):
    campaign = load(campaign_path); check_files(campaign['code_files'])
    require(mode in {'bootstrap', 'successor', 'rescue', 'ready-cpu'}, 'unknown controller mode')
    ids = sorted(set(map(str, wait_ids)), key=int)
    require(all(re.fullmatch(r'\d+', j) for j in ids), 'invalid finite successor IDs')
    require(0 <= consecutive_rescues <= MAX_CONSECUTIVE_RESCUES, 'consecutive rescue bound exceeded')
    if mode in {'rescue', 'ready-cpu'}:
        require(parent and re.fullmatch(r'\d+', str(parent)), 'controller predecessor required')
    if mode == 'successor':
        require(parent and ids, 'successor requires actual live science')
    dependency = ('afternotok:' + str(parent) if mode == 'rescue' else
                  '?'.join('afterany:' + j for j in ids) if mode == 'successor' else
                  'afterany:' + str(parent) if mode == 'ready-cpu' else None)
    return seal({'schema': 'verification-autodriver-card-v3', 'campaign_path': str(campaign_path),
        'campaign_sha256': campaign['sha256'], 'result_root': campaign['result_root'],
        'mode': mode, 'parent_job_id': str(parent) if parent else None, 'wait_job_ids': ids,
        'consecutive_rescues': consecutive_rescues, 'dependency': dependency,
        'program_files': files or program_files(), 'scientific_code_files': campaign['code_files'],
        'required_models': list(MODEL_KEYS),
        'resources': {'partition': 'lrd_all_serial', 'qos': 'normal', 'cpus': 4, 'gpus': 0,
                      'memory_gb': 24, 'walltime_seconds': 14400, 'billing_core_hours_max': 0},
        'claim': 'all seven required claim/model cells progress through the adopted source-bound scientific gates',
        'artifact': 'immutable handoffs, validated completion/failure events, measured next-stage cards and generated paper exhibits',
        'price_basis': 'bounded receipt validation and serial postprocessing; no inference in this allocation; GPU successors use measured finite role/context budgets',
        'authorization': 'user: script all runs and sbatches so progress survives Codex failure tonight; autonomous approved all-seven campaign',
        'failure_interpretation': 'execution, unmet prices and scientific gate limits remain distinct; no completion-to-effect inference',
        'recovery': 'at most two consecutive controller failures; zero blind scientific allocation retries; no timers or daemons'})


def submit_controller(card, *, test_only=False):
    campaign = load(card['campaign_path'])
    require(card['campaign_sha256'] == campaign['sha256'] and card['scientific_code_files'] == campaign['code_files'], 'controller campaign differs')
    check_files(campaign['code_files']); check_files(card['program_files'])
    root = private_path(card['result_root']); slot = root / 'automation/submissions' / card['sha256']
    save(slot / 'CARD.json', card)
    receipt_path = slot / 'SUBMISSION.json'
    if receipt_path.exists():
        receipt = load(receipt_path)
        require(receipt['controller_card_sha256'] == card['sha256'], 'foreign controller receipt')
        return receipt
    comment = 'cv3auto:' + card['sha256']
    if (slot / 'INTENT.json').exists() and not test_only:
        intent = load(slot / 'INTENT.json'); matched = comment_matches(comment)
        require(matched['partition'] == 'lrd_all_serial', 'controller intent partition differs')
        return save(receipt_path, {'schema': 'verification-autodriver-submission-v3',
            'controller_card_sha256': card['sha256'], 'job_id': matched['job_id'], 'reconciled': True,
            'matching_accounting': matched, 'intent_sha256': intent['sha256']})
    free = dispatch.serial_slots()['free']
    needed = 3 if card['mode'] == 'bootstrap' else 1
    require(free >= needed, 'controller needs its allocation, failure rescue and successor serial slots')
    logs = root / 'automation/logs'; logs.mkdir(parents=True, exist_ok=True)
    argv = ['sbatch', '--account=iscrc_miosr', '--partition=lrd_all_serial', '--qos=normal', '--nodes=1',
            '--ntasks=1', '--cpus-per-task=4', '--mem=24G', '--time=04:00:00', '--no-requeue',
            '--job-name=cv3-auto-' + card['mode'], '--comment=' + comment, '--chdir=' + str(REPO),
            '--output=' + str(logs / '%x-%j.out'), '--error=' + str(logs / '%x-%j.err')]
    if card['dependency']:
        argv.append('--dependency=' + card['dependency'])
    argv.extend([str(SHELL), str(slot / 'CARD.json')])
    snapshot = native.preflight()
    configuration = command(['scontrol', 'show', 'config'])
    require(re.search(r'DependencyParameters\s*=.*kill_invalid_depend', configuration),
            'live scheduler must cancel impossible failure rescues')
    snapshot = seal({**snapshot, 'controller_scheduler': {'DependencyParameters': 'kill_invalid_depend',
                    'MinJobAge': re.search(r'MinJobAge\s*=\s*(\d+)', configuration).group(1)}})
    response = subprocess.run([argv[0], '--test-only', *argv[1:]], check=True, capture_output=True, text=True, timeout=30)
    if test_only:
        return {'test_only': True, 'scheduler_output': response.stdout + response.stderr, 'controller_card': str(slot / 'CARD.json')}
    # Persist preflight only after validation; a rejected pre-intent attempt is retryable.
    saved = save(slot / 'PREFLIGHT.json', snapshot)
    intent = save(slot / 'INTENT.json', {'schema': 'verification-autodriver-intent-v3',
        'controller_card_sha256': card['sha256'], 'preflight_sha256': saved['sha256'],
        'comment': comment, 'epoch': time.time()})
    raw = command([argv[0], '--parsable', *argv[1:]])
    job = raw.split(';', 1)[0]; require(re.fullmatch(r'\d+', job), 'invalid controller job ID')
    observed = command(['scontrol', 'show', 'job', job])
    require('JobId=' + job in observed and 'Account=iscrc_miosr' in observed and 'Partition=lrd_all_serial' in observed,
            'controller submission readback differs')
    return save(receipt_path, {'schema': 'verification-autodriver-submission-v3',
        'controller_card_sha256': card['sha256'], 'job_id': job, 'sbatch_response': raw,
        'read_back': observed, 'intent_sha256': intent['sha256']})


def observe_terminal(campaign_path, root, campaign, submissions, states, controller):
    known = {}
    for path in (root / 'health/events').glob('*.json'):
        event = load(path)
        if event.get('campaign_sha256') == campaign['sha256']:
            known.setdefault((event['predecessor_job_id'], event['job_card_sha256']), []).append(event)
    events = []
    for row in submissions:
        state = states.get(row['job_id'], {})
        if state.get('state') not in TERMINAL:
            continue
        identity = (row['job_id'], row['card']['sha256'])
        prior = known.get(identity, [])
        if prior:
            events.extend({'job_id': row['job_id'], 'state': e['state'], 'event_sha256': e['sha256']} for e in prior)
            continue
        card = row['card']; work = health.expected_work(card, root)
        observer = seal({'schema': 'verification-health-card-v3', 'campaign_path': str(campaign_path),
            'campaign_sha256': campaign['sha256'], 'result_root': str(root), 'predecessor_job_id': row['job_id'],
            'predecessor_submission_path': str(row['submission_path']), 'predecessor_submission_sha256': row['submission']['sha256'],
            'job_card_path': str(row['card_path']), 'job_card_sha256': card['sha256'], 'expected_work': work,
            'observer_files': controller['program_files'], 'claim': card['claim'],
            'artifact': 'actual terminal accounting, exact expected artifacts and retained compatible prices',
            'controller_owned': True, 'observer_allocation': 'existing finite allocated controller; no additional callback job',
            'scientific_dispatch': 'observer emits facts; separately source-sealed controller advances authorized gates'})
        path = root / 'automation/observer-cards' / (observer['sha256'] + '.json'); save(path, observer)
        event = health.check(path)
        events.append({'job_id': row['job_id'], 'state': event['state'], 'event_sha256': event['sha256']})
    return events


def publish(campaign_path, root, campaign, run_dir, value):
    from moe_exp.verification_study.paper import export
    lock = root / 'health/paper-export.lock'; lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        provenance = export(campaign_path, root)
    result = save(run_dir / 'RESULT.json', {'schema': 'verification-autodriver-result-v3',
        'campaign_sha256': campaign['sha256'], 'controller_job_id': os.environ['SLURM_JOB_ID'],
        'observed_epoch': time.time(), 'paper_provenance_sha256': provenance['sha256'], **value})
    save(campaign_path.parent / 'AUTOMATION_STATUS.json', result, replace=True)
    science = value.get('science_observations', {})
    running = [j for j, s in science.items() if s.get('state') == 'RUNNING']
    pending = [j for j, s in science.items() if s.get('state') == 'PENDING']
    lines = ['# Automatic campaign continuation', '',
             'Generated from source-bound controller receipts. Scheduler completion alone does not establish a scientific result.', '',
             f"Current controller: `{os.environ['SLURM_JOB_ID']}`. State: `{value['status']}`.", '',
             f'Scientific jobs observed: {len(running)} running; {len(pending)} pending.', '',
             f"Successor: `{value.get('successor_job_id') or 'none'}`. Failure rescue: `{value.get('rescue_job_id') or 'none'}`.", '',
             'All seven models retain the same required gates. The controller has no timer or persistent daemon.', '',
             f"Receipt: `{run_dir / 'RESULT.json'}`.", '',
             'Read `report/verification-study-v3/AUTOMATION_STATUS.json` for exact actions, errors, job IDs and source bindings.', '']
    target = REPO / 'paper/wiki/AUTOMATION_STATUS.md'
    temporary = target.with_suffix('.md.part'); temporary.write_text('\n'.join(lines)); temporary.replace(target)
    return result


def accepted_successor(root, parent, current=None):
    """A scheduler-accepted successor exists even if HANDOFF was never saved."""
    rows = []
    for path in (root / 'automation/submissions').glob('*/CARD.json'):
        card = load(path)
        if card['parent_job_id'] != str(parent) or card['mode'] not in {'successor', 'ready-cpu'}:
            continue
        receipt = path.parent / 'SUBMISSION.json'
        if not receipt.exists() and not (path.parent / 'INTENT.json').exists():
            continue
        value = load(receipt) if receipt.exists() else submit_controller(card)
        require(value['controller_card_sha256'] == card['sha256'], 'parent successor receipt changed')
        if value['job_id'] != current:
            rows.append(value)
    require(len(rows) <= 1, 'multiple accepted parent handoffs require reconciliation')
    return rows[0] if rows else None


def advance_frontier(campaign_path, root, campaign, controller, states, passes=3):
    """Consume completions racing with dispatch before deciding to stop."""
    import verification_study_v3_autostages_r1 as stages
    require(passes > 0, 'finite stage pass budget required')
    consumed = []
    for index in range(passes):
        frontier = stages.next_actions(campaign_path, root, submit=True, observations=states)
        rows = scientific_submissions(root, campaign)
        fresh = observations([r['job_id'] for r in rows])
        transitions = [j for j, value in fresh.items() if value.get('state') in TERMINAL and
                       not value.get('live') and (states.get(j, {}).get('state') not in TERMINAL or states.get(j, {}).get('live'))]
        if not transitions:
            return frontier, fresh, consumed
        consumed += observe_terminal(campaign_path, root, campaign, rows, fresh, controller)
        states = fresh
        if index == passes - 1:
            # This immediate successor is justified by exact newly terminal work,
            # not an empty periodic polling loop.
            frontier = dict(frontier, ready_cpu_actions=True, unconsumed_terminal_job_ids=transitions)
            return frontier, fresh, consumed
    raise AssertionError('finite frontier loop has no result')


@contextmanager
def orchestration_lock(root, seconds=120):
    """Brief overlap at handoff waits safely instead of losing the next chain."""
    path = root / 'discovery-orchestration.lock'
    with path.open('a') as handle:
        deadline = time.monotonic() + seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'controller lock contention exceeded finite recovery wait')
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(card_path):
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu()
    require(getpass.getuser() == 'lmolfett' and socket.gethostname().endswith('leonardo.local'), 'wrong allocated host/user')
    controller = load(card_path, 'verification-autodriver-card-v3'); root = Path(controller['result_root'])
    campaign_path = Path(controller['campaign_path']); campaign = load(campaign_path)
    require(campaign['sha256'] == controller['campaign_sha256'] and set(campaign['models']) == set(MODEL_KEYS), 'all-seven campaign differs')
    check_files(campaign['code_files']); check_files(controller['program_files'])
    current = os.environ['SLURM_JOB_ID']; run_dir = root / 'automation/runs' / current
    # A crash after accepting a successor cannot create a second independent chain.
    if controller['mode'] == 'rescue':
        receipt = accepted_successor(root, controller['parent_job_id'], current)
        if receipt:
            observed = observations([receipt['job_id']]).get(receipt['job_id'], {})
            if observed.get('live') or observed.get('state') == 'COMPLETED':
                return save(run_dir / 'RESULT.json', {'schema': 'verification-autodriver-result-v3',
                    'campaign_sha256': campaign['sha256'], 'controller_job_id': current,
                    'status': 'EXISTING_HANDOFF_REUSED', 'successor_job_id': receipt['job_id'], 'new_allocations': 0})
    rescue = None
    if controller['consecutive_rescues'] < MAX_CONSECUTIVE_RESCUES:
        rescue_card = controller_card(campaign_path, 'rescue', parent=current,
            consecutive_rescues=controller['consecutive_rescues'] + 1, files=controller['program_files'])
        rescue = submit_controller(rescue_card)
        save(run_dir / 'RESCUE.json', {'schema': 'verification-autodriver-rescue-armed-v3',
            'controller_card_sha256': controller['sha256'], 'rescue_card_sha256': rescue_card['sha256'],
            'rescue_job_id': rescue['job_id'], 'dependency': rescue_card['dependency']})
    with orchestration_lock(root):
        save(run_dir / 'START.json', {'schema': 'verification-autodriver-start-v3', 'campaign_sha256': campaign['sha256'],
            'controller_card_sha256': controller['sha256'], 'controller_job_id': current, 'started_epoch': time.time()})
        repaired = reconcile_scientific_intents(root, campaign)
        rows = scientific_submissions(root, campaign); states = observations([r['job_id'] for r in rows])
        events = observe_terminal(campaign_path, root, campaign, rows, states, controller)
        frontier, states, raced_events = advance_frontier(campaign_path, root, campaign, controller, states)
        events += raced_events
        ids = sorted([j for j, v in states.items() if v.get('live')], key=int)
        successor = None
        if ids:
            successor_card = controller_card(campaign_path, 'successor', parent=current,
                                               wait_ids=ids, files=controller['program_files'])
            successor = submit_controller(successor_card)
        elif frontier.get('ready_cpu_actions'):
            require(frontier.get('actions') or frontier.get('unconsumed_terminal_job_ids'),
                    'empty frontier cannot start an immediate controller loop')
            successor_card = controller_card(campaign_path, 'ready-cpu', parent=current, files=controller['program_files'])
            successor = submit_controller(successor_card)
        if successor:
            save(run_dir / 'HANDOFF.json', {'schema': 'verification-autodriver-handoff-v3',
                'controller_job_id': current, 'successor_job_id': successor['job_id'],
                'successor_card_sha256': successor_card['sha256'], 'wait_job_ids': ids,
                'submission_path': str(root / 'automation/submissions' / successor_card['sha256'] / 'SUBMISSION.json')})
        status = ('WAITING_FOR_SCIENCE' if successor else
                  'CAMPAIGN_COMPLETE' if frontier.get('status') == 'COMPLETE' else 'STOPPED_WITH_EXPLICIT_ACTIONS')
        value = {'status': status, 'frontier': frontier, 'reconciled_submission_job_ids': repaired,
                 'terminal_events': events, 'science_observations': states,
                 'successor_job_id': successor['job_id'] if successor else None,
                 'rescue_job_id': rescue['job_id'] if rescue else None,
                 'scientific_success': 'ONLY_FROM_VALIDATED_ARTIFACTS_AND_GATES'}
        result = publish(campaign_path, root, campaign, run_dir, value)
        if status == 'STOPPED_WITH_EXPLICIT_ACTIONS' and frontier.get('errors'):
            from verification_study_v3_codex_review_r1 import review
            diagnostic = review(campaign_path, frontier)
            save(run_dir / 'CODEX_DIAGNOSTIC.json', {
                'schema': 'verification-autodriver-diagnostic-v3', 'campaign_sha256': campaign['sha256'],
                'status': diagnostic['status'], 'private_review_sha256': diagnostic.get('sha256'),
                'interpretation': 'read-only recommendation; no extra scientific or repair allocations'})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bootstrap', 'run', 'inspect'))
    parser.add_argument('--campaign', type=Path, default=REPO / 'report/verification-study-v3/CAMPAIGN.json')
    parser.add_argument('--card', type=Path)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'bootstrap':
        card = controller_card(args.campaign.resolve(), 'bootstrap')
        result = submit_controller(card, test_only=args.test_only)
    elif args.command == 'run':
        try:
            result = run(args.card.resolve())
        except Exception as error:
            card = load(args.card); root = Path(card['result_root'])
            save(root / 'automation/runs' / os.environ['SLURM_JOB_ID'] / 'FAILURE.json', {
                'schema': 'verification-autodriver-failure-v3', 'campaign_sha256': card['campaign_sha256'],
                'controller_job_id': os.environ['SLURM_JOB_ID'], 'controller_card_sha256': card['sha256'],
                'error_type': type(error).__name__, 'error': str(error)[:2000],
                'status': 'CONTROLLER_FAILURE_RESCUE_ARMED' if (root / 'automation/runs' / os.environ['SLURM_JOB_ID'] / 'RESCUE.json').exists() else 'CONTROLLER_FAILURE_NO_ARMED_RESCUE',
                'consecutive_rescues': card['consecutive_rescues'],
                'observed_epoch': time.time(), 'scientific_null': False})
            raise
    else:
        campaign = load(args.campaign); check_files(campaign['code_files'])
        rows = scientific_submissions(Path(campaign['result_root']), campaign)
        result = {'campaign_sha256': campaign['sha256'], 'science_observations': observations([r['job_id'] for r in rows])}
    print(json.dumps({k: v for k, v in result.items() if k in {'status', 'job_id', 'successor_job_id', 'rescue_job_id', 'test_only', 'scheduler_output', 'science_observations'}}, default=str), flush=True)


if __name__ == '__main__':
    main()
