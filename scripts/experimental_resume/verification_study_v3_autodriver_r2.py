"""Finite r2 controller: clean sbatch transport and held, prearmed startup.

The scientific executor and r1 operational artifacts remain unchanged. Every
accepted controller has at most two deterministic failure rescues before its
primary is released. Interrupted acceptance, prearming and release resume the
same immutable identities; unavailable serial slots never retire protection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import verification_study_v3_autodriver_r1 as previous
from moe_exp.verification_study import campaign as native
from moe_exp.verification_study.contracts import check_files, file_sha, load, private_path, require, save, seal
from moe_exp.verification_study.registry import MODEL_KEYS, REPO

SCRIPT = Path(__file__).resolve()
SHELL = SCRIPT.with_suffix('.sbatch')
MAX_CONSECUTIVE_RESCUES = 2
TERMINAL = previous.TERMINAL
dispatch = previous.dispatch
health = previous.health
scientific_submissions = previous.scientific_submissions
observations = previous.observations
comment_matches = previous.comment_matches
observe_terminal = previous.observe_terminal
publish = previous.publish
orchestration_lock = previous.orchestration_lock


class SerialSlotLimit(ValueError):
    """Finite resource deferral; no protected predecessor is cancelled."""


@contextmanager
def submission_transport():
    """Clean only sbatch subprocess environments, including unchanged helpers."""
    from verification_study_v3_transport_r2 import clean_submission_environment
    original = subprocess.run
    if getattr(original, '_verification_submission_transport_r2', False) is True:
        yield
        return

    def run(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and args and Path(str(args[0])).name == 'sbatch':
            kwargs['env'] = clean_submission_environment(kwargs.get('env'))
        return original(args, *positional, **kwargs)

    run._verification_submission_transport_r2 = True
    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original


def command(argv, timeout=30):
    with submission_transport():
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def program_files():
    from verification_study_v3_transport_r2 import source_files
    files = previous.program_files()
    files.update(source_files())
    for path in (SCRIPT, SHELL, SCRIPT.with_name('verification_study_v3_autostages_r2.py')):
        files[str(path)] = file_sha(path)
    return files


def controller_card(campaign_path, mode, *, parent=None, wait_ids=(), consecutive_rescues=0,
                    failure_ancestor_job_ids=None, files=None):
    value = previous.controller_card(campaign_path, mode, parent=parent, wait_ids=wait_ids,
        consecutive_rescues=consecutive_rescues, files=files or program_files())
    ancestors = list(map(str, failure_ancestor_job_ids or ([parent] if mode == 'rescue' else [])))
    require(all(re.fullmatch(r'\d+', job) for job in ancestors) and len(set(ancestors)) == len(ancestors),
            'invalid/duplicate failure ancestor identity')
    if mode == 'rescue':
        require(ancestors and ancestors[0] == str(parent) and len(ancestors) == consecutive_rescues,
                'rescue must bind every failure ancestor')
        value['dependency'] = 'afternotok:' + ':'.join(ancestors)
    else:
        require(not ancestors and consecutive_rescues == 0, 'new primary cannot inherit a failure counter')
    return seal({**value, 'operational_version': 'autodriver-r2',
        'failure_ancestor_job_ids': ancestors, 'failure_chain_root_job_id': ancestors[-1] if ancestors else None,
        'launch_policy': 'held acceptance; exact two-rescue prearm; release deepest-to-primary',
        'submission_environment_policy': 'child-only removal of inherited SLURM_*, SRUN_* and SBATCH_*',
        'slot_deferral': 'preserve existing protection; no controller cancellation to manufacture slots'})


def slot_for(card):
    return private_path(card['result_root']) / 'automation/submissions' / card['sha256']


def rescue_card(card, receipt):
    if card['consecutive_rescues'] == MAX_CONSECUTIVE_RESCUES:
        return None
    return controller_card(Path(card['campaign_path']), 'rescue', parent=receipt['job_id'],
        consecutive_rescues=card['consecutive_rescues'] + 1,
        failure_ancestor_job_ids=[receipt['job_id'], *card['failure_ancestor_job_ids']], files=card['program_files'])


def validate_card(card):
    campaign = load(card['campaign_path'])
    require(card.get('operational_version') == 'autodriver-r2' and
            card['campaign_sha256'] == campaign['sha256'] and
            card['scientific_code_files'] == campaign['code_files'], 'controller r2 campaign differs')
    require(0 <= card['consecutive_rescues'] <= MAX_CONSECUTIVE_RESCUES, 'rescue counter outside finite chain')
    check_files(campaign['code_files']); check_files(card['program_files'])
    return campaign


def accepted_receipt(card):
    """Reconcile only an exact accepted identity; absence never permits retry."""
    slot = slot_for(card); receipt_path = slot / 'SUBMISSION.json'
    if receipt_path.exists():
        receipt = load(receipt_path, 'verification-autodriver-submission-v3')
        require(receipt['controller_card_sha256'] == card['sha256'], 'foreign controller receipt')
        return receipt
    intent_path = slot / 'INTENT.json'
    if not intent_path.exists():
        return None
    intent = load(intent_path)
    require(intent['controller_card_sha256'] == card['sha256'], 'foreign controller intent')
    matched = comment_matches('cv3auto:' + card['sha256'])
    require(matched['partition'] == 'lrd_all_serial', 'controller intent partition differs')
    return save(receipt_path, {'schema': 'verification-autodriver-submission-v3',
        'controller_card_sha256': card['sha256'], 'job_id': matched['job_id'], 'reconciled': True,
        'matching_accounting': matched, 'intent_sha256': intent['sha256'], 'submitted_held': True})


def missing_chain_slots(card):
    """Count only unaccepted members, reusing partially committed chains."""
    receipt = accepted_receipt(card)
    if receipt is None:
        return MAX_CONSECUTIVE_RESCUES - card['consecutive_rescues'] + 1
    if (slot_for(card) / 'RELEASE.json').exists():
        return 0
    child = rescue_card(card, receipt)
    return missing_chain_slots(child) if child else 0


def readback(card, receipt):
    text = command(['scontrol', 'show', 'job', str(receipt['job_id'])])
    require('JobId=' + str(receipt['job_id']) in text and 'Account=iscrc_miosr' in text and
            'Partition=lrd_all_serial' in text and 'Comment=cv3auto:' + card['sha256'] in text,
            'controller job readback identity differs')
    return text


def dependency_ids(text, kind):
    match = re.search(r'(?:^|\s)Dependency=([^\s]+)', text)
    require(match is not None, 'controller readback has no dependency')
    dependency = re.sub(r'\([^)]*\)', '', match.group(1))
    require('?' not in dependency, 'failure rescue dependency cannot use OR')
    ids = []
    for clause in dependency.split(','):
        parts = clause.split(':')
        require(parts[0] == kind and len(parts) > 1 and all(re.fullmatch(r'\d+', part) for part in parts[1:]),
                'controller dependency kind/identity differs')
        ids.extend(parts[1:])
    require(len(set(ids)) == len(ids), 'duplicate failure dependency identity')
    return ids


def _submit_held(card, *, test_only=False):
    slot = slot_for(card); save(slot / 'CARD.json', card)
    receipt = accepted_receipt(card)
    if receipt is not None:
        return receipt
    root = private_path(card['result_root']); logs = root / 'automation/logs'; logs.mkdir(parents=True, exist_ok=True)
    comment = 'cv3auto:' + card['sha256']
    argv = ['sbatch', '--account=iscrc_miosr', '--partition=lrd_all_serial', '--qos=normal', '--nodes=1',
        '--ntasks=1', '--cpus-per-task=4', '--mem=24G', '--time=04:00:00', '--no-requeue', '--hold',
        '--kill-on-invalid-dep=yes', '--job-name=cv3-auto-r2-' + card['mode'], '--comment=' + comment,
        '--chdir=' + str(REPO), '--output=' + str(logs / '%x-%j.out'), '--error=' + str(logs / '%x-%j.err')]
    if card['dependency']:
        argv.append('--dependency=' + card['dependency'])
    argv.extend([str(SHELL), str(slot / 'CARD.json')])
    snapshot = native.preflight(); configuration = command(['scontrol', 'show', 'config'])
    minimum_age = re.search(r'MinJobAge\s*=\s*(\d+)', configuration)
    require(minimum_age is not None, 'live controller scheduler configuration unavailable')
    snapshot = seal({**snapshot, 'controller_scheduler': {'kill_on_invalid_dep': True,
        'MinJobAge': minimum_age.group(1)}, 'submission_environment_policy': card['submission_environment_policy']})
    with submission_transport():
        response = subprocess.run([argv[0], '--test-only', *argv[1:]], check=True,
            capture_output=True, text=True, timeout=30)
    if test_only:
        return {'test_only': True, 'scheduler_output': response.stdout + response.stderr,
                'controller_card': str(slot / 'CARD.json'), 'protected_chain_members':
                MAX_CONSECUTIVE_RESCUES - card['consecutive_rescues'] + 1}
    saved = save(slot / 'PREFLIGHT.json', snapshot)
    intent = save(slot / 'INTENT.json', {'schema': 'verification-autodriver-intent-v3',
        'controller_card_sha256': card['sha256'], 'preflight_sha256': saved['sha256'],
        'comment': comment, 'submitted_held': True, 'epoch': time.time()})
    raw = command([argv[0], '--parsable', *argv[1:]])
    job = raw.split(';', 1)[0]; require(re.fullmatch(r'\d+', job), 'invalid controller job ID')
    observed = readback(card, {'job_id': job})
    require('JobState=PENDING' in observed and ('Reason=JobHeldUser' in observed or 'Priority=0' in observed),
            'new controller must remain held until startup protection is committed')
    return save(slot / 'SUBMISSION.json', {'schema': 'verification-autodriver-submission-v3',
        'controller_card_sha256': card['sha256'], 'job_id': job, 'sbatch_response': raw,
        'read_back': observed, 'intent_sha256': intent['sha256'], 'submitted_held': True})


def ensure_prearmed(card, receipt):
    """No early return on a primary receipt: missing rescue journals resume."""
    slot = slot_for(card); child = rescue_card(card, receipt); children = []
    if child:
        child_receipt = submit_controller(child)
        children.append({'controller_card_sha256': child['sha256'], 'job_id': child_receipt['job_id'],
            'submission_path': str(slot_for(child) / 'SUBMISSION.json'),
            'submission_sha256': child_receipt['sha256'],
            'prearm_path': str(slot_for(child) / 'PREARM.json'),
            'prearm_sha256': load(slot_for(child) / 'PREARM.json')['sha256'],
            'release_path': str(slot_for(child) / 'RELEASE.json'),
            'release_sha256': load(slot_for(child) / 'RELEASE.json')['sha256']})
    return save(slot / 'PREARM.json', {'schema': 'verification-autodriver-prearmed-chain-v3',
        'controller_card_sha256': card['sha256'], 'primary_job_id': receipt['job_id'],
        'submission_sha256': receipt['sha256'], 'consecutive_rescues': card['consecutive_rescues'],
        'max_consecutive_rescues': MAX_CONSECUTIVE_RESCUES, 'children': children,
        'failure_ancestor_job_ids': card['failure_ancestor_job_ids'], 'dependency': card['dependency'],
        'all_child_dependencies_registered_before_primary_release': True})


def release_controller(card):
    """Release a fully protected existing controller, never allocate here."""
    validate_card(card); slot = slot_for(card); receipt = accepted_receipt(card)
    require(receipt is not None, 'release requires an accepted held controller')
    path = slot / 'RELEASE.json'
    if path.exists():
        saved = load(path)
        require(saved['controller_card_sha256'] == card['sha256'] and
                saved['submission_sha256'] == receipt['sha256'], 'foreign controller release receipt')
        return saved
    proof = load(slot / 'PREARM.json', 'verification-autodriver-prearmed-chain-v3')
    require(proof['controller_card_sha256'] == card['sha256'] and proof['primary_job_id'] == receipt['job_id'] and
            proof['submission_sha256'] == receipt['sha256'], 'release lacks exact prearmed primary proof')
    child = rescue_card(card, receipt)
    require(len(proof['children']) == (1 if child else 0), 'release chain has missing/extra rescue')
    intent_path = slot / 'RELEASE_INTENT.json'
    try:
        observed = readback(card, receipt)
    except subprocess.CalledProcessError:
        require(intent_path.exists(), 'aged controller without release intent requires reconciliation')
        matched = comment_matches('cv3auto:' + card['sha256'])
        require(matched['job_id'] == receipt['job_id'] and matched['partition'] == 'lrd_all_serial' and
                matched.get('state', '').split(' ', 1)[0].rstrip('+') in TERMINAL,
                'aged release intent lacks exact terminal accounting')
        observed = ('JobId=' + receipt['job_id'] + ' Account=iscrc_miosr Partition=lrd_all_serial Comment=cv3auto:' +
                    card['sha256'] + ' JobState=' + matched['state'] + ' Reason=None ReconciledFromAccounting=True')
    held = 'JobState=PENDING' in observed and ('Reason=JobHeldUser' in observed or 'Priority=0' in observed)
    if child:
        bound = proof['children'][0]
        require(bound['controller_card_sha256'] == child['sha256'] and
                load(bound['submission_path'])['sha256'] == bound['submission_sha256'] and
                load(bound['prearm_path'])['sha256'] == bound['prearm_sha256'] and
                load(bound['release_path'])['sha256'] == bound['release_sha256'], 'prearmed rescue receipt differs')
        child_receipt = load(bound['submission_path'])
        if held:
            child_text = readback(child, child_receipt)
            require('JobState=PENDING' in child_text and
                    set(dependency_ids(child_text, 'afternotok')) == set(child['failure_ancestor_job_ids']),
                    'prearmed rescue is not waiting on every exact failure ancestor')
    if not intent_path.exists():
        require(held,
                'untracked release of primary requires explicit reconciliation')
        save(intent_path, {'schema': 'verification-autodriver-release-intent-v3',
            'controller_card_sha256': card['sha256'], 'submission_sha256': receipt['sha256'],
            'prearm_sha256': proof['sha256'], 'job_id': receipt['job_id']})
    intent = load(intent_path)
    require(intent['controller_card_sha256'] == card['sha256'] and intent['prearm_sha256'] == proof['sha256'],
            'changed release intent')
    if held:
        command(['scontrol', 'release', str(receipt['job_id'])])
        observed = readback(card, receipt)
    require('Reason=JobHeldUser' not in observed, 'controller remains held after release')
    return save(path, {'schema': 'verification-autodriver-release-v3',
        'controller_card_sha256': card['sha256'], 'submission_sha256': receipt['sha256'],
        'prearm_sha256': proof['sha256'], 'release_intent_sha256': intent['sha256'],
        'job_id': receipt['job_id'], 'read_back': observed})


def submit_controller(card, *, test_only=False, defer_release=False):
    validate_card(card)
    if (slot_for(card) / 'RELEASE.json').exists() and not test_only:
        release_controller(card)
        return accepted_receipt(card)
    needed = missing_chain_slots(card)
    free = dispatch.serial_slots()['free']
    if needed > free:
        raise SerialSlotLimit(f'protected finite controller chain needs {needed} free serial slots; observed {free}')
    receipt = _submit_held(card, test_only=test_only)
    if test_only:
        return receipt
    ensure_prearmed(card, receipt)
    if not defer_release:
        release_controller(card)
    return receipt


def reconcile_scientific_intents(root, campaign):
    repaired = []
    for path in sorted((root / 'submissions').glob('*/INTENT.json')):
        if (path.parent / 'SUBMISSION.json').exists():
            continue
        card = load(path.parent / 'JOB_CARD.json')
        require(card['campaign_sha256'] == campaign['sha256'] and card['code_files'] == campaign['code_files'],
                'unresolved foreign scientific intent')
        intent = load(path); require(intent['job_card_sha256'] == card['sha256'], 'intent/card mismatch')
        prefix = ('cv3t:' if 'transport_executor' in card else 'cv3x:' if 'extension' in card else
                  'cv3r:' if 'reader_executor' in card else 'cv3:')
        matched = comment_matches(intent.get('comment', prefix + card['sha256']))
        require(matched['partition'] == card['resources']['partition'], 'scientific intent partition differs')
        proof = save(path.parent / 'INTENT_ACCOUNTING_RECONCILIATION.json', {
            'schema': 'verification-intent-accounting-reconciliation-v3', 'job_card_sha256': card['sha256'],
            'intent_sha256': intent['sha256'], 'matching': matched, 'new_allocations': 0})
        receipt = save(path.parent / 'SUBMISSION.json', {'schema': 'verification-submission-v3',
            'job_card_sha256': card['sha256'], 'job_id': matched['job_id'], 'reconciled': True,
            'accounting_reconciliation_sha256': proof['sha256'], 'preflight_sha256': intent['preflight_sha256']})
        repaired.append(receipt['job_id'])
    return repaired


def accepted_successor(root, parents, current=None):
    parents = {str(parents)} if isinstance(parents, (str, int)) else set(map(str, parents))
    candidates = []
    for path in (root / 'automation/submissions').glob('*/CARD.json'):
        card = load(path)
        if card['parent_job_id'] not in parents or card['mode'] not in {'successor', 'ready-cpu'}:
            continue
        if not (path.parent / 'SUBMISSION.json').exists() and not (path.parent / 'INTENT.json').exists():
            continue
        require(card.get('operational_version') == 'autodriver-r2', 'accepted foreign driver version needs separate reconciliation')
        receipt_path = path.parent / 'SUBMISSION.json'
        if receipt_path.exists() and load(receipt_path)['job_id'] == current:
            continue
        candidates.append(card)
    require(len(candidates) <= 1, 'multiple accepted ancestor handoffs require reconciliation')
    return submit_controller(candidates[0]) if candidates else None


def advance_frontier(campaign_path, root, campaign, controller, states, passes=3):
    import verification_study_v3_autostages_r2 as stages
    require(passes > 0, 'finite stage pass budget required'); consumed = []
    for index in range(passes):
        frontier = stages.next_actions(campaign_path, root, submit=True, observations=states)
        rows = scientific_submissions(root, campaign); fresh = observations([r['job_id'] for r in rows])
        transitions = [j for j, value in fresh.items() if value.get('state') in TERMINAL and not value.get('live') and
            (states.get(j, {}).get('state') not in TERMINAL or states.get(j, {}).get('live'))]
        if not transitions:
            return frontier, fresh, consumed
        consumed += observe_terminal(campaign_path, root, campaign, rows, fresh, controller); states = fresh
        if index == passes - 1:
            return dict(frontier, ready_cpu_actions=True, unconsumed_terminal_job_ids=transitions), fresh, consumed
    raise AssertionError('finite frontier loop has no result')


def pin_runtime_environment(root, current):
    """Reassert private paths after srun TaskProlog, before tokenizer imports."""
    from verification_study_v3_transport_r2 import pin_runtime_environment as pin
    return pin(private_path(root / 'automation/runtime' / current))


def run(card_path):
    from moe_exp.verification_study.closeout import require_cpu
    require_cpu()
    current = os.environ['SLURM_JOB_ID']; controller = load(card_path, 'verification-autodriver-card-v3')
    root = private_path(controller['result_root']); pin_runtime_environment(root, current)
    require(getpass.getuser() == 'lmolfett' and socket.gethostname().endswith('leonardo.local'), 'wrong allocated host/user')
    campaign_path = Path(controller['campaign_path']); campaign = validate_card(controller)
    require(set(campaign['models']) == set(MODEL_KEYS), 'all-seven campaign differs')
    receipt = accepted_receipt(controller); require(receipt and receipt['job_id'] == current, 'untracked controller allocation')
    proof = load(slot_for(controller) / 'PREARM.json')
    require(proof['controller_card_sha256'] == controller['sha256'] and proof['primary_job_id'] == current,
            'allocated controller lacks startup prearm proof')
    run_dir = root / 'automation/runs' / current
    children = proof['children']; rescue = children[0] if children else None
    if controller['mode'] == 'rescue':
        next_receipt = accepted_successor(root, controller['failure_ancestor_job_ids'], current)
        if next_receipt:
            state = observations([next_receipt['job_id']]).get(next_receipt['job_id'], {})
            if state.get('live') or state.get('state') == 'COMPLETED':
                return save(run_dir / 'RESULT.json', {'schema': 'verification-autodriver-result-v3',
                    'campaign_sha256': campaign['sha256'], 'controller_job_id': current,
                    'status': 'EXISTING_HANDOFF_REUSED', 'successor_job_id': next_receipt['job_id'], 'new_allocations': 0})
    with orchestration_lock(root), submission_transport():
        save(run_dir / 'START.json', {'schema': 'verification-autodriver-start-v3', 'campaign_sha256': campaign['sha256'],
            'controller_card_sha256': controller['sha256'], 'controller_job_id': current, 'started_epoch': time.time(),
            'operational_version': 'autodriver-r2', 'prearm_sha256': proof['sha256']})
        repaired = reconcile_scientific_intents(root, campaign)
        rows = scientific_submissions(root, campaign); states = observations([r['job_id'] for r in rows])
        events = observe_terminal(campaign_path, root, campaign, rows, states, controller)
        frontier, states, raced_events = advance_frontier(campaign_path, root, campaign, controller, states); events += raced_events
        ids = sorted([j for j, value in states.items() if value.get('live')], key=int)
        successor = None; successor_card = None; deferred = None
        if ids:
            successor_card = controller_card(campaign_path, 'successor', parent=current, wait_ids=ids, files=controller['program_files'])
        elif frontier.get('ready_cpu_actions'):
            require(frontier.get('actions') or frontier.get('unconsumed_terminal_job_ids'), 'empty immediate frontier forbidden')
            successor_card = controller_card(campaign_path, 'ready-cpu', parent=current, files=controller['program_files'])
        if successor_card:
            try:
                successor = submit_controller(successor_card)
            except SerialSlotLimit as error:
                deferred = save(run_dir / 'HANDOFF_DEFERRED.json', {'schema': 'verification-autodriver-slot-deferral-v3',
                    'controller_card_sha256': controller['sha256'], 'next_card_sha256': successor_card['sha256'],
                    'reason': str(error), 'existing_prearm_sha256': proof['sha256'], 'cancelled_jobs': [], 'new_allocations': 0})
        if successor:
            save(run_dir / 'HANDOFF.json', {'schema': 'verification-autodriver-handoff-v3', 'controller_job_id': current,
                'successor_job_id': successor['job_id'], 'successor_card_sha256': successor_card['sha256'], 'wait_job_ids': ids,
                'submission_path': str(slot_for(successor_card) / 'SUBMISSION.json'),
                'prearm_sha256': load(slot_for(successor_card) / 'PREARM.json')['sha256'],
                'release_sha256': load(slot_for(successor_card) / 'RELEASE.json')['sha256']})
        status = ('WAITING_FOR_SCIENCE' if successor else 'STOPPED_SERIAL_SLOT_LIMIT' if deferred else
                  'CAMPAIGN_COMPLETE' if frontier.get('status') == 'COMPLETE' else 'STOPPED_WITH_EXPLICIT_ACTIONS')
        value = {'status': status, 'frontier': frontier, 'reconciled_submission_job_ids': repaired,
            'terminal_events': events, 'science_observations': states, 'successor_job_id': successor['job_id'] if successor else None,
            'rescue_job_id': rescue['job_id'] if rescue else None, 'slot_deferral_sha256': deferred['sha256'] if deferred else None,
            'scientific_success': 'ONLY_FROM_VALIDATED_ARTIFACTS_AND_GATES', 'operational_version': 'autodriver-r2'}
        result = publish(campaign_path, root, campaign, run_dir, value)
        if status == 'STOPPED_WITH_EXPLICIT_ACTIONS' and frontier.get('errors'):
            from verification_study_v3_codex_review_r1 import review
            diagnostic = review(campaign_path, frontier)
            save(run_dir / 'CODEX_DIAGNOSTIC.json', {'schema': 'verification-autodriver-diagnostic-v3',
                'campaign_sha256': campaign['sha256'], 'status': diagnostic['status'],
                'private_review_sha256': diagnostic.get('sha256'), 'interpretation': 'read-only; zero repair allocations'})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bootstrap', 'release', 'run', 'inspect'))
    parser.add_argument('--campaign', type=Path, default=REPO / 'report/verification-study-v3/CAMPAIGN.json')
    parser.add_argument('--card', type=Path); parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--defer-release', action='store_true'); args = parser.parse_args()
    if args.command == 'bootstrap':
        card = controller_card(args.campaign.resolve(), 'bootstrap')
        result = submit_controller(card, test_only=args.test_only, defer_release=args.defer_release)
        if not args.test_only:
            result = {**result, 'controller_card': str(slot_for(card) / 'CARD.json'),
                'prearm_sha256': load(slot_for(card) / 'PREARM.json')['sha256'], 'primary_held': args.defer_release}
    elif args.command == 'release':
        require(args.card is not None and not args.test_only, 'release requires exact existing card')
        result = release_controller(load(args.card.resolve()))
    elif args.command == 'run':
        try:
            result = run(args.card.resolve())
        except Exception as error:
            card = load(args.card); root = Path(card['result_root']); current = os.environ['SLURM_JOB_ID']
            prearm = slot_for(card) / 'PREARM.json'
            save(root / 'automation/runs' / current / 'FAILURE.json', {'schema': 'verification-autodriver-failure-v3',
                'campaign_sha256': card['campaign_sha256'], 'controller_job_id': current, 'controller_card_sha256': card['sha256'],
                'error_type': type(error).__name__, 'error': str(error)[:2000], 'operational_version': 'autodriver-r2',
                'status': 'CONTROLLER_FAILURE_PREARMED_RESCUE' if prearm.exists() and load(prearm)['children'] else 'CONSECUTIVE_RESCUE_LIMIT_REACHED',
                'consecutive_rescues': card['consecutive_rescues'], 'observed_epoch': time.time(), 'scientific_null': False})
            raise
    else:
        campaign = load(args.campaign); check_files(campaign['code_files'])
        rows = scientific_submissions(Path(campaign['result_root']), campaign)
        result = {'campaign_sha256': campaign['sha256'], 'science_observations': observations([r['job_id'] for r in rows])}
    print(json.dumps({key: value for key, value in result.items() if key in {
        'status', 'job_id', 'controller_card', 'prearm_sha256', 'primary_held', 'successor_job_id', 'rescue_job_id',
        'test_only', 'scheduler_output', 'science_observations'}}, default=str), flush=True)


if __name__ == '__main__':
    main()
