"""Resume an exact protected r4-to-r5 semantic interpretation handoff.

Accept and verify the held replacement and both rescues before retiring three
genuinely unstarted controllers. Keep every scientific allocation and raw draw.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
import verification_study_v3_autodriver_r4 as old
import verification_study_v3_autodriver_r5 as driver
import verification_study_v3_semantic_format_r1 as fmt

CAMPAIGN = REPO / 'report/verification-study-v3/CAMPAIGN.json'
TARGETS = ('59662113', '59662139', '59662142')


def live_unstarted(job, card):
    text = driver.command(['scontrol', 'show', 'job', job, '-o'])
    fields = dict(re.findall(r'(\w+)=([^ ]*)', text))
    require(fields.get('JobId') == job and fields.get('UserId', '').startswith('lmolfett(')
            and fields.get('Account') == 'iscrc_miosr' and fields.get('Partition') == 'lrd_all_serial'
            and fields.get('Comment') == 'cv3auto:' + card['sha256'], 'foreign transfer target')
    require(fields.get('JobState') == 'PENDING' and fields.get('Reason') == 'JobHeldUser'
            and fields.get('StartTime') == 'Unknown' and fields.get('RunTime') == '00:00:00'
            and fields.get('AllocTRES') == '(null)', 'transfer target is not held and unstarted')
    return text


def cancellation_accounting(jobs):
    text = driver.command(['sacct', '-nPX', '-j', ','.join(jobs),
        '-o', 'JobID,State,ExitCode,ElapsedRaw,Start,End,AllocCPUS,Account,Partition,Comment%128'])
    result = {}
    for line in text.splitlines():
        fields = line.split('|')
        if fields[0] not in jobs:
            continue
        require(len(fields) >= 10 and fields[0] not in result, 'ambiguous transfer accounting')
        result[fields[0]] = {'job_id': fields[0], 'state': fields[1].split(' ')[0].rstrip('+'),
            'exit_code': fields[2], 'elapsed_seconds': int(fields[3]), 'start': fields[4],
            'end': fields[5], 'allocated_cpus': int(fields[6]), 'account': fields[7],
            'partition': fields[8], 'comment': fields[9].strip()}
    return result


def validate_cancelled(value, job, card):
    require(value['job_id'] == job and value['state'] == 'CANCELLED'
            and value['exit_code'] == '0:0' and value['elapsed_seconds'] == 0
            and value['allocated_cpus'] == 0 and value['start'] in {'None', 'Unknown'}
            and value['account'] == 'iscrc_miosr' and value['partition'] == 'lrd_all_serial'
            and value['comment'] == 'cv3auto:' + card['sha256'],
            'old controller was not an exact zero-allocation cancellation')


def live_rescue(card, submission):
    text = driver.readback(card, submission)
    fields = dict(re.findall(r'(\w+)=([^\s]*)', text))
    require(fields.get('JobId') == submission['job_id']
            and fields.get('UserId', '').startswith('lmolfett(')
            and fields.get('Account') == 'iscrc_miosr' and fields.get('Partition') == 'lrd_all_serial'
            and fields.get('Comment') == 'cv3auto:' + card['sha256'], 'foreign replacement rescue')
    require(fields.get('JobState') == 'PENDING' and fields.get('Reason') == 'Dependency',
            'replacement rescue must be live, pending and released before old retirement')
    require(set(driver.dependency_ids(text, 'afternotok')) == set(card['failure_ancestor_job_ids']),
            'replacement rescue actual failure ancestors differ')
    return text


def validate_protection(card, submission, *, observe_live=True):
    slot = driver.slot_for(card); proof = load(slot / 'PREARM.json')
    require(proof['controller_card_sha256'] == card['sha256']
            and proof['primary_job_id'] == submission['job_id']
            and proof['submission_sha256'] == submission['sha256'], 'replacement prearm differs')
    current = (card, submission, proof); seen = []
    while current:
        c, s, p = current; driver.validate_card(c)
        require(p['controller_card_sha256'] == c['sha256'] and p['primary_job_id'] == s['job_id'],
                'rescue protection identity differs')
        require(p['submission_sha256'] == s['sha256'], 'rescue prearm submission differs')
        if observe_live:
            if c['consecutive_rescues']:
                live_rescue(c, s)
            else:
                live_unstarted(s['job_id'], c)
        seen.append(s['job_id'])
        children = p['children']
        require(len(children) == (0 if c['consecutive_rescues'] == 2 else 1),
                'replacement requires exactly two failure rescues')
        if not children:
            current = None; continue
        child = children[0]; cs = Path(child['submission_path']).parent
        cc = load(cs / 'CARD.json'); cr = load(cs / 'SUBMISSION.json'); cp = load(cs / 'PREARM.json')
        release = load(cs / 'RELEASE.json')
        require(cc['sha256'] == child['controller_card_sha256'] and cc['parent_job_id'] == s['job_id']
                and cr['sha256'] == child['submission_sha256'] and cr['job_id'] == child['job_id']
                and cp['sha256'] == child['prearm_sha256'] and release['sha256'] == child['release_sha256']
                and release['controller_card_sha256'] == cc['sha256']
                and release['submission_sha256'] == cr['sha256'] and release['job_id'] == cr['job_id'],
                'replacement rescue journal differs')
        current = (cc, cr, cp)
    require(len(seen) == 3 and len(set(seen)) == 3, 'duplicate or incomplete replacement protection')
    return proof


def transfer():
    campaign = load(CAMPAIGN); check_files(campaign['code_files'])
    readiness = load(CAMPAIGN.parent / 'AUTOMATION_READINESS-r5.json')
    check_files(readiness['code_files'])
    require(readiness['campaign_sha256'] == campaign['sha256']
            and readiness['status'] == 'PASS', 'r5 readiness not qualified')
    binding = fmt.load_binding(CAMPAIGN)
    hold = load(CAMPAIGN.parent / 'SEMANTIC_FORMAT_HOLD-r1.json')
    require(hold['campaign_sha256'] == campaign['sha256']
            and tuple(v['job_id'] for v in hold['targets']) == TARGETS, 'foreign hold proof')
    root = Path(campaign['result_root'])
    with driver.orchestration_lock(root):
        card = driver.controller_card(CAMPAIGN, 'bootstrap')
        submission = driver.submit_controller(card, defer_release=True)
        release_started = any((driver.slot_for(card) / name).exists()
                              for name in ('RELEASE.json', 'RELEASE_INTENT.json'))
        if release_started:
            require((CAMPAIGN.parent / 'SEMANTIC_FORMAT_TRANSFER_INTENT-r1.json').exists()
                    and (CAMPAIGN.parent / 'HEALTH_RECONCILIATION-semantic-r5.json').exists(),
                    'primary release started without the committed exact retirement proof')
        observe_live = not release_started
        if release_started and not (driver.slot_for(card) / 'RELEASE.json').exists():
            # A committed intent can precede the scheduler mutation. A still-held
            # primary needs both actual rescues checked again before release.
            try:
                observed = driver.readback(card, submission)
                fields = dict(re.findall(r'(\w+)=([^\s]*)', observed))
                observe_live = (fields.get('JobState') == 'PENDING' and
                                (fields.get('Reason') == 'JobHeldUser' or fields.get('Priority') == '0'))
            except subprocess.CalledProcessError:
                matched = driver.comment_matches('cv3auto:' + card['sha256'])
                require(matched['job_id'] == submission['job_id']
                        and matched['partition'] == 'lrd_all_serial'
                        and matched.get('state', '').split(' ', 1)[0].rstrip('+') in driver.TERMINAL,
                        'aged transfer release intent lacks exact terminal accounting')
        prearm = validate_protection(card, submission, observe_live=observe_live)
        intent = save(CAMPAIGN.parent / 'SEMANTIC_FORMAT_TRANSFER_INTENT-r1.json', {
            'schema': 'verification-semantic-format-transfer-intent-v3',
            'campaign_sha256': campaign['sha256'], 'source_file_sha256': file_sha(__file__),
            'hold_sha256': hold['sha256'], 'readiness_sha256': readiness['sha256'],
            'semantic_format_binding_sha256': binding['sha256'], 'targets': hold['targets'],
            'replacement_controller_card_sha256': card['sha256'],
            'replacement_submission_sha256': submission['sha256'], 'replacement_job_id': submission['job_id'],
            'replacement_prearm_sha256': prearm['sha256']})
        targets = []
        for value in hold['targets']:
            slot = Path(value['slot']); old_card = load(slot / 'CARD.json'); old_sub = load(slot / 'SUBMISSION.json')
            old.validate_card(old_card)
            require(old_card['sha256'] == value['controller_card_sha256']
                    and old_sub['sha256'] == value['submission_sha256']
                    and old_sub['job_id'] == value['job_id'], 'held target journal changed')
            targets.append((value['job_id'], old_card, old_sub, slot))
        observations = cancellation_accounting(TARGETS)
        pending = [row for row in targets if observations.get(row[0], {}).get('state') != 'CANCELLED']
        if pending:
            live_unstarted(submission['job_id'], card)
            for job, old_card, *_ in pending:
                live_unstarted(job, old_card)
            for job, old_card, *_ in reversed(pending):
                live_unstarted(job, old_card)
                driver.command(['scancel', job])
        after = cancellation_accounting(TARGETS)
        rows = []
        for job, old_card, old_sub, slot in targets:
            require(job in after, 'cancelled controller accounting missing')
            validate_cancelled(after[job], job, old_card)
            rows.append({**after[job], 'card_path': str(slot / 'CARD.json'),
                'card_sha256': old_card['sha256'], 'submission_path': str(slot / 'SUBMISSION.json'),
                'submission_sha256': old_sub['sha256']})
        health_path = CAMPAIGN.parent / 'HEALTH_RECONCILIATION-semantic-r5.json'
        if health_path.exists():
            health = load(health_path)
            require(health['schema'] == 'verification-health-observer-reconciliation-v3'
                    and health['campaign_sha256'] == campaign['sha256']
                    and health['transfer_intent_sha256'] == intent['sha256']
                    and health['cancelled_unstarted_observers'] == rows and health['targets'] == list(TARGETS)
                    and health['replacement_controller_job_id'] == submission['job_id']
                    and health['replacement_card_sha256'] == card['sha256']
                    and health['replacement_prearm_sha256'] == prearm['sha256']
                    and health['native_science_cancellations'] == [], 'existing transfer cancellation proof differs')
        else:
            health = save(health_path, {'schema': 'verification-health-observer-reconciliation-v3',
                'campaign_sha256': campaign['sha256'], 'targets': list(TARGETS),
                'cancelled_unstarted_observers': rows, 'native_science_cancellations': [],
                'replacement_controller_job_id': submission['job_id'], 'replacement_card_sha256': card['sha256'],
                'replacement_prearm_sha256': prearm['sha256'], 'transfer_intent_sha256': intent['sha256']})
        release = driver.release_controller(card)
        result = save(CAMPAIGN.parent / 'SEMANTIC_FORMAT_TRANSFER-r1.json', {
            'schema': 'verification-semantic-format-transfer-v3', 'campaign_sha256': campaign['sha256'],
            'intent_sha256': intent['sha256'], 'semantic_format_binding_sha256': binding['sha256'],
            'replacement_job_id': submission['job_id'], 'replacement_release_sha256': release['sha256'],
            'health_reconciliation_sha256': health['sha256'], 'cancelled_unstarted_controllers': rows,
            'experimental_cancellations': [], 'additional_reader_draws': 0})
        return result


if __name__ == '__main__':
    value = transfer()
    print(json.dumps({'status': 'PROTECTED_SEMANTIC_HANDOFF', 'replacement_job_id': value['replacement_job_id'],
                      'sha256': value['sha256'], 'additional_reader_draws': 0}))
