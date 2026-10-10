"""Reversible hold of one exact unstarted controller chain for a format amendment.

No experimental allocation is stopped, resubmitted or cancelled. Persisted
intent and observed holds survive a Codex interruption.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
from moe_exp.verification_study.contracts import check_files, file_sha, load, require, save
import verification_study_v3_autodriver_r4 as driver

CAMPAIGN = REPO / 'report/verification-study-v3/CAMPAIGN.json'
EXPECTED = ('59662113', '59662139', '59662142')


def readback(job, card):
    value = driver.command(['scontrol', 'show', 'job', job, '-o'])
    fields = dict(re.findall(r'(\w+)=([^ ]*)', value))
    require(fields.get('JobId') == job and fields.get('UserId', '').startswith('lmolfett(')
            and fields.get('Account') == 'iscrc_miosr'
            and fields.get('Partition') == 'lrd_all_serial'
            and fields.get('Comment') == 'cv3auto:' + card['sha256'], 'foreign controller readback')
    require(fields.get('JobState') == 'PENDING' and fields.get('StartTime') == 'Unknown'
            and fields.get('AllocTRES') == '(null)' and fields.get('RunTime') == '00:00:00',
            'only genuinely unstarted controllers can be held')
    return value, fields


def main():
    campaign = load(CAMPAIGN); check_files(campaign['code_files'])
    root = Path(campaign['result_root'])
    with driver.orchestration_lock(root):
        handoff_path = root / 'automation/runs/59661283/HANDOFF.json'
        handoff = load(handoff_path)
        require(handoff['controller_job_id'] == '59661283'
                and handoff['successor_job_id'] == EXPECTED[0], 'handoff changed')
        accounting = subprocess.run(['sacct', '-nPX', '-j', '59659714,59661283',
            '-o', 'JobID,State,ExitCode,ElapsedRaw'], check=True, capture_output=True,
            text=True, timeout=30).stdout
        rows = {r.split('|')[0]: r.split('|')[1:] for r in accounting.strip().splitlines()}
        require(all(j in rows and rows[j][:2] == ['COMPLETED', '0:0']
                    for j in ('59659714', '59661283')), 'reader or parent lacks successful terminal accounting')
        pending = [Path(handoff['submission_path']).parent]; chain = []
        while pending:
            slot = pending.pop(0); card = load(slot / 'CARD.json')
            driver.validate_card(card)
            sub = load(slot / 'SUBMISSION.json'); proof = load(slot / 'PREARM.json')
            require(sub['job_id'] == proof['primary_job_id']
                    and proof['controller_card_sha256'] == card['sha256'], 'prearm identity differs')
            chain.append((sub['job_id'], card, slot, sub, proof))
            pending.extend(Path(c['submission_path']).parent for c in proof['children'])
        require(tuple(row[0] for row in chain) == EXPECTED, 'not the exact approved amendment hold chain')
        before = {job: readback(job, card)[0] for job, card, *_ in chain}
        intent = save(CAMPAIGN.parent / 'SEMANTIC_FORMAT_HOLD_INTENT-r1.json', {
            'schema': 'verification-semantic-format-hold-intent-v3',
            'campaign_sha256': campaign['sha256'], 'source_file_sha256': file_sha(__file__),
            'handoff_sha256': handoff['sha256'], 'targets': [
                {'job_id': j, 'controller_card_sha256': c['sha256'],
                 'submission_sha256': s['sha256'], 'prearm_sha256': p['sha256'], 'slot': str(slot)}
                for j, c, slot, s, p in chain],
            'reason': 'prospective versioned semantic JSON-fence interpretation before nomination or treatment',
            'experimental_jobs': 'all native discovery, audit and raw reader allocations continue unchanged'})
        for job, card, *_ in chain:
            _, fields = readback(job, card)
            if fields.get('Reason') != 'JobHeldUser':
                driver.command(['scontrol', 'hold', job])
            _, fields = readback(job, card)
            require(fields.get('Reason') == 'JobHeldUser', 'controller hold was not observed')
        after = {job: readback(job, card)[0] for job, card, *_ in chain}
        result = save(CAMPAIGN.parent / 'SEMANTIC_FORMAT_HOLD-r1.json', {
            'schema': 'verification-semantic-format-hold-v3',
            'campaign_sha256': campaign['sha256'], 'intent_sha256': intent['sha256'],
            'source_file_sha256': file_sha(__file__), 'targets': intent['targets'],
            'observed_holds': after, 'no_experimental_job_changed': True,
            'parent_accounting': accounting})
        print(json.dumps({'status': 'EXACT_CONTROLLER_CHAIN_HELD', 'jobs': list(EXPECTED),
                          'sha256': result['sha256'], 'experimental_allocations': 0}))


if __name__ == '__main__':
    main()
