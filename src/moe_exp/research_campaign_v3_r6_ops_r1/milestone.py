"""Protected report-only milestone, independent of scientific validation."""
from pathlib import Path
import os
import time

from moe_exp.research_campaign_v3_r6.contracts import (
    allocated, digest, file_sha, load, locked, publish, require, seal,
)
from .policy import check_binding

OLD_OWNERS = ('60454552', '60462231', '60462233', '60462236')
BEGIN = '2026-10-15T21:59:59'  # Submission child TZ=UTC; October 15 23:59:59 Rome.


def marked(value):
    value = {k: v for k, v in value.items() if k not in ('comment', 'sha256')}
    value['ops_report_only'] = True
    value['comment'] = 'rc3:' + digest(value)
    return seal(value)


def terminal_observation(scheduler, jid):
    for _ in range(10):
        value = scheduler.observe(jid)
        if value['terminal']:
            return value
        time.sleep(1)
    raise ValueError('Terminal accounting not yet verified: ' + jid)


def protect(campaign, scheduler=None):
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm, card, submit_card
    from moe_exp.research_campaign_v3_r6.controller import _check, jobs
    scheduler = scheduler or Slurm()
    binding = check_binding(campaign)
    root = Path(campaign['root']); directory = root / 'ops_r1/milestone'
    with locked(root, 'ops_milestone'):
        complete = directory / 'PROTECTION.json'
        if complete.exists():
            return load(complete)
        preflight = load(root / 'OPS_R1_PREFLIGHT.json')
        require(preflight['passed'] and preflight['binding_sha256'] == binding['sha256'], 'Report adapter preflight missing')
        snapshot = scheduler.snapshot()
        require(len([j for j in snapshot['jobs'] if j['partition'] == 'lrd_all_serial']) <= 6,
                'No room for protected milestone replacement')
        check = _check(campaign, scheduler, jobs(campaign, scheduler), hours=28, bytes_=1024**2)
        require(check['admitted'], 'Protected report-only milestone cannot be admitted')
        # Holding retains ownership while preventing the sealed paper-writing job
        # from becoming runnable during this replacement transaction.
        hold_path = directory / 'LEGACY_SNAPSHOT_HOLD.json'
        if not hold_path.exists():
            before = scheduler.observe(OLD_OWNERS[0])
            require(before['state'] == 'PENDING' and before['elapsed_seconds'] == 0 and
                    before['start'] in ('None', 'Unknown', 'N/A', '', None), 'Legacy snapshot already started')
            scheduler.command(['scontrol', 'hold', OLD_OWNERS[0]])
            raw = scheduler.command(['scontrol', 'show', 'job', '-o', OLD_OWNERS[0]])
            require('Reason=JobHeldUser' in raw, 'Legacy snapshot hold was not verified')
            publish(hold_path, {'job_id': OLD_OWNERS[0], 'before': before, 'after_readback': raw})
        snapshot_card = marked(card(campaign, 'snapshot', minutes=180, hold=True, begin=BEGIN))
        snap = submit_card(campaign, snapshot_card, scheduler)
        primary_card = marked(card(campaign, 'controller', minutes=60, hold=True, generation=0,
                                   dependency='afterany:' + snap['job_id']))
        primary = submit_card(campaign, primary_card, scheduler)
        values, accepted = [snapshot_card, primary_card], [snap, primary]
        ancestors = [primary['job_id']]
        for index in (1, 2):
            value = marked(card(campaign, 'controller', shard=index, hold=True, generation=0,
                                minutes=180 if index == 2 else 60, ancestors=ancestors,
                                dependency='afternotok:' + ':'.join(reversed(ancestors))))
            receipt = submit_card(campaign, value, scheduler)
            values.append(value); accepted.append(receipt); ancestors.append(receipt['job_id'])
        ownership = publish(directory / 'ACCEPTED.json', {
            'snapshot': snap, 'primary': primary, 'guards': accepted[2:],
            'binding_sha256': binding['sha256'], 'report_only': True, 'reservation_hours': 28,
            'begin_utc': BEGIN, 'begin_local': '2026-10-15 23:59:59 Europe/Rome',
        })
        # All four successors have verified immutable cards before any retirement.
        for value, receipt in zip(values, accepted):
            scheduler.verify(receipt['job_id'], value)
        retirements = []
        for jid in OLD_OWNERS:
            path = directory / 'retirements' / (jid + '.json')
            if path.exists():
                retirements.append(load(path)); continue
            before = scheduler.observe(jid)
            if not before['terminal']:
                require(before['state'] == 'PENDING' and before['elapsed_seconds'] == 0 and
                        before['start'] in ('None', 'Unknown', 'N/A', '', None), 'Legacy operational owner is not safely pending')
                scheduler.command(['scancel', jid])
            after = terminal_observation(scheduler, jid)
            require(after['state'] == 'CANCELLED' and after['elapsed_seconds'] == 0 and
                    after['start'] in ('None', 'Unknown', 'N/A', '', None), 'Zero-runtime retirement not proven')
            retirements.append(publish(path, {'job_id': jid, 'before': before, 'after': after,
                                              'accepted_successors_sha256': ownership['sha256']}))
        for value, receipt in reversed(list(zip(values, accepted))):
            path = root / 'submissions' / value['sha256'] / 'RELEASED.json'
            if not path.exists():
                scheduler.release(receipt['job_id'])
                proof = scheduler.verify(receipt['job_id'], value)
                require(proof['fields'].get('Reason') != 'JobHeldUser', 'Milestone successor remains held')
                publish(path, {'job_id': receipt['job_id'], 'proof': proof})
        return publish(complete, {'accepted_sha256': ownership['sha256'], 'retirements': retirements,
                                 'bounded_admission': check, 'scientific_work_released': False})


def export_fallback(campaign):
    """Retain completed r5 report plus r6 evidence without regenerating cell banks."""
    root = Path(campaign['root']); directory = root / 'ops_r1/milestone/output'
    path = directory / 'MILESTONE_REPORT.json'
    if path.exists():
        value = load(path)
        require(file_sha(directory / 'report.md') == value['report_md_sha256'], 'Milestone report bytes differ')
        return value
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = Path(campaign['resume_from']) / 'output/REPORT.json'
    previous = load(source)
    partial_path = root / 'PARTIAL_VALIDATION_EVIDENCE.json'
    partial = load(partial_path)
    references = {str(source): file_sha(source), str(partial_path): file_sha(partial_path)}
    for name in ('OPS_R1_VALIDATION_COMPLETED.json', 'OPS_R1_VALIDATION_TERMINAL.json', 'READINESS.json', 'LAUNCH.json'):
        current = root / name
        if current.exists():
            references[str(current)] = file_sha(current)
    statepath = directory / 'STATE.json'
    state = load(statepath) if statepath.exists() else publish(statepath, {
        'source_report_sha256': previous['sha256'], 'partial_evidence_sha256': partial['sha256'],
        'references': references, 'observed_epoch': time.time(), 'SCIENCE_COMPLETE': False,
    })
    text = ('# October 15 report-only milestone\n\n'
            'The retained r5 report is complete; the seven-model scientific study remains incomplete.\n\n'
            f"Retained report: {source}\n\n"
            'R6 retains all seven native fixtures, six REAP qualifications and the GPT/Nemotron coordination banks. '
            'Compact accuracy–memory curves and new causal steering effects are not established. '
            'The operations validation outcome and latest launch receipt are referenced below.\n\n' +
            '\n'.join('- ' + name + ': ' + sha for name, sha in state['references'].items()) + '\n')
    output = directory / 'report.md'
    if output.exists():
        require(output.read_text() == text, 'Interrupted milestone report differs')
    else:
        with output.open('x') as handle:
            handle.write(text)
    return publish(path, {'status': 'complete', 'REPORT_COMPLETE': True, 'SCIENCE_COMPLETE': False,
        'source_report_sha256': state['source_report_sha256'], 'partial_evidence_sha256': state['partial_evidence_sha256'],
        'state_sha256': state['sha256'], 'references': state['references'], 'report_md_sha256': file_sha(output),
        'manuscript_updated': False, 'remote_writes': 0, 'scientific_completion_not_inferred': True})


def worker(campaign, value, scheduler=None):
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm
    allocated(); check_binding(campaign)
    require(value['ops_report_only'], 'Report-only milestone flag is missing')
    scheduler = scheduler or Slurm()
    root = Path(campaign['root']); base = root / 'ops_r1/milestone'
    path = base / 'workers' / value['sha256'] / 'RESULT.json'
    if path.exists():
        return load(path)
    # Snapshot plus primary permit one full-export recovery. Further failure
    # owners use the already completed compact fallback rather than retrying
    # the failed full export or creating a scientific successor.
    full_export_allowed = value['stage'] == 'snapshot' or value['shard_index'] == 0
    if (root / 'OPS_R1_VALIDATION_COMPLETED.json').exists() and full_export_allowed:
        from .policy import require_readiness
        from moe_exp.research_campaign_v3_r6.export import closeout
        require_readiness(campaign)
        closeout(campaign, final=False)
        report = load(root / 'snapshots/2026-10-15/REPORT.json')
    else:
        report = export_fallback(campaign)
    owners = load(base / 'ACCEPTED.json')
    if value['stage'] == 'controller':
        current = os.environ['SLURM_JOB_ID']
        for item in owners['guards']:
            jid = item['job_id']
            if jid == current:
                continue
            afterpath = base / 'guard_retirements' / (jid + '.json')
            if afterpath.exists():
                continue
            before = scheduler.observe(jid)
            if before['terminal']:
                continue
            require(before['state'] == 'PENDING' and before['elapsed_seconds'] == 0,
                    'Report-only guard is active')
            scheduler.command(['scancel', jid])
            after = terminal_observation(scheduler, jid)
            require(after['state'] == 'CANCELLED' and after['elapsed_seconds'] == 0, 'Milestone guard retirement differs')
            publish(afterpath, {'before': before, 'after': after, 'report_sha256': report['sha256']})
    return publish(path, {'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'],
                         'report_sha256': report['sha256'], 'scientific_work_released': False})
