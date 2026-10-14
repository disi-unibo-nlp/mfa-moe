"""Same protected transfer contract, independent reader-price amendment journal."""
from pathlib import Path
from moe_exp.research_campaign_v3_r6.contracts import load,locked,publish,require
from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import require_readiness,retirable
from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1 import check_binding
def transfer_latest(campaign, scheduler=None):
    """Replace only zero-runtime scientific controllers, after accepting all owners."""
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm, protected_controller
    from moe_exp.research_campaign_v3_r6.controller import _check, jobs
    from moe_exp.research_campaign_v3_r6_ops_r1.milestone import terminal_observation
    require_readiness(campaign)
    scheduler = scheduler or Slurm()
    root = Path(campaign['root']); directory = root / 'reader_price_failure_id_r1_transfer'
    with locked(root, 'controller'):
        done = directory / 'TRANSFER.json'
        if done.exists():
            return load(done)
        require((root / 'LAUNCH.json').exists(), 'Original protected launch is not accepted')
        intent_path = directory / 'INTENT.json'
        if intent_path.exists():
            intent = load(intent_path)
        else:
            observed = jobs(campaign, scheduler)
            owners = [j for j in observed if not j['accounting']['terminal'] and
                      j['card']['stage'] == 'controller' and not j['card'].get('ops_report_only')]
            if not owners or not all(retirable(j['card'], j['accounting']) for j in owners):
                return {'status': 'waiting_for_pending_scientific_owners', 'job_submissions': 0}
            newest = max((load(p) for p in (root / 'handoffs').glob('*.json')), key=lambda h: h['generation'])
            expected = {newest['primary']['job_id'], *[r['job_id'] for r in newest['guards']]}
            require({j['accepted']['job_id'] for j in owners} == expected,
                    'Current scientific ownership does not match the immutable handoff')
            snapshot = scheduler.snapshot()
            require(len([j for j in snapshot['jobs'] if j['partition'] == 'lrd_all_serial']) <= 7,
                    'No room to accept three scientific replacement owners')
            admission = _check(campaign, scheduler, observed, hours=16, bytes_=0)
            require(admission['admitted'], 'Controller ownership transfer cannot be admitted')
            intent = publish(intent_path, {'generation': newest['generation'] + 1,
                'owners': owners, 'wait_ids': sorted(set(newest['wait_ids']) | expected),
                'bounded_admission': admission, 'binding_sha256': check_binding(campaign)['sha256']})
        for old in intent['owners']:
            jid = old['accepted']['job_id']
            hold = directory / 'held' / (jid + '.json')
            retired = directory / 'retired' / (jid + '.json')
            if hold.exists() or retired.exists():
                continue
            before = scheduler.observe(jid)
            require(retirable(old['card'], before), 'Scientific owner started before hold')
            scheduler.command(['scontrol', 'hold', jid])
            raw = scheduler.command(['scontrol', 'show', 'job', '-o', jid])
            require('Reason=JobHeldUser' in raw, 'Scientific owner hold not verified')
            publish(hold, {'job_id': jid, 'before': before, 'after': raw})
        handoff = protected_controller(campaign, scheduler, generation=intent['generation'],
            wait_ids=intent['wait_ids'], release=False)
        accepted = [handoff['primary'], *handoff['guards']]
        for receipt in accepted:
            value = load(root / 'cards' / (receipt['card_sha256'] + '.json'))
            scheduler.verify(receipt['job_id'], value)
        all_accepted = publish(directory / 'ACCEPTED.json', {'handoff': handoff,
            'binding_sha256': intent['binding_sha256'], 'original_intent_sha256': intent['sha256']})
        for old in intent['owners']:
            jid = old['accepted']['job_id']; path = directory / 'retired' / (jid + '.json')
            if path.exists():
                continue
            before = scheduler.observe(jid)
            require(retirable(old['card'], before), 'Scientific owner is not safely pending')
            scheduler.command(['scancel', jid]); after = terminal_observation(scheduler, jid)
            require(after['state'] == 'CANCELLED' and after['elapsed_seconds'] == 0 and
                    after['start'] in ('None', 'Unknown', 'N/A', '', None), 'Zero-runtime retirement not verified')
            publish(path, {'job_id': jid, 'before': before, 'after': after,
                           'accepted_replacements_sha256': all_accepted['sha256']})
        protected_controller(campaign, scheduler, generation=intent['generation'], release=True)
        for receipt in accepted:
            value = load(root / 'cards' / (receipt['card_sha256'] + '.json'))
            proof = scheduler.verify(receipt['job_id'], value)
            require(proof['fields'].get('Reason') != 'JobHeldUser', 'New scientific owner remains held')
        return publish(done, {'status': 'complete', 'accepted_sha256': all_accepted['sha256'],
            'binding_sha256': intent['binding_sha256'], 'preserved_reporting_owners':
            ['60509618', '60509619', '60509620', '60509621'], 'automatic_retries': 0})
