"""Preserve protected serial owners while admitting CPU work on Booster CPUs."""
from pathlib import Path
import re

from moe_exp.research_campaign_v3_r6.contracts import (
    REPO, check_files, digest, load, locked, publish, require, seal,
)
from moe_exp.research_campaign_v3_r6_ops_r1 import policy as base

BINDING = 'CPU_DISPATCH_R1_BINDING.json'
PROOF = 'CPU_DISPATCH_R1_PREFLIGHT.json'
WRAPPER = REPO / 'scripts/research_campaign_v3_r6_cpu_dispatch_r1/job.sbatch'
SERIAL_OWNERS = frozenset(('controller', 'snapshot', 'validate'))


def check_binding(campaign):
    original = base.check_binding(campaign)
    binding = load(Path(campaign['root']) / BINDING)
    check_files(binding['files'])
    require(binding['campaign_sha256'] == campaign['sha256'] and
            binding['ops_binding_sha256'] == original['sha256'], 'CPU dispatch ancestry differs')
    require(binding['remote_writes'] is False and binding['manuscript_writes'] is False and
            binding['additional_gpu_reservations'] == 0, 'CPU dispatch scope differs')
    return binding


def require_readiness(campaign):
    binding = check_binding(campaign)
    complete = base.require_readiness(campaign)
    proof = load(Path(campaign['root']) / PROOF)
    require(proof['passed'] and proof['binding_sha256'] == binding['sha256'] and
            proof['base_validation_sha256'] == complete['sha256'] and
            proof['actual_graph_sha256'] == load(Path(campaign['root']) / 'IMPORT_GRAPH_VALIDATION.json')['sha256'],
            'Matching CPU dispatch and full scientific proofs are required')
    return proof


def amend_card(value, binding_sha):
    # The independently protected report-only snapshot retains its accepted card.
    if value.get('ops_report_only'):
        return value
    amended = {k: v for k, v in value.items() if k not in ('sha256', 'comment')}
    amended['cpu_dispatch_binding_sha256'] = binding_sha
    amended['cpu_dispatch_original_partition'] = value['partition']
    if value['partition'] == 'lrd_all_serial' and value['stage'] not in SERIAL_OWNERS:
        require(value['gpus'] == 0 and value['cpus'] == 4 and value['memory_gib'] == 24,
                'Unregistered CPU resource shape')
        amended['partition'] = 'boost_usr_prod'
    amended['comment'] = 'rc3:' + digest(amended)
    return seal(amended)


def retirable(value, accounting):
    return (value['stage'] == 'controller' and not value.get('ops_report_only') and
            accounting['state'] == 'PENDING' and accounting['elapsed_seconds'] == 0 and
            accounting['start'] in ('None', 'Unknown', 'N/A', '', None))


def install(campaign):
    from moe_exp.research_campaign_v3_r6 import controller, scheduler
    binding = check_binding(campaign)
    if getattr(scheduler.Slurm, '_cpu_dispatch_installed', False):
        require(scheduler.Slurm._cpu_dispatch_binding == binding['sha256'], 'CPU dispatch binding differs')
        return
    require(not getattr(scheduler.Slurm, '_ops_r1_installed', False),
            'Install the CPU dispatch amendment at the process entrypoint')
    native_verify = scheduler.Slurm.verify
    original_wrapper = base.WRAPPER
    base.install(campaign)
    original_card, original_verify = scheduler.card, scheduler.Slurm.verify
    base.WRAPPER = WRAPPER  # Process-local; none of the pending source files changes.

    def card(*args, **kwargs):
        return amend_card(original_card(*args, **kwargs), binding['sha256'])

    def verify(self, jid, value):
        if value.get('ops_report_only'):
            proof = native_verify(self, jid, value)
            require(value['ops_binding_sha256'] == binding['ops_binding_sha256'] and
                    proof['fields']['Command'] == str(original_wrapper), 'Reporting owner differs')
            return proof
        require(value.get('cpu_dispatch_binding_sha256') == binding['sha256'], 'CPU worker amendment missing')
        proof = original_verify(self, jid, value)
        if value['partition'] == 'boost_usr_prod' and value['gpus'] == 0:
            tres = proof['fields'].get('ReqTRES', '')
            require(not re.search(r'(?:^|,)gres/gpu=', tres), 'CPU worker reserved a GPU')
            require(proof['resources']['billing_units'] <= value['billing_units_upper'],
                    'CPU worker exceeds the preserved accounting price')
        return proof

    scheduler.card = controller.card = card
    scheduler.Slurm.verify = verify
    scheduler.Slurm._cpu_dispatch_installed = True
    scheduler.Slurm._cpu_dispatch_binding = binding['sha256']


def transfer(campaign, scheduler=None):
    """Replace only zero-runtime scientific controllers, after accepting all owners."""
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm, protected_controller
    from moe_exp.research_campaign_v3_r6.controller import _check, jobs
    from moe_exp.research_campaign_v3_r6_ops_r1.milestone import terminal_observation
    require_readiness(campaign)
    scheduler = scheduler or Slurm()
    root = Path(campaign['root']); directory = root / 'cpu_dispatch_r1'
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
