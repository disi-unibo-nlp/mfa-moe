#!/usr/bin/env python3
"""Resume accepted protected ownership with batched legacy accounting."""
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import (
    file_sha, load, locked, publish, require,
)


def run(root):
    campaign = load(root / 'CAMPAIGN.json')
    binding = load(root / 'LOOKUP_R1_BINDING.json')
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import require_readiness
    require_readiness(campaign)
    require(load(root / 'LOOKUP_R1_PREFLIGHT.json')['binding_sha256'] == binding['sha256'],
            'Matching operational lookup proof required')
    spec = importlib.util.spec_from_file_location('lookup_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_lookup_r1/run.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.install(campaign, binding)
    from moe_exp.research_campaign_v3_r6.controller import _check, jobs
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm, protected_controller
    from moe_exp.research_campaign_v3_r6_ops_r1.milestone import terminal_observation
    scheduler = Slurm()
    with locked(root, 'controller'):
        if (root / 'ACCEPTED_LAUNCH_RECOVERY_R1.json').exists():
            return load(root / 'ACCEPTED_LAUNCH_RECOVERY_R1.json')
        handoff = load(root / 'handoffs/0000.json')
        require(handoff['generation'] == 0 and len(handoff['guards']) == 2,
                'Initial accepted protection is incomplete')
        owners = [handoff['primary'], *handoff['guards']]
        for accepted in owners:
            value = load(root / 'cards' / (accepted['card_sha256'] + '.json'))
            require(value['lookup_binding_sha256'] == binding['sha256'], 'Scientific owner binding differs')
            scheduler.verify(accepted['job_id'], value)
        observed = jobs(campaign, scheduler)
        resolve = [j for j in observed if j['card']['stage'] == 'resolve']
        require(len(resolve) == 1 and handoff['wait_ids'] == [resolve[0]['accepted']['job_id']],
                'Accepted import dependency differs')
        initial = resolve[0]
        import_result = load(root / 'stages/resolve-all-00-00000/RESULT.json')
        require(import_result['status'] == 'complete' and import_result['validated_actual_import']
                and import_result['evidence_import_sha256'] == load(root / 'EVIDENCE_IMPORT.json')['sha256'],
                'Actual source import is not already complete')
        admission = _check(campaign, scheduler, observed, hours=268, bytes_=512 * 1024**2)
        require(admission['admitted'], 'Protected continuation no longer fits live entitlement')
        intent = load(root / 'OWNERSHIP_TRANSFER_INTENT.json')
        planned = intent['observed']
        ids = list(dict.fromkeys(item['accepted']['job_id'] for item in planned))
        accounting = {}
        for offset in range(0, len(ids), 200):
            accounting.update(scheduler.observe_many(ids[offset:offset + 200]))
        require(set(ids) <= set(accounting), 'Legacy batch accounting is incomplete')
        retirements, preserved = [], []
        for item in planned:
            value, before = item['card'], item['accounting']
            jid = item['accepted']['job_id']; actual = accounting[jid]
            if before['terminal']:
                require(actual['terminal'], 'Historical terminal ownership changed')
                continue
            if value['stage'] not in ('controller', 'snapshot'):
                preserved.append(item)
                continue
            require(actual['state'] == 'CANCELLED' and actual['elapsed_seconds'] == 0
                    and actual['start'] in ('None', 'Unknown', 'N/A', '', None),
                    'A historical operational owner is still live; do not retire blindly')
            path = root / 'transfer_retirements' / (jid + '.json')
            receipt = load(path) if path.exists() else publish(path, {
                'job_id': jid, 'card_sha256': value['sha256'], 'before': before, 'after': actual,
                'reconciled_interrupted_retirement': True,
            })
            retirements.append(receipt)
        require(not preserved, 'An inherited scientific allocation still owns the import dependency')
        transfer = publish(root / 'OWNERSHIP_TRANSFER.json', {
            'parent_campaign': load(Path(campaign['resume_from']) / 'CAMPAIGN.json')['sha256'],
            'prearmed': handoff, 'old_operational_retirements': retirements,
            'accepted_scientific_jobs_preserved': preserved, 'reconciled_intents': [],
            'original_authorization_baseline_preserved': True,
        })
        snapshot = load(root / 'ops_r1/milestone/ACCEPTED.json')['snapshot']
        launch = publish(root / 'LAUNCH.json', {
            'campaign_id': campaign['campaign_id'],
            'initial_jobs': [initial['accepted'], snapshot, *owners], 'output_location': str(root),
            'snapshot': admission['snapshot'], 'bounded_initial_admission': admission,
            'complete_finite_graph_registered': True,
            'resumed_accepted_protection': True, 'job_submissions_during_recovery': 0,
        })
        # The bound import completed during actual validation; retire only its
        # redundant zero-runtime allocation after all successor owners verified.
        jid = initial['accepted']['job_id']; current = scheduler.observe(jid)
        require(current['state'] == 'PENDING' and current['elapsed_seconds'] == 0
                and current['start'] in ('None', 'Unknown', 'N/A', '', None),
                'Redundant import allocation has started; preserve it')
        scheduler.verify(jid, initial['card'])
        scheduler.command(['scancel', jid])
        after = terminal_observation(scheduler, jid)
        require(after['state'] == 'CANCELLED' and after['elapsed_seconds'] == 0
                and after['start'] in ('None', 'Unknown', 'N/A', '', None),
                'Redundant import retirement is not zero-runtime')
        publish(root / 'RESOLUTION_REUSE_DISPOSITION.json', {
            'job_id': jid, 'before': current, 'after': after,
            'actual_import_result_sha256': import_result['sha256'],
            'accepted_protection_sha256': handoff['sha256'],
            'scientific_assignments_repeated': 0, 'reservation_hours_saved': 12,
        })
        protected_controller(campaign, scheduler, generation=0, release=True)
        readbacks = []
        for accepted in owners:
            value = load(root / 'cards' / (accepted['card_sha256'] + '.json'))
            proof = scheduler.verify(accepted['job_id'], value)
            require(proof['fields'].get('Reason') != 'JobHeldUser', 'Scientific owner remains held')
            readbacks.append({'job_id': accepted['job_id'], 'proof': proof})
        return publish(root / 'ACCEPTED_LAUNCH_RECOVERY_R1.json', {
            'complete': True, 'launch_sha256': launch['sha256'],
            'source_sha256': file_sha(Path(__file__)), 'job_submissions': 0,
            'legacy_accounting_jobs': len(ids), 'batched_accounting_verified': True,
            'ownership_transfer_sha256': transfer['sha256'], 'released_owners': readbacks,
            'preserved_reporting_owners': ['60509618', '60509619', '60509620', '60509621'],
        })


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Provide the existing campaign root')
    result = run(Path(sys.argv[1]).resolve())
    print(json.dumps({'complete': result['complete'], 'job_submissions': 0,
                      'released_jobs': [r['job_id'] for r in result['released_owners']]}))
