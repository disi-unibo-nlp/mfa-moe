#!/usr/bin/env python3
"""Bound allocation interface over the unchanged reader/export/lookup sources."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import check_files, digest, load, require, seal


def install(campaign, binding, value=None, physical=None):
    from moe_exp.research_campaign_v3_r6 import scheduler, controller
    from moe_exp.research_campaign_v3_r6_ops_r1 import policy as base
    from moe_exp.research_campaign_v3_r6_cpu_allocation_r1 import install as install_guard, verification_target
    native_verify = scheduler.Slurm.verify
    spec = importlib.util.spec_from_file_location('preserved_lookup_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_lookup_r1/run.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.install(campaign, load(Path(campaign['root']) / 'LOOKUP_R1_BINDING.json'))
    original_card, original_verify = scheduler.card, scheduler.Slurm.verify
    base.WRAPPER = REPO / 'scripts/research_campaign_v3_r6_cpu_allocation_r1/job.sbatch'
    def card(*args, **kwargs):
        original = original_card(*args, **kwargs)
        if original.get('ops_report_only'):
            return original
        amended = {k: v for k, v in original.items() if k not in ('sha256', 'comment')}
        amended['cpu_allocation_binding_sha256'] = binding['sha256']
        amended['comment'] = 'rc3:' + digest(amended)
        return seal(amended)
    def verify(self, jid, candidate):
        if candidate.get('ops_report_only'):
            return original_verify(self, jid, candidate)
        if 'cpu_allocation_binding_sha256' not in candidate:
            require(candidate.get('lookup_binding_sha256') == binding['lookup_binding_sha256'],
                    'Legacy worker lookup binding differs')
            proof = native_verify(self, jid, candidate)
            require(proof['fields']['Command'] ==
                    str(REPO / 'scripts/research_campaign_v3_r6_lookup_r1/job.sbatch'),
                    'Legacy scientific entrypoint differs')
            return proof
        require(candidate['cpu_allocation_binding_sha256'] == binding['sha256'],
                'Worker CPU allocation binding differs')
        return original_verify(self, jid, candidate)
    scheduler.card = controller.card = card
    scheduler.Slurm.verify = verify
    if value is not None:
        require(value.get('cpu_allocation_binding_sha256') == binding['sha256'],
                'Immutable worker lacks its allocation amendment')
        def guard_verify(jid, logical):
            target, submitted = verification_target(jid, logical, physical or logical)
            return scheduler.Slurm().verify(target, submitted)
        install_guard(value, guard_verify)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command'); parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    require(args.command in ('worker', 'status', 'dry-run', 'transfer', 'recover-preinit'), 'No competing launch or validation retry')
    value = load(args.card) if args.card else None
    physical = value
    if value and value.get('array_cards'):
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or (Path(value['root']) if value else None)
    require(root is not None, 'Campaign or immutable card required')
    campaign = load(root / 'CAMPAIGN.json')
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import require_readiness
    require_readiness(campaign)
    lookup = load(root / 'LOOKUP_R1_BINDING.json'); check_files(lookup['files'])
    binding = load(root / 'CPU_ALLOCATION_R1_BINDING.json'); check_files(binding['files'])
    proof = load(root / 'CPU_ALLOCATION_R1_PREFLIGHT.json')
    require(binding['campaign_sha256'] == campaign['sha256']
            and binding['lookup_binding_sha256'] == lookup['sha256']
            and proof['passed'] and proof['binding_sha256'] == binding['sha256']
            and proof['lookup_proof_sha256'] == load(root / 'LOOKUP_R1_PREFLIGHT.json')['sha256'],
            'Matching full scientific and allocation-interface proofs required')
    install(campaign, binding, value, physical)
    if args.command == 'transfer':
        from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import transfer
        print(json.dumps(transfer(campaign), sort_keys=True))
        return
    if args.command == 'recover-preinit':
        from moe_exp.research_campaign_v3_r6.contracts import publish, locked
        from moe_exp.research_campaign_v3_r6.controller import jobs, _check, stage_key
        from moe_exp.research_campaign_v3_r6.scheduler import Slurm, card, submit_card
        scheduler = Slurm()
        with locked(root, 'controller'):
            observed = jobs(campaign, scheduler)
            candidates = [j for j in observed if j['accepted']['job_id'] == '60564471']
            require(len(candidates) == 1, 'Exact original rejected CPU worker is required')
            original = candidates[0]; old = original['card']
            directory = root / 'stages' / stage_key(old['stage'], old['model'], old['shard_index'], old['threshold_index'])
            require(old['stage'] == 'analysis_pilot' and old['model'] == 'qwen35'
                    and old['attempt'] == 0 and original['accounting']['state'] == 'FAILED'
                    and not (directory / 'START.json').exists() and not (directory / 'RESULT.json').exists(),
                    'Only this proven pre-initialization rejection may recover')
            admission = _check(campaign, scheduler, observed, hours=12, bytes_=0)
            require(admission['admitted'], 'The existing single deterministic recovery cannot be admitted')
            replacement = card(campaign, old['stage'], old['model'], old['shard_index'], old['threshold_index'],
                               attempt=1, minutes=old['minutes'])
            publish(directory / 'INFRASTRUCTURE_REPLACEMENT.json', {
                'original': original, 'replacement': replacement,
                'proof': 'Allocation guard rejected the real CPU-only Booster step before stage or scientific initialization; one original deterministic assignment, no stochastic redraw',
            })
            accepted = submit_card(campaign, replacement, scheduler)
            receipt = publish(root / 'CPU_ALLOCATION_R1_RECOVERY.json', {
                'original_job': '60564471', 'replacement': accepted,
                'binding_sha256': binding['sha256'], 'bounded_admission': admission,
                'stochastic_assignments_repeated': 0, 'further_recoveries': 0,
            })
            print(json.dumps({'replacement_job_id': accepted['job_id'], 'sha256': receipt['sha256']}))
        return
    spec = importlib.util.spec_from_file_location('preserved_ops_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_ops_r1/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
