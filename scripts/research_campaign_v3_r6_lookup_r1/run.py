#!/usr/bin/env python3
"""Source-bound read-only ancestry acceleration over both existing amendments."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import check_files, digest, load, require, seal


def amend_card(value, binding):
    if value.get('ops_report_only'):
        return value
    value = {k: v for k, v in value.items() if k not in ('sha256', 'comment')}
    value['lookup_binding_sha256'] = binding
    value['comment'] = 'rc3:' + digest(value)
    return seal(value)


def install(campaign, binding):
    from moe_exp.research_campaign_v3_r6 import controller, scheduler
    from moe_exp.research_campaign_v3_r6_ops_r1 import policy as base
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1 import policy as cpu
    from moe_exp.research_campaign_v3_r6_lookup_r1 import install_forecasts
    native_verify = scheduler.Slurm.verify
    cpu.install(campaign)
    original_card, original_verify = scheduler.card, scheduler.Slurm.verify
    base.WRAPPER = REPO / 'scripts/research_campaign_v3_r6_lookup_r1/job.sbatch'

    def card(*args, **kwargs):
        return amend_card(original_card(*args, **kwargs), binding['sha256'])

    def verify(self, jid, value):
        if value.get('ops_report_only'):
            return original_verify(self, jid, value)
        if 'lookup_binding_sha256' not in value:
            require(value.get('cpu_dispatch_binding_sha256') == binding['cpu_dispatch_binding_sha256'],
                    'Legacy scientific owner lacks the CPU amendment')
            proof = native_verify(self, jid, value)
            require(proof['fields']['Command'] == str(cpu.WRAPPER), 'Legacy scientific wrapper differs')
            return proof
        require(value['lookup_binding_sha256'] == binding['sha256'], 'Worker lookup binding differs')
        return original_verify(self, jid, value)

    scheduler.card = controller.card = card
    scheduler.Slurm.verify = verify
    install_forecasts()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    require(args.command in ('worker', 'submit', 'status', 'closeout', 'dry-run', 'transfer'),
            'No validation retry or publication dispatch')
    value = load(args.card) if args.card else None
    if value and value.get('array_cards'):
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or (Path(value['root']) if value else None)
    require(root is not None, 'Campaign or immutable card required')
    campaign = load(root / 'CAMPAIGN.json')
    binding = load(root / 'LOOKUP_R1_BINDING.json')
    check_files(binding['files'])
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import require_readiness
    cpu_proof = require_readiness(campaign)
    proof = load(root / 'LOOKUP_R1_PREFLIGHT.json')
    require(binding['campaign_sha256'] == campaign['sha256'] and proof['passed']
            and proof['binding_sha256'] == binding['sha256']
            and proof['cpu_dispatch_proof_sha256'] == cpu_proof['sha256'],
            'Matching full readiness and lookup proofs are required')
    if value:
        require(value.get('lookup_binding_sha256') == binding['sha256'], 'Worker lookup proof missing')
    install(campaign, binding)
    if args.command == 'transfer':
        from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import transfer
        print(json.dumps(transfer(campaign), sort_keys=True))
        return
    spec = importlib.util.spec_from_file_location('preserved_ops_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_ops_r1/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
