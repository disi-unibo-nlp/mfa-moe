#!/usr/bin/env python3
"""Use existing qualified reader prices while preserving failed job receipts."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import check_files, digest, load, require, seal


def amend_card(original, binding):
    if original.get('ops_report_only'):
        return original
    amended = {k: v for k, v in original.items() if k not in ('sha256', 'comment')}
    amended['reader_price_binding_sha256'] = binding['sha256']
    amended['comment'] = 'rc3:' + digest(amended)
    return seal(amended)


def install(campaign, binding, value=None, physical=None):
    from moe_exp.research_campaign_v3_r6 import scheduler, controller
    from moe_exp.research_campaign_v3_r6_ops_r1 import policy as base
    from moe_exp.research_campaign_v3_r6_reader_price_r1 import install_resolution
    native_verify = scheduler.Slurm.verify
    spec = importlib.util.spec_from_file_location('preserved_cpu_allocation_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_cpu_allocation_r1/run.py')
    previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    previous.install(campaign, load(Path(campaign['root']) / 'CPU_ALLOCATION_R1_BINDING.json'), value, physical)
    original_card, original_verify = scheduler.card, scheduler.Slurm.verify
    base.WRAPPER = REPO / 'scripts/research_campaign_v3_r6_reader_price_r1/job.sbatch'
    def card(*args, **kwargs):
        return amend_card(original_card(*args, **kwargs), binding)
    def verify(self, jid, candidate):
        if candidate.get('ops_report_only'):
            return original_verify(self, jid, candidate)
        if 'reader_price_binding_sha256' not in candidate:
            require(candidate.get('cpu_allocation_binding_sha256') == binding['cpu_allocation_binding_sha256'],
                    'Legacy allocation binding differs')
            proof = native_verify(self, jid, candidate)
            require(proof['fields']['Command'] ==
                    str(REPO / 'scripts/research_campaign_v3_r6_cpu_allocation_r1/job.sbatch'),
                    'Legacy scientific wrapper differs')
            return proof
        require(candidate['reader_price_binding_sha256'] == binding['sha256'], 'Reader price binding differs')
        return original_verify(self, jid, candidate)
    scheduler.card = controller.card = card
    scheduler.Slurm.verify = verify
    install_resolution(campaign)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command'); parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    require(args.command in ('resolve-price', 'transfer', 'worker', 'status', 'dry-run'),
            'No new reader draw, competing launch or validation retry')
    physical = load(args.card) if args.card else None
    value = physical
    if value and value.get('array_cards'):
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or (Path(value['root']) if value else None)
    require(root is not None, 'Campaign or immutable card required')
    campaign = load(root / 'CAMPAIGN.json')
    from moe_exp.research_campaign_v3_r6_reader_price_r1 import check_binding
    binding = check_binding(campaign)
    proof = load(root / 'READER_PRICE_R1_PREFLIGHT.json')
    require(proof['passed'] and proof['binding_sha256'] == binding['sha256']
            and binding['campaign_sha256'] == campaign['sha256'], 'Matching price reuse proof required')
    if args.command == 'resolve-price':
        from moe_exp.research_campaign_v3_r6_reader_price_r1 import reconcile
        result = reconcile(campaign)
        print(json.dumps({'complete': result['complete'], 'new_scientific_draws': 0, 'sha256': result['sha256']}))
        return
    require(value is None or value.get('reader_price_binding_sha256') == binding['sha256'],
            'Worker reader-price amendment missing')
    install(campaign, binding, value, physical)
    if args.command == 'transfer':
        from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import transfer
        # This is a new amendment of the latest pending handoff, independent of
        # the already completed CPU-allocation ownership transfer.
        directory = root / 'cpu_dispatch_r1'
        require((directory / 'TRANSFER.json').exists(), 'Previous transfer must be complete')
        from moe_exp.research_campaign_v3_r6_reader_price_r1_transfer import transfer_latest
        print(json.dumps(transfer_latest(campaign), sort_keys=True))
        return
    spec = importlib.util.spec_from_file_location('preserved_ops_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_ops_r1/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__': main()
