#!/usr/bin/env python3
"""Bound embedded-failure identity correction over unchanged pricing sources."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]
from moe_exp.research_campaign_v3_r6.contracts import digest, load, require, seal
from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1 import check_binding


def amend_card(value, binding):
    if value.get('ops_report_only'):
        return value
    payload = {k: v for k, v in value.items() if k not in ('sha256', 'comment')}
    payload['reader_price_failure_id_binding_sha256'] = binding['sha256']
    payload['comment'] = 'rc3:' + digest(payload)
    return seal(payload)


def install(campaign, binding, value=None, physical=None):
    from moe_exp.research_campaign_v3_r6 import controller, scheduler
    from moe_exp.research_campaign_v3_r6_ops_r1 import policy as base
    from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1 import install as install_identity
    native_verify = scheduler.Slurm.verify
    spec = importlib.util.spec_from_file_location('preserved_price_entry',
        REPO / 'scripts/research_campaign_v3_r6_reader_price_r1/run.py')
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    parent.install(campaign, load(Path(campaign['root']) / 'READER_PRICE_R1_BINDING.json'), value, physical)
    original_card, original_verify = scheduler.card, scheduler.Slurm.verify
    base.WRAPPER = REPO / 'scripts/research_campaign_v3_r6_reader_price_failure_id_r1/job.sbatch'
    def card(*args, **kwargs):
        return amend_card(original_card(*args, **kwargs), binding)
    def verify(self, jid, candidate):
        if candidate.get('ops_report_only') or 'reader_price_binding_sha256' not in candidate:
            return original_verify(self, jid, candidate)
        if 'reader_price_failure_id_binding_sha256' not in candidate:
            require(candidate['reader_price_binding_sha256'] == binding['reader_price_binding_sha256'],
                    'Legacy pricing binding differs')
            proof = native_verify(self, jid, candidate)
            require(proof['fields']['Command'] ==
                    str(REPO / 'scripts/research_campaign_v3_r6_reader_price_r1/job.sbatch'),
                    'Legacy pricing wrapper differs')
            return proof
        require(candidate['reader_price_failure_id_binding_sha256'] == binding['sha256'],
                'Worker failure-identity amendment differs')
        return original_verify(self, jid, candidate)
    scheduler.card = controller.card = card
    scheduler.Slurm.verify = verify
    install_identity()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command'); parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    require(args.command in ('verify-resolution', 'transfer', 'worker', 'status', 'dry-run'),
            'No new draw, launch or validation retry')
    physical = load(args.card) if args.card else None
    value = physical
    if value and value.get('array_cards'):
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or (Path(value['root']) if value else None)
    require(root is not None, 'Campaign or immutable card required')
    campaign = load(root / 'CAMPAIGN.json'); binding = check_binding(campaign)
    proof = load(root / 'READER_PRICE_FAILURE_ID_R1_PREFLIGHT.json')
    require(proof['passed'] and proof['binding_sha256'] == binding['sha256'], 'Matching failure-identity proof required')
    require(value is None or value.get('reader_price_failure_id_binding_sha256') == binding['sha256'],
            'Worker failure-identity amendment missing')
    install(campaign, binding, value, physical)
    if args.command == 'verify-resolution':
        from moe_exp.research_campaign_v3_r6 import controller, failure_domains
        require(controller._stage_state(root, 'reader_price')['status'] == 'complete', 'Price view remains unavailable')
        require(failure_domains.blocked(campaign, 'reader_price', 'all') is None and
                failure_domains.blocked(campaign, 'annotate', 'nemotron') is None, 'Exact resolved disposition remains blocked')
        print(json.dumps({'verified': True, 'new_scientific_draws': 0, 'slurm_submissions': 0}))
        return
    if args.command == 'transfer':
        from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1_transfer import transfer_latest
        print(json.dumps(transfer_latest(campaign), sort_keys=True)); return
    spec = importlib.util.spec_from_file_location('preserved_ops_entry',
        REPO / 'scripts/research_campaign_v3_r6_ops_r1/run.py')
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent); parent.main()


if __name__ == '__main__': main()
