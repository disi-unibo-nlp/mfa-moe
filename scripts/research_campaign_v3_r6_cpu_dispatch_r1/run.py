#!/usr/bin/env python3
"""Additional source-bound CPU dispatch; no new campaign or validation attempt."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts/experimental_resume')]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    from moe_exp.research_campaign_v3_r6.contracts import load, require
    from moe_exp.research_campaign_v3_r6_cpu_dispatch_r1.policy import install, require_readiness, transfer
    require(args.command in ('worker', 'submit', 'status', 'closeout', 'dry-run', 'transfer'),
            'This amendment creates no validation retry or publication dispatch')
    value = load(args.card) if args.card else None
    if value and value.get('array_cards'):
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or (Path(value['root']) if value else None)
    require(root is not None, 'Campaign or immutable card required')
    campaign = load(root / 'CAMPAIGN.json')
    proof = require_readiness(campaign)
    binding = load(root / 'CPU_DISPATCH_R1_BINDING.json')
    if value:
        require(value.get('cpu_dispatch_binding_sha256') == binding['sha256'],
                'Worker lacks the additional source-bound CPU dispatch amendment')
    install(campaign)
    if args.command == 'transfer':
        print(json.dumps(transfer(campaign), sort_keys=True))
        return
    spec = importlib.util.spec_from_file_location('preserved_ops_r1_entrypoint',
        REPO / 'scripts/research_campaign_v3_r6_ops_r1/run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
