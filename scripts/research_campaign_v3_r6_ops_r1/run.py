#!/usr/bin/env python3
"""Verified operations entrypoint for the unchanged r6 scientific campaign."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))


def validate_submit(campaign):
    from moe_exp.research_campaign_v3_r6.contracts import load, publish, require
    from moe_exp.research_campaign_v3_r6.scheduler import Slurm, card, serial_capacity, submit_card
    from moe_exp.research_campaign_v3_r6.controller import _check, jobs
    from moe_exp.research_campaign_v3_r6_ops_r1.policy import check_binding
    root = Path(campaign['root']); binding = check_binding(campaign)
    path = root / 'OPS_R1_VALIDATION_LAUNCH.json'
    if path.exists():
        return load(path)
    require(not (root / 'OPS_R1_VALIDATION_TERMINAL.json').exists(), 'One validation allocation is exhausted')
    preflight = load(root / 'OPS_R1_PREFLIGHT.json')
    require(preflight['passed'] and preflight['binding_sha256'] == binding['sha256'], 'Operations preflight missing')
    scheduler = Slurm(); snapshot = scheduler.snapshot()
    value = card(campaign, 'validate', minutes=180, attempt=4)
    require(all(serial_capacity(snapshot, [value], handoff_slots=0).values()), 'Validation exceeds live serial limits')
    admission = _check(campaign, scheduler, jobs(campaign, scheduler), hours=12, bytes_=2 * 1024**3)
    require(admission['admitted'], 'Authorized validation cannot be admitted')
    receipt = submit_card(campaign, value, scheduler)
    return publish(path, {'allocation': receipt, 'card_sha256': value['sha256'],
        'binding_sha256': binding['sha256'], 'bounded_admission': admission,
        'reservation_local_hours': 12, 'reservation_with_margin': 15, 'automatic_retries': 0,
        'scientific_release_requires_complete_validation': True})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--card', type=Path)
    args, _ = parser.parse_known_args()
    from moe_exp.research_campaign_v3_r6.contracts import load, require
    require(args.campaign is not None or args.card is not None, 'Campaign or immutable card is required')
    value = load(args.card) if args.card else None
    if value and value.get('array_cards'):
        import os
        value = value['array_cards'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root = args.campaign or Path(value['root'])
    campaign = load(root / 'CAMPAIGN.json')
    if args.command == 'prepare':
        from moe_exp.research_campaign_v3_r6_ops_r1.binding import prepare
        result = prepare(campaign)
    elif args.command == 'preflight':
        from moe_exp.research_campaign_v3_r6_ops_r1.preflight import run
        result = run(campaign)
    else:
        from moe_exp.research_campaign_v3_r6_ops_r1.policy import check_binding, install, require_readiness
        validation = args.command == 'validate-import' or (args.command == 'worker' and value and value['stage'] == 'validate')
        if validation:
            check_binding(campaign)
        else:
            install(campaign)
        if value is not None:
            require(value.get('ops_binding_sha256') == load(root / 'OPS_R1_BINDING.json')['sha256'],
                    'Immutable worker card lacks operations binding')
        if args.command == 'protect-milestone':
            from moe_exp.research_campaign_v3_r6_ops_r1.milestone import protect
            result = protect(campaign)
        elif args.command == 'validate-submit':
            result = validate_submit(campaign)
        elif args.command == 'validate-import' or (args.command == 'worker' and value is not None and value['stage'] == 'validate'):
            spec = importlib.util.spec_from_file_location('validation_candidate_r5',
                REPO / 'scripts/research_campaign_v3_r6_validation_candidate_r5.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            result = module.run(campaign)
        elif args.command == 'worker' and value is not None and value.get('ops_report_only'):
            from moe_exp.research_campaign_v3_r6_ops_r1.milestone import worker
            result = worker(campaign, value)
        else:
            require(args.command not in ('prepare', 'publish'), 'Sealed preparation or publication command rejected')
            if args.command in ('submit', 'worker', 'closeout'):
                require_readiness(campaign)
            spec = importlib.util.spec_from_file_location('sealed_r6_launcher', REPO / 'scripts/research_campaign_v3_r6/run.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module.main()
            return
    if isinstance(result, dict) and 'files' in result:
        result = {'sha256': result['sha256'], 'file_count': len(result['files'])}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
