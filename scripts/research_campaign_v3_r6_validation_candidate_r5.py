#!/usr/bin/env python3
"""Single bound validation of the installed operational amendment."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))


def run(campaign, *, launch_after=True):
    from moe_exp.research_campaign_v3_r6.contracts import load, publish
    from moe_exp.research_campaign_v3_r6.controller import stage_key
    from moe_exp.research_campaign_v3_r6_ops_r1.validation import validate
    root = Path(campaign['root']); binding = load(root / 'OPS_R1_BINDING.json')
    directory = root / 'stages' / stage_key('validate')
    path = directory / 'START-4.json'
    if not path.exists():
        publish(path, {'job_id': os.environ['SLURM_JOB_ID'], 'ops_binding_sha256': binding['sha256']})
    proof = validate(campaign)
    result_path = directory / 'RESULT.json'
    if not result_path.exists():
        publish(result_path, {'stage': 'validate', 'model': 'all', 'shard_index': 0, 'threshold_index': 0,
            'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'], 'failure_scope': None,
            'detail': {'readiness_sha256': proof['readiness_sha256'], 'ops_validation_sha256': proof['sha256']}})
    if launch_after:
        from moe_exp.research_campaign_v3_r6.controller import launch
        try:
            launched = launch(campaign)
            publish(root / 'OPS_R1_LAUNCH_VERIFIED.json', {'launch_sha256': launched['sha256'],
                'validation_sha256': proof['sha256'], 'binding_sha256': binding['sha256']})
        except Exception as exc:
            failure = root / 'OPS_R1_LAUNCH_FAILURE.json'
            if not failure.exists():
                publish(failure, {'error_type': type(exc).__name__, 'error': str(exc)[:4000],
                    'validation_passed': True, 'unresolved_intents_must_be_reconciled': True})
            raise
    return proof


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--no-launch', action='store_true')
    args = parser.parse_args()
    from moe_exp.research_campaign_v3_r6.contracts import load
    print(json.dumps(run(load(args.campaign / 'CAMPAIGN.json'), launch_after=not args.no_launch), sort_keys=True))


if __name__ == '__main__':
    main()
