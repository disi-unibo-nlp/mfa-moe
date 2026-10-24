#!/usr/bin/env python3
"""Prepare v3 native specs; unresolved checkpoints stay pending, never proxies."""
import argparse
import json
from pathlib import Path
from measurement_study.campaign_v3_plan import build_campaign_specs
from measurement_study.production_plan import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-campaign',type=Path,required=True)
    p.add_argument('--overrides',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=json.loads(a.base_campaign.read_text())
    overrides=json.loads(a.overrides.read_text()) if a.overrides else None
    value=dict(schema_version=3,corpus='card-v3',sources=build_campaign_specs(base,overrides),
               storage_root='/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3',
               gpu_hour_ceiling=None,max_concurrent_gpu_jobs=16,capture_runtime_factor=1.25,
               max_shard_hours=16,freeze_status='requires_explicit_T1_source_final_receipt')
    if a.output.exists() and json.loads(a.output.read_text())!=value:
        raise ValueError('Campaign metadata already exists with different inputs; choose new output')
    atomic(a.output,value)
    print(json.dumps({key:spec['native_backend_readiness'] for key,spec in value['sources'].items()}))

if __name__=='__main__':main()
