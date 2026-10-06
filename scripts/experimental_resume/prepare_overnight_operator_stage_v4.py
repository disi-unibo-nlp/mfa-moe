"""Prepare qualified C/fresh execution and complete semantic resource envelopes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('C', 'FRESH'), required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_viz':
        raise RuntimeError('stage preparation requires authorized CPU Slurm')
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    root = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
    pin_qualified_worker(root / 'addenda/ordered/9727c10299b71e7a/moe_exp_src')
    import overnight_routing_runner_v2 as run
    from overnight_routing_adjudicated_v4 import install
    install()
    from prepare_overnight_ab_v1 import measurement_reserve
    doc = run.common.DOC
    prefix = 'OVERNIGHT_DISCOVERY_C' if args.stage == 'C' else 'OVERNIGHT_FRESH_COMPARISON'
    design = doc / (prefix + ('_DESIGN_v1.json' if args.stage == 'C' else '_DESIGN_v2.json'))
    manifest_path, price_path = doc / (prefix + '_MANIFEST_v2.json'), doc / (prefix + '_PRICE_v2.json')
    manifest, price = run.prepare(design, manifest_path, price_path,
                                  rows_per_shard=4 if args.stage == 'C' else 0,
                                  max_wall_seconds=7200 if args.stage == 'C' else 8100)
    entry = Path(__file__).with_name('overnight_routing_entry_v4.py')
    for flag in ('--cpu-preflight', '--startup-import-check'):
        subprocess.run([sys.executable, '-B', str(entry), 'overnight_routing_runner_v2',
                        '--manifest', str(manifest_path), flag], check=True)
    reserve = measurement_reserve(manifest['expected_requests'])
    execution_files = [Path(__file__).with_name(name) for name in
                       ('overnight_routing_run_v4.sbatch', 'overnight_routing_entry_v4.py',
                        'diagnose_mechanism_validation_v3.py', 'build_price_overnight_semantics_v4.sbatch',
                        'rate_overnight_semantics_v2.sbatch', 'analyze_overnight_semantics_v2.sbatch',
                        'dispatch_overnight_stage_v4.py', 'dispatch_overnight_stage_v4.sbatch')]
    body = {'schema': 'overnight-complete-resource-envelope-v2', 'stage': args.stage,
            'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
            'generation_gpu_hours': price['estimated_complete_gpu_hours'], 'semantic_measurement': reserve,
            'total_generation_plus_semantic_gpu_hours': price['estimated_complete_gpu_hours'] + reserve['measurement_gpu_hour_ceiling'],
            'execution_files': {str(path.resolve()): run.base.file_sha(path) for path in execution_files},
            'authorization': 'RESOURCE_AUTHORIZATION_2026-10-02_v2.json plus approved parallel plan2026-10-04',
            'dense_labels': 'Independent secondary stage requires exact generated-frame price before GPU submission.',
            'status': 'PASS_COMPLETE_GENERATION_AND_SEMANTIC_ENVELOPE'}
    envelope = run.common.write_once(doc / (prefix + '_ENVELOPE_v2.json'), body)
    print(json.dumps({'stage': args.stage, 'status': 'PASS_CPU_PREFLIGHT_AND_ENVELOPE',
                      'manifest_sha256': manifest['sha256'], 'envelope_sha256': envelope['sha256'],
                      'shards': len(manifest['shards']), 'requests': manifest['expected_requests'],
                      'generation_gpu_hours': price['estimated_complete_gpu_hours'],
                      'semantic_gpu_upper_bound': reserve['measurement_gpu_hour_ceiling']}), flush=True)


if __name__ == '__main__':
    main()
