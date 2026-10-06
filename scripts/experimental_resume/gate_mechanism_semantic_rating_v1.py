"""Read-only launch gate for the priced 1024-token arm-blind reader stage.

Print a reviewable sbatch command only after generation, blind frame and exact
complete-stage Qwen3.8 price are sealed. This program never submits a job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

import run_boundary_micro_screen as base
import run_mechanism_validation_v1 as generation
import build_mechanism_blind_frame_v1 as builder
import price_mechanism_semantics_v1 as pricing
import rate_mechanism_semantics_v1 as rating

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')
SBATCH = rating.REPO / 'scripts/experimental_resume/rate_mechanism_semantics_v1.sbatch'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def gate(manifest, gen_price, stage, arm_map, frame, reader_price):
    require(manifest['schema'] == 'mechanism-validation-manifest-v1' and
            gen_price['schema'] == 'mechanism-validation-price-v1' and
            gen_price['status'] == 'PASS_COMPLETE_STAGE' and
            gen_price['manifest_sha256'] == manifest['sha256'] and
            stage['schema'] == 'mechanism-validation-stage-completion-v1' and
            stage['status'] == 'COMPLETE_UNGRADED_GENERATION' and
            stage['manifest_sha256'] == manifest['sha256'] and
            stage['price_sha256'] == gen_price['sha256'] and
            stage['counts']['assigned'] == manifest['expected_requests'],
            'generation stage is unsealed or bound to another price')
    require(arm_map['schema'] == 'mechanism-1024-arm-map-v1' and
            frame['schema'] == 'mechanism-1024-blind-frame-v1' and
            arm_map['manifest_sha256'] == manifest['sha256'] and
            arm_map['generation_price_sha256'] == gen_price['sha256'] and
            arm_map['stage_completion_sha256'] == frame['stage_completion_sha256'] ==
            stage['sha256'] and
            arm_map['builder_sha256'] == frame['builder_sha256'] ==
            base.file_sha(builder.__file__) and
            arm_map['rubric_sha256'] == frame['rubric_sha256'] ==
            base.file_sha(rating.RUBRIC) and
            frame['generation_manifest_sha256'] == manifest['sha256'] and
            frame['assigned'] == manifest['expected_requests'] and
            len(arm_map['records']) == manifest['expected_requests'],
            'blind frame or all-assigned map differs from generation stage')
    plan = builder.expected_assignments(manifest)
    require([r['uid'] for r in arm_map['records']] == [r['uid'] for r in plan] and
            all(all(row[key] == expected[key] for key in expected)
                for row, expected in zip(arm_map['records'], plan, strict=True)),
            'arm map omits or reassigns an intention-to-treat cell')
    blind_ids = [r['blind_id'] for r in frame['records']]
    require(blind_ids == sorted(blind_ids) and len(set(blind_ids)) == len(blind_ids) and
            set(blind_ids) == {r['blind_id'] for r in arm_map['records']
                               if r['measurement_status'] == 'gradeable'} and
            len({r['blind_id'] for r in arm_map['records']}) == len(plan),
            'blind IDs, gradeable status or reader population differ')
    rating.validate(frame, reader_price)
    prior_price = base.sealed(pricing.PRIOR_PRICE)
    prior_binding = base.sealed(pricing.PRIOR_BINDING)
    prior_summary = base.sealed(pricing.PRIOR_SUMMARY)
    require(reader_price['prior_price_sha256'] == prior_price['sha256'] and
            reader_price['prior_binding_sha256'] == prior_binding['sha256'] and
            reader_price['prior_summary_sha256'] == prior_summary['sha256'] and
            prior_binding['price_sha256'] == prior_price['sha256'] and
            prior_summary['binding_sha256'] == prior_binding['sha256'] and
            reader_price['gpus'] == 2 and
            reader_price['max_wall_seconds_per_job'] == 7200,
            'reader price differs from sealed throughput source or GPU profile')
    return {'manifest_sha256': manifest['sha256'],
            'generation_stage_sha256': stage['sha256'],
            'frame_sha256': frame['sha256'],
            'price_sha256': reader_price['sha256'],
            'assigned': manifest['expected_requests'],
            'gradeable': len(frame['records']),
            'ratings': reader_price['ratings'],
            'shards': len(reader_price['shards']),
            'estimated_complete_gpu_hours':
                reader_price['estimated_complete_gpu_hours'],
            'gpu_hour_comparison_ceiling': reader_price['gpu_hour_ceiling'],
            'stage_label': manifest['stage_label']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=generation.MANIFEST)
    parser.add_argument('--generation-price', type=Path, default=generation.PRICE)
    parser.add_argument('--generation-out', type=Path, required=True)
    parser.add_argument('--semantic-out', type=Path, required=True)
    parser.add_argument('--rating-out', type=Path, required=True)
    parser.add_argument('--allow-existing-rating-out', action='store_true')
    args = parser.parse_args()
    manifest = base.sealed(args.manifest)
    expected_suffix = manifest['sha256'][:16]
    require(args.generation_out.resolve() ==
            ROOT / f'mechanism-validation-v1-{expected_suffix}' and
            args.semantic_out.resolve() ==
            ROOT / f'mechanism-1024-semantics-v1-{expected_suffix}' and
            args.rating_out.resolve() ==
            ROOT / f'mechanism-1024-readers-v1-{expected_suffix}' and
            base.file_sha(SBATCH),
            'output roots or rating wrapper differ from manifest-bound locations')
    if args.rating_out.exists():
        require(args.allow_existing_rating_out and
                all(path.name.startswith('shard-') for path in args.rating_out.iterdir()),
                'rating output already exists; inspect same-bound recovery first')
    gen_price = base.sealed(args.generation_price)
    stage = base.sealed(args.generation_out / 'STAGE_COMPLETION.json')
    arm_map = base.sealed(args.semantic_out / 'ARM_MAP.json')
    frame = base.sealed(args.semantic_out / 'BLIND_FRAME.json')
    reader_price = base.sealed(args.semantic_out / 'READER_PRICE.json')
    report = gate(manifest, gen_price, stage, arm_map, frame, reader_price)
    array = f"0-{report['shards'] - 1}" if report['shards'] > 1 else '0'
    command = ['sbatch', f'--array={array}',
               '--export=ALL,MECHANISM_BLIND_FRAME=' + str(args.semantic_out / 'BLIND_FRAME.json') +
               ',MECHANISM_READER_PRICE=' + str(args.semantic_out / 'READER_PRICE.json') +
               ',MECHANISM_READER_OUT=' + str(args.rating_out), str(SBATCH)]
    print(json.dumps({**report, 'status': 'READY_FOR_REVIEW_NOT_SUBMITTED',
                      'rating_sbatch_sha256': base.file_sha(SBATCH),
                      'suggested_command': shlex.join(command)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
