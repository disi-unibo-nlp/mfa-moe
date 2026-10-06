"""Bind a sealed 1024-token mechanism stage to arm-blind semantic inputs.

Run on a CPU Slurm step after STAGE_COMPLETION.json exists. The frame contains
only the five reader-visible text fields; assignment and failure metadata stay
in ARM_MAP.json, which the rating driver never opens.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base
import run_mechanism_validation_v1 as generation
import prepare_mechanism_start_frame_v1 as starts

REPO = generation.REPO
RUBRIC = REPO / 'report/experimental-resume-v1/MECHANISM_1024_SEMANTIC_RUBRIC_v1.md'
TOKENIZER = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
                 'models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/'
                 '95a723d08a9490559dae23d0cff1d9466213d989')
THINK_END_ID = 248069
ALLOWLIST = ['transition', 'problem', 'full_emitted_prefix',
             'triggering_sentence', 'continuation']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_once(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        require(base.sealed(path) == value, 'existing blinded artifact differs')
    else:
        base.atomic_json(path, value)
    return value


def expected_assignments(manifest):
    plan = []
    for row in manifest['rows']:
        for seed in manifest['seeds']:
            key = f"{row['uid']}|{seed}"
            order = manifest['arm_order_by_uid_seed'][key]
            require(len(order) == 4 and set(order) == set(generation.ARMS),
                    'four-arm schedule differs')
            for position, arm in enumerate(order):
                plan.append({'uid': 'mechanism-v1|' + base.digest([
                    manifest['sha256'], row['uid'], seed, arm])[:24],
                    'prefix_uid': row['uid'], 'family': row['family'],
                    'transition': row['transition'], 'question': row['question'],
                    'seed': seed, 'arm': arm, 'execution_position': position})
    require(len(plan) == manifest['expected_requests'] and
            len({r['uid'] for r in plan}) == len(plan), 'assignment count differs')
    return plan


def complete_outputs(manifest, price, run_out):
    """Read every priced shard, assigned receipt, route array and stage seal."""
    stage = base.sealed(run_out / 'STAGE_COMPLETION.json')
    require(stage['schema'] == 'mechanism-validation-stage-completion-v1' and
            stage['status'] == 'COMPLETE_UNGRADED_GENERATION' and
            stage['manifest_sha256'] == manifest['sha256'] and
            stage['price_sha256'] == price['sha256'] and
            stage['shards'] == len(price['shards']) and
            stage['counts']['assigned'] == manifest['expected_requests'],
            'complete generation stage is unsealed or rebound')
    plan = expected_assignments(manifest)
    found, receipts = [], []
    for shard_index, shard in enumerate(price['shards']):
        directory = run_out / f'shard-{shard_index:03d}'
        binding = base.sealed(directory / 'BINDING.json')
        summary = base.sealed(directory / 'SUMMARY.json')
        completion = base.sealed(directory / 'MECHANISM_COMPLETION.json')
        expected = plan[shard['start_row'] * 8:shard['end_row'] * 8]
        require(binding['schema'] == 'routing-boundary-micro-screen-binding-v1' and
                binding['manifest_sha256'] == manifest['sha256'] and
                binding['driver_sha256'] == manifest['base_driver_sha256'] and
                summary['schema'] == 'routing-boundary-micro-screen-summary-v1' and
                summary['status'] == 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION' and
                summary['manifest_sha256'] == manifest['sha256'] and
                summary['binding_sha256'] == binding['sha256'] and
                summary['requests'] == len(expected) and
                completion['schema'] == 'mechanism-validation-completion-v1' and
                completion['status'] == 'COMPLETE_UNGRADED_GENERATION' and
                completion['manifest_sha256'] == manifest['sha256'] and
                completion['binding_sha256'] == binding['sha256'] and
                completion['source_summary_sha256'] == summary['sha256'] and
                completion['counts']['assigned'] == len(expected),
                'shard binding, summary or completion differs')
        shard_found = []
        for batch_index in range(summary['batches']):
            assignment = base.sealed(directory / f'batch-{batch_index:03d}-assignment.json')
            batch = base.sealed(directory / f'batch-{batch_index:03d}.json')
            array = directory / f'batch-{batch_index:03d}.npz'
            require(assignment['schema'] ==
                    'routing-boundary-micro-screen-assignment-v1' and
                    assignment['batch_index'] == batch_index and
                    assignment['manifest_sha256'] == manifest['sha256'] and
                    assignment['binding_sha256'] == binding['sha256'] and
                    batch['schema'] == 'routing-boundary-micro-screen-batch-v1' and
                    batch['batch_index'] == batch_index and
                    batch['manifest_sha256'] == manifest['sha256'] and
                    batch['binding_sha256'] == binding['sha256'] and
                    base.file_sha(array) == batch['array_sha256'] and
                    [r['uid'] for r in assignment['requests']] ==
                    [r['uid'] for r in batch['outputs']],
                    'assigned batch, output or route array differs')
            shard_found.extend(batch['outputs'])
            receipts.append({'shard': shard_index, 'batch': batch_index,
                             'assignment_sha256': assignment['sha256'],
                             'batch_sha256': batch['sha256'],
                             'array_sha256': batch['array_sha256']})
        require([r['uid'] for r in shard_found] == [r['uid'] for r in expected],
                'shard omits or reorders assigned results')
        found.extend(shard_found)
    require([r['uid'] for r in found] == [r['uid'] for r in plan] and
            len(found) == manifest['expected_requests'],
            'stage omits assigned results')
    return stage, plan, found, receipts


def build(manifest, price, source, run_out, out, tokenizer, salt=None):
    require(manifest['schema'] == 'mechanism-validation-manifest-v1' and
            manifest['horizon'] == 1024 and
            price['schema'] == 'mechanism-validation-price-v1' and
            price['status'] == 'PASS_COMPLETE_STAGE' and
            price['manifest_sha256'] == manifest['sha256'] and
            source['schema'] == 'mechanism-start-frame-v1' and
            source['sha256'] == manifest['frame_sha256'],
            'mechanism source or complete-stage price differs')
    stage, plan, outputs, receipts = complete_outputs(manifest, price, run_out)
    originals = {r['uid']: r for r in source['records']}
    require(len(originals) == len(source['records']), 'duplicate native start')
    map_path = out / 'ARM_MAP.json'
    if map_path.exists():
        previous = base.sealed(map_path)
        require(previous['manifest_sha256'] == manifest['sha256'] and
                previous['stage_completion_sha256'] == stage['sha256'],
                'existing arm map belongs to another stage')
        salt = bytes.fromhex(previous['blind_salt_hex'])
    else:
        salt = salt or os.urandom(32)
    require(len(salt) == 32, 'blind salt must be 32 bytes')
    map_rows, blind_rows = [], []
    for expected, observed in zip(plan, outputs, strict=True):
        require(all(observed.get(key) == expected[key] for key in
                    ('uid', 'prefix_uid', 'family', 'transition', 'question',
                     'seed', 'arm', 'execution_position')),
                'observed result identity differs from frozen assignment')
        tokens = observed['tokens']
        require(isinstance(tokens, list) and len(tokens) <= 1024 and
                all(type(x) is int and x >= 0 for x in tokens),
                'invalid emitted token sequence')
        closure = THINK_END_ID in tokens
        reasoning = tokens[:tokens.index(THINK_END_ID)] if closure else tokens
        if observed['error']:
            status = 'generation_error'
        elif not observed['routed_present']:
            status = 'missing_routed_array'
        elif not reasoning:
            status = 'zero_reasoning_tokens'
        else:
            status = 'gradeable'
        blind_id = hmac.new(salt, expected['uid'].encode(), hashlib.sha256).hexdigest()[:32]
        map_rows.append({**expected, 'blind_id': blind_id,
                         'measurement_status': status,
                         'finish': observed['finish'],
                         'stop_reason': observed['stop_reason'],
                         'error': observed['error'],
                         'emitted_tokens': len(tokens),
                         'reasoning_tokens': len(reasoning),
                         'closed_reasoning': closure,
                         'hit_1024_cap': len(tokens) == 1024,
                         'routed_present': observed['routed_present'],
                         'action_dose': observed.get('action_dose'),
                         'inactive_native_checks': observed.get('inactive_native_checks'),
                         'missing_telemetry_ranks': observed.get('missing_telemetry_ranks')})
        if status == 'gradeable':
            original = originals[expected['prefix_uid']]
            data = original['reader_input']
            native = next(r for r in manifest['rows']
                          if r['uid'] == expected['prefix_uid'])
            require(original['transition'] == expected['transition'] and
                    original['analysis_meta']['prompt_ids_sha256'] ==
                    native['prompt_ids_sha256'] and
                    original['analysis_meta']['prefix_ids_sha256'] ==
                    native['prefix_ids_sha256'] and
                    original['analysis_meta']['prefix_text_sha256'] ==
                    hashlib.sha256(data['emitted_prefix'].encode()).hexdigest() and
                    data['emitted_prefix'].rstrip().endswith(
                        data['triggering_sentence'].rstrip()),
                    'pretreatment reader frame differs from generation start')
            blind_rows.append({'blind_id': blind_id, 'reader_input': {
                'transition': expected['transition'], 'problem': data['problem'],
                'full_emitted_prefix': data['emitted_prefix'],
                'triggering_sentence': data['triggering_sentence'],
                'continuation': tokenizer.decode(reasoning, skip_special_tokens=False)}})
    blind_rows.sort(key=lambda row: row['blind_id'])
    require(len({r['blind_id'] for r in map_rows}) == len(map_rows) and
            len(blind_rows) <= len(map_rows), 'blind ID collision')
    map_body = {'schema': 'mechanism-1024-arm-map-v1',
                'builder_sha256': base.file_sha(__file__),
                'manifest_sha256': manifest['sha256'],
                'generation_price_sha256': price['sha256'],
                'stage_completion_sha256': stage['sha256'],
                'source_frame_sha256': source['sha256'],
                'rubric_sha256': base.file_sha(RUBRIC),
                'tokenizer_config_sha256': base.file_sha(TOKENIZER / 'tokenizer_config.json'),
                'blind_salt_hex': salt.hex(), 'batch_receipts': receipts,
                'records': map_rows}
    frame_body = {'schema': 'mechanism-1024-blind-frame-v1',
                  'builder_sha256': base.file_sha(__file__),
                  'generation_manifest_sha256': manifest['sha256'],
                  'stage_completion_sha256': stage['sha256'],
                  'source_frame_sha256': source['sha256'],
                  'rubric_sha256': base.file_sha(RUBRIC),
                  'reader_input_allowlist': ALLOWLIST,
                  'continuation_max_tokens': 1024,
                  'assigned': len(plan), 'records': blind_rows}
    out.mkdir(parents=True, exist_ok=True)
    arm_map = write_once(map_path, map_body)
    frame = write_once(out / 'BLIND_FRAME.json', frame_body)
    return arm_map, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=generation.MANIFEST)
    parser.add_argument('--generation-price', type=Path, default=generation.PRICE)
    parser.add_argument('--run-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'),
            'token decoding requires CPU Slurm step')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    arm_map, frame = build(base.sealed(args.manifest), base.sealed(args.generation_price),
                           base.sealed(starts.FRAME), args.run_out, args.out, tokenizer)
    print(json.dumps({'map_sha256': arm_map['sha256'],
                      'frame_sha256': frame['sha256'],
                      'assigned': frame['assigned'],
                      'gradeable': len(frame['records'])}), flush=True)


if __name__ == '__main__':
    main()
