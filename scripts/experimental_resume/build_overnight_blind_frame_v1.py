"""Bind arbitrary frozen overnight arms to exact arm-blind semantic inputs."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base
import overnight_routing_runner_v1 as generation
import rate_overnight_semantics_v1 as rating

REPO = generation.REPO
SOURCE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
              'steering-v1/runs/routing-control-v1/dense-discovery/'
              'TRANSITION_V22_FULL_PREFIX_START_FRAME.json')
TOKENIZER = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
                 'models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/'
                 '95a723d08a9490559dae23d0cff1d9466213d989')
THINK_END_ID = 248069
ALLOWLIST = sorted(rating.ALLOWLIST)
require = rating.require


def expected_assignments(manifest):
    names = [a['name'] for a in manifest['arms']]
    require(len(set(names)) == len(names) and {'native', 'native_duplicate'} <= set(names),
            'distinct assigned arms and native repeats required')
    for row in manifest['rows']:
        for seed in manifest['seeds']:
            order = manifest['arm_order_by_uid_seed'][f"{row['uid']}|{seed}"]
            require(len(order) == len(names) and set(order) == set(names), 'incomplete arm schedule')
    plan = [meta for _, meta in generation.request_metadata(manifest, manifest['rows'], manifest['arms'])]
    require(len(plan) == manifest['expected_requests'] and len({r['uid'] for r in plan}) == len(plan),
            'assignment count or UID uniqueness differs')
    return plan


def complete_outputs(manifest, price, run_out):
    require(manifest['schema'] == 'overnight-routing-manifest-v1' and
            manifest['horizon'] in (256, 1024) and manifest['seeds'] == [0, 1] and
            price['schema'] == 'overnight-routing-price-v1' and
            price['manifest_sha256'] == manifest['sha256'] and
            price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY' and
            price['shards'] == manifest['shards'], 'frozen generation/price differs')
    require(all(base.file_sha(path) == sha for path, sha in manifest['code_files'].items()),
            'generation code changed after assignment')
    stage = base.sealed(run_out / 'STAGE_COMPLETION.json')
    require(stage['schema'] == 'overnight-routing-stage-completion-v1' and
            stage['status'] == 'COMPLETE_UNGRADED_DISCOVERY_GENERATION' and
            stage['manifest_sha256'] == manifest['sha256'] and
            stage['counts']['assigned'] == manifest['expected_requests'] and
            len(stage['shard_completion_sha256s']) == len(manifest['shards']),
            'generation stage lacks exact completion seal')
    plan, found, receipts = expected_assignments(manifest), [], []
    for index, shard in enumerate(manifest['shards']):
        directory = run_out / f'shard-{index:03d}'
        binding = base.sealed(directory / 'BINDING.json')
        summary = base.sealed(directory / 'SUMMARY.json')
        completion = base.sealed(directory / 'OVERNIGHT_COMPLETION.json')
        expected = [meta for _, meta in generation.request_metadata(manifest,
            manifest['rows'][shard['start_row']:shard['end_row']], manifest['arms'])]
        require(binding['schema'] == 'routing-boundary-micro-screen-binding-v1' and
                summary['schema'] == 'routing-boundary-micro-screen-summary-v1' and
                summary['status'] == 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION' and
                completion['schema'] == 'overnight-routing-completion-v1' and
                completion['status'] == 'COMPLETE_UNGRADED_DISCOVERY_GENERATION' and
                binding['manifest_sha256'] == manifest['sha256'] and
                summary['manifest_sha256'] == manifest['sha256'] and
                summary['binding_sha256'] == binding['sha256'] and
                summary['requests'] == len(expected) and
                completion['sha256'] == stage['shard_completion_sha256s'][index] and
                completion['manifest_sha256'] == manifest['sha256'] and
                completion['source_summary_sha256'] == summary['sha256'] and
                completion['shard'] == index and completion['counts']['assigned'] == len(expected),
                'shard binding, count or completion changed')
        rows = []
        for batch_index in range(summary['batches']):
            assignment = base.sealed(directory / f'batch-{batch_index:03d}-assignment.json')
            batch = base.sealed(directory / f'batch-{batch_index:03d}.json')
            require(assignment['schema'] == 'routing-boundary-micro-screen-assignment-v1' and
                    batch['schema'] == 'routing-boundary-micro-screen-batch-v1' and
                    assignment['manifest_sha256'] == batch['manifest_sha256'] == manifest['sha256'] and
                    assignment['binding_sha256'] == batch['binding_sha256'] == binding['sha256'] and
                    assignment['batch_index'] == batch['batch_index'] == batch_index and
                    base.file_sha(directory / f'batch-{batch_index:03d}.npz') == batch['array_sha256'] and
                    [r['uid'] for r in assignment['requests']] == [r['uid'] for r in batch['outputs']],
                    'batch assignments, route arrays or seal differs')
            rows.extend(batch['outputs'])
            receipts.append({'shard': index, 'batch': batch_index,
                             'assignment_sha256': assignment['sha256'],
                             'batch_sha256': batch['sha256'], 'array_sha256': batch['array_sha256']})
        require([r['uid'] for r in rows] == [r['uid'] for r in expected], 'shard omitted/reordered assignment')
        for assigned, observed in zip(expected, rows, strict=True):
            require(all(observed.get(k) == v for k, v in assigned.items()), 'assigned metadata differs')
        found.extend(rows)
    require([r['uid'] for r in found] == [r['uid'] for r in plan], 'stage omits assigned results')
    return stage, plan, found, receipts


def reader_context(original, native):
    data, meta = original['reader_input'], original['analysis_meta']
    require(set(data) == {'problem', 'emitted_prefix', 'triggering_sentence'} and
            all(isinstance(v, str) for v in data.values()) and
            all(original[k] == native[k] for k in ('uid', 'family', 'attempt_id', 'transition', 'sentence_index')) and
            data['problem'] == native['question'] and
            original['prefix_tokens'] == native['prefix_tokens'] == len(native['prefix_ids']) and
            meta['prefix_ids_sha256'] == native['prefix_ids_sha256'] == base.digest(native['prefix_ids']) and
            native['prompt_ids_sha256'] == base.digest(native['prompt_ids']) and
            meta['native_trace_sha256'] == native['trace_sha256'] and
            meta['tokenizer_sha256'] == native['tokenizer_sha256'] and
            meta['prefix_text_sha256'] == native['prefix_text_sha256'] ==
                hashlib.sha256(data['emitted_prefix'].encode()).hexdigest() and
            data['emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()),
            'pretreatment problem/prefix/trigger does not match frozen generation')
    return data


def build(manifest, price, source, run_out, out, tokenizer, salt=None):
    require(source['schema'] == 'transition-v22-full-prefix-start-frame-v1' and
            source['sha256'] == '6c10499bf71c7223b5c90fa6c2a1b13e13e9fb5c9c2af584d3ba67b4272290b2',
            'fixed discovery start frame differs')
    stage, plan, outputs, receipts = complete_outputs(manifest, price, run_out)
    originals = {r['uid']: r for r in source['records']}
    natives = {r['uid']: r for r in manifest['rows']}
    require(len(originals) == len(source['records']) and len(natives) == len(manifest['rows']),
            'duplicate pretreatment start UID')
    contexts = {uid: reader_context(originals[uid], native) for uid, native in natives.items()}
    map_path = out / 'ARM_MAP.json'
    if map_path.exists():
        previous = base.sealed(map_path)
        require(previous['manifest_sha256'] == manifest['sha256'] and
                previous['stage_completion_sha256'] == stage['sha256'], 'existing arm-map rebound')
        salt = bytes.fromhex(previous['blind_salt_hex'])
    else:
        salt = salt or os.urandom(32)
    require(len(salt) == 32, 'blind salt must contain 32 bytes')
    map_rows, blind_rows = [], []
    for expected, observed in zip(plan, outputs, strict=True):
        tokens = observed['tokens']
        require(isinstance(tokens, list) and len(tokens) <= manifest['horizon'] and
                all(type(t) is int and t >= 0 for t in tokens), 'invalid generated token sequence')
        closure = THINK_END_ID in tokens
        reasoning = tokens[:tokens.index(THINK_END_ID)] if closure else tokens
        status = ('generation_error' if observed['error'] else 'missing_routed_array' if not
                  observed['routed_present'] else 'zero_reasoning_tokens' if not reasoning else 'gradeable')
        blind_id = hmac.new(salt, expected['uid'].encode(), hashlib.sha256).hexdigest()[:32]
        map_rows.append({**expected, 'blind_id': blind_id, 'measurement_status': status,
                         'finish': observed['finish'], 'stop_reason': observed['stop_reason'],
                         'error': observed['error'], 'emitted_tokens': len(tokens),
                         'reasoning_tokens': len(reasoning), 'closed_reasoning': closure,
                         'hit_cap': len(tokens) == manifest['horizon'],
                         'routed_present': observed['routed_present'],
                         'action_dose': observed.get('action_dose'),
                         'inactive_native_checks': observed.get('inactive_native_checks'),
                         'missing_telemetry_ranks': observed.get('missing_telemetry_ranks')})
        if status == 'gradeable':
            data = contexts[expected['prefix_uid']]
            blind_rows.append({'blind_id': blind_id, 'reader_input': {
                'transition': expected['transition'], 'problem': data['problem'],
                'full_emitted_prefix': data['emitted_prefix'], 'triggering_sentence': data['triggering_sentence'],
                'continuation': tokenizer.decode(reasoning, skip_special_tokens=False)}})
    blind_rows.sort(key=lambda r: r['blind_id'])
    require(len({r['blind_id'] for r in map_rows}) == len(map_rows), 'blind ID collision')
    out.mkdir(parents=True, exist_ok=True)
    shared = {'builder_sha256': base.file_sha(__file__), 'stage_completion_sha256': stage['sha256'],
              'source_frame_sha256': source['sha256'], 'rubric_sha256': base.file_sha(rating.RUBRIC)}
    shared['analysis_driver_sha256'] = base.file_sha(Path(__file__).with_name('analyze_overnight_semantics_v1.py'))
    arm_map = rating.save(map_path, {**shared, 'schema': 'overnight-semantic-arm-map-v1',
        'manifest_sha256': manifest['sha256'], 'generation_price_sha256': price['sha256'],
        'blind_salt_hex': salt.hex(), 'batch_receipts': receipts, 'records': map_rows,
        'continuation_max_tokens': manifest['horizon'],
        'tokenizer_config_sha256': base.file_sha(TOKENIZER / 'tokenizer_config.json')}, existing_ok=True)
    frame = rating.save(out / 'BLIND_FRAME.json', {**shared, 'schema': 'overnight-semantic-blind-frame-v1',
        'generation_manifest_sha256': manifest['sha256'], 'reader_input_allowlist': ALLOWLIST,
        'continuation_max_tokens': manifest['horizon'], 'assigned': len(plan), 'records': blind_rows}, existing_ok=True)
    return arm_map, frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--generation-price', type=Path, required=True)
    p.add_argument('--run-out', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'), 'decoding requires CPU Slurm step')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    arm_map, frame = build(base.sealed(args.manifest), base.sealed(args.generation_price),
                           base.sealed(SOURCE), args.run_out, args.out, tokenizer)
    print(json.dumps({'map_sha256': arm_map['sha256'], 'frame_sha256': frame['sha256'],
                      'assigned': frame['assigned'], 'gradeable': len(frame['records'])}), flush=True)


if __name__ == '__main__':
    main()
