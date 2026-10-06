"""Bind every assigned eligible-screen result and make arm-blind rating inputs.

This runs on CPU Slurm after generation.  It never supplies arm, policy, seed,
expert route, family or future native text to the semantic reader.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path

import run_boundary_micro_screen as base
import run_eligible_deactivation_serial_v4 as diagnostic_v4

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
FRAME = ROOT / 'runs/routing-control-v1/dense-discovery/TRANSITION_V22_FULL_PREFIX_START_FRAME.json'
RUBRIC = REPO / 'report/experimental-resume-v1/CAUSAL_ELIGIBLE_IMMEDIATE_SEMANTIC_RUBRIC_v1.md'
TOKENIZER = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
                 'models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/'
                 '95a723d08a9490559dae23d0cff1d9466213d989')
THINK_END_ID = 248069
KINDS = {'routing-eligible-deactivation-serial-v4': 'deactivate-v4|'}


def write_once(path, body):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if base.sealed(path) != value:
            raise ValueError('bound blind output differs: ' + str(path))
    else:
        tmp = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
        os.replace(tmp, path)
    return value


def assignments(manifest):
    prefix = KINDS[manifest['schema']]
    for row in manifest['rows']:
        for seed in manifest['seeds']:
            for arm in diagnostic_v4.ordered_arms(manifest, row, seed):
                uid = prefix + base.digest([manifest['sha256'], row['uid'], seed, arm['name']])[:24]
                yield {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                       'transition': row['transition'], 'canonical_question': row['canonical_question'],
                       'seed': seed, 'arm': arm['name'], 'role': arm['role']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('blind-frame decoding requires CPU Slurm allocation and step')
    manifest, source = base.sealed(args.manifest), base.sealed(FRAME)
    if manifest['schema'] not in KINDS or len(manifest['rows']) != 13:
        raise ValueError('unexpected eligible-screen assignment manifest')
    if source['schema'] != 'transition-v22-full-prefix-start-frame-v1':
        raise ValueError('source full-prefix frame changed')
    source_by_uid = {row['uid']: row for row in source['records']}
    for row in manifest['rows']:
        original = source_by_uid[row['uid']]
        if (original['family'] != row['family'] or original['transition'] != row['transition'] or
                original['reader_input']['problem'] != row['question'] or
                hashlib.sha256(original['reader_input']['emitted_prefix'].encode()).hexdigest() != row['prefix_text_sha256'] or
                original['analysis_meta']['prefix_ids_sha256'] != row['prefix_ids_sha256']):
            raise ValueError('rated pretreatment prefix differs from exact native replay')
    binding, summary = base.sealed(args.run_out / 'BINDING.json'), base.sealed(args.run_out / 'SUMMARY.json')
    if (binding['manifest_sha256'] != manifest['sha256'] or
            summary['manifest_sha256'] != manifest['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            summary['requests'] != manifest['expected_requests'] or
            summary['status'] != 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION'):
        raise ValueError('generation binding, completion or assignment count differs')
    plan = list(assignments(manifest))
    if len(plan) != manifest['expected_requests'] or len(plan) % 6:
        raise ValueError('six-arm assignment count differs')
    by_uid, receipts = {}, []
    for index in range(len(plan) // 6):
        expected = plan[index * 6:(index + 1) * 6]
        assignment = base.sealed(args.run_out / f'batch-{index:03d}-assignment.json')
        batch = base.sealed(args.run_out / f'batch-{index:03d}.json')
        array = args.run_out / f'batch-{index:03d}.npz'
        if (assignment['manifest_sha256'] != manifest['sha256'] or
                assignment['binding_sha256'] != binding['sha256'] or
                [r['uid'] for r in assignment['requests']] != [r['uid'] for r in expected] or
                batch['manifest_sha256'] != manifest['sha256'] or
                batch['binding_sha256'] != binding['sha256'] or
                [r['uid'] for r in batch['outputs']] != [r['uid'] for r in expected] or
                base.file_sha(array) != batch['array_sha256']):
            raise ValueError('changed or missing assigned checkpoint: ' + str(index))
        receipts.append({'batch': index, 'assignment_sha256': assignment['sha256'],
                         'result_sha256': batch['sha256'], 'array_sha256': batch['array_sha256']})
        by_uid.update({row['uid']: row for row in batch['outputs']})
    if len(by_uid) != len(plan):
        raise ValueError('missing or duplicate assigned outcomes')
    negative_mode = manifest['schema'] == 'routing-eligible-deactivation-serial-v4'
    telemetry_by_rank, telemetry_hashes, telemetry_parse = {}, {}, {}
    if negative_mode:
        from moe_steer import reduce
        telemetry_dir = args.run_out / 'telemetry'
        telemetry_by_rank, telemetry_parse = reduce.load_telemetry(telemetry_dir)
        telemetry_hashes = {str(rank): base.file_sha(telemetry_dir / f'rank{rank}.jsonl')
                            for rank in (0, 1)}
        if set(telemetry_by_rank) != {0, 1}:
            raise ValueError('base-hook telemetry missing a tensor-parallel rank')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out / 'ARM_MAP.json'
    if mapping_path.exists():
        old = base.sealed(mapping_path)
        if old['manifest_sha256'] != manifest['sha256'] or old['generation_binding_sha256'] != binding['sha256']:
            raise ValueError('blind map is bound to another run')
        salt = bytes.fromhex(old['blind_salt_hex'])
    else:
        salt = os.urandom(32)
    frame_rows, map_rows = [], []
    for item in plan:
        output = by_uid[item['uid']]
        base_hook_dose = {}
        if negative_mode:
            for rank in (0, 1):
                record = telemetry_by_rank[rank].get(item['uid'])
                if record is None:
                    if not output.get('error'):
                        raise ValueError('assigned successful request lacks base-hook dose')
                    continue
                if record.get('ordered_action_dose') is not None:
                    raise ValueError('negative base-hook request unexpectedly used ordered control')
                base_hook_dose[str(rank)] = {
                    'policy': record.get('policy'),
                    'cpu_active_rows': record.get('cpu_active_rows'),
                    'cpu_pulse_rows': record.get('cpu_pulse_rows'),
                    'rows': record.get('rows'),
                    'prefix_len': record.get('prefix_len'),
                    'preemptions': record.get('preemptions'),
                    'counters': record.get('counters'),
                }
        blind_id = hmac.new(salt, item['uid'].encode(), hashlib.sha256).hexdigest()[:32]
        tokens = output.get('tokens', [])
        if output.get('error'):
            status = 'generation_error'
        elif not output.get('routed_present'):
            status = 'missing_routed_array'
        elif not tokens:
            status = 'zero_emitted_tokens'
        else:
            status = 'gradeable'
        closure = THINK_END_ID in tokens
        reasoning = tokens[:tokens.index(THINK_END_ID)] if closure else tokens
        map_rows.append({**item, 'blind_id': blind_id, 'measurement_status': status,
                         'finish': output.get('finish'), 'stop_reason': output.get('stop_reason'),
                         'emitted_tokens': len(tokens), 'reasoning_tokens': len(reasoning),
                         'closed_reasoning': closure, 'action_dose': output.get('action_dose'),
                         'base_hook_dose': base_hook_dose,
                         'inactive_native_checks': output.get('inactive_native_checks')})
        if status == 'gradeable':
            original = source_by_uid[item['prefix_uid']]
            frame_rows.append({'blind_id': blind_id, 'reader_input': {
                'transition': item['transition'], 'problem': original['reader_input']['problem'],
                'full_emitted_prefix': original['reader_input']['emitted_prefix'],
                'triggering_sentence': original['reader_input']['triggering_sentence'],
                'continuation': tokenizer.decode(reasoning, skip_special_tokens=False)}})
    frame_rows.sort(key=lambda row: row['blind_id'])
    map_body = {'schema': 'eligible-immediate-arm-map-v1',
                'builder_sha256': base.file_sha(__file__),
                'diagnostic_driver_sha256': base.file_sha(diagnostic_v4.__file__),
                'manifest_sha256': manifest['sha256'],
                'generation_binding_sha256': binding['sha256'],
                'generation_summary_sha256': summary['sha256'],
                'source_frame_sha256': source['sha256'],
                'rubric_sha256': base.file_sha(RUBRIC),
                'batch_receipts': receipts, 'blind_salt_hex': salt.hex(),
                'base_hook_telemetry_sha256_by_rank': telemetry_hashes,
                'base_hook_telemetry_parse': telemetry_parse,
                'tokenizer_config_sha256': base.file_sha(TOKENIZER / 'tokenizer_config.json'),
                'records': map_rows}
    map_value = write_once(mapping_path, map_body)
    frame_body = {'schema': 'eligible-immediate-blind-frame-v1',
                  'builder_sha256': base.file_sha(__file__),
                  'diagnostic_driver_sha256': base.file_sha(diagnostic_v4.__file__),
                  'generation_manifest_sha256': manifest['sha256'],
                  'generation_summary_sha256': summary['sha256'],
                  'source_frame_sha256': source['sha256'],
                  'rubric_sha256': base.file_sha(RUBRIC),
                  'reader_input_allowlist': ['transition', 'problem', 'full_emitted_prefix',
                                             'triggering_sentence', 'continuation'],
                  'continuation_max_tokens': 256,
                  'records': frame_rows}
    frame_value = write_once(args.out / 'BLIND_FRAME.json', frame_body)
    print(json.dumps({'map': str(mapping_path), 'map_sha256': map_value['sha256'],
                      'frame': str(args.out / 'BLIND_FRAME.json'),
                      'frame_sha256': frame_value['sha256'],
                      'assigned': len(plan), 'gradeable': len(frame_rows),
                      'ungradeable': len(plan) - len(frame_rows)}))


if __name__ == '__main__':
    main()
