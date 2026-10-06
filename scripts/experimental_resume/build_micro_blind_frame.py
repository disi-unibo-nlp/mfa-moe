"""Make an arm-blind semantic-rating frame from sealed micro-screen receipts.

Every assigned request remains in the separate sealed analysis map, including
failures and batches without a completion. Only gradeable reasoning continuations
appear in the rater frame. Run on a CPU Slurm node after a GPU stage finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import socket

THINK_END_ID = 248069
SCOUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
             'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/'
             'dense-discovery/CANDIDATE_PREFIX_SCOUT_v1.json')
UNITS = SCOUT.with_name('UNITS.json')
TOKENIZER = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
                 'models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/'
                 '95a723d08a9490559dae23d0cff1d9466213d989')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('JSON seal differs: ' + str(path))
    return value


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError('existing blind output differs: ' + str(path))
    else:
        part = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
        part.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
        os.replace(part, path)
    return value


def assigned(manifest):
    for row in manifest['rows']:
        for seed in manifest['seeds']:
            for arm in manifest['arms']:
                uid = 'micro-v1|' + digest([manifest['sha256'], row['uid'], seed, arm['name']])[:24]
                yield {'uid': uid, 'prefix_uid': row['uid'], 'family': row['family'],
                       'question': row['question'], 'seed': seed, 'arm': arm['name'],
                       'role': arm['role'], 'prompt_len': len(row['prompt_ids']) + len(row['prefix_ids'])}


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('blind-frame decoding belongs on CPU Slurm')
    if args.batch_size < 6 or args.batch_size % 6:
        raise ValueError('batch size must retain complete six-arm groups')
    manifest = sealed(args.manifest)
    scout = sealed(SCOUT)
    units = sealed(UNITS)
    if manifest['prefix_scout_sha256'] != scout['sha256']:
        raise ValueError('original-problem source differs from generation manifest')
    scout_rows = {r['uid']: r for r in scout['records']}
    unit_rows = {(r['family'], r['attempt_id'], r['sentence_index']): r
                 for r in units['records']}
    binding = sealed(args.run_out / 'BINDING.json')
    if binding['manifest_sha256'] != manifest['sha256']:
        raise ValueError('generation binding and manifest differ')
    plan = list(assigned(manifest))
    if len(plan) != manifest['expected_requests']:
        raise ValueError('assigned request count differs from manifest')
    rows = {r['uid']: r for r in manifest['rows']}
    by_uid = {}
    receipts = []
    for i in range(math.ceil(len(plan) / args.batch_size)):
        expected = plan[i * args.batch_size:(i + 1) * args.batch_size]
        assignment_path = args.run_out / f'batch-{i:03d}-assignment.json'
        result_path = args.run_out / f'batch-{i:03d}.json'
        if assignment_path.exists():
            assignment = sealed(assignment_path)
            if (assignment['manifest_sha256'] != manifest['sha256'] or
                assignment['binding_sha256'] != binding['sha256'] or
                [r['uid'] for r in assignment['requests']] != [r['uid'] for r in expected]):
                raise ValueError('batch assignment changed')
        if not result_path.exists():
            receipts.append({'batch': i, 'status': 'NO_RESULT_RECEIPT',
                             'assignment_sha256': sealed(assignment_path)['sha256']
                             if assignment_path.exists() else None})
            continue
        result = sealed(result_path)
        array_path = args.run_out / f'batch-{i:03d}.npz'
        if (result['manifest_sha256'] != manifest['sha256'] or
            result['binding_sha256'] != binding['sha256'] or
            [r['uid'] for r in result['outputs']] != [r['uid'] for r in expected] or
            not array_path.exists() or file_sha(array_path) != result['array_sha256']):
            raise ValueError('batch outputs or routed arrays changed')
        receipts.append({'batch': i, 'status': 'SEALED_RESULT',
                         'result_sha256': result['sha256'], 'array_sha256': result['array_sha256']})
        by_uid.update({r['uid']: r for r in result['outputs']})
    if len(by_uid) != len({x['uid'] for x in by_uid.values()}):
        raise ValueError('duplicate result UID')
    summary_path = args.run_out / 'SUMMARY.json'
    summary = sealed(summary_path) if summary_path.exists() else None
    if summary and (summary['manifest_sha256'] != manifest['sha256'] or
                    summary['requests'] != len(plan)):
        raise ValueError('summary and assigned count differ')
    if summary and summary['status'] == 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION' and len(by_uid) != len(plan):
        raise ValueError('complete summary has missing results')

    # The blinded frame contains no source request UID, family, seed, action, or arm.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    map_path = args.out / 'ARM_MAP.json'
    if map_path.exists():
        existing = sealed(map_path)
        if existing['manifest_sha256'] != manifest['sha256'] or existing['generation_binding_sha256'] != binding['sha256']:
            raise ValueError('blind map belongs to another run')
        salt = bytes.fromhex(existing['blind_salt_hex'])
    else:
        salt = os.urandom(32)
    frame_rows, map_rows = [], []
    for item in plan:
        output = by_uid.get(item['uid'])
        blind_id = hmac.new(salt, item['uid'].encode(), hashlib.sha256).hexdigest()[:32]
        if output is None:
            status = 'missing_result'
        elif output.get('error'):
            status = 'generation_error'
        elif not output.get('routed_present'):
            status = 'missing_routed_array'
        elif not output.get('tokens'):
            status = 'zero_emitted_tokens'
        else:
            status = 'gradeable'
        map_rows.append({**item, 'blind_id': blind_id, 'measurement_status': status,
                         'finish': output.get('finish') if output else None,
                         'stop_reason': output.get('stop_reason') if output else None,
                         'error': output.get('error') if output else None,
                         'emitted_tokens': len(output.get('tokens', [])) if output else 0,
                         'action_dose': output.get('action_dose') if output else None,
                         'inactive_native_checks': output.get('inactive_native_checks') if output else None})
        if status != 'gradeable':
            continue
        source = rows[item['prefix_uid']]
        scout_source = scout_rows.get(source['uid'])
        if scout_source is None or source['prefix_ids'] != scout_source['prefix_ids']:
            raise ValueError('native prefix source differs from scout')
        unit = unit_rows.get((scout_source['family'], scout_source['attempt_id'],
                              scout_source['sentence_index']))
        if (unit is None or unit['trace_sha256'] != scout_source['trace_sha256'] or
            unit['inputs']['problem_statement'] != scout_source['problem']):
            raise ValueError('audited trigger sentence differs from dense native unit')
        emitted = output['tokens']
        closure = THINK_END_ID in emitted
        reasoning_ids = emitted[:emitted.index(THINK_END_ID)] if closure else emitted
        frame_rows.append({'blind_id': blind_id,
                           'reader_input': {'problem': scout_source['problem'],
                                            'previous_sentence': unit['inputs']['previous_sentence'],
                                            'triggering_sentence': unit['inputs']['sentence'],
                                            'full_emitted_prefix': tokenizer.decode(source['prefix_ids'],
                                                                                     skip_special_tokens=False),
                                            'continuation': tokenizer.decode(reasoning_ids,
                                                                             skip_special_tokens=False)}})
    frame_rows.sort(key=lambda x: x['blind_id'])
    map_body = {'schema': 'routing-micro-screen-arm-map-v1',
                'manifest_sha256': manifest['sha256'],
                'generation_binding_sha256': binding['sha256'],
                'generation_summary_sha256': summary['sha256'] if summary else None,
                'batch_receipts': receipts, 'blind_salt_hex': salt.hex(),
                'native_units_sha256': units['sha256'],
                'tokenizer_path': str(TOKENIZER),
                'tokenizer_config_sha256': file_sha(TOKENIZER / 'tokenizer_config.json'),
                'records': map_rows}
    mapping = write_once(map_path, map_body)
    frame_body = {'schema': 'routing-micro-screen-blind-frame-v1',
                  'generation_manifest_sha256': manifest['sha256'],
                  'opaque_id_scheme': 'HMAC-SHA256; salt held only in separate arm map',
                  'reader_input_allowlist': ['problem', 'previous_sentence', 'triggering_sentence',
                                             'full_emitted_prefix', 'continuation'],
                  'full_emitted_prefix_max_native_tokens': 8192,
                  'continuation_max_tokens': manifest['max_tokens'],
                  'records': frame_rows}
    frame = write_once(args.out / 'BLIND_FRAME.json', frame_body)
    print(json.dumps({'arm_map': str(map_path), 'arm_map_sha256': mapping['sha256'],
                      'blind_frame': str(args.out / 'BLIND_FRAME.json'),
                      'blind_frame_sha256': frame['sha256'],
                      'assigned': len(plan), 'gradeable': len(frame_rows),
                      'incomplete_batches': sum(r['status'] != 'SEALED_RESULT' for r in receipts)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    run(parser.parse_args())
