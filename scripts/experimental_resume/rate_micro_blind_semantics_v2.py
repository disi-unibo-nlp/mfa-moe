"""Separate pre-treatment starts and arm-blind targets for pilot continuations.

Only BLIND_FRAME.json is read.  The arm map is intentionally inaccessible to
this driver; joining assignments belongs to a later, separate CPU analysis.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
FRAME = ROOT / ('steering-v1/runs/routing-control-v1/'
                'micro-screen-blind-qual4-ac4c9651e71fe067/BLIND_FRAME.json')
RUBRIC = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_SEMANTIC_RUBRIC_v2.md'
PRICE = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.1.json'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
             'models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
BATCH = 12
MAX_TOKENS = 1024
ALLOWLIST = {'problem', 'previous_sentence', 'triggering_sentence',
             'full_emitted_prefix', 'continuation'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed sealed JSON: {path}')
    return value


def messages(row, task):
    if set(row) != {'blind_id', 'reader_input'} or set(row['reader_input']) != ALLOWLIST:
        raise ValueError('arm-blind reader allowlist changed')
    if task not in ('start', 'target'):
        raise ValueError('unknown rating task')
    data = row['reader_input']
    if not all(isinstance(x, str) for x in data.values()):
        raise ValueError('nontext reader field')
    if not data['full_emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()):
        raise ValueError('triggering sentence is not the end of the native prefix')
    if task == 'start':
        system = ('Rate only the starting condition of visible mathematical reasoning. '
                  'Use the original problem and already emitted native prefix. '
                  'Return exactly {"start": true} or {"start": false}; no explanation. '
                  'True only if the final triggering sentence states a complete specific '
                  'candidate or computed value still unchecked in the full prefix. '
                  'An unfinished expression, generic plan or already checked candidate '
                  'is false. Ambiguity is false. No future continuation is provided.')
    else:
        system = ('Rate only the NEW reasoning continuation. Use the original problem '
                  'and already emitted native prefix to identify the candidate in the '
                  'triggering sentence. Return exactly {"target": true} or '
                  '{"target": false}; no explanation. True only if the continuation '
                  'substantively tests THAT candidate against an original constraint '
                  'or by an independent calculation. Verification words, restatement, '
                  'unrelated calculation or final answer alone are false. If reasoning '
                  'closes first, or evidence is ambiguous, return false.')
    user = ('Original problem:\n' + data['problem'] +
            '\n\nAlready emitted native reasoning prefix:\n' + data['full_emitted_prefix'] +
            '\n\nTriggering sentence (already in prefix):\n' + data['triggering_sentence'])
    if task == 'target':
        user += '\n\nOne assigned new reasoning continuation:\n' + data['continuation']
    user += '\n\nJSON:'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_rating(raw, task):
    if '</think>' in raw:
        raw = raw.rsplit('</think>', 1)[1]
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if (task not in ('start', 'target') or not isinstance(value, dict) or
            set(value) != {task} or type(value[task]) is not bool):
        return None
    return value


def rating_seed(uid, reader, task):
    return int(digest(['micro-blind-semantic-v2', task, uid, reader])[:8], 16) % 2_000_000_000


def start_groups(rows):
    """Collapse 48 assignments to four frozen prefixes before reading a continuation."""
    grouped = {}
    for row in rows:
        data = row['reader_input']
        key = digest([data['problem'], data['full_emitted_prefix'],
                      data['triggering_sentence']])[:32]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    if len(grouped) != 4 or any(len(group) != 12 for group in grouped.values()):
        raise ValueError('expected four unique native prefixes and 12 arms/seeds each')
    return [(key, grouped[key][0]) for key in sorted(grouped)]


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError(f'existing result differs: {path}')
    else:
        part = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
        part.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(part, path)
    return value


def run(out):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('semantic inference requires GPU Slurm')
    frame, price = sealed(FRAME), sealed(PRICE)
    rows = frame['records']
    if (frame['schema'] != 'routing-micro-screen-blind-frame-v1' or len(rows) != 48 or
            set(frame['reader_input_allowlist']) != ALLOWLIST or
            len({r['blind_id'] for r in rows}) != 48):
        raise ValueError('blind frame population changed')
    groups = start_groups(rows)
    if (price['status'] != 'PASS_COMPLETE_STAGE' or
            price['frame_sha256'] != frame['sha256'] or
            price['driver_sha256'] != file_sha(__file__) or
            price['rubric_sha256'] != file_sha(RUBRIC) or
            price['ratings'] != 104 or price['start_ratings'] != 8 or
            price['target_ratings'] != 96 or
            price['max_decode_tokens'] != 104 * MAX_TOKENS):
        raise ValueError('sealed complete-stage price does not bind code/rubric/frame')
    binding_body = {'schema': 'micro-blind-semantic-binding-v2',
                    'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
                    'driver_sha256': file_sha(__file__), 'rubric_sha256': file_sha(RUBRIC),
                    'model': str(MODEL), 'batch_size': BATCH, 'readers': 2,
                    'max_tokens_per_rating': MAX_TOKENS,
                    'sampler': {'temperature': .2, 'top_p': .95,
                                'enable_thinking': True, 'reasoning_effort': 'low'},
                    'scope': 'four pre-treatment starts and 48 arm-blind targets; two Qwen draws each'}
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = write_once(out / 'BINDING.json', binding_body)
    (out / 'batches').mkdir(exist_ok=True)
    (out / 'assignments').mkdir(exist_ok=True)
    pending = []
    start_rows = [(key, row) for key, row in groups]
    for reader in (0, 1):
        path = out / 'batches' / f'start-reader{reader}.json'
        if path.exists():
            saved = sealed(path)
            if (saved['binding_sha256'] != binding['sha256'] or
                    [r['rating_id'] for r in saved['records']] != [key for key, _ in start_rows]):
                raise ValueError('saved start batch differs')
        else:
            pending.append(('start', 0, reader, start_rows, path))
    for start in range(0, len(rows), BATCH):
        block = [(r['blind_id'], r) for r in rows[start:start+BATCH]]
        for reader in (0, 1):
            path = out / 'batches' / f'target-{start:03d}-reader{reader}.json'
            if path.exists():
                saved = sealed(path)
                if (saved['binding_sha256'] != binding['sha256'] or
                        [r['rating_id'] for r in saved['records']] !=
                        [key for key, _ in block]):
                    raise ValueError('saved batch belongs to different assignments')
            else:
                pending.append(('target', start, reader, block, path))
    summary_path = out / 'SUMMARY.json'
    if summary_path.exists():
        previous = sealed(summary_path)
        if (pending or previous['binding_sha256'] != binding['sha256'] or
                previous['ratings'] != 104 or previous['rows'] != 48):
            raise ValueError('completed rating summary differs from same-bound batches')
        print(json.dumps({'summary': str(summary_path), 'sha256': previous['sha256'],
                          'status': 'same-manifest-complete'}), flush=True)
        return
    load_seconds = 0.0
    if pending:
        from vllm import LLM, SamplingParams
        t0 = time.monotonic()
        model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                    dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=49152,
                    max_num_seqs=16, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config='vllm', language_model_only=True,
                    attention_config={'backend': 'FLASH_ATTN'})
        load_seconds = time.monotonic() - t0
    for task, start, reader, block, path in pending:
        assignment_name = f'{task}-{start:03d}-reader{reader}'
        receipt = out / 'assignments' / f'{assignment_name}.json'
        if receipt.exists():
            raise RuntimeError(f'incomplete previous assignment requires adjudication: {receipt}')
        write_once(receipt, {'schema': 'micro-blind-rating-assignment-v2',
                             'binding_sha256': binding['sha256'],
                             'job_id': os.environ['SLURM_JOB_ID'],
                             'task': task, 'start': start, 'reader': reader,
                             'rating_ids': [key for key, _ in block],
                             'status': 'attempted_before_generation'})
        params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                 seed=rating_seed(key, reader, task)) for key, _ in block]
        t0 = time.monotonic()
        try:
            outputs = model.chat([messages(r, task) for _, r in block], sampling_params=params,
                                 chat_template_kwargs={'enable_thinking': True,
                                                       'reasoning_effort': 'low'},
                                 use_tqdm=False)
        except Exception as exc:
            write_once(out / 'assignments' / f'{assignment_name}-failure.json',
                       {'schema': 'micro-blind-rating-failure-v2',
                        'binding_sha256': binding['sha256'],
                        'job_id': os.environ['SLURM_JOB_ID'], 'task': task, 'start': start,
                        'reader': reader, 'exception_type': type(exc).__name__,
                        'exception_message': str(exc)[:500]})
            raise
        if len(outputs) != len(block):
            raise ValueError('judge output count differs from assigned count')
        records = []
        for (key, _), output in zip(block, outputs, strict=True):
            completion = output.outputs[0]
            records.append({'rating_id': key,
                            'rating': parse_rating(completion.text, task),
                            'finish_reason': completion.finish_reason,
                            'prompt_tokens': len(output.prompt_token_ids),
                            'generated_tokens': len(completion.token_ids),
                            'raw_completion': completion.text})
        write_once(path, {'schema': 'micro-blind-semantic-batch-v2',
                          'binding_sha256': binding['sha256'], 'task': task, 'start': start,
                          'reader': reader, 'job_id': os.environ['SLURM_JOB_ID'],
                          'elapsed_seconds': time.monotonic() - t0,
                          'records': records})
        print(json.dumps({'task': task, 'start': start, 'reader': reader,
                          'valid': sum(x['rating'] is not None and
                                       x['finish_reason'] == 'stop' for x in records)}), flush=True)
    batch_rows = [sealed(out / 'batches' / f'start-reader{reader}.json') for reader in (0, 1)]
    for start in range(0, len(rows), BATCH):
        for reader in (0, 1):
            batch_rows.append(sealed(out / 'batches' / f'target-{start:03d}-reader{reader}.json'))
    all_records = [r for batch in batch_rows for r in batch['records']]
    if len(all_records) != 104:
        raise ValueError('rating count differs')
    summary = write_once(out / 'SUMMARY.json',
                         {'schema': 'micro-blind-semantic-summary-v2',
                          'binding_sha256': binding['sha256'],
                          'frame_sha256': frame['sha256'],
                          'batch_sha256': [x['sha256'] for x in batch_rows],
                          'ratings': 104, 'start_ratings': 8, 'target_ratings': 96,
                          'rows': 48, 'unique_native_prefixes': 4,
                          'valid_stopped_ratings': sum(r['rating'] is not None and
                                                       r['finish_reason'] == 'stop'
                                                       for r in all_records),
                          'prompt_tokens': sum(r['prompt_tokens'] for r in all_records),
                          'generated_tokens': sum(r['generated_tokens'] for r in all_records),
                          'load_seconds_this_job': load_seconds,
                          'status': 'COMPLETE_ARM_BLIND_LLM_AUDIT'})
    print(json.dumps({'summary': str(out / 'SUMMARY.json'),
                      'sha256': summary['sha256'], 'valid': summary['valid_stopped_ratings']}),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
