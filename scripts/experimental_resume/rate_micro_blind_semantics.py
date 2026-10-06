"""Two arm-blind Qwen ratings of the sealed causal micro-screen continuations.

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
RUBRIC = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_SEMANTIC_RUBRIC_v1.md'
PRICE = REPO / 'report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v1.json'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
             'models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
BATCH = 12
MAX_TOKENS = 768
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


def messages(row):
    if set(row) != {'blind_id', 'reader_input'} or set(row['reader_input']) != ALLOWLIST:
        raise ValueError('arm-blind reader allowlist changed')
    data = row['reader_input']
    if not all(isinstance(x, str) for x in data.values()):
        raise ValueError('nontext reader field')
    if not data['full_emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()):
        raise ValueError('triggering sentence is not the end of the native prefix')
    system = (
        'Rate visible mathematical reasoning independently. Use only the original '
        'problem, already emitted native prefix and this one new continuation. '
        'The triggering sentence is repeated for focus and is not an outcome. '
        'Return exactly one JSON object with booleans start and target; no explanation. '
        'start=true only if the triggering sentence states a complete specific candidate '
        'or computed value still unchecked in the full prefix. An unfinished expression, '
        'generic plan or previously checked candidate is false. target=true only if the '
        'new continuation substantively checks THAT candidate against an original '
        'constraint or by an independent calculation. Verification words, a restatement, '
        'an unrelated calculation or a final answer alone are false. If reasoning closes '
        'before that check, target=false. Ambiguity is false. '
        'The two judgments are independent: target may be true when start is false.'
    )
    user = ('Original problem:\n' + data['problem'] +
            '\n\nAlready emitted native reasoning prefix:\n' + data['full_emitted_prefix'] +
            '\n\nPrevious sentence:\n' + data['previous_sentence'] +
            '\n\nTriggering sentence (already in prefix):\n' + data['triggering_sentence'] +
            '\n\nOne assigned new reasoning continuation:\n' + data['continuation'] +
            '\n\nJSON:')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_rating(raw):
    if '</think>' in raw:
        raw = raw.rsplit('</think>', 1)[1]
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if (not isinstance(value, dict) or set(value) != {'start', 'target'} or
            any(type(value[k]) is not bool for k in ('start', 'target'))):
        return None
    return value


def rating_seed(blind_id, reader):
    return int(digest(['micro-blind-semantic-v1', blind_id, reader])[:8], 16) % 2_000_000_000


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
    if (price['status'] != 'PASS_COMPLETE_STAGE' or
            price['frame_sha256'] != frame['sha256'] or
            price['driver_sha256'] != file_sha(__file__) or
            price['rubric_sha256'] != file_sha(RUBRIC) or
            price['ratings'] != 96 or price['max_decode_tokens'] != 96 * MAX_TOKENS):
        raise ValueError('sealed complete-stage price does not bind code/rubric/frame')
    binding_body = {'schema': 'micro-blind-semantic-binding-v1',
                    'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
                    'driver_sha256': file_sha(__file__), 'rubric_sha256': file_sha(RUBRIC),
                    'model': str(MODEL), 'batch_size': BATCH, 'readers': 2,
                    'max_tokens_per_rating': MAX_TOKENS,
                    'sampler': {'temperature': .2, 'top_p': .95,
                                'enable_thinking': True, 'reasoning_effort': 'low'},
                    'scope': 'arm-blind pilot LLM audit of 48 assigned continuations'}
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = write_once(out / 'BINDING.json', binding_body)
    (out / 'batches').mkdir(exist_ok=True)
    (out / 'assignments').mkdir(exist_ok=True)
    pending = []
    for start in range(0, len(rows), BATCH):
        block = rows[start:start+BATCH]
        for reader in (0, 1):
            path = out / 'batches' / f'{start:03d}-reader{reader}.json'
            if path.exists():
                saved = sealed(path)
                if (saved['binding_sha256'] != binding['sha256'] or
                        [r['blind_id'] for r in saved['records']] !=
                        [r['blind_id'] for r in block]):
                    raise ValueError('saved batch belongs to different assignments')
            else:
                pending.append((start, reader, block, path))
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
    for start, reader, block, path in pending:
        receipt = out / 'assignments' / f'{start:03d}-reader{reader}.json'
        if receipt.exists():
            raise RuntimeError(f'incomplete previous assignment requires adjudication: {receipt}')
        write_once(receipt, {'schema': 'micro-blind-rating-assignment-v1',
                             'binding_sha256': binding['sha256'],
                             'job_id': os.environ['SLURM_JOB_ID'],
                             'start': start, 'reader': reader,
                             'blind_ids': [r['blind_id'] for r in block],
                             'status': 'attempted_before_generation'})
        params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                 seed=rating_seed(r['blind_id'], reader)) for r in block]
        t0 = time.monotonic()
        try:
            outputs = model.chat([messages(r) for r in block], sampling_params=params,
                                 chat_template_kwargs={'enable_thinking': True,
                                                       'reasoning_effort': 'low'},
                                 use_tqdm=False)
        except Exception as exc:
            write_once(out / 'assignments' / f'{start:03d}-reader{reader}-failure.json',
                       {'schema': 'micro-blind-rating-failure-v1',
                        'binding_sha256': binding['sha256'],
                        'job_id': os.environ['SLURM_JOB_ID'], 'start': start,
                        'reader': reader, 'exception_type': type(exc).__name__,
                        'exception_message': str(exc)[:500]})
            raise
        if len(outputs) != len(block):
            raise ValueError('judge output count differs from assigned count')
        records = []
        for row, output in zip(block, outputs, strict=True):
            completion = output.outputs[0]
            records.append({'blind_id': row['blind_id'],
                            'rating': parse_rating(completion.text),
                            'finish_reason': completion.finish_reason,
                            'prompt_tokens': len(output.prompt_token_ids),
                            'generated_tokens': len(completion.token_ids),
                            'raw_completion': completion.text})
        write_once(path, {'schema': 'micro-blind-semantic-batch-v1',
                          'binding_sha256': binding['sha256'], 'start': start,
                          'reader': reader, 'job_id': os.environ['SLURM_JOB_ID'],
                          'elapsed_seconds': time.monotonic() - t0,
                          'records': records})
        print(json.dumps({'start': start, 'reader': reader,
                          'valid': sum(x['rating'] is not None and
                                       x['finish_reason'] == 'stop' for x in records)}), flush=True)
    batch_rows = []
    for start in range(0, len(rows), BATCH):
        for reader in (0, 1):
            batch_rows.append(sealed(out / 'batches' / f'{start:03d}-reader{reader}.json'))
    all_records = [r for batch in batch_rows for r in batch['records']]
    if len(all_records) != 96:
        raise ValueError('rating count differs')
    summary = write_once(out / 'SUMMARY.json',
                         {'schema': 'micro-blind-semantic-summary-v1',
                          'binding_sha256': binding['sha256'],
                          'frame_sha256': frame['sha256'],
                          'batch_sha256': [x['sha256'] for x in batch_rows],
                          'ratings': 96, 'rows': 48,
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
