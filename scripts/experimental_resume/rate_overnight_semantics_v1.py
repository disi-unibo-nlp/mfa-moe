"""Two-reader arm-blind Qwen3.8 audit of frozen 256/1024-token continuations.

This program reads only BLIND_FRAME.json and the complete-stage reader price.
Each shard has append-only attempt receipts and one committed result per
blind ID and reader. An interrupted call needs explicit --resume-uncommitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import run_boundary_micro_screen as base

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
RUBRIC = REPO / 'report/experimental-resume-v1/OVERNIGHT_SEMANTIC_RUBRIC_2026-10-04_v1.md'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
             'models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
ALLOWLIST = {'transition', 'problem', 'full_emitted_prefix',
             'triggering_sentence', 'continuation'}
BATCH = 8
MAX_TOKENS = 1024
MAX_MODEL_LEN = 49152


def require(ok, message):
    if not ok:
        raise ValueError(message)


def messages(row):
    require(set(row) == {'blind_id', 'reader_input'} and
            set(row['reader_input']) == ALLOWLIST, 'reader-visible allowlist differs')
    d = row['reader_input']
    require(all(isinstance(value, str) for value in d.values()) and
            d['transition'] in ('candidate_to_verify', 'approach_to_commit') and
            d['full_emitted_prefix'].rstrip().endswith(d['triggering_sentence'].rstrip()),
            'invalid blind transition, prefix or trigger')
    if d['transition'] == 'candidate_to_verify':
        criterion = ('True only if the NEW reasoning continuation substantively tests '
                     'the same previously unchecked candidate or value against an '
                     'original constraint or by an independent calculation. Checking '
                     'words, repetition, unrelated calculation, and a final answer '
                     'alone are false.')
    else:
        criterion = ('True only if the NEW reasoning continuation commits to the '
                     'identified approach through a concrete planning or implementation '
                     'step advancing the original problem. Restatement, generic planning, '
                     'or an unrelated approach are false. If the approach is unclear '
                     'from the prefix, return false.')
    system = ('Rate only the new reasoning continuation after the triggering sentence. '
              'The prefix and trigger are context, not outcome evidence. ' + criterion +
              ' Closure without the behavior and ambiguity are false. Return exactly '
              '{"target":true} or {"target":false}, with no explanation.')
    user = ('Transition: ' + d['transition'] + '\n\nOriginal problem:\n' + d['problem'] +
            '\n\nAlready emitted native reasoning prefix:\n' + d['full_emitted_prefix'] +
            '\n\nTriggering sentence (already in prefix):\n' + d['triggering_sentence'] +
            '\n\nOne assigned new reasoning continuation:\n' + d['continuation'] +
            '\n\nJSON:')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_rating(raw):
    if '</think>' in raw:
        raw = raw.rsplit('</think>', 1)[1]
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) and set(value) == {'target'} and \
        type(value['target']) is bool else None


def rating_seed(blind_id, reader):
    return int(base.digest(['overnight-semantic-semantic-v1', blind_id, reader])[:8], 16) % 2_000_000_000


def save(path, body, *, existing_ok=False):
    value = {**body, 'sha256': base.digest(body)}
    if path.exists():
        if existing_ok and base.sealed(path) == value:
            return value
        raise FileExistsError(path)
    temp = path.with_name(path.name + f'.partial-{os.getpid()}-{time.monotonic_ns()}')
    with temp.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return value


def validate(frame, price):
    rows = frame['records']
    require(frame['schema'] == 'overnight-semantic-blind-frame-v1' and
            frame['continuation_max_tokens'] in (256, 1024) and
            frame['rubric_sha256'] == base.file_sha(RUBRIC) and
            set(frame['reader_input_allowlist']) == ALLOWLIST and
            len(rows) <= frame['assigned'] and
            len({r['blind_id'] for r in rows}) == len(rows) and
            price['schema'] == 'overnight-semantic-reader-price-v1' and
            price['status'] == 'PASS_COMPLETE_STAGE' and
            price['frame_sha256'] == frame['sha256'] and
            price['rating_driver_sha256'] == base.file_sha(__file__) and
            price['rubric_sha256'] == base.file_sha(RUBRIC) and
            price['model_snapshot'] == str(MODEL) and
            price['rows'] == len(rows) and price['ratings'] == 2 * len(rows) and
            price['max_decode_tokens'] == 2 * len(rows) * MAX_TOKENS and
            price['batch_size'] == BATCH and
            price['gpus'] == 2 and price['max_wall_seconds_per_job'] == 7200 and
            price['pricing_driver_sha256'] == base.file_sha(
                REPO / 'scripts/experimental_resume/price_overnight_semantics_v1.py') and
            price['estimated_complete_gpu_hours'] <= price['gpu_hour_ceiling'] and
            len(price['prompt_token_lengths']) == len(rows) and
            price['prompt_tokens_exact_twice'] ==
            2 * sum(price['prompt_token_lengths']) and
            max(price['prompt_token_lengths'], default=0) + MAX_TOKENS <= MAX_MODEL_LEN,
            'reader frame, code or complete-stage price differs')
    for row in rows:
        messages(row)
    if not rows:
        require(price['shards'] == [] and price['estimated_complete_gpu_hours'] == 0,
                'empty gradeable frame must not reserve reader GPUs')
        return
    require(price['shards'] and price['shards'][0]['start'] == 0 and
            price['shards'][-1]['end'] == len(rows) and
            all(a['end'] == b['start'] for a, b in zip(price['shards'], price['shards'][1:])) and
            all(0 <= s['start'] < s['end'] <= len(rows) and
                s['start'] % BATCH == 0 and
                (s['end'] == len(rows) or s['end'] % BATCH == 0)
                for s in price['shards']), 'reader shards do not cover exact frame')


def attempts(out, start, reader, block, binding_sha):
    ledger = []
    for path in sorted((out / 'attempts').glob(f'{start:06d}-reader{reader}-attempt*.json')):
        item = base.sealed(path)
        require(item['schema'] == 'overnight-semantic-reader-attempt-v1' and
                item['binding_sha256'] == binding_sha and
                item['start'] == start and item['reader'] == reader and
                item['attempt_index'] == len(ledger) and
                item['recovery_of_uncommitted_attempts'] == [x['sha256'] for x in ledger] and
                item['blind_ids'] == [r['blind_id'] for r in block] and
                item['seeds'] == [rating_seed(r['blind_id'], reader) for r in block] and
                path.name == f'{start:06d}-reader{reader}-attempt{len(ledger):03d}.json',
                'reader attempt ledger differs')
        ledger.append(item)
    return ledger


def committed(out, start, reader, block, binding_sha):
    path = out / 'batches' / f'{start:06d}-reader{reader}.json'
    ledger = attempts(out, start, reader, block, binding_sha)
    if not path.exists():
        return None
    item = base.sealed(path)
    index = item['attempt_index']
    require(item['schema'] == 'overnight-semantic-reader-batch-v1' and
            item['binding_sha256'] == binding_sha and
            item['start'] == start and item['reader'] == reader and
            type(index) is int and 0 <= index < len(ledger) and
            item['attempt_sha256'] == ledger[index]['sha256'] and
            [r['blind_id'] for r in item['records']] == [r['blind_id'] for r in block],
            'reader batch differs from committed assignment')
    for row in block:
        receipt = base.sealed(out / 'assignments' /
                              f"{row['blind_id']}-reader{reader}-attempt{index:03d}.json")
        require(receipt['binding_sha256'] == binding_sha and
                receipt['blind_id'] == row['blind_id'] and
                receipt['reader'] == reader and
                receipt['attempt_sha256'] == ledger[index]['sha256'],
                'per-row reader attempt receipt differs')
    return item


def shard_batches(frame, price, shard_index):
    shard = price['shards'][shard_index]
    return [(start, frame['records'][start:min(start + BATCH, shard['end'])])
            for start in range(shard['start'], shard['end'], BATCH)]


def shard_summary(out, frame, price, binding, shard_index):
    batches = []
    for start, block in shard_batches(frame, price, shard_index):
        for reader in (0, 1):
            item = committed(out, start, reader, block, binding['sha256'])
            if item is None:
                return None
            batches.append(item)
    records = [row for item in batches for row in item['records']]
    body = {'schema': 'overnight-semantic-reader-shard-summary-v1',
            'binding_sha256': binding['sha256'], 'shard_index': shard_index,
            'batch_sha256s': [item['sha256'] for item in batches],
            'rows': price['shards'][shard_index]['end'] - price['shards'][shard_index]['start'],
            'ratings': len(records),
            'parsed_stop': sum(r['rating'] is not None and r['finish_reason'] == 'stop'
                               for r in records),
            'generated_tokens': sum(r['generated_tokens'] for r in records),
            'status': 'COMPLETE_ARM_BLIND_LLM_AUDIT'}
    return save(out / 'SUMMARY.json', body, existing_ok=True)


def run_shard(frame, price, out, shard_index, resume_uncommitted, deadline_epoch):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            not socket.gethostname().startswith('login'),
            'semantic inference requires GPU Slurm step')
    require(0 <= shard_index < len(price['shards']), 'invalid priced shard')
    out = out / f'shard-{shard_index:03d}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('attempts', 'assignments', 'batches', 'failures'):
        (out / name).mkdir(exist_ok=True)
    binding = save(out / 'BINDING.json', {
        'schema': 'overnight-semantic-reader-binding-v1',
        'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
        'driver_sha256': base.file_sha(__file__),
        'rubric_sha256': base.file_sha(RUBRIC),
        'shard_index': shard_index, 'model_snapshot': str(MODEL),
        'batch_size': BATCH, 'readers': 2, 'max_tokens_per_rating': MAX_TOKENS,
        'sampler': {'temperature': .2, 'top_p': .95, 'thinking': True,
                    'reasoning_effort': 'low'},
        'reader_input_allowlist': frame['reader_input_allowlist']}, existing_ok=True)
    pending = []
    for start, block in shard_batches(frame, price, shard_index):
        for reader in (0, 1):
            if committed(out, start, reader, block, binding['sha256']) is None:
                ledger = attempts(out, start, reader, block, binding['sha256'])
                require(not ledger or resume_uncommitted,
                        'uncommitted reader attempt needs explicit recovery flag')
                pending.append((start, reader, block, ledger))
    if not pending:
        summary = shard_summary(out, frame, price, binding, shard_index)
        print(json.dumps({'status': 'COMPLETE_ARM_BLIND_LLM_AUDIT',
                          'summary_sha256': summary['sha256']}), flush=True)
        return
    if deadline_epoch and time.time() + 1200 >= deadline_epoch:
        print(json.dumps({'status': 'PARTIAL_CHECKPOINTED_BEFORE_MODEL_LOAD',
                          'pending_reader_batches': len(pending)}), flush=True)
        return
    from vllm import LLM, SamplingParams
    model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=MAX_MODEL_LEN,
                max_num_seqs=16, max_num_batched_tokens=8192,
                gpu_memory_utilization=.85, enforce_eager=True,
                generation_config='vllm', language_model_only=True,
                attention_config={'backend': 'FLASH_ATTN'})
    for start, reader, block, ledger in pending:
        if deadline_epoch and time.time() + 600 >= deadline_epoch:
            print(json.dumps({'status': 'PARTIAL_CHECKPOINTED',
                              'next_start': start, 'next_reader': reader}), flush=True)
            break
        index = len(ledger)
        attempt = save(out / 'attempts' /
                       f'{start:06d}-reader{reader}-attempt{index:03d}.json', {
            'schema': 'overnight-semantic-reader-attempt-v1',
            'binding_sha256': binding['sha256'], 'start': start, 'reader': reader,
            'attempt_index': index,
            'recovery_of_uncommitted_attempts': [a['sha256'] for a in ledger],
            'blind_ids': [r['blind_id'] for r in block],
            'seeds': [rating_seed(r['blind_id'], reader) for r in block],
            'job_id': os.environ['SLURM_JOB_ID'], 'state': 'started_before_model_chat'})
        for row in block:
            save(out / 'assignments' /
                 f"{row['blind_id']}-reader{reader}-attempt{index:03d}.json", {
                'schema': 'overnight-semantic-reader-assignment-v1',
                'binding_sha256': binding['sha256'],
                'blind_id': row['blind_id'], 'reader': reader,
                'attempt_index': index, 'attempt_sha256': attempt['sha256'],
                'job_id': os.environ['SLURM_JOB_ID']})
        params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                 seed=rating_seed(r['blind_id'], reader)) for r in block]
        started = time.monotonic()
        try:
            outputs = model.chat([messages(row) for row in block],
                                 sampling_params=params,
                                 chat_template_kwargs={'enable_thinking': True,
                                                       'reasoning_effort': 'low'},
                                 use_tqdm=False)
        except Exception as exc:
            save(out / 'failures' /
                 f'{start:06d}-reader{reader}-attempt{index:03d}-failure.json', {
                'schema': 'overnight-semantic-reader-failure-v1',
                'binding_sha256': binding['sha256'],
                'attempt_sha256': attempt['sha256'],
                'job_id': os.environ['SLURM_JOB_ID'],
                'exception_type': type(exc).__name__,
                'exception_message': str(exc)[:500]})
            raise
        require(len(outputs) == len(block), 'reader output count differs')
        records = []
        for position, (row, output) in enumerate(zip(block, outputs, strict=True)):
            require(len(output.prompt_token_ids) ==
                    price['prompt_token_lengths'][start + position],
                    'reader chat-template prompt length differs from exact price')
            completion = output.outputs[0]
            records.append({'blind_id': row['blind_id'],
                            'rating': parse_rating(completion.text),
                            'finish_reason': completion.finish_reason,
                            'prompt_tokens': len(output.prompt_token_ids),
                            'generated_tokens': len(completion.token_ids),
                            'raw_completion': completion.text})
        save(out / 'batches' / f'{start:06d}-reader{reader}.json', {
            'schema': 'overnight-semantic-reader-batch-v1',
            'binding_sha256': binding['sha256'], 'start': start, 'reader': reader,
            'attempt_index': index, 'attempt_sha256': attempt['sha256'],
            'elapsed_seconds': time.monotonic() - started, 'records': records})
        print(json.dumps({'start': start, 'reader': reader,
                          'rated': len(records)}), flush=True)
    summary = shard_summary(out, frame, price, binding, shard_index)
    if summary:
        print(json.dumps({'status': summary['status'],
                          'summary_sha256': summary['sha256']}), flush=True)


def seal_stage(frame, price, out):
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, shard in enumerate(price['shards']):
        directory = out / f'shard-{index:03d}'
        binding = base.sealed(directory / 'BINDING.json')
        require(binding['schema'] == 'overnight-semantic-reader-binding-v1' and
                binding['frame_sha256'] == frame['sha256'] and
                binding['price_sha256'] == price['sha256'] and
                binding['driver_sha256'] == base.file_sha(__file__) and
                binding['shard_index'] == index, 'reader shard binding differs')
        summary = shard_summary(directory, frame, price, binding, index)
        require(summary is not None and summary['ratings'] == 2 * (shard['end'] - shard['start']),
                'reader shard has incomplete assignment coverage')
        summaries.append(summary)
    body = {'schema': 'overnight-semantic-reader-stage-summary-v1',
            'frame_sha256': frame['sha256'], 'price_sha256': price['sha256'],
            'shard_summary_sha256s': [s['sha256'] for s in summaries],
            'rows': len(frame['records']), 'ratings': 2 * len(frame['records']),
            'parsed_stop': sum(s['parsed_stop'] for s in summaries),
            'status': 'COMPLETE_ARM_BLIND_LLM_AUDIT'}
    return save(out / 'STAGE_SUMMARY.json', body, existing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--price', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--seal-stage', action='store_true')
    parser.add_argument('--resume-uncommitted', action='store_true')
    parser.add_argument('--deadline-epoch', type=float, default=0)
    args = parser.parse_args()
    frame, price = base.sealed(args.frame), base.sealed(args.price)
    validate(frame, price)
    if args.seal_stage:
        require(args.shard_index is None, 'seal-stage takes no shard index')
        summary = seal_stage(frame, price, args.out)
        print(json.dumps({'status': summary['status'],
                          'summary_sha256': summary['sha256']}), flush=True)
    else:
        require(args.shard_index is not None, 'priced shard index required')
        run_shard(frame, price, args.out, args.shard_index,
                  args.resume_uncommitted, args.deadline_epoch)


if __name__ == '__main__':
    main()
