"""Rate all eligible same-prefix continuations with two arm-blind Qwen readers.

The input is only the sealed BLIND_FRAME.json.  Assignment identities and routing
results are deliberately absent.  A separate analysis joins ratings to arms.
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
STAGE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp')
RUBRIC = REPO / 'report/experimental-resume-v1/CAUSAL_ELIGIBLE_IMMEDIATE_SEMANTIC_RUBRIC_v1.md'
MODEL = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf/hub/'
             'models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
ALLOWLIST = frozenset({'transition', 'problem', 'full_emitted_prefix',
                       'triggering_sentence', 'continuation'})
TRANSITIONS = frozenset({'candidate_to_verify', 'approach_to_commit'})
BATCH = 12
READERS = (0, 1)
MAX_TOKENS = 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('invalid sealed JSON: ' + str(path))
    return value


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    part.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(part, path)


def write_once(path, body):
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError('bound output differs: ' + str(path))
    else:
        atomic_json(path, value)
    return value


def validate(frame, price, frame_path):
    rows = frame.get('records', [])
    if (frame.get('schema') != 'eligible-immediate-blind-frame-v1' or
            not 1 <= len(rows) <= 156 or
            len({r['blind_id'] for r in rows}) != len(rows) or
            set(frame.get('reader_input_allowlist', [])) != ALLOWLIST or
            frame.get('rubric_sha256') != file_sha(RUBRIC) or
            frame.get('continuation_max_tokens') != 256):
        raise ValueError('eligible blind frame or rubric differs')
    for row in rows:
        if set(row) != {'blind_id', 'reader_input'} or not isinstance(row['blind_id'], str):
            raise ValueError('unknown blind rating row')
        data = row['reader_input']
        if set(data) != ALLOWLIST or data['transition'] not in TRANSITIONS or any(
                not isinstance(value, str) for value in data.values()):
            raise ValueError('reader input exceeds frozen allowlist')
        if not data['full_emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()):
            raise ValueError('triggering sentence does not end the native prefix')
    if (price.get('schema') != 'eligible-immediate-blind-rating-price-v1' or
            price.get('status') != 'PASS_COMPLETE_STAGE' or
            price.get('frame_sha256') != frame['sha256'] or
            price.get('frame_file_sha256') != file_sha(frame_path) or
            price.get('driver_sha256') != file_sha(__file__) or
            price.get('rubric_sha256') != file_sha(RUBRIC) or
            price.get('assigned_requests') != 156 or
            price.get('gradeable_requests') != len(rows) or
            price.get('ratings') != len(rows) * len(READERS) or
            price.get('max_decode_tokens') != len(rows) * len(READERS) * MAX_TOKENS):
        raise ValueError('blind-rating complete-stage price differs')
    return rows


def messages(row):
    data = row['reader_input']
    if data['transition'] == 'candidate_to_verify':
        rule = ('True only if new reasoning substantively tests the same previously '
                'unchecked candidate or value against an original constraint or '
                'independent calculation. Verification words, repetition, unrelated '
                'arithmetic, and a final answer alone are false.')
    else:
        rule = ('True only if new reasoning commits to the identified approach with '
                'a concrete planning or implementation step that advances the '
                'original problem. Restating the method, generic planning language, '
                'and switching to an unrelated approach are false. If the approach '
                'is not identifiable from the prefix, return false.')
    system = ('Rate only the new reasoning continuation before reasoning closure. '
              'The triggering sentence cannot satisfy the outcome. Closure without '
              'the target and ambiguity are false. ' + rule +
              ' Return exactly {"target": true} or {"target": false}, no explanation.')
    user = ('Original problem:\n' + data['problem'] +
            '\n\nAlready emitted native reasoning prefix:\n' + data['full_emitted_prefix'] +
            '\n\nTriggering sentence (context only):\n' + data['triggering_sentence'] +
            '\n\nOne assigned new reasoning continuation:\n' + data['continuation'] +
            '\n\nJSON:')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def parse_rating(raw):
    # Qwen thinking is outside the requested JSON.  A missing closing marker
    # leaves no valid visible answer and is conservatively unscored.
    if '</think>' in raw:
        raw = raw.rsplit('</think>', 1)[1]
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {'target'} or type(value['target']) is not bool:
        return None
    return value['target']


def rating_seed(blind_id, reader):
    return int(digest(['eligible-immediate-semantic-v1', blind_id, reader])[:8], 16) % 2_000_000_000


def attempt_accounting(attempt_dir, binding_sha256, batches):
    """Expose failed/interrupted judge work separately from observed successful tokens."""
    attempts = [sealed(path) for path in sorted(attempt_dir.glob('*-attempt[0-9][0-9][0-9].json'))]
    failures = [sealed(path) for path in sorted(attempt_dir.glob('*-failure.json'))]
    if any(row['binding_sha256'] != binding_sha256 for row in attempts + failures):
        raise ValueError('rating attempt belongs to another binding')
    attempt_by_hash = {row['sha256']: row for row in attempts}
    attempt_hashes = set(attempt_by_hash)
    successful = {batch['attempt_receipt_sha256'] for batch in batches}
    failed = {row['attempt_receipt_sha256'] for row in failures}
    if (not successful <= attempt_hashes or not failed <= attempt_hashes or
            successful & failed or len(failed) != len(failures)):
        raise ValueError('rating attempt receipt accounting is inconsistent')
    unresolved = attempt_hashes - successful - failed
    return {'attempts': len(attempts), 'successful_attempts': len(successful),
            'failed_attempts': len(failed), 'interrupted_attempts': len(unresolved),
            'known_failed_attempt_seconds': sum(row['elapsed_seconds'] for row in failures),
            'unknown_decode_token_upper_bound':
                sum(row['max_decode_tokens_at_risk'] for row in failures) +
                sum(len(attempt_by_hash[key]['blind_ids']) * MAX_TOKENS for key in unresolved),
            'cost_caveat': 'Failed/interrupted attempts have unknown actual token counts; '
                           'their upper bound and actual Slurm allocation time are reported separately.'}


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('eligible semantic inference requires GPU Slurm')
    if not args.frame.resolve().is_relative_to(STAGE) or not args.out.resolve().is_relative_to(STAGE):
        raise ValueError('frame and output must be in owned staging directory')
    frame, price = sealed(args.frame), sealed(args.price)
    rows = validate(frame, price, args.frame)
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_ELIGIBLE_RATING_PREFLIGHT',
                          'frame_sha256': frame['sha256'], 'ratings': len(rows) * len(READERS)}))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_body = {'schema': 'eligible-immediate-blind-rating-binding-v1',
                    'frame_sha256': frame['sha256'], 'frame_file_sha256': file_sha(args.frame),
                    'price_sha256': price['sha256'], 'driver_sha256': file_sha(__file__),
                    'rubric_sha256': file_sha(RUBRIC), 'model': str(MODEL),
                    'batch_size': BATCH, 'readers': len(READERS),
                    'max_tokens_per_rating': MAX_TOKENS,
                    'sampler': {'temperature': .2, 'top_p': .95, 'enable_thinking': True,
                                'reasoning_effort': 'low'}}
    binding = write_once(args.out / 'BINDING.json', binding_body)
    (args.out / 'batches').mkdir(exist_ok=True)
    (args.out / 'attempts').mkdir(exist_ok=True)
    planned = []
    for start in range(0, len(rows), BATCH):
        block = rows[start:start + BATCH]
        ids = [r['blind_id'] for r in block]
        for reader in READERS:
            path = args.out / 'batches' / f'target-{start:03d}-reader{reader}.json'
            if path.exists():
                saved = sealed(path)
                if (saved['binding_sha256'] != binding['sha256'] or
                        [r['blind_id'] for r in saved['records']] != ids):
                    raise ValueError('same-bound completed rating batch differs')
            else:
                planned.append((start, reader, block, path))
    summary_path = args.out / 'SUMMARY.json'
    if summary_path.exists():
        summary = sealed(summary_path)
        if (planned or summary['binding_sha256'] != binding['sha256'] or
                summary['ratings'] != len(rows) * len(READERS)):
            raise ValueError('same-bound rating summary differs')
        print(json.dumps({'status': 'SAME_BOUND_COMPLETE', 'summary': str(summary_path)}))
        return
    from vllm import LLM, SamplingParams
    t0 = time.monotonic()
    model = LLM(model=str(MODEL), tokenizer=str(MODEL), tensor_parallel_size=2,
                dtype='bfloat16', kv_cache_dtype='bfloat16', max_model_len=49152,
                max_num_seqs=16, max_num_batched_tokens=8192,
                gpu_memory_utilization=.85, enforce_eager=True,
                generation_config='vllm', language_model_only=True,
                attention_config={'backend': 'FLASH_ATTN'})
    load_seconds = time.monotonic() - t0
    for start, reader, block, path in planned:
        if args.deadline_epoch and time.time() + 440 >= args.deadline_epoch:
            print(json.dumps({'status': 'CHECKPOINTED_BEFORE_NEXT_BATCH',
                              'start': start, 'reader': reader}), flush=True)
            break
        stem = f'target-{start:03d}-reader{reader}'
        previous = sorted((args.out / 'attempts').glob(stem + '-attempt*.json'))
        attempt = len(previous)
        receipt = args.out / 'attempts' / f'{stem}-attempt{attempt:03d}.json'
        write_once(receipt, {'schema': 'eligible-immediate-rating-attempt-v1',
                             'binding_sha256': binding['sha256'],
                             'job_id': os.environ['SLURM_JOB_ID'], 'start': start,
                             'reader': reader, 'blind_ids': [r['blind_id'] for r in block],
                             'status': 'attempted_before_generation'})
        params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                 seed=rating_seed(row['blind_id'], reader)) for row in block]
        begin = time.monotonic()
        try:
            outputs = model.chat([messages(row) for row in block], sampling_params=params,
                                 chat_template_kwargs={'enable_thinking': True,
                                                       'reasoning_effort': 'low'},
                                 use_tqdm=False)
            if len(outputs) != len(block):
                raise ValueError('rating output count differs from assignment')
            records = []
            for row, output in zip(block, outputs, strict=True):
                completion = output.outputs[0]
                records.append({'blind_id': row['blind_id'],
                                'rating': parse_rating(completion.text),
                                'finish_reason': completion.finish_reason,
                                'prompt_tokens': len(output.prompt_token_ids),
                                'generated_tokens': len(completion.token_ids),
                                'raw_completion': completion.text})
            write_once(path, {'schema': 'eligible-immediate-rating-batch-v1',
                              'binding_sha256': binding['sha256'], 'start': start,
                              'reader': reader, 'job_id': os.environ['SLURM_JOB_ID'],
                              'attempt_receipt_sha256': sealed(receipt)['sha256'],
                              'elapsed_seconds': time.monotonic() - begin,
                              'records': records})
        except Exception as error:
            write_once(args.out / 'attempts' / f'{stem}-attempt{attempt:03d}-failure.json',
                       {'schema': 'eligible-immediate-rating-failure-v1',
                        'binding_sha256': binding['sha256'],
                        'attempt_receipt_sha256': sealed(receipt)['sha256'],
                        'job_id': os.environ['SLURM_JOB_ID'], 'start': start,
                        'reader': reader, 'elapsed_seconds': time.monotonic() - begin,
                        'max_decode_tokens_at_risk': len(block) * MAX_TOKENS,
                        'exception_type': type(error).__name__,
                        'exception_message': str(error)[:500]})
            raise
        print(json.dumps({'start': start, 'reader': reader,
                          'valid': sum(r['rating'] is not None and
                                       r['finish_reason'] == 'stop' for r in records)}), flush=True)
    paths = [args.out / 'batches' / f'target-{start:03d}-reader{reader}.json'
             for start in range(0, len(rows), BATCH) for reader in READERS]
    if any(not path.exists() for path in paths):
        print(json.dumps({'status': 'PARTIAL_CHECKPOINTED',
                          'completed_batches': sum(path.exists() for path in paths),
                          'total_batches': len(paths)}), flush=True)
        return
    batches = [sealed(path) for path in paths]
    attempts = attempt_accounting(args.out / 'attempts', binding['sha256'], batches)
    records = [row for batch in batches for row in batch['records']]
    if len(records) != len(rows) * len(READERS):
        raise ValueError('rating count differs')
    summary = write_once(summary_path, {'schema': 'eligible-immediate-rating-summary-v1',
                                        'binding_sha256': binding['sha256'],
                                        'frame_sha256': frame['sha256'],
                                        'batch_sha256': [batch['sha256'] for batch in batches],
                                        'ratings': len(records), 'rows': len(rows),
                                        'valid_stopped_ratings': sum(r['rating'] is not None and
                                                                     r['finish_reason'] == 'stop'
                                                                     for r in records),
                                        'prompt_tokens': sum(r['prompt_tokens'] for r in records),
                                        'generated_tokens': sum(r['generated_tokens'] for r in records),
                                        'attempt_accounting': attempts,
                                        'load_seconds_this_job': load_seconds,
                                        'status': 'COMPLETE_ARM_BLIND_LLM_AUDIT'})
    print(json.dumps({'status': 'COMPLETE', 'summary': str(summary_path),
                      'sha256': summary['sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--price', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, default=0.)
    parser.add_argument('--cpu-preflight', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
