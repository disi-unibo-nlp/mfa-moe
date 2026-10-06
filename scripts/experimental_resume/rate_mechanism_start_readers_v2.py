"""Arm-blind two-draw Qwen3.8 audit with explicit crash-attempt recovery."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time

from prepare_mechanism_start_frame_v1 import FRAME, REPO, SELECTION, digest, file_sha, sealed, write_once
import rate_transition_v22_fullprefix_starts_v2 as prior_reader

PRICE = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v2.json'
BATCH = 16
MAX_TOKENS = 1024


def rating_seed(uid, reader):
    return int(digest(['mechanism-start-rating-v1', uid, reader])[:8], 16) % 2_000_000_000


def row_result(output):
    one = output.outputs[0]
    return {'rating': prior_reader.parse_rating(one.text),
            'finish_reason': one.finish_reason,
            'generated_tokens': len(one.token_ids),
            'raw_completion': one.text}


def validate_inputs(frame, selection, price):
    if (frame['schema'] != 'mechanism-start-frame-v1' or
            frame['selection_sha256'] != selection['sha256'] or
            frame['families_in_frozen_pool'] != 128 or
            frame['rows'] != len(selection['records']) or
            [r['uid'] for r in frame['records']] != [r['uid'] for r in selection['records']] or
            price['schema'] != 'mechanism-start-reader-price-v2' or
            price['status'] != 'PASS_COMPLETE_20_GPUH' or
            price['frame_sha256'] != frame['sha256'] or
            price['selection_sha256'] != selection['sha256'] or
            price['rating_driver_sha256'] != file_sha(__file__) or
            price['message_source_sha256'] != file_sha(prior_reader.__file__) or
            price['rubric_sha256'] != file_sha(prior_reader.RUBRIC) or
            price['rows'] != frame['rows'] or price['ratings'] != 2 * frame['rows']):
        raise ValueError('start rating not bound to complete sealed stage')
    if len({r['uid'] for r in frame['records']}) != frame['rows']:
        raise ValueError('duplicate mechanism start UID')
    for row in frame['records']:
        prior_reader.messages(row)  # exact visible-input allowlist and complete trigger


def save(path, body, *, existing_ok=False):
    """Commit one sealed file atomically; preserve orphan .partial files."""
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if existing_ok and sealed(path) == value:
            return value
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}-{time.monotonic_ns()}')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return value


def attempt_ledger(out, start, reader, block, binding_sha):
    """Read every prior attempt and select the next index without erasing one."""
    paths = sorted((out / 'attempts').glob(f'{start:06d}-reader{reader}-attempt*.json'))
    attempts = []
    for path in paths:
        value = sealed(path)
        if (value['schema'] != 'mechanism-start-reader-attempt-v2' or
                value['binding_sha256'] != binding_sha or
                value['start'] != start or value['reader'] != reader or
                value['recovery_of_uncommitted_attempts'] !=
                [a['sha256'] for a in attempts] or
                value['uids'] != [r['uid'] for r in block] or
                value['seeds'] != [rating_seed(r['uid'], reader) for r in block] or
                path.name != f"{start:06d}-reader{reader}-attempt{value['attempt_index']:03d}.json"):
            raise ValueError('changed or foreign reader attempt ledger')
        attempts.append(value)
    if [a['attempt_index'] for a in attempts] != list(range(len(attempts))):
        raise ValueError('reader attempt indices are not contiguous')
    return attempts


def committed_reader(out, start, reader, block, binding_sha):
    path = out / 'batches' / f'{start:06d}-reader{reader}.json'
    if not path.exists():
        attempt_ledger(out, start, reader, block, binding_sha)
        return None
    saved = sealed(path)
    attempts = attempt_ledger(out, start, reader, block, binding_sha)
    index = saved.get('attempt_index')
    if (saved['schema'] != 'mechanism-start-reader-batch-v2' or
            saved['binding_sha256'] != binding_sha or saved['start'] != start or
            saved['reader'] != reader or
            [r['uid'] for r in saved['records']] != [r['uid'] for r in block] or
            type(index) is not int or index < 0 or index >= len(attempts)):
        raise ValueError('reader result rebound to another assignment or attempt')
    if saved['attempt_sha256'] != attempts[index]['sha256']:
        raise ValueError('committed reader result does not cite its attempt')
    for row in block:
        receipt = out / 'assignments' / f"{row['uid']}-reader{reader}-attempt{index:03d}.json"
        entry = sealed(receipt)
        if (entry['binding_sha256'] != binding_sha or entry['uid'] != row['uid'] or
                entry['reader'] != reader or entry['attempt_index'] != index or
                entry['attempt_sha256'] != attempts[index]['sha256']):
            raise ValueError('committed reader lacks exact per-UID attempt receipt')
    return saved


def run(out):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('mechanism reader inference requires GPU Slurm step')
    frame, selection, price = sealed(FRAME), sealed(SELECTION), sealed(PRICE)
    validate_inputs(frame, selection, price)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('batches', 'assignments', 'attempts', 'loads'):
        (out / name).mkdir(exist_ok=True)
    binding_body = {
        'schema': 'mechanism-start-reader-binding-v2',
        'frame_sha256': frame['sha256'], 'selection_sha256': selection['sha256'],
        'price_sha256': price['sha256'], 'driver_sha256': file_sha(__file__),
        'message_source_sha256': file_sha(prior_reader.__file__),
        'rubric_sha256': file_sha(prior_reader.RUBRIC),
        'model_snapshot': str(prior_reader.MODEL), 'model_revision': prior_reader.MODEL.name,
        'batch_size': BATCH, 'max_tokens_per_rating': MAX_TOKENS, 'readers': 2,
        'sampler': {'temperature': .2, 'top_p': .95, 'thinking': True,
                    'reasoning_effort': 'low'},
        'visible_input_allowlist': frame['visible_input_allowlist'],
    }
    binding = save(out / 'BINDING.json', binding_body, existing_ok=True)
    rows = frame['records']
    pending = []
    for start in range(0, len(rows), BATCH):
        path = out / 'batches' / f'{start:06d}.json'
        if path.exists():
            saved = sealed(path)
            if (saved['binding_sha256'] != binding['sha256'] or
                    [r['uid'] for r in saved['records']] !=
                    [r['uid'] for r in rows[start:start+BATCH]]):
                raise ValueError('completed batch rebound to another assignment')
            block = rows[start:start+BATCH]
            readers = [committed_reader(out, start, reader, block, binding['sha256'])
                       for reader in (0, 1)]
            if (any(r is None for r in readers) or
                    saved['reader_timings'] != [r['timing'] for r in readers] or
                    saved['records'] != [
                        {'uid': row['uid'], 'transition': row['transition'],
                         'readers': [readers[0]['records'][i]['result'],
                                     readers[1]['records'][i]['result']]}
                        for i, row in enumerate(block)]):
                raise ValueError('completed batch differs from committed reader results')
        else:
            pending.append((start, path))
    needs_model = any(committed_reader(out, start, reader, rows[start:start+BATCH],
                                       binding['sha256']) is None
                      for start, _ in pending for reader in (0, 1))
    model = None
    if needs_model:
        from vllm import LLM
        load_start = time.monotonic()
        model = LLM(model=str(prior_reader.MODEL), tokenizer=str(prior_reader.MODEL),
                    tensor_parallel_size=2, dtype='bfloat16', kv_cache_dtype='bfloat16',
                    max_model_len=49152, max_num_seqs=32, max_num_batched_tokens=8192,
                    gpu_memory_utilization=.85, enforce_eager=True,
                    generation_config='vllm', language_model_only=True,
                    attention_config={'backend': 'FLASH_ATTN'})
        save(out / 'loads' / f"{os.environ['SLURM_JOB_ID']}.json",
             {'schema': 'mechanism-start-reader-load-v2',
              'binding_sha256': binding['sha256'],
              'job_id': os.environ['SLURM_JOB_ID'],
              'load_seconds': time.monotonic() - load_start})
    for start, batch_path in pending:
        block = rows[start:start+BATCH]
        both, timings = [], []
        for reader in (0, 1):
            saved = committed_reader(out, start, reader, block, binding['sha256'])
            if saved is None:
                from vllm import SamplingParams
                attempts = attempt_ledger(out, start, reader, block, binding['sha256'])
                index = len(attempts)
                attempt = save(out / 'attempts' /
                               f'{start:06d}-reader{reader}-attempt{index:03d}.json',
                               {'schema': 'mechanism-start-reader-attempt-v2',
                                'binding_sha256': binding['sha256'],
                                'start': start, 'reader': reader,
                                'attempt_index': index,
                                'recovery_of_uncommitted_attempts': [a['sha256'] for a in attempts],
                                'uids': [r['uid'] for r in block],
                                'seeds': [rating_seed(r['uid'], reader) for r in block],
                                'job_id': os.environ['SLURM_JOB_ID'],
                                'state': 'started_before_model_chat'})
                for row in block:
                    save(out / 'assignments' / f"{row['uid']}-reader{reader}-attempt{index:03d}.json",
                         {'schema': 'mechanism-start-rating-assignment-v2',
                          'binding_sha256': binding['sha256'], 'uid': row['uid'],
                          'reader': reader, 'start': start, 'attempt_index': index,
                          'attempt_sha256': attempt['sha256'],
                          'state': 'attempted_before_model_chat',
                          'job_id': os.environ['SLURM_JOB_ID']})
                params = [SamplingParams(temperature=.2, top_p=.95, max_tokens=MAX_TOKENS,
                                         seed=rating_seed(row['uid'], reader)) for row in block]
                t0 = time.monotonic()
                outputs = model.chat([prior_reader.messages(row) for row in block],
                                     sampling_params=params,
                                     chat_template_kwargs={'enable_thinking': True,
                                                           'reasoning_effort': 'low'},
                                     use_tqdm=False)
                if len(outputs) != len(block):
                    raise ValueError('reader model output count differs')
                timing = {'reader': reader, 'job_id': os.environ['SLURM_JOB_ID'],
                          'wall_seconds': time.monotonic() - t0,
                          'prompt_tokens': sum(len(o.prompt_token_ids) for o in outputs),
                          'generated_tokens': sum(len(o.outputs[0].token_ids) for o in outputs)}
                path = out / 'batches' / f'{start:06d}-reader{reader}.json'
                saved = save(path, {'schema': 'mechanism-start-reader-batch-v2',
                                    'binding_sha256': binding['sha256'],
                                    'start': start, 'reader': reader,
                                    'attempt_index': index,
                                    'attempt_sha256': attempt['sha256'],
                                    'timing': timing,
                                    'records': [{'uid': row['uid'], 'result': row_result(outputs[i])}
                                                for i, row in enumerate(block)]})
            committed_reader(out, start, reader, block, binding['sha256'])
            both.append([r['result'] for r in saved['records']])
            timings.append(saved['timing'])
        save(batch_path, {'schema': 'mechanism-start-rating-batch-v2',
                          'binding_sha256': binding['sha256'], 'start': start,
                          'reader_timings': timings,
                          'records': [{'uid': row['uid'], 'transition': row['transition'],
                                       'readers': [both[0][i], both[1][i]]}
                                      for i, row in enumerate(block)]})
        print(json.dumps({'complete': min(start+BATCH, len(rows)), 'of': len(rows)}), flush=True)
    completed, timings = [], []
    for start in range(0, len(rows), BATCH):
        saved = sealed(out / 'batches' / f'{start:06d}.json')
        completed += saved['records']
        timings += saved['reader_timings']
    if [r['uid'] for r in completed] != [r['uid'] for r in rows]:
        raise ValueError('incomplete reader assignment coverage')
    counts = Counter(rows=len(completed), ratings=2 * len(completed))
    for row in completed:
        readers = row['readers']
        counts['generated_tokens'] += sum(r['generated_tokens'] for r in readers)
        counts['parsed_stop'] += sum(r['rating'] is not None and r['finish_reason'] == 'stop'
                                     for r in readers)
        counts['pair_covered'] += all(r['rating'] is not None and r['finish_reason'] == 'stop'
                                      for r in readers)
        counts['start_agree_true'] += all(r['rating'] == {'start': True} and
                                           r['finish_reason'] == 'stop' for r in readers)
    committed_attempts = set()
    all_attempts = []
    for start in range(0, len(rows), BATCH):
        block = rows[start:start+BATCH]
        for reader in (0, 1):
            saved = committed_reader(out, start, reader, block, binding['sha256'])
            attempts = attempt_ledger(out, start, reader, block, binding['sha256'])
            committed_attempts.add(saved['attempt_sha256'])
            all_attempts.extend(attempts)
    counts['physical_attempts_recorded'] = len(all_attempts)
    counts['uncommitted_attempts_recorded'] = sum(
        a['sha256'] not in committed_attempts for a in all_attempts)
    summary = save(out / 'SUMMARY.json',
                         {'schema': 'mechanism-start-reader-summary-v2',
                          'binding_sha256': binding['sha256'],
                          'counts': dict(counts), 'reader_timings': timings,
                          'attempt_sha256s': [a['sha256'] for a in all_attempts],
                          'scope': 'All frozen start assignments, two same-model Qwen3.8 draws. One committed result per UID/reader. Uncommitted attempts may have consumed compute; exact physical inference once-only is not claimed. LLM audit, not human truth or causal control.'},
                         existing_ok=True)
    print(json.dumps({'summary': str(out / 'SUMMARY.json'),
                      'sha256': summary['sha256'], 'counts': summary['counts']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
