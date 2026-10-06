"""Arm-blind two-draw Qwen3.8 audit of frozen mechanism starting conditions."""
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

PRICE = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v1.json'
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
            price['schema'] != 'mechanism-start-reader-price-v1' or
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


def save(path, body):
    if Path(path).exists():
        raise FileExistsError(path)
    return write_once(path, body)


def run(out):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('mechanism reader inference requires GPU Slurm step')
    frame, selection, price = sealed(FRAME), sealed(SELECTION), sealed(PRICE)
    validate_inputs(frame, selection, price)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'WRITER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('batches', 'assignments', 'loads'):
        (out / name).mkdir(exist_ok=True)
    binding_body = {
        'schema': 'mechanism-start-reader-binding-v1',
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
    binding = write_once(out / 'BINDING.json', binding_body)
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
        else:
            pending.append((start, path))
    needs_model = any(not (out / 'batches' / f'{start:06d}-reader{reader}.json').exists()
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
             {'schema': 'mechanism-start-reader-load-v1',
              'binding_sha256': binding['sha256'],
              'job_id': os.environ['SLURM_JOB_ID'],
              'load_seconds': time.monotonic() - load_start})
    for start, batch_path in pending:
        block = rows[start:start+BATCH]
        both, timings = [], []
        for reader in (0, 1):
            path = out / 'batches' / f'{start:06d}-reader{reader}.json'
            if path.exists():
                saved = sealed(path)
                if (saved['binding_sha256'] != binding['sha256'] or
                        saved['start'] != start or saved['reader'] != reader or
                        [r['uid'] for r in saved['records']] != [r['uid'] for r in block]):
                    raise ValueError('reader batch rebound to another assignment')
            else:
                from vllm import SamplingParams
                for row in block:
                    receipt = out / 'assignments' / f"{row['uid']}-reader{reader}.json"
                    if receipt.exists():
                        raise RuntimeError('incomplete attempted rating needs manual adjudication: ' + str(receipt))
                for row in block:
                    save(out / 'assignments' / f"{row['uid']}-reader{reader}.json",
                         {'schema': 'mechanism-start-rating-assignment-v1',
                          'binding_sha256': binding['sha256'], 'uid': row['uid'],
                          'reader': reader, 'start': start,
                          'state': 'attempted_before_generation',
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
                saved = save(path, {'schema': 'mechanism-start-reader-batch-v1',
                                    'binding_sha256': binding['sha256'],
                                    'start': start, 'reader': reader, 'timing': timing,
                                    'records': [{'uid': row['uid'], 'result': row_result(outputs[i])}
                                                for i, row in enumerate(block)]})
            for row in block:
                if not (out / 'assignments' / f"{row['uid']}-reader{reader}.json").exists():
                    raise ValueError('saved reader batch lacks assignment receipt')
            both.append([r['result'] for r in saved['records']])
            timings.append(saved['timing'])
        save(batch_path, {'schema': 'mechanism-start-rating-batch-v1',
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
    summary = write_once(out / 'SUMMARY.json',
                         {'schema': 'mechanism-start-reader-summary-v1',
                          'binding_sha256': binding['sha256'],
                          'counts': dict(counts), 'reader_timings': timings,
                          'scope': 'All frozen start assignments, two same-model Qwen3.8 draws; start agreement is an LLM audit, not human truth or causal control.'})
    print(json.dumps({'summary': str(out / 'SUMMARY.json'),
                      'sha256': summary['sha256'], 'counts': summary['counts']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
