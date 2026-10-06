"""Four-channel Qwen start audit for the separate 220-family extension.

Modes: live qualification, one priced GPU shard, or CPU-only complete-stage seal.
Each primary/veto draw is an append-only physical attempt until a result commits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import mechanism_extension_reader_contract_v1 as contract
import seal_mechanism_extension_reader_v1 as sealer
from freeze_mechanism_extension_220_v1 import digest, sealed


def require_gpu_step():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('extension reader inference requires a GPU Slurm step')


def expected_manifest():
    family, selection, frame, source = contract.source_inputs()
    body = sealer.prepare_body(family, selection, frame, source)
    manifest = sealed(contract.QUAL_MANIFEST)
    contract.require(manifest == {**body, 'sha256': digest(body)},
                     'extension qualification manifest differs from current code or source')
    return family, selection, frame, source, manifest


def expected_final():
    family, selection, frame, source, manifest = expected_manifest()
    result = sealed(contract.QUAL_RESULT)
    body = sealer.rebind_body(family, selection, frame, source, manifest, result)
    price = sealed(contract.FINAL_PRICE)
    contract.require(price == {**body, 'sha256': digest(body)} and
                     price['status'] == 'PASS_COMPLETE_STAGE',
                     'extension reader price is not sealed PASS for this code/profile')
    return family, selection, frame, source, manifest, result, price


def tokenizer_ids(tokenizer, row, kind):
    value = tokenizer.apply_chat_template(
        contract.messages(row, kind), tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort='low')
    if isinstance(value, dict):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    contract.require(isinstance(value, (list, tuple)) and len(value) >= 16 and
                     all(type(token) is int and token >= 0 for token in value),
                     'qualification chat template did not produce exact token IDs')
    return list(value)


def make_model():
    from vllm import LLM
    p = contract.PROFILE
    return LLM(model=p['model_snapshot'], tokenizer=p['model_snapshot'],
               tensor_parallel_size=p['tensor_parallel_size'], dtype=p['dtype'],
               kv_cache_dtype=p['kv_cache_dtype'], max_model_len=p['max_model_len'],
               max_num_seqs=p['max_num_seqs'],
               max_num_batched_tokens=p['max_num_batched_tokens'],
               gpu_memory_utilization=p['gpu_memory_utilization'],
               enforce_eager=p['enforce_eager'],
               generation_config=p['generation_config'],
               language_model_only=p['language_model_only'],
               attention_config={'backend': p['attention_backend']})


def make_params(block, kind, reader):
    from vllm import SamplingParams
    return [SamplingParams(temperature=contract.PROFILE['temperature'],
                           top_p=contract.PROFILE['top_p'],
                           max_tokens=contract.MAX_TOKENS,
                           seed=contract.rating_seed(row['uid'], kind, reader))
            for row in block]


def chat(model, block, kind, reader):
    return model.chat([contract.messages(row, kind) for row in block],
                      sampling_params=make_params(block, kind, reader),
                      chat_template_kwargs={'enable_thinking': True,
                                            'reasoning_effort': 'low'},
                      use_tqdm=False)


def row_result(output, kind):
    contract.require(len(output.outputs) == 1, 'reader output multiplicity differs')
    one = output.outputs[0]
    contract.require(one.finish_reason in ('stop', 'length') and
                     len(one.token_ids) <= contract.MAX_TOKENS,
                     'reader output exceeds cap or has unknown finish reason')
    value = {'finish_reason': one.finish_reason,
             'generated_tokens': len(one.token_ids), 'raw_completion': one.text}
    parsed = contract.parse(one.text, kind)
    if kind == 'primary':
        value['rating'] = parsed
    else:
        value['already_completed'] = (None if parsed is None else
                                      parsed['already_completed'])
    return value


def qualify():
    require_gpu_step()
    _, _, frame, source, manifest = expected_manifest()
    if contract.QUAL_RESULT.exists():
        result = sealed(contract.QUAL_RESULT)
        sealer.check_qualification(result, manifest)
        print(json.dumps({'qualification': str(contract.QUAL_RESULT),
                          'sha256': result['sha256'], 'status': result['status']}))
        return result
    directory = contract.OUTPUT_ROOT / 'qualification'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt_path = directory / 'ATTEMPT.json'
        if attempt_path.exists():
            raise RuntimeError('uncommitted qualification attempt requires fresh remaining-cost review')
        contract.save(attempt_path, {
            'schema': 'mechanism-extension-reader-qualification-attempt-v1',
            'qualification_manifest_sha256': manifest['sha256'],
            'source_exact_price_sha256': source['sha256'],
            'job_id': os.environ['SLURM_JOB_ID'],
            'state': 'started_before_model_load'})
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(contract.primary.MODEL,
                                                  local_files_only=True)
        indices = manifest['sample_indices']
        block = [frame['records'][i] for i in indices]
        expected = {(kind, row['uid']): tokenizer_ids(tokenizer, row, kind)
                    for kind in contract.KINDS for row in block}
        contract.require(all(len(ids) + contract.MAX_TOKENS <=
                             contract.PROFILE['max_model_len']
                             for ids in expected.values()),
                         'qualification sample exceeds model context')
        started = time.monotonic()
        model = make_model()
        load_seconds = time.monotonic() - started
        records = []
        generated = 0
        for kind in contract.KINDS:
            for reader in contract.READERS:
                outputs = chat(model, block, kind, reader)
                contract.require(len(outputs) == len(block),
                                 'qualification model output count differs')
                for row, output in zip(block, outputs):
                    ids = expected[kind, row['uid']]
                    contract.require(list(output.prompt_token_ids) == ids,
                                     'qualification vLLM/HF prompt token IDs differ')
                    result = row_result(output, kind)
                    generated += result['generated_tokens']
                    records.append({
                        'kind': kind, 'reader': reader, 'uid': row['uid'],
                        'seed': contract.rating_seed(row['uid'], kind, reader),
                        'prompt_tokens': len(ids),
                        'prompt_ids_sha256': digest(ids),
                        'generated_tokens': result['generated_tokens'],
                        'finish_reason': result['finish_reason'],
                        'parsed': (result.get('rating') is not None if kind == 'primary'
                                   else result.get('already_completed') is not None),
                        'completion_sha256': hashlib.sha256(
                            result['raw_completion'].encode()).hexdigest(),
                    })
        body = {
            'schema': 'mechanism-extension-reader-qualification-result-v1',
            'status': 'PASS', 'pass': True,
            'qualification_manifest_sha256': manifest['sha256'],
            'source_exact_price_sha256': source['sha256'],
            'rating_driver_sha256': contract.file_sha(__file__),
            'contract_sha256': contract.file_sha(contract.__file__),
            'model_profile': contract.PROFILE,
            'job_id': os.environ['SLURM_JOB_ID'],
            'load_seconds': load_seconds,
            'sample_uids': manifest['sample_uids'],
            'reader_channels': manifest['reader_channels'],
            'ratings': len(records), 'generated_tokens': generated,
            'prompt_ids_exact_pass': True,
            'model_output_count_pass': True, 'cap_and_finish_pass': True,
            'records': records,
            'scope': 'Small live engine and prompt-ID qualification, not start eligibility or mechanism outcome.',
        }
        result = contract.save(contract.QUAL_RESULT, body)
        sealer.check_qualification(result, manifest)
        print(json.dumps({'qualification': str(contract.QUAL_RESULT),
                          'sha256': result['sha256'], 'status': result['status'],
                          'ratings': result['ratings']}), flush=True)
        return result


def attempt_ledger(out, start, kind, reader, block, binding_sha):
    prefix = f'{start:06d}-{kind}-reader{reader}-attempt'
    paths = sorted((out / 'attempts').glob(prefix + '*.json'))
    attempts = []
    for path in paths:
        value = sealed(path)
        contract.require(
            value['schema'] == 'mechanism-extension-reader-attempt-v1' and
            value['binding_sha256'] == binding_sha and
            value['start'] == start and value['kind'] == kind and
            value['reader'] == reader and
            value['recovery_of_uncommitted_attempts'] ==
            [a['sha256'] for a in attempts] and
            value['uids'] == [r['uid'] for r in block] and
            value['seeds'] == [contract.rating_seed(r['uid'], kind, reader)
                               for r in block] and
            path.name == prefix + f"{value['attempt_index']:03d}.json",
            'foreign or changed extension reader attempt ledger')
        attempts.append(value)
    contract.require([a['attempt_index'] for a in attempts] == list(range(len(attempts))),
                     'reader attempt indices are not contiguous')
    return attempts


def committed_channel(out, start, kind, reader, block, binding_sha):
    path = out / 'batches' / f'{start:06d}-{kind}-reader{reader}.json'
    attempts = attempt_ledger(out, start, kind, reader, block, binding_sha)
    if not path.exists():
        return None
    saved = sealed(path)
    index = saved.get('attempt_index')
    contract.require(
        saved['schema'] == 'mechanism-extension-reader-channel-batch-v1' and
        saved['binding_sha256'] == binding_sha and
        saved['start'] == start and saved['kind'] == kind and
        saved['reader'] == reader and
        [r['uid'] for r in saved['records']] == [r['uid'] for r in block] and
        type(index) is int and 0 <= index < len(attempts) and
        saved['attempt_sha256'] == attempts[index]['sha256'] and
        index == len(attempts) - 1,
        'committed extension reader channel differs from its final attempt')
    for row in block:
        entry = sealed(out / 'assignments' /
                       f"{row['uid']}-{kind}-reader{reader}-attempt{index:03d}.json")
        contract.require(
            entry['schema'] == 'mechanism-extension-rating-assignment-v1' and
            entry['binding_sha256'] == binding_sha and
            entry['uid'] == row['uid'] and entry['kind'] == kind and
            entry['reader'] == reader and entry['start'] == start and
            entry['attempt_index'] == index and
            entry['attempt_sha256'] == attempts[index]['sha256'],
            'committed extension reader lacks exact per-UID attempt receipt')
    return saved


def combined_batch(out, start, block, binding_sha):
    path = out / 'batches' / f'{start:06d}.json'
    channels = {(kind, reader): committed_channel(
        out, start, kind, reader, block, binding_sha)
        for kind in contract.KINDS for reader in contract.READERS}
    if any(value is None for value in channels.values()):
        contract.require(not path.exists(), 'combined batch exists without four committed channels')
        return None
    records = [
        {'uid': row['uid'], 'transition': row['transition'],
         'primary_readers': [channels['primary', reader]['records'][i]['result']
                             for reader in contract.READERS],
         'veto_readers': [channels['veto', reader]['records'][i]['result']
                          for reader in contract.READERS]}
        for i, row in enumerate(block)]
    timings = [channels[kind, reader]['timing']
               for kind in contract.KINDS for reader in contract.READERS]
    body = {'schema': 'mechanism-extension-rating-batch-v1',
            'binding_sha256': binding_sha, 'start': start,
            'channel_batch_sha256s': [channels[kind, reader]['sha256']
                                      for kind in contract.KINDS
                                      for reader in contract.READERS],
            'channel_timings': timings, 'records': records}
    if path.exists():
        contract.require(sealed(path) == {**body, 'sha256': digest(body)},
                         'completed extension batch differs from committed channels')
        return sealed(path)
    return contract.save(path, body)


def binding_body(family, selection, frame, source, manifest, result, price,
                 shard_index):
    shard = price['shards'][shard_index]
    return {
        'schema': 'mechanism-extension-reader-shard-binding-v1',
        'extension_family_freeze_sha256': family['sha256'],
        'selection_sha256': selection['sha256'], 'frame_sha256': frame['sha256'],
        'source_exact_price_sha256': source['sha256'],
        'qualification_manifest_sha256': manifest['sha256'],
        'qualification_result_sha256': result['sha256'],
        'final_price_sha256': price['sha256'],
        'rating_driver_sha256': contract.file_sha(__file__),
        'contract_sha256': contract.file_sha(contract.__file__),
        'primary_message_source_sha256': contract.file_sha(contract.primary.__file__),
        'veto_message_source_sha256': contract.file_sha(contract.veto.__file__),
        'model_profile': contract.PROFILE,
        'shard_index': shard_index, 'start_row': shard['start_row'],
        'end_row': shard['end_row'], 'batch_size': contract.BATCH,
        'reader_channels': manifest['reader_channels'],
        'visible_input_allowlist': frame['visible_input_allowlist'],
        'output_root': str(contract.OUTPUT_ROOT),
        'scope': 'All frozen extension starts assigned in this price shard; no rating-dependent replacement.',
    }


def shard_directory(shard_index):
    return contract.OUTPUT_ROOT / f'shard-{shard_index:03d}'


def shard_rows(frame, price, shard_index):
    contract.require(type(shard_index) is int and 0 <= shard_index < len(price['shards']),
                     'shard index is not in complete-stage price')
    shard = price['shards'][shard_index]
    return shard, frame['records'][shard['start_row']:shard['end_row']]


def channel_result(out, start, kind, reader, block, binding, model):
    saved = committed_channel(out, start, kind, reader, block, binding['sha256'])
    if saved is not None:
        return saved
    from vllm import SamplingParams  # noqa: F401  (fail before assigning work if vLLM is absent)
    attempts = attempt_ledger(out, start, kind, reader, block, binding['sha256'])
    index = len(attempts)
    attempt = contract.save(
        out / 'attempts' / f'{start:06d}-{kind}-reader{reader}-attempt{index:03d}.json',
        {'schema': 'mechanism-extension-reader-attempt-v1',
         'binding_sha256': binding['sha256'], 'start': start,
         'kind': kind, 'reader': reader, 'attempt_index': index,
         'recovery_of_uncommitted_attempts': [a['sha256'] for a in attempts],
         'uids': [r['uid'] for r in block],
         'seeds': [contract.rating_seed(r['uid'], kind, reader) for r in block],
         'job_id': os.environ['SLURM_JOB_ID'],
         'state': 'started_before_model_chat'})
    for row in block:
        contract.save(out / 'assignments' /
                      f"{row['uid']}-{kind}-reader{reader}-attempt{index:03d}.json",
                      {'schema': 'mechanism-extension-rating-assignment-v1',
                       'binding_sha256': binding['sha256'], 'uid': row['uid'],
                       'kind': kind, 'reader': reader, 'start': start,
                       'attempt_index': index, 'attempt_sha256': attempt['sha256'],
                       'state': 'attempted_before_model_chat',
                       'job_id': os.environ['SLURM_JOB_ID']})
    t0 = time.monotonic()
    try:
        outputs = chat(model, block, kind, reader)
        contract.require(len(outputs) == len(block), 'extension reader output count differs')
        results = [row_result(output, kind) for output in outputs]
    except Exception as exc:
        contract.save(out / 'failures' /
                      f'{start:06d}-{kind}-reader{reader}-attempt{index:03d}.json',
                      {'schema': 'mechanism-extension-reader-failure-v1',
                       'binding_sha256': binding['sha256'],
                       'attempt_sha256': attempt['sha256'],
                       'exception_type': type(exc).__name__,
                       'exception_message': str(exc)[:500],
                       'job_id': os.environ['SLURM_JOB_ID']})
        raise
    timing = {'kind': kind, 'reader': reader,
              'job_id': os.environ['SLURM_JOB_ID'],
              'wall_seconds': time.monotonic() - t0,
              'prompt_tokens': sum(len(o.prompt_token_ids) for o in outputs),
              'generated_tokens': sum(r['generated_tokens'] for r in results)}
    saved = contract.save(out / 'batches' /
                          f'{start:06d}-{kind}-reader{reader}.json',
                          {'schema': 'mechanism-extension-reader-channel-batch-v1',
                           'binding_sha256': binding['sha256'],
                           'start': start, 'kind': kind, 'reader': reader,
                           'attempt_index': index,
                           'attempt_sha256': attempt['sha256'],
                           'timing': timing,
                           'records': [{'uid': row['uid'], 'result': results[i]}
                                       for i, row in enumerate(block)]})
    committed_channel(out, start, kind, reader, block, binding['sha256'])
    return saved


def summarize_records(completed, family_by_uid):
    counts = Counter(rows=len(completed), ratings=4 * len(completed),
                     primary_ratings=2 * len(completed),
                     strict_veto_sensitivity_ratings=2 * len(completed))
    accepted_families, strict_families = set(), set()
    for row in completed:
        primary, veto = row['primary_readers'], row['veto_readers']
        contract.require(len(primary) == len(veto) == 2 and
                         row['uid'] in family_by_uid,
                         'reader summary row has missing channel or unknown UID')
        counts['generated_tokens'] += sum(r['generated_tokens'] for r in primary + veto)
        counts['primary_parsed_stop'] += sum(r['rating'] is not None and
                                             r['finish_reason'] == 'stop'
                                             for r in primary)
        counts['veto_parsed_stop'] += sum(r['already_completed'] is not None and
                                          r['finish_reason'] == 'stop'
                                          for r in veto)
        primary_ok = all(r['rating'] == {'start': True} and
                         r['finish_reason'] == 'stop' for r in primary)
        strict_ok = primary_ok and all(r['already_completed'] is False and
                                       r['finish_reason'] == 'stop' for r in veto)
        counts['primary_accepted_starts'] += primary_ok
        counts['strict_sensitivity_starts'] += strict_ok
        if primary_ok:
            accepted_families.add(family_by_uid[row['uid']])
        if strict_ok:
            strict_families.add(family_by_uid[row['uid']])
    counts['primary_accepted_families'] = len(accepted_families)
    counts['strict_sensitivity_families'] = len(strict_families)
    return dict(counts)


def verify_shard(out, frame, selection, source, manifest, result, price,
                 family, shard_index):
    shard, rows = shard_rows(frame, price, shard_index)
    expected_binding = binding_body(family, selection, frame, source,
                                    manifest, result, price, shard_index)
    binding = sealed(out / 'BINDING.json')
    contract.require(binding == {**expected_binding, 'sha256': digest(expected_binding)},
                     'reader shard binding differs from final price')
    completed, timings, all_attempts, committed_attempts = [], [], [], set()
    expected_batches = set()
    known_attempts = {}
    for start in range(shard['start_row'], shard['end_row'], contract.BATCH):
        block = frame['records'][start:min(start + contract.BATCH, shard['end_row'])]
        saved = combined_batch(out, start, block, binding['sha256'])
        contract.require(saved is not None, 'reader shard has incomplete four-channel batch')
        completed.extend(saved['records'])
        timings.extend(saved['channel_timings'])
        expected_batches.add(f'{start:06d}.json')
        for kind in contract.KINDS:
            for reader in contract.READERS:
                channel = committed_channel(out, start, kind, reader, block,
                                            binding['sha256'])
                attempts = attempt_ledger(out, start, kind, reader, block,
                                          binding['sha256'])
                expected_batches.add(f'{start:06d}-{kind}-reader{reader}.json')
                committed_attempts.add(channel['attempt_sha256'])
                all_attempts.extend(attempts)
                for attempt in attempts:
                    key = (start, kind, reader, attempt['attempt_index'])
                    known_attempts[key] = attempt
    contract.require({p.name for p in (out / 'batches').glob('*.json')} == expected_batches and
                     [r['uid'] for r in completed] == [r['uid'] for r in rows] and
                     len({r['uid'] for r in completed}) == len(rows),
                     'reader shard batch files or assigned UIDs differ')
    expected_attempt_names = {
        f"{start:06d}-{kind}-reader{reader}-attempt{index:03d}.json"
        for start, kind, reader, index in known_attempts}
    contract.require({p.name for p in (out / 'attempts').glob('*.json')} ==
                     expected_attempt_names,
                     'reader shard has an unrecognized physical attempt receipt')
    assignment_receipts = 0
    for path in (out / 'assignments').glob('*.json'):
        entry = sealed(path)
        key = (entry['start'], entry['kind'], entry['reader'],
               entry['attempt_index'])
        attempt = known_attempts.get(key)
        contract.require(attempt is not None and
                         entry['schema'] == 'mechanism-extension-rating-assignment-v1' and
                         entry['binding_sha256'] == binding['sha256'] and
                         entry['uid'] in attempt['uids'] and
                         entry['attempt_sha256'] == attempt['sha256'] and
                         path.name == f"{entry['uid']}-{entry['kind']}-reader{entry['reader']}-attempt{entry['attempt_index']:03d}.json",
                         'reader shard has an unrecognized assignment receipt')
        assignment_receipts += 1
    attempt_keys_by_sha = {attempt['sha256']: key
                           for key, attempt in known_attempts.items()}
    for path in (out / 'failures').glob('*.json'):
        entry = sealed(path)
        key = attempt_keys_by_sha.get(entry['attempt_sha256'])
        contract.require(entry['schema'] == 'mechanism-extension-reader-failure-v1' and
                         entry['binding_sha256'] == binding['sha256'] and
                         key is not None and
                         path.name == f'{key[0]:06d}-{key[1]}-reader{key[2]}-attempt{key[3]:03d}.json',
                         'reader shard has an unrecognized failure receipt')
    family_by_uid = {r['uid']: r['family'] for r in selection['records']}
    counts = summarize_records(completed, family_by_uid)
    counts['physical_channel_attempts_recorded'] = len(all_attempts)
    counts['physical_assignment_receipts_recorded'] = assignment_receipts
    counts['uncommitted_channel_attempts_recorded'] = sum(
        a['sha256'] not in committed_attempts for a in all_attempts)
    summary_body = {
        'schema': 'mechanism-extension-reader-shard-summary-v1',
        'binding_sha256': binding['sha256'],
        'shard_index': shard_index, 'start_row': shard['start_row'],
        'end_row': shard['end_row'], 'counts': counts,
        'channel_timings': timings,
        'attempt_sha256s': [a['sha256'] for a in all_attempts],
        'scope': ('All assigned extension primary and strict-veto ratings in this '
                  'priced shard. Uncommitted physical attempts remain recorded; '
                  'invalid/capped ratings remain assigned. No human truth claim.'),
    }
    summary = out / 'SUMMARY.json'
    if summary.exists():
        contract.require(sealed(summary) == {**summary_body, 'sha256': digest(summary_body)},
                         'extension reader shard summary differs from receipts')
    return summary_body


def run_shard(shard_index):
    require_gpu_step()
    family, selection, frame, source, manifest, result, price = expected_final()
    shard, _ = shard_rows(frame, price, shard_index)
    out = shard_directory(shard_index)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('batches', 'assignments', 'attempts', 'failures', 'loads'):
            (out / name).mkdir(exist_ok=True)
        body = binding_body(family, selection, frame, source, manifest,
                            result, price, shard_index)
        binding = contract.save(out / 'BINDING.json', body, existing_ok=True)
        pending = []
        for start in range(shard['start_row'], shard['end_row'], contract.BATCH):
            block = frame['records'][start:min(start + contract.BATCH, shard['end_row'])]
            if combined_batch(out, start, block, binding['sha256']) is None:
                pending.append((start, block))
        needs_model = any(committed_channel(out, start, kind, reader, block,
                                            binding['sha256']) is None
                          for start, block in pending
                          for kind in contract.KINDS for reader in contract.READERS)
        model = None
        if needs_model:
            started = time.monotonic()
            model = make_model()
            contract.save(out / 'loads' / f"{os.environ['SLURM_JOB_ID']}.json",
                          {'schema': 'mechanism-extension-reader-load-v1',
                           'binding_sha256': binding['sha256'],
                           'job_id': os.environ['SLURM_JOB_ID'],
                           'load_seconds': time.monotonic() - started})
        for start, block in pending:
            for kind in contract.KINDS:
                for reader in contract.READERS:
                    channel_result(out, start, kind, reader, block, binding, model)
            combined_batch(out, start, block, binding['sha256'])
            print(json.dumps({'shard': shard_index,
                              'complete_through_row': min(start + contract.BATCH,
                                                          shard['end_row']),
                              'of': shard['end_row']}), flush=True)
        summary_body = verify_shard(out, frame, selection, source, manifest,
                                    result, price, family, shard_index)
        summary = contract.save(out / 'SUMMARY.json', summary_body, existing_ok=True)
        print(json.dumps({'shard_summary': str(out / 'SUMMARY.json'),
                          'sha256': summary['sha256'],
                          'counts': summary['counts']}), flush=True)
        return summary


def seal_stage():
    family, selection, frame, source, manifest, result, price = expected_final()
    contract.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (contract.OUTPUT_ROOT / 'STAGE_SEAL.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summaries, completed = [], []
        for index in range(len(price['shards'])):
            out = shard_directory(index)
            body = verify_shard(out, frame, selection, source, manifest,
                                result, price, family, index)
            value = sealed(out / 'SUMMARY.json')
            contract.require(value == {**body, 'sha256': digest(body)},
                             'extension reader shard summary absent or changed')
            summaries.append(value)
            for start in range(body['start_row'], body['end_row'], contract.BATCH):
                completed.extend(sealed(out / 'batches' / f'{start:06d}.json')['records'])
        contract.require([r['uid'] for r in completed] ==
                         [r['uid'] for r in frame['records']],
                         'complete extension reader stage lacks exact assigned UIDs')
        family_by_uid = {r['uid']: r['family'] for r in selection['records']}
        counts = summarize_records(completed, family_by_uid)
        counts['physical_channel_attempts_recorded'] = sum(
            s['counts']['physical_channel_attempts_recorded'] for s in summaries)
        counts['uncommitted_channel_attempts_recorded'] = sum(
            s['counts']['uncommitted_channel_attempts_recorded'] for s in summaries)
        contract.require(counts['rows'] == price['rows'] and
                         counts['ratings'] == price['ratings'],
                         'complete reader stage differs from priced assignments')
        body = {
            'schema': 'mechanism-extension-reader-stage-completion-v1',
            'status': 'COMPLETE_ALL_ASSIGNED_RATINGS',
            'extension_family_freeze_sha256': family['sha256'],
            'selection_sha256': selection['sha256'],
            'frame_sha256': frame['sha256'],
            'source_exact_price_sha256': source['sha256'],
            'final_price_sha256': price['sha256'],
            'qualification_result_sha256': result['sha256'],
            'rating_driver_sha256': contract.file_sha(__file__),
            'shard_summary_sha256s': [s['sha256'] for s in summaries],
            'counts': counts,
            'scope': ('All frozen exploratory extension starts retained. Primary '
                      'two-reader accepted families/starts and strict-veto '
                      'sensitivity denominators are distinct. Not the '
                      'registered 128-family validation or a causal effect.'),
        }
        value = contract.save(contract.OUTPUT_ROOT / 'STAGE_COMPLETION.json',
                              body, existing_ok=True)
        print(json.dumps({'stage_completion': str(contract.OUTPUT_ROOT /
                                                  'STAGE_COMPLETION.json'),
                          'sha256': value['sha256'],
                          'counts': value['counts']}), flush=True)
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--qualify', action='store_true')
    modes.add_argument('--shard-index', type=int)
    modes.add_argument('--seal-stage', action='store_true')
    args = parser.parse_args()
    if args.qualify:
        qualify()
    elif args.seal_stage:
        seal_stage()
    else:
        run_shard(args.shard_index)
