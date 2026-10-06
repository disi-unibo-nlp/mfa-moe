"""Bound saved generated continuations to blind seven-class labels and route IDs.

This is an offline descriptive measurement. Class labels are not substantive
behavior votes, and unweighted expert selections are not router gate weights.
No generated continuation is chosen by its result or by an earlier pilot effect.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import hmac
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import label_dense_discovery as qualified
import mechanism_extension_reader_contract_v2 as id_contract
from freeze_mechanism_extension_220_v1 import digest, sealed

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
TOKENIZER = REPO.parent / 'cache/hf/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989'
DISCOVERY_FRAME = ROOT / 'runs/routing-control-v1/dense-discovery/TRANSITION_V22_FULL_PREFIX_START_FRAME.json'
THINK_END = 248069
BATCH = 32
MAX_TOKENS = 1024
WINDOW = 64
PROFILE = {'tensor_parallel_size': 2, 'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16',
           'max_model_len': 49152, 'max_num_seqs': 32, 'max_num_batched_tokens': 8192,
           'gpu_memory_utilization': .85, 'enforce_eager': True, 'generation_config': 'vllm',
           'language_model_only': True, 'attention_config': {'backend': 'FLASH_ATTN'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, body):
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(sealed(path) == value, 'existing dense artifact differs: ' + str(path))
        return value
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}-{time.monotonic_ns()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return value


def require_step(gpu=False):
    require(bool(os.environ.get('SLURM_JOB_ID')) and bool(os.environ.get('SLURM_STEP_ID')) and
            not socket.gethostname().startswith('login'), 'dense work requires a Slurm step')
    if gpu:
        require(bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'dense labeling requires allocated GPUs')


def measurement_code():
    import qualify_dense_judge
    from moe_exp.models import token_replay
    from moe_exp.correlation_pipeline import spans
    from moe_exp.correlation_pipeline.dynamics import classes, routing, common
    from moe_exp.routing_control import analysis, trajectory_analysis_v1
    modules = [__file__, qualified.__file__, qualify_dense_judge.__file__, id_contract.__file__,
               token_replay.__file__, spans.__file__, classes.__file__, routing.__file__, common.__file__,
               analysis.__file__, trajectory_analysis_v1.__file__]
    modules.append(Path(__file__).with_name('overnight_routing_entry_v1.py'))
    modules.extend(Path(__file__).with_name(name) for name in
                   ('prepare_generated_dense_v1.sbatch', 'label_generated_dense_v1.sbatch',
                    'analyze_generated_dense_v1.sbatch'))
    return {str(Path(p).resolve()): file_sha(p) for p in modules}


def qualification():
    value = sealed(qualified.PARITY)
    require(file_sha(qualified.PROGRAM) == qualified.PROGRAM_SHA and
            value['schema'] == 'dense-judge-parity-v1' and
            value['program_sha256'] == qualified.PROGRAM_SHA and
            value['model_path'] == str(qualified.MODEL) and len(value['records']) == 200,
            'pinned seven-class labeler qualification differs')
    covered = sum(r['new_label'] in qualified.CLASSES and r['finish_reason'] == 'stop' for r in value['records'])
    agreed = sum(r['new_label'] == r['historical_label'] and r['finish_reason'] == 'stop' for r in value['records'])
    require(covered >= 190 and agreed >= 140 and value['generation_seconds'] > 0 and
            value['total_generated_tokens'] > 0, 'seven-class operational parity failed')
    return value


def instruction():
    require(file_sha(qualified.PROGRAM) == qualified.PROGRAM_SHA, 'dense reader rubric changed')
    return json.loads(qualified.PROGRAM.read_text())['classify']['signature']['instructions'] + \
        '\nReturn only the one label. Do not include an explanation.'


def output_path(manifest, parent):
    return Path(parent) / ('dense-generated-v1-' + manifest['sha256'][:12] + '-' +
                           digest(measurement_code())[:12])


def source_outputs(manifest, price, run_out):
    """Check every exact assignment and raw array before any offline labeling."""
    import numpy as np
    schema = manifest['schema']
    require(schema in ('overnight-routing-manifest-v1', 'overnight-routing-manifest-v2'),
            'only sealed overnight v1/v2 generation is supported')
    version = schema.rsplit('-', 1)[1]
    runner = importlib.import_module('overnight_routing_runner_' + version)
    require(manifest['code_files'].get(str(Path(runner.__file__).resolve())) == file_sha(runner.__file__) and
            all(file_sha(path) == sha for path, sha in manifest['code_files'].items()),
            'generation runner or source code differs from frozen manifest')
    require(price['manifest_sha256'] == manifest['sha256'] and price['shards'] == manifest['shards'] and
            price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY', 'generation price differs')
    stage = sealed(Path(run_out) / 'STAGE_COMPLETION.json')
    require(stage['manifest_sha256'] == manifest['sha256'] and
            stage['schema'] == 'overnight-routing-stage-completion-' + version and
            stage['counts']['assigned'] == manifest['expected_requests'], 'generation incomplete')
    expected = [meta for _, meta in runner.request_metadata(manifest, manifest['rows'], manifest.get('arms'))]
    found, array_sources, receipts = [], {}, []
    for i, shard in enumerate(manifest['shards']):
        directory = Path(run_out) / f'shard-{i:03d}'
        binding, summary, completion = [sealed(directory / name) for name in
                                         ('BINDING.json', 'SUMMARY.json', 'OVERNIGHT_COMPLETION.json')]
        require(binding['manifest_sha256'] == summary['manifest_sha256'] == completion['manifest_sha256'] == manifest['sha256'] and
                summary['binding_sha256'] == binding['sha256'] and
                completion['sha256'] == stage['shard_completion_sha256s'][i] and
                completion['source_summary_sha256'] == summary['sha256'], 'generation shard receipt differs')
        for j in range(summary['batches']):
            assignment = sealed(directory / f'batch-{j:03d}-assignment.json')
            batch = sealed(directory / f'batch-{j:03d}.json')
            arrays = directory / f'batch-{j:03d}.npz'
            require(assignment['manifest_sha256'] == batch['manifest_sha256'] == manifest['sha256'] and
                    assignment['binding_sha256'] == batch['binding_sha256'] == binding['sha256'] and
                    batch['array_sha256'] == file_sha(arrays) and
                    [r['uid'] for r in assignment['requests']] == [r['uid'] for r in batch['outputs']],
                    'saved generation batch/array differs')
            with np.load(arrays, allow_pickle=False) as archive:
                require(set(archive.files) == {r['uid'] for r in batch['outputs'] if r['routed_present']},
                        'route archive has missing or unassigned arrays')
                for row in batch['outputs']:
                    require(row['uid'] not in array_sources, 'duplicate generated UID')
                    if row['routed_present']:
                        routes = archive[row['uid']]
                        require(routes.shape == (len(row['tokens']), 40, 8) and
                                np.issubdtype(routes.dtype, np.integer) and
                                ((routes >= 0) & (routes < 256)).all() and
                                (np.diff(np.sort(routes, axis=-1), axis=-1) > 0).all(),
                                'raw top8 expert IDs disagree with emitted tokens')
                    array_sources[row['uid']] = {'path': str(arrays.resolve()), 'sha256': batch['array_sha256'],
                                                 'key': row['uid'], 'available': row['routed_present'],
                                                 'signals': ['executed_expert_ids'] if row['routed_present'] else []}
            found.extend(batch['outputs'])
            receipts.append({'assignment_sha256': assignment['sha256'], 'batch_sha256': batch['sha256'],
                             'array_sha256': batch['array_sha256'],
                             'assignment_path': str(directory / f'batch-{j:03d}-assignment.json'),
                             'batch_path': str(directory / f'batch-{j:03d}.json')})
    require(len(found) == len(expected) == manifest['expected_requests'] and
            len({r['uid'] for r in found}) == len(found), 'generation assignment coverage differs')
    for assigned, observed in zip(expected, found, strict=True):
        require(all(observed.get(k) == v for k, v in assigned.items()), 'generated assignment metadata changed')
    return stage, expected, found, array_sources, receipts


def seal_generation_if_needed(manifest, manifest_path, run_out):
    path = Path(run_out) / 'STAGE_COMPLETION.json'
    if path.exists():
        return sealed(path)
    require(manifest['schema'] in ('overnight-routing-manifest-v1', 'overnight-routing-manifest-v2'),
            'unsupported generation schema')
    version = manifest['schema'].rsplit('-', 1)[1]
    # Engine validation resolves its qualified worker namespace in a fresh child.
    # This process keeps the repository's current offline analysis modules.
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('overnight_routing_entry_v1.py')),
                    'overnight_routing_runner_' + version, '--manifest', str(manifest_path),
                    '--out', str(run_out), '--seal-stage'], check=True)
    return sealed(path)


def sentence_layout(tokenizer, prompt_ids, tokens):
    """Actual token IDs -> DecodeStream offsets -> dense deterministic sentence units."""
    from moe_exp.models.token_replay import make_token_replay
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), 'invalid continuation tokens')
    closure = tokens.index(THINK_END) if THINK_END in tokens else len(tokens)
    if closure == 0:
        return {'text': '', 'offsets': [], 'token_owner': [], 'sentences': [],
                'reasoning_end': 0, 'closed_reasoning': closure < len(tokens)}
    text, replay = make_token_replay(tokenizer, prompt_ids, tokens[:closure])
    trace = SimpleNamespace(cot_text=text, metadata={'reasoning_content': text, 'token_replay': replay})
    layout = saved_layout(trace)
    owners = [None] * closure
    sentences = []
    for unit, owned in zip(layout['units'], layout['unit_tokens'], strict=True):
        index = unit['index']
        require(unit['text'] == text[unit['start']:unit['end']], 'sentence exact text changed')
        for token in owned:
            require(owners[token] is None, 'token assigned to two sentences')
            owners[token] = index
        complete = (index + 1 < len(layout['units']) or closure < len(tokens) or
                    unit['text'].endswith(('.', '!', '?')) or '\n' in text[unit['end']:])
        sentences.append({'sentence_index': index, 'segment': 0, 'char_start': unit['start'],
                          'char_end': unit['end'], 'text': unit['text'], 'token_indices': owned,
                          'token_start': min(owned) if owned else None,
                          'token_end': max(owned) + 1 if owned else None,
                          'complete': complete})
    require(len(replay['completion_offsets']) == closure, 'token replay lost emitted IDs')
    return {'text': text, 'offsets': replay['completion_offsets'], 'token_owner': owners,
            'sentences': sentences, 'reasoning_end': closure, 'closed_reasoning': closure < len(tokens)}


def make_blind_inputs(layout, problem, trigger, uid, salt):
    """Only these four literal text fields enter the qualified offline classifier."""
    units, records = layout['sentences'], []
    for i, row in enumerate(units):
        blind = hmac.new(salt, f'{uid}|{i}'.encode(), hashlib.sha256).hexdigest()
        records.append({'blind_id': blind, 'inputs': {
            'problem_statement': problem,
            'previous_sentence': units[i - 1]['text'] if i else trigger,
            'sentence': row['text'],
            'next_sentence': units[i + 1]['text'] if i + 1 < len(units) else
                             ('<END OF RESPONSE>' if layout['closed_reasoning'] else '<END OF OBSERVED WINDOW>')}})
        row['blind_id'] = blind
    return records


def build_frame(manifest, price, run_out, directory, tokenizer):
    stage, assigned, outputs, arrays, receipts = source_outputs(manifest, price, run_out)
    source_path = Path(manifest.get('source_frame_path', DISCOVERY_FRAME))
    source = sealed(source_path)
    require(source['sha256'] == manifest.get('source_frame_sha256',
            '6c10499bf71c7223b5c90fa6c2a1b13e13e9fb5c9c2af584d3ba67b4272290b2'), 'native start frame differs')
    originals = {r['uid']: r for r in source['records']}
    natives = {r['uid']: r for r in manifest['rows']}
    map_path = directory / 'ARM_MAP.json'
    salt = bytes.fromhex(sealed(map_path)['blind_salt_hex']) if map_path.exists() else os.urandom(32)
    records, blind_rows = [], []
    for expected, result in zip(assigned, outputs, strict=True):
        native = natives[expected['prefix_uid']]
        original = originals[native['uid']]
        data, meta = original['reader_input'], original['analysis_meta']
        require(set(data) == {'problem', 'emitted_prefix', 'triggering_sentence'} and
                meta['prefix_ids_sha256'] == native['prefix_ids_sha256'] == digest(native['prefix_ids']) and
                native['prompt_ids_sha256'] == digest(native['prompt_ids']) and
                original['family'] == native['family'] and original['transition'] == native['transition'] and
                original['prefix_tokens'] == len(native['prefix_ids']) and
                hashlib.sha256(data['emitted_prefix'].encode()).hexdigest() == meta['prefix_text_sha256'] and
                data['emitted_prefix'].rstrip().endswith(data['triggering_sentence'].rstrip()),
                'native prefix and offline context are not exactly aligned')
        tokens = result['tokens']
        require(len(tokens) <= manifest['horizon'], 'generation exceeds assigned horizon')
        layout = sentence_layout(tokenizer, native['prompt_ids'], tokens)
        blind_rows.extend(make_blind_inputs(layout, data['problem'], data['triggering_sentence'], expected['uid'], salt))
        records.append({**expected, **layout, 'emitted_token_ids': tokens,
                        'emitted_token_ids_sha256': digest(tokens), 'array': arrays[expected['uid']],
                        'finish_reason': result['finish'], 'error': result['error'],
                        'hit_cap': len(tokens) == manifest['horizon'],
                        'action_dose': result.get('action_dose'),
                        'inactive_native_checks': result.get('inactive_native_checks'),
                        'missing_telemetry_ranks': result.get('missing_telemetry_ranks')})
    blind_rows.sort(key=lambda r: r['blind_id'])
    require(len({r['blind_id'] for r in blind_rows}) == len(blind_rows), 'blind sentence identity collision')
    codes = measurement_code()
    shared = {'generation_manifest_sha256': manifest['sha256'], 'generation_stage_sha256': stage['sha256'],
              'source_frame_path': str(source_path), 'source_frame_sha256': source['sha256'],
              'code_files': codes, 'code_digest': digest(codes), 'horizon': manifest['horizon'],
              'generation_run_out': str(Path(run_out).resolve())}
    arm_map = save(map_path, {**shared, 'schema': 'generated-dense-arm-map-v1', 'assigned': len(assigned),
                             'blind_salt_hex': salt.hex(), 'generation_batch_receipts': receipts,
                             'tokenizer_config_sha256': file_sha(TOKENIZER / 'tokenizer_config.json'),
                             'records': records})
    frame = save(directory / 'BLIND_FRAME.json', {**shared, 'schema': 'generated-dense-blind-frame-v1',
                 'assigned': len(assigned), 'sentences': len(blind_rows), 'records': blind_rows,
                 'input_allowlist': ['problem_statement', 'previous_sentence', 'sentence', 'next_sentence'],
                 'scope': 'Offline dense class audit with observed next sentence; never online detector input.'})
    return arm_map, frame


def price_frame(frame, tokenizer, max_wall_seconds):
    parity = qualification()
    require(type(max_wall_seconds) is int and max_wall_seconds >= 3600, 'invalid complete label job wall')
    prompts, rubric = [], instruction()
    for row in frame['records']:
        ids = id_contract.exact_token_ids(tokenizer.apply_chat_template(
            qualified.messages(row, rubric), tokenize=True, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort='low'))
        require(len(ids) + MAX_TOKENS <= PROFILE['max_model_len'], 'dense prompt exceeds qualified model context')
        prompts.append({'blind_id': row['blind_id'], 'prompt_tokens': len(ids), 'prompt_ids_sha256': digest(ids)})
    decode_rate = parity['total_generated_tokens'] / parity['generation_seconds'] * .65
    timing_path = REPO / 'report/experimental-resume-v1/MECHANISM_START_READER_PRICE_v2.json'
    timing = sealed(timing_path)
    prefill_rate = timing['bounded_prefill_tps']
    require(prefill_rate > 0, 'missing qualified prefill timing')
    load, shutdown, retry, overhead = parity['load_seconds'] * 1.25, 196., 1.25, 900.
    usable = max_wall_seconds - load - shutdown - overhead
    require(usable > 0, 'dense load/overhead exceeds shard wall')
    shards, begin, seconds = [], 0, 0.
    for start in range(0, len(prompts), BATCH):
        block = prompts[start:start + BATCH]
        work = retry * (sum(r['prompt_tokens'] for r in block) / prefill_rate + len(block) * MAX_TOKENS / decode_rate)
        require(work <= usable, 'one dense32-sentence batch exceeds wall; revise allocation')
        if seconds and seconds + work > usable:
            shards.append({'start': begin, 'stop': start, 'work_seconds': seconds,
                           'complete_wall_seconds': seconds + load + shutdown + overhead})
            begin, seconds = start, 0.
        seconds += work
    if prompts:
        shards.append({'start': begin, 'stop': len(prompts), 'work_seconds': seconds,
                       'complete_wall_seconds': seconds + load + shutdown + overhead})
    loads = len(shards) + (max(1, math.ceil(len(shards) * .25)) if shards else 0)
    total_seconds = sum(s['work_seconds'] for s in shards) + loads * (load + shutdown) + len(shards) * overhead
    historical = Counter(r['historical_label'] for r in parity['records'])
    agreement = Counter(r['historical_label'] for r in parity['records']
                        if r['new_label'] == r['historical_label'] and r['finish_reason'] == 'stop')
    return {'schema': 'generated-dense-label-price-v1', 'status': 'PASS_COMPLETE_LABEL_STAGE',
            'frame_sha256': frame['sha256'], 'code_files': measurement_code(),
            'parity_sha256': parity['sha256'], 'qualified_program_sha256': qualified.PROGRAM_SHA,
            'historical_parity_class_agreement': {name: {'assigned': n, 'agree': agreement[name],
                                                        'fraction': agreement[name] / n}
                                                 for name, n in sorted(historical.items())},
            'model_path': str(qualified.MODEL), 'model_profile': PROFILE,
            'temperature': 0., 'max_tokens': MAX_TOKENS, 'batch_size': BATCH,
            'sentences': len(prompts), 'prompt_tokens_exact': sum(p['prompt_tokens'] for p in prompts),
            'maximum_decode_tokens': len(prompts) * MAX_TOKENS, 'prompt_records': prompts,
            'max_wall_seconds': max_wall_seconds, 'gpus_per_job': 2,
            'observed_parity_decode_tokens_per_second': parity['total_generated_tokens'] / parity['generation_seconds'],
            'bounded_decode_tokens_per_second': decode_rate, 'bounded_prefill_tokens_per_second': prefill_rate,
            'prefill_rate_source': str(timing_path), 'prefill_rate_source_sha256': timing['sha256'],
            'repeat_factor': retry, 'cold_load_seconds': load, 'shutdown_seconds': shutdown,
            'per_shard_overhead_seconds': overhead, 'cold_loads_including_reserve': loads,
            'shards': shards, 'complete_stage_GPU_h': 2 * total_seconds / 3600,
            'scope': 'All assigned sentence prompts, exact prefill, full1024-token reader caps, retries, loads/shutdown and overhead; no native generation.'}


def inputs(directory):
    directory = Path(directory)
    frame, price = sealed(directory / 'BLIND_FRAME.json'), sealed(directory / 'LABEL_PRICE.json')
    require(frame['schema'] == 'generated-dense-blind-frame-v1' and
            price['schema'] == 'generated-dense-label-price-v1' and
            price['frame_sha256'] == frame['sha256'] and price['status'] == 'PASS_COMPLETE_LABEL_STAGE' and
            frame['code_files'] == price['code_files'] == measurement_code() and
            frame['code_digest'] == digest(measurement_code()) and
            price['parity_sha256'] == qualification()['sha256'] and
            price['model_profile'] == PROFILE and price['model_path'] == str(qualified.MODEL) and
            len(frame['records']) == frame['sentences'] == price['sentences'] and
            [r['blind_id'] for r in frame['records']] == [r['blind_id'] for r in price['prompt_records']],
            'dense frame, exact price, code or qualification binding changed')
    require(sealed(Path(price['prefill_rate_source']))['sha256'] == price['prefill_rate_source_sha256'],
            'dense prefill timing source changed')
    require(directory.name == 'dense-generated-v1-' + frame['generation_manifest_sha256'][:12] + '-' +
            frame['code_digest'][:12], 'dense output directory is not bound to source and code')
    require(sum(s['stop'] - s['start'] for s in price['shards']) == frame['sentences'] and
            all(a['stop'] == b['start'] for a, b in zip(price['shards'], price['shards'][1:])) and
            (not price['shards'] or price['shards'][0]['start'] == 0 and
             price['shards'][-1]['stop'] == frame['sentences']), 'dense price misses assigned rows')
    return frame, price


def label_binding(frame, price, shard_index):
    require(type(shard_index) is int and 0 <= shard_index < len(price['shards']), 'dense shard index outside price')
    return {'schema': 'generated-dense-label-binding-v1', 'frame_sha256': frame['sha256'],
            'price_sha256': price['sha256'], 'code_digest': frame['code_digest'],
            'shard_index': shard_index, 'shard': price['shards'][shard_index]}


def committed_labels(out, start, stop, binding, prompt_records):
    path = out / 'batches' / f'{start:07d}.json'
    if not path.exists():
        return None
    value = sealed(path)
    attempt = sealed(out / 'attempts' / f'{start:07d}.json')
    require(value['schema'] == 'generated-dense-label-batch-v1' and
            value['binding_sha256'] == attempt['binding_sha256'] == binding['sha256'] and
            value['attempt_sha256'] == attempt['sha256'] and
            value['start'] == attempt['start'] == start and value['stop'] == attempt['stop'] == stop and
            [r['blind_id'] for r in value['records']] == [r['blind_id'] for r in prompt_records[start:stop]] and
            attempt['blind_ids'] == [r['blind_id'] for r in value['records']],
            'dense committed labels differ from assignment receipt')
    for result, prompt in zip(value['records'], prompt_records[start:stop], strict=True):
        require(result['prompt_ids_sha256'] == prompt['prompt_ids_sha256'] and
                result['prompt_tokens'] == prompt['prompt_tokens'] and
                result['finish_reason'] in ('stop', 'length') and
                type(result['generated_tokens']) is int and 0 <= result['generated_tokens'] <= MAX_TOKENS and
                (result['finish_reason'] != 'length' or result['generated_tokens'] == MAX_TOKENS) and
                result['label'] == qualified.parse_label(result['raw_completion']),
                'dense label token identity or parser receipt differs')
    return value


def label_shard(directory, shard_index):
    require_step(gpu=True)
    directory = Path(directory)
    frame, price = inputs(directory)
    body = label_binding(frame, price, shard_index)
    shard = price['shards'][shard_index]
    out = directory / f'labels-shard-{shard_index:03d}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'WRITER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = save(out / 'BINDING.json', body)
        for folder in ('batches', 'attempts', 'loads'):
            (out / folder).mkdir(exist_ok=True)
        pending = []
        for start in range(shard['start'], shard['stop'], BATCH):
            stop = min(start + BATCH, shard['stop'])
            if committed_labels(out, start, stop, binding, price['prompt_records']) is None:
                require(not (out / 'attempts' / f'{start:07d}.json').exists(),
                        'uncommitted dense model call requires a fresh remaining-cost recovery seal')
                pending.append((start, stop))
        model = None
        if pending:
            from vllm import LLM, SamplingParams
            save(out / 'loads' / (os.environ['SLURM_JOB_ID'] + '-attempt.json'),
                 {'schema': 'generated-dense-load-attempt-v1', 'binding_sha256': binding['sha256'],
                  'job_id': os.environ['SLURM_JOB_ID'], 'state': 'before_model_load'})
            began = time.monotonic()
            model = LLM(model=str(qualified.MODEL), tokenizer=str(qualified.MODEL), **PROFILE)
            save(out / 'loads' / (os.environ['SLURM_JOB_ID'] + '-complete.json'),
                 {'schema': 'generated-dense-load-result-v1', 'binding_sha256': binding['sha256'],
                  'job_id': os.environ['SLURM_JOB_ID'], 'load_seconds': time.monotonic() - began})
            params = SamplingParams(temperature=0., max_tokens=MAX_TOKENS)
        rubric = instruction()
        for start, stop in pending:
            block = frame['records'][start:stop]
            attempt = save(out / 'attempts' / f'{start:07d}.json',
                           {'schema': 'generated-dense-label-attempt-v1', 'binding_sha256': binding['sha256'],
                            'job_id': os.environ['SLURM_JOB_ID'], 'start': start, 'stop': stop,
                            'blind_ids': [r['blind_id'] for r in block], 'state': 'before_model_chat'})
            began = time.monotonic()
            outputs = model.chat([qualified.messages(row, rubric) for row in block], sampling_params=params,
                                 chat_template_kwargs={'enable_thinking': True, 'reasoning_effort': 'low'},
                                 use_tqdm=False)
            require(len(outputs) == len(block), 'dense label output count differs')
            records = []
            for row, output, prompt in zip(block, outputs, price['prompt_records'][start:stop], strict=True):
                require(len(output.outputs) == 1 and digest(list(output.prompt_token_ids)) == prompt['prompt_ids_sha256'],
                        'dense live prompt IDs differ from exact CPU price')
                one = output.outputs[0]
                require(one.finish_reason in ('stop', 'length') and len(one.token_ids) <= MAX_TOKENS,
                        'dense reader cap or finish reason differs')
                records.append({'blind_id': row['blind_id'], 'label': qualified.parse_label(one.text),
                                'finish_reason': one.finish_reason, 'generated_tokens': len(one.token_ids),
                                'raw_completion': one.text, 'prompt_tokens': len(output.prompt_token_ids),
                                'prompt_ids_sha256': digest(list(output.prompt_token_ids))})
            save(out / 'batches' / f'{start:07d}.json',
                 {'schema': 'generated-dense-label-batch-v1', 'binding_sha256': binding['sha256'],
                  'attempt_sha256': attempt['sha256'], 'job_id': os.environ['SLURM_JOB_ID'],
                  'start': start, 'stop': stop, 'wall_seconds': time.monotonic() - began, 'records': records})
            committed_labels(out, start, stop, binding, price['prompt_records'])
            print(json.dumps({'shard': shard_index, 'complete_through': stop, 'stop': shard['stop']}), flush=True)
        return seal_label_shard(directory, frame, price, shard_index)


def seal_label_shard(directory, frame, price, shard_index):
    out = directory / f'labels-shard-{shard_index:03d}'
    body = label_binding(frame, price, shard_index)
    binding = sealed(out / 'BINDING.json')
    require(binding == {**body, 'sha256': digest(body)}, 'dense label shard binding differs')
    shard, rows, receipts = price['shards'][shard_index], [], []
    expected_files = set()
    for start in range(shard['start'], shard['stop'], BATCH):
        batch = committed_labels(out, start, min(start + BATCH, shard['stop']), binding, price['prompt_records'])
        require(batch is not None, 'dense label shard is incomplete')
        rows.extend(batch['records']); receipts.append(batch['sha256'])
        expected_files.add(f'{start:07d}.json')
    require({p.name for p in (out / 'batches').glob('*.json')} == expected_files and
            {p.name for p in (out / 'attempts').glob('*.json')} == expected_files,
            'dense shard has foreign batch or attempt receipts')
    value = save(out / 'SUMMARY.json', {'schema': 'generated-dense-label-summary-v1',
                 'binding_sha256': binding['sha256'], 'sentences': len(rows), 'batch_sha256s': receipts,
                 'parsed_stop': sum(r['label'] in qualified.CLASSES and r['finish_reason'] == 'stop' for r in rows),
                 'generated_tokens': sum(r['generated_tokens'] for r in rows),
                 'status': 'COMPLETE_ALL_ASSIGNED_DENSE_LABELS'})
    return value, rows


def seal_labels(directory):
    frame, price = inputs(directory)
    records, summaries = [], []
    for index in range(len(price['shards'])):
        summary, rows = seal_label_shard(directory, frame, price, index)
        records.extend(rows); summaries.append(summary['sha256'])
    require([r['blind_id'] for r in records] == [r['blind_id'] for r in frame['records']],
            'dense full stage omits or reorders sentence IDs')
    result = save(directory / 'LABEL_COMPLETION.json',
                  {'schema': 'generated-dense-label-completion-v1', 'frame_sha256': frame['sha256'],
                   'price_sha256': price['sha256'], 'sentences': len(records),
                   'shard_summary_sha256s': summaries,
                   'parsed_stop': sum(r['label'] in qualified.CLASSES and r['finish_reason'] == 'stop' for r in records),
                   'generated_tokens': sum(r['generated_tokens'] for r in records),
                   'status': 'COMPLETE_ALL_ASSIGNED_DENSE_LABELS',
                   'scope': 'Operational qualified LLM class audit; no human ground truth or substantive-event assertion.'})
    return result, records


def id_routing_profiles(ids, window=WINDOW):
    """Reuse the native ID reducer; frequency kinematics are explicitly unweighted."""
    import numpy as np
    from moe_exp.correlation_pipeline.dynamics.routing import RoutingReducer
    ids = np.asarray(ids)
    require(ids.ndim == 3 and ids.shape[1:] == (40, 8) and np.issubdtype(ids.dtype, np.integer) and
            ((ids >= 0) & (ids < 256)).all() and (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all(),
            'route profile requires exact unique40-layer/top8 selections')
    rows_by_layer = []
    for layer in range(40):
        reducer = RoutingReducer(256, 8, layer=layer, model='qwen36-generated',
                                 config={'windows': [window], 'stride': window, 'lags': [1],
                                         'shuffle_replicates': 0})
        rows_by_layer.append(reducer.push(ids[:, layer, :]))
    n = len(ids) // window
    profiles, frequencies = [], []
    for index in range(n):
        layer_rows = [rows[index] for rows in rows_by_layer]
        distribution = np.zeros((40, 256), dtype=float)
        for layer, row in enumerate(layer_rows):
            for expert, rate in row['expert_rates'].items():
                distribution[layer, int(expert)] = rate / 8
        require(np.allclose(distribution.sum(1), 1), 'selection frequency normalization differs')
        frequencies.append(distribution)
        profiles.append({'token_start': index * window, 'token_end': (index + 1) * window,
                         'expert_set_turnover': float(np.mean([r['set_turnover'] for r in layer_rows])),
                         'selection_frequency_entropy': float(np.mean([r['selection_entropy'] for r in layer_rows]))})
    velocity = [float(np.abs(right - left).sum(axis=1).mean() / (2 * window))
                for left, right in zip(frequencies, frequencies[1:])]
    acceleration = [float((right - left) / window) for left, right in zip(velocity, velocity[1:])]
    adjacent = ((ids[1:, :, :, None] == ids[:-1, :, None, :]).any(axis=-1).mean()
                if len(ids) >= 2 else None)
    return {'reasoning_tokens': len(ids), 'window_tokens': window, 'complete_windows': n,
            'tail_tokens_outside_complete_windows': len(ids) - n * window,
            'mean_adjacent_expert_set_turnover': None if adjacent is None else float(1 - adjacent),
            'windows': profiles, 'selection_frequency_velocity': velocity,
            'selection_frequency_acceleration': acceleration,
            'selection_frequency_velocity_units': 'TV of normalized unweighted selection frequencies per emitted token',
            'selection_frequency_acceleration_units': 'Change in selection-frequency speed per emitted token',
            'gate_distribution_kinematics': None,
            'gate_distribution_status': 'UNAVAILABLE_RAW_ARCHIVES_CONTAIN_IDS_ONLY',
            'scope': 'Identity-use dynamics only. Equal weighting describes selections, not actual executed mixture/gate weights.'}


def joined_sentences(record, labels):
    """Reject token/text mismatches; invalid or incomplete labels remain explicit gaps."""
    require(digest(record['emitted_token_ids']) == record['emitted_token_ids_sha256'], 'generated token IDs changed')
    end = record['reasoning_end']
    expected_end = record['emitted_token_ids'].index(THINK_END) if THINK_END in record['emitted_token_ids'] else len(record['emitted_token_ids'])
    require(end == expected_end and len(record['offsets']) == len(record['token_owner']) == end,
            'reasoning closure/token offsets changed')
    require([r['sentence_index'] for r in record['sentences']] == list(range(len(record['sentences']))),
            'sentence units must remain dense before measurement gaps')
    valid, all_rows, owned = [], [], set()
    for row in record['sentences']:
        index = row['sentence_index']
        require(row['text'] == record['text'][row['char_start']:row['char_end']] and
                row['token_indices'] == [i for i, owner in enumerate(record['token_owner']) if owner == index] and
                not owned.intersection(row['token_indices']), 'sentence text or token ownership changed')
        owned.update(row['token_indices'])
        vote = labels[row['blind_id']]
        status = ('incomplete_tail' if not row['complete'] else 'unowned_sentence' if not row['token_indices'] else
                  'parsed_stop' if vote['label'] in qualified.CLASSES and vote['finish_reason'] == 'stop' else
                  'unparsed_or_capped_label')
        measured = {**row, 'label': vote['label'], 'label_finish_reason': vote['finish_reason'],
                    'measurement_status': status}
        all_rows.append(measured)
        if status == 'parsed_stop':
            valid.append(measured)
    require({owner for owner in record['token_owner'] if owner is not None} ==
            {row['sentence_index'] for row in record['sentences'] if row['token_indices']},
            'token owner refers to absent sentence')
    return valid, all_rows


def analyze(directory):
    import numpy as np
    from moe_exp.routing_control.analysis import class_summary
    from moe_exp.routing_control.trajectory_analysis_v1 import _combine_class_summaries
    require_step()
    frame, price = inputs(directory)
    completion, rated = seal_labels(directory)
    labels = {r['blind_id']: r for r in rated}
    arm_map = sealed(directory / 'ARM_MAP.json')
    require(arm_map['generation_manifest_sha256'] == frame['generation_manifest_sha256'] and
            arm_map['generation_stage_sha256'] == frame['generation_stage_sha256'] and
            arm_map['code_files'] == frame['code_files'] and
            len(arm_map['records']) == frame['assigned'] and
            {s['blind_id'] for r in arm_map['records'] for s in r['sentences']} == set(labels),
            'dense label/arm-map assignment join differs')
    require(sealed(Path(frame['generation_run_out']) / 'STAGE_COMPLETION.json')['sha256'] == frame['generation_stage_sha256'],
            'generation stage changed after blind labeling')
    for receipt in arm_map['generation_batch_receipts']:
        require(sealed(Path(receipt['assignment_path']))['sha256'] == receipt['assignment_sha256'] and
                sealed(Path(receipt['batch_path']))['sha256'] == receipt['batch_sha256'],
                'generation raw assignment/result changed after blind labeling')
    by_archive = defaultdict(list)
    for record in arm_map['records']:
        by_archive[record['array']['path']].append(record)
    request_results, groups, expert_counts, expert_tokens = [], defaultdict(list), {}, Counter()
    for path, records in by_archive.items():
        archive_sha = file_sha(path)
        require(all(r['array']['sha256'] == archive_sha for r in records), 'raw route archive changed before analysis')
        with np.load(path, allow_pickle=False) as archive:
            for record in records:
                valid, sentence_rows = joined_sentences(record, labels)
                key = (record['transition'], record['arm'])
                groups[key].append(valid)
                class_result = class_summary(valid)
                profiles = None
                if record['array']['available']:
                    routed = archive[record['array']['key']]
                    require(routed.shape == (len(record['emitted_token_ids']), 40, 8), 'route array/token join changed')
                    routed = routed[:record['reasoning_end']]
                    profiles = id_routing_profiles(routed)
                    labels_by_index = {r['sentence_index']: r['label'] for r in valid}
                    classes = [labels_by_index.get(owner, 'UNMEASURED') for owner in record['token_owner']]
                    for name in (*qualified.CLASSES, 'UNMEASURED'):
                        selected = routed[np.asarray([label == name for label in classes], dtype=bool)]
                        group = (*key, name)
                        if group not in expert_counts:
                            expert_counts[group] = np.zeros((40, 256), dtype=np.int64)
                        for layer in range(40):
                            expert_counts[group][layer] += np.bincount(selected[:, layer, :].reshape(-1), minlength=256)
                        expert_tokens[group] += len(selected)
                request_results.append({'uid': record['uid'], 'prefix_uid': record['prefix_uid'],
                    'family': record['family'], 'transition': record['transition'], 'arm': record['arm'], 'seed': record['seed'],
                    'generation_error': record['error'], 'finish_reason': record['finish_reason'], 'hit_cap': record['hit_cap'],
                    'emitted_tokens': len(record['emitted_token_ids']), 'reasoning_tokens': record['reasoning_end'],
                    'reasoning_closed': record['closed_reasoning'], 'sentence_count': len(sentence_rows),
                    'sentence_measurement_counts': dict(Counter(r['measurement_status'] for r in sentence_rows)),
                    'measured_class_counts': dict(Counter(r['label'] for r in valid)),
                    'class_summary': class_result, 'routing_identity_profiles': profiles,
                    'sparse_intervention_dose_by_rank': record['action_dose'],
                    'dose_scope': 'Saved aggregate action telemetry only; ranks retained separately, never summed as independent observations.',
                    'sentences': sentence_rows})
    require(len(request_results) == frame['assigned'] and
            {r['uid'] for r in request_results} == {r['uid'] for r in arm_map['records']}, 'analysis lost assigned continuations')
    counters_path = directory / 'EXPERT_SELECTION_COUNTS.npz'
    arrays, array_index = {}, []
    for index, (key, counts) in enumerate(sorted(expert_counts.items())):
        name = f'group{index:04d}'
        arrays[name] = counts
        array_index.append({'array_key': name, 'transition': key[0], 'arm': key[1], 'class': key[2],
                            'reasoning_token_observations': expert_tokens[key], 'shape': [40, 256],
                            'normalization': 'Counts / reasoning_token_observations = per-token selection rate; rates sum8 per layer.'})
    if counters_path.exists():
        with np.load(counters_path, allow_pickle=False) as old:
            require(set(old.files) == set(arrays) and all(np.array_equal(old[k], v) for k, v in arrays.items()),
                    'same-bound aggregate selection counts differ')
    else:
        temp = counters_path.with_name(counters_path.name + f'.partial-{os.getpid()}')
        with temp.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, counters_path)
    summaries = [{'transition': key[0], 'arm': key[1], 'assigned_requests': len(rows),
                  'class_summary': _combine_class_summaries(rows)} for key, rows in sorted(groups.items())]
    return save(directory / 'TRAJECTORY_RESULT.json', {
        'schema': 'generated-dense-trajectory-result-v1', 'frame_sha256': frame['sha256'],
        'arm_map_sha256': arm_map['sha256'], 'label_completion_sha256': completion['sha256'],
        'label_price_sha256': price['sha256'], 'code_files': measurement_code(),
        'assigned_continuations': frame['assigned'], 'assigned_sentences': frame['sentences'],
        'parsed_stop_labels': completion['parsed_stop'], 'horizon': frame['horizon'],
        'records': request_results, 'summaries_by_transition_arm': summaries,
        'expert_selection_counts_path': str(counters_path), 'expert_selection_counts_sha256': file_sha(counters_path),
        'expert_selection_array_index': array_index,
        'available_metrics': ['seven-class transitions', 'observed dwell with window censoring', 'reentry/loops within contiguous labeled blocks',
                              'expert identity turnover', 'expert selection frequencies/entropy', 'selection-frequency velocity/acceleration at64tokens',
                              'saved sparse intervention dose telemetry'],
        'unavailable_metrics': ['full router gate-weight trajectories', 'gate-distribution TV velocity/acceleration', 'router logit margins',
                                'per-sentence substantive behavior votes', 'accuracy or original-prompt utility'],
        'inference_status': 'DESCRIPTIVE_SECONDARY_MEASUREMENTS_NO_NEW_SIGNIFICANCE_CLAIMS',
        'limitations': 'LLM class audit may misclassify classes. Invalid/incomplete sentence labels break adjacency; no sparse gaps are joined. Counts are observed-window descriptors; independent semantic effects remain in their separate blinded reader analysis. No gate weights are inferred from expert IDs.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('path', 'prepare', 'label', 'analyze'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--generation-price', type=Path)
    parser.add_argument('--run-out', type=Path)
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    parser.add_argument('--shard-index', type=int)
    args = parser.parse_args()
    if args.stage in ('path', 'prepare'):
        require(args.manifest is not None and args.out_root is not None, 'manifest and out-root required')
        manifest = sealed(args.manifest)
        directory = output_path(manifest, args.out_root)
        if args.stage == 'path':
            print(directory)
            return
        require_step()
        require(args.generation_price is not None and args.run_out is not None, 'generation price and run output required')
        require(str(directory.resolve()).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/'), 'dense output must remain in owned work tree')
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'PREPARE.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            seal_generation_if_needed(manifest, args.manifest, args.run_out)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
            _, frame = build_frame(manifest, sealed(args.generation_price), args.run_out, directory, tokenizer)
            judge_tokenizer = AutoTokenizer.from_pretrained(qualified.MODEL, local_files_only=True)
            price = save(directory / 'LABEL_PRICE.json', price_frame(frame, judge_tokenizer, args.max_wall_seconds))
        print(json.dumps({'directory': str(directory), 'sentences': frame['sentences'],
                          'shards': len(price['shards']), 'GPU_h': price['complete_stage_GPU_h']}), flush=True)
        return
    require(args.directory is not None, 'prepared dense directory required')
    if args.stage == 'label':
        result, _ = label_shard(args.directory, args.shard_index)
    else:
        result = analyze(args.directory)
    print(json.dumps({'status': 'COMPLETE', 'stage': args.stage, 'sha256': result['sha256']}), flush=True)


if __name__ == '__main__':
    main()
