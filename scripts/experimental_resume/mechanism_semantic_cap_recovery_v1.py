"""Isolated, arm-blind larger-cap replay of every failed original semantic rating.

The 760-assignment primary artifact is immutable. Selection uses only the
original reader's finish/parser status. Replays are a measurement sensitivity,
not exact continuations: batch occupancy can change numerics despite CRN seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time

import run_boundary_micro_screen as base
import rate_mechanism_semantics_v1 as original
import analyze_mechanism_semantics_v1 as analysis

REPO = original.REPO
DOC = REPO / 'report/experimental-resume-v1'
RUNS = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')
SOURCE = RUNS / 'mechanism-1024-semantics-v1-40ea298e178d0863'
RATINGS = RUNS / 'mechanism-1024-readers-v1-40ea298e178d0863'
PLAN = DOC / 'MECHANISM_SEMANTIC_CAP_RECOVERY_MANIFEST_v1.json'
CAP = 4096
QUAL_COUNT = 4
BOOTSTRAPS = 50000
SAMPLER = {'temperature': .2, 'top_p': .95, 'enable_thinking': True,
           'reasoning_effort': 'low', 'max_tokens': CAP}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    return original.save(path, body, existing_ok=True)


def source_artifacts():
    return {'frame': SOURCE / 'BLIND_FRAME.json', 'price': SOURCE / 'READER_PRICE.json',
            'reader_summary': RATINGS / 'STAGE_SUMMARY.json', 'arm_map': SOURCE / 'ARM_MAP.json',
            'generation_manifest': DOC / 'MECHANISM_VALIDATION_MANIFEST_v1.json',
            'original_analysis': DOC / 'MECHANISM_1024_SEMANTIC_ITT_v1.json'}


def code_files():
    files = [Path(__file__), Path(original.__file__), Path(analysis.__file__),
             Path(base.__file__), original.RUBRIC]
    files += [Path(__file__).with_name(name) for name in
              ('prepare_mechanism_semantic_cap_recovery_v1.sbatch',
               'run_mechanism_semantic_cap_recovery_v1.sbatch',
               'analyze_mechanism_semantic_cap_recovery_v1.sbatch')]
    return {str(path.resolve()): base.file_sha(path) for path in files}


def select_failures(frame, votes):
    """No arm map, outcome, agreement or target-value selection is permitted."""
    selected = []
    for index, row in enumerate(frame['records']):
        original.messages(row)
        for reader in (0, 1):
            record = votes[row['blind_id']][reader]
            require(record['rating'] == original.parse_rating(record['raw_completion']),
                    'stored original parser result differs')
            valid = record['finish_reason'] == 'stop' and record['rating'] is not None
            if valid:
                continue
            require(record['finish_reason'] in ('length', 'stop'), 'unknown original finish')
            if record['finish_reason'] == 'length':
                require(record['generated_tokens'] == 1024, 'original length stop differs from cap')
            selected.append({'uid': row['blind_id'] + '|reader' + str(reader),
                             'blind_id': row['blind_id'], 'reader': reader,
                             'source_index': index, 'reader_input': row['reader_input'],
                             'seed': original.rating_seed(row['blind_id'], reader),
                             'selection_reason': 'length_cap' if record['finish_reason'] == 'length'
                                                 else 'malformed_stop',
                             'original_record_sha256': base.digest(record),
                             'original_raw_completion': record['raw_completion'],
                             'original_generated_tokens': record['generated_tokens']})
    return sorted(selected, key=lambda row: row['uid'])


def projected(requests, original_price, cap=CAP, wall=7200):
    """Reuse exact original prompt lengths; charge full caps, loads and retries."""
    prefill = original_price['bounded_prefill_tokens_per_second']
    decode = original_price['bounded_decode_tokens_per_second']
    repeat = original_price['repeat_work_factor']
    overhead = original_price['cold_load_seconds'] + original_price['shutdown_seconds']
    usable = wall - overhead - 900
    require(requests and usable > 0, 'empty or unpriceable replay stage')
    shards = []; start = 0; accumulated = 0.
    for offset in range(0, len(requests), original.BATCH):
        block = requests[offset:offset + original.BATCH]
        cost = repeat * (sum(r['prompt_tokens'] for r in block) / prefill + len(block) * cap / decode)
        require(cost <= usable, 'one complete replay batch exceeds job walltime')
        if accumulated and accumulated + cost > usable:
            shards.append({'start': start, 'end': offset, 'work_seconds': accumulated})
            start = offset; accumulated = 0.
        accumulated += cost
    shards.append({'start': start, 'end': len(requests), 'work_seconds': accumulated})
    reserve = max(1, math.ceil(len(shards) * (repeat - 1)))
    total = sum(s['work_seconds'] for s in shards) + (len(shards) + reserve) * overhead
    return {'shards': shards, 'ratings': len(requests), 'max_decode_tokens': len(requests) * cap,
            'prefill_tokens': sum(r['prompt_tokens'] for r in requests),
            'max_context_tokens': max(r['prompt_tokens'] for r in requests) + cap,
            'max_wall_seconds': wall, 'gpus': 2, 'batch_size': original.BATCH,
            'cold_load_seconds': original_price['cold_load_seconds'],
            'shutdown_seconds': original_price['shutdown_seconds'], 'reserve_seconds_per_job': 900,
            'repeat_work_factor': repeat, 'recovery_loads': reserve,
            'bounded_prefill_tps': prefill, 'bounded_decode_tps': decode,
            'estimated_complete_gpu_hours': 2 * total / 3600}


def prepare(path, ceiling):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz', 'prepare requires CPU Slurm')
    from transformers import AutoTokenizer
    from price_transition_ratings_v3 import count_prompt_tokens
    sources = {k: base.sealed(p) for k, p in source_artifacts().items()}
    frame, old_price = sources['frame'], sources['price']
    stage, votes = analysis.reader_votes(frame, old_price, RATINGS)
    requests = select_failures(frame, votes)
    require(Counter(r['selection_reason'] for r in requests) ==
            Counter(length_cap=179, malformed_stop=1), 'observed frozen failed-rating population differs')
    tokenizer = AutoTokenizer.from_pretrained(original.MODEL, local_files_only=True)
    for row in requests:
        ids = tokenizer.apply_chat_template(original.messages({'blind_id': row['blind_id'],
            'reader_input': row['reader_input']}), tokenize=True, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort='low')
        require(isinstance(ids, list) and all(type(t) is int for t in ids), 'unexpected tokenization type')
        row['prompt_tokens'] = count_prompt_tokens(ids)
        row['prompt_ids_sha256'] = base.digest(ids)
        row['messages_sha256'] = base.digest(original.messages({'blind_id': row['blind_id'],
                                                              'reader_input': row['reader_input']}))
        require(row['prompt_tokens'] == old_price['prompt_token_lengths'][row['source_index']] and
                row['prompt_tokens'] + CAP <= original.MAX_MODEL_LEN,
                'same-prompt tokenization differs or longer cap exceeds original context')
    qualification = [i for i, r in enumerate(requests) if r['selection_reason'] == 'length_cap'][:QUAL_COUNT]
    generation_price = projected(requests, old_price)
    qualification_price = projected([requests[i] for i in qualification], old_price, wall=3600)
    total = generation_price['estimated_complete_gpu_hours'] + qualification_price['estimated_complete_gpu_hours']
    require(total <= ceiling, 'complete qualification plus replay price exceeds ceiling')
    body = {'schema': 'mechanism-semantic-cap-recovery-manifest-v1',
            'status': 'FROZEN_PRICED_REQUIRES_GPU_QUALIFICATION',
            'sources': {k: {'path': str(p), 'sha256': sources[k]['sha256']}
                        for k, p in source_artifacts().items()},
            'source_reader_stage_sha256': stage['sha256'], 'code_files': code_files(),
            'model_snapshot': str(original.MODEL), 'sampler': SAMPLER,
            'engine_profile': {'tensor_parallel_size': 2, 'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16',
                'max_model_len': original.MAX_MODEL_LEN, 'max_num_seqs': 16, 'max_num_batched_tokens': 8192,
                'gpu_memory_utilization': .85, 'enforce_eager': True, 'generation_config': 'vllm',
                'language_model_only': True, 'attention_config': {'backend': 'FLASH_ATTN'}},
            'requests': requests, 'qualification_indices': qualification,
            'selection_rule': 'Every original non-valid-stop rating, using only finish/parser status; no arm-map or outcome filtering.',
            'qualification_rule': 'First four capped rating UIDs in lexicographic order; all execute, one exceeds original cap.',
            'qualification_price': qualification_price, 'generation_price': generation_price,
            'complete_gpu_hour_ceiling': ceiling, 'estimated_complete_gpu_hours': total,
            'analysis': {'original_endpoint_unchanged': True, 'contrasts': analysis.PAIRS,
                         'scopes': analysis.SCOPES, 'bootstrap_replicates': BOOTSTRAPS,
                         'bootstrap_seed': analysis.BOOT_SEED, 'multiplicity': 12,
                         'repaired_endpoint': 'both valid readers positive, all assigned rows; remaining failures false',
                         'capped_only_sensitivity': '179 cap replays only; descriptive point estimates without additional tests'},
            'limitations': ['Post-result measurement sensitivity, never replacement of sealed primary.',
                'Same prompt/model/seed/profile but changed batch occupancy; exact replay prefix equality is measured, not assumed.',
                '4096 caps may remain unresolved; no automatic further extension or semantic reader redesign.']}
    return save(path, body)


def validate(plan):
    require(plan['schema'] == 'mechanism-semantic-cap-recovery-manifest-v1' and
            plan['sampler'] == SAMPLER and plan['model_snapshot'] == str(original.MODEL) and
            plan['code_files'] == code_files(), 'recovery plan/code/profile changed')
    for source in plan['sources'].values():
        require(base.sealed(Path(source['path']))['sha256'] == source['sha256'], 'sealed original source changed')
    rows = plan['requests']; require(len({r['uid'] for r in rows}) == len(rows), 'duplicate recovery UID')
    require(rows == sorted(rows, key=lambda r: r['uid']) and
            Counter(r['selection_reason'] for r in rows) == Counter(length_cap=179, malformed_stop=1),
            'recovery selection differs')
    for row in rows:
        require(row['seed'] == original.rating_seed(row['blind_id'], row['reader']) and
                row['uid'] == row['blind_id'] + '|reader' + str(row['reader']) and
                row['prompt_tokens'] + CAP <= original.MAX_MODEL_LEN and
                base.digest(original.messages({'blind_id': row['blind_id'], 'reader_input': row['reader_input']})) ==
                row['messages_sha256'], 'replay prompt/seed/context differs')
    old_price = base.sealed(Path(plan['sources']['price']['path']))
    require(plan['qualification_indices'] == [i for i, r in enumerate(rows) if r['selection_reason'] == 'length_cap'][:QUAL_COUNT] and
            plan['generation_price'] == projected(rows, old_price) and
            plan['qualification_price'] == projected([rows[i] for i in plan['qualification_indices']], old_price, wall=3600),
            'complete price or fixed qualification differs')
    require(plan['estimated_complete_gpu_hours'] == sum(plan[k]['estimated_complete_gpu_hours'] for k in
            ('generation_price', 'qualification_price')) and
            plan['estimated_complete_gpu_hours'] <= plan['complete_gpu_hour_ceiling'], 'unpriced recovery')


def indices(plan, mode, shard):
    if mode == 'qualification':
        require(shard == 0, 'qualification is one fixed job')
        return plan['qualification_indices']
    require(mode == 'generation' and 0 <= shard < len(plan['generation_price']['shards']), 'invalid recovery shard')
    s = plan['generation_price']['shards'][shard]
    return list(range(s['start'], s['end']))


def directory(out, mode, shard):
    return out / ('qualification' if mode == 'qualification' else f'shard-{shard:03d}')


def binding_for(plan, mode, shard):
    return {'schema': 'mechanism-semantic-cap-recovery-binding-v1', 'manifest_sha256': plan['sha256'],
            'mode': mode, 'shard': shard, 'indices': indices(plan, mode, shard), 'sampler': SAMPLER}


def completed(plan, out, mode, shard):
    dest = directory(out, mode, shard); binding = base.sealed(dest / 'BINDING.json')
    expected = binding_for(plan, mode, shard)
    require(binding == {**expected, 'sha256': base.digest(expected)}, 'recovery output binding differs')
    records = []; hashes = []
    for start in range(0, len(expected['indices']), original.BATCH):
        block = expected['indices'][start:start + original.BATCH]
        item = base.sealed(dest / f'batch-{start:04d}.json')
        attempt = base.sealed(dest / f'attempt-{start:04d}-{item["attempt_index"]:03d}.json')
        require(item['binding_sha256'] == binding['sha256'] and item['indices'] == block and
                item['attempt_sha256'] == attempt['sha256'] and attempt['binding_sha256'] == binding['sha256'] and
                attempt['indices'] == block and attempt['seeds'] == [plan['requests'][i]['seed'] for i in block] and
                [r['uid'] for r in item['records']] ==
                [plan['requests'][i]['uid'] for i in block], 'recovery batch/attempt identity differs')
        hashes.append(item['sha256'])
        for record, index in zip(item['records'], block, strict=True):
            source = plan['requests'][index]
            require(record['source_record_sha256'] == source['original_record_sha256'] and
                    record['reader'] == source['reader'] and record['blind_id'] == source['blind_id'] and
                    record['prompt_ids_sha256'] == source['prompt_ids_sha256'] and
                    record['rating'] == original.parse_rating(record['raw_completion']) and
                    0 <= record['generated_tokens'] <= CAP and
                    record['generated_tokens'] == len(record['token_ids']) and
                    record['finish_reason'] in ('stop', 'length') and
                    (record['finish_reason'] != 'length' or record['generated_tokens'] == CAP),
                    'recovery cap/parser/prompt/record differs')
        records.extend(item['records'])
    return records, hashes


def qualification(plan, out):
    value = base.sealed(out / 'QUALIFICATION.json')
    records, hashes = completed(plan, out, 'qualification', 0)
    require(value['manifest_sha256'] == plan['sha256'] and value['pass'] is True and
            value['batch_sha256s'] == hashes and len(records) == QUAL_COUNT and
            any(r['generated_tokens'] > 1024 for r in records), 'larger-cap fixture did not qualify')
    return value


def run(plan, out, mode, shard, resume, deadline):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and
            not socket.gethostname().startswith('login'), 'replay inference requires GPU Slurm')
    if mode == 'generation': qualification(plan, out)
    dest = directory(out, mode, shard); dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / 'WRITER.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = save(dest / 'BINDING.json', binding_for(plan, mode, shard))
    pending = []
    for start in range(0, len(binding['indices']), original.BATCH):
        block = binding['indices'][start:start + original.BATCH]
        if (dest / f'batch-{start:04d}.json').exists(): continue
        ledger = sorted(dest.glob(f'attempt-{start:04d}-*.json'))
        require(not ledger or resume, 'uncommitted replay requires explicit --resume-uncommitted')
        for number, path in enumerate(ledger):
            old = base.sealed(path)
            require(old['binding_sha256'] == binding['sha256'] and old['indices'] == block and
                    old['attempt_index'] == number, 'replay attempt ledger differs')
        pending.append((start, block, len(ledger)))
    if pending:
        price = plan['qualification_price'] if mode == 'qualification' else plan['generation_price']
        def needed(block):
            return price['repeat_work_factor'] * (sum(plan['requests'][i]['prompt_tokens'] for i in block) /
                price['bounded_prefill_tps'] + len(block) * CAP / price['bounded_decode_tps'])
        require(not deadline or time.time() + price['cold_load_seconds'] + needed(pending[0][1]) +
                price['shutdown_seconds'] + 240 < deadline, 'insufficient cold-load/first-batch reserve')
        from vllm import LLM, SamplingParams
        model = LLM(model=str(original.MODEL), tokenizer=str(original.MODEL), **plan['engine_profile'])
        for start, block, attempt_index in pending:
            require(not deadline or time.time() + needed(block) + price['shutdown_seconds'] + 240 < deadline,
                    'checkpointed before replay batch deadline')
            reqs = [plan['requests'][i] for i in block]
            attempt = save(dest / f'attempt-{start:04d}-{attempt_index:03d}.json',
                {'binding_sha256': binding['sha256'], 'indices': block, 'attempt_index': attempt_index,
                 'job_id': os.environ['SLURM_JOB_ID'], 'seeds': [r['seed'] for r in reqs],
                 'state': 'durable_before_inference'})
            started = time.monotonic()
            try:
                outputs = model.chat([original.messages({'blind_id': r['blind_id'], 'reader_input': r['reader_input']})
                                      for r in reqs],
                    sampling_params=[SamplingParams(temperature=.2, top_p=.95, max_tokens=CAP, seed=r['seed']) for r in reqs],
                    chat_template_kwargs={'enable_thinking': True, 'reasoning_effort': 'low'}, use_tqdm=False)
                require(len(outputs) == len(reqs), 'replay output count differs')
                records = []
                for req, output in zip(reqs, outputs, strict=True):
                    completion = output.outputs[0]; old = req['original_raw_completion']; new = completion.text
                    require(base.digest(output.prompt_token_ids) == req['prompt_ids_sha256'], 'exact replay prompt IDs differ')
                    lcp = 0
                    for a, b in zip(old, new):
                        if a != b: break
                        lcp += 1
                    records.append({'uid': req['uid'], 'blind_id': req['blind_id'], 'reader': req['reader'],
                        'source_record_sha256': req['original_record_sha256'], 'rating': original.parse_rating(new),
                        'finish_reason': completion.finish_reason, 'generated_tokens': len(completion.token_ids),
                        'token_ids': list(completion.token_ids), 'prompt_ids_sha256': base.digest(output.prompt_token_ids),
                        'raw_completion': new, 'old_completion_is_exact_prefix': new.startswith(old),
                        'longest_common_prefix_characters': lcp, 'old_completion_characters': len(old)})
                save(dest / f'batch-{start:04d}.json', {'binding_sha256': binding['sha256'], 'indices': block,
                    'attempt_index': attempt_index, 'attempt_sha256': attempt['sha256'], 'records': records,
                    'elapsed_seconds': time.monotonic() - started})
            except Exception as exc:
                save(dest / f'failure-{start:04d}-{attempt_index:03d}.json', {'attempt_sha256': attempt['sha256'],
                    'exception_type': type(exc).__name__, 'message': str(exc)[:1000]})
                raise
            print(json.dumps({'mode': mode, 'shard': shard, 'batch': start, 'records': len(records)}), flush=True)
    records, hashes = completed(plan, out, mode, shard)
    body = {'schema': 'mechanism-semantic-cap-recovery-summary-v1', 'manifest_sha256': plan['sha256'],
            'mode': mode, 'shard': shard, 'records': len(records), 'batch_sha256s': hashes,
            'valid_stop': sum(r['finish_reason'] == 'stop' and r['rating'] is not None for r in records),
            'over_original_cap': sum(r['generated_tokens'] > 1024 for r in records),
            'old_prefix_exact': sum(r['old_completion_is_exact_prefix'] for r in records)}
    save(dest / 'SUMMARY.json', body)
    if mode == 'qualification':
        require(body['over_original_cap'] > 0, 'larger cap not exercised; qualification failed')
        save(out / 'QUALIFICATION.json', {**body, 'pass': True})


def point_contrasts(records):
    starts = analysis._cell_values(records, 'both_positive'); result = []
    for scope in analysis.SCOPES:
        subset = [r for r in starts if scope == 'all' or r['transition'] == scope]
        for left, right in analysis.PAIRS:
            result.append({'scope': scope, 'arm': left, 'reference': right,
                           'estimate': sum(r['means'][left] - r['means'][right] for r in subset) / len(subset)})
    return result


def analyze(plan, out, dest):
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz',
            'family-bootstrap sensitivity requires CPU Slurm')
    qual = qualification(plan, out); new = {}; hashes = []
    for shard in range(len(plan['generation_price']['shards'])):
        records, batch_hashes = completed(plan, out, 'generation', shard); hashes.extend(batch_hashes)
        for record in records:
            require(record['uid'] not in new, 'duplicate repaired rating UID')
            new[record['uid']] = record
    require(set(new) == {r['uid'] for r in plan['requests']}, 'incomplete all-invalid repair population')
    frame = base.sealed(Path(plan['sources']['frame']['path'])); price = base.sealed(Path(plan['sources']['price']['path']))
    stage, old_votes = analysis.reader_votes(frame, price, RATINGS)
    all_votes = copy.deepcopy(old_votes); cap_votes = copy.deepcopy(old_votes)
    for req in plan['requests']:
        old = old_votes[req['blind_id']][req['reader']]
        require(base.digest(old) == req['original_record_sha256'], 'original individual rating changed')
        all_votes[req['blind_id']][req['reader']] = new[req['uid']]
        if req['selection_reason'] == 'length_cap': cap_votes[req['blind_id']][req['reader']] = new[req['uid']]
    generation = base.sealed(Path(plan['sources']['generation_manifest']['path']))
    arm_map = base.sealed(Path(plan['sources']['arm_map']['path']))
    repaired = analysis.join(generation, arm_map, frame, all_votes)
    capped = analysis.join(generation, arm_map, frame, cap_votes)
    primary = base.sealed(Path(plan['sources']['original_analysis']['path']))
    require([r['uid'] for r in repaired] == [r['uid'] for r in primary['records']], 'assigned ITT population changed')
    valid_before = sum(r['finish_reason'] == 'stop' and r['rating'] is not None
                       for pair in old_votes.values() for r in pair.values())
    valid_after = sum(r['finish_reason'] == 'stop' and r['rating'] is not None
                      for pair in all_votes.values() for r in pair.values())
    result = {'schema': 'mechanism-semantic-cap-recovery-sensitivity-v1', 'manifest_sha256': plan['sha256'],
              'qualification_sha256': qual['sha256'], 'generation_batch_sha256s': hashes,
              'original_analysis_sha256': primary['sha256'], 'original_reader_stage_sha256': stage['sha256'],
              'status': 'COMPLETE_SEPARATE_MEASUREMENT_SENSITIVITY', 'assigned': len(repaired),
              'before_after_coverage': {'assigned_rating_pairs': len(repaired),
                  'assigned_ratings': 2 * len(frame['records']),
                  'valid_ratings_before': valid_before, 'valid_ratings_after': valid_after,
                  'valid_pairs_before': sum(r['reader_pair_valid'] for r in primary['records']),
                  'valid_pairs_after': sum(r['reader_pair_valid'] for r in repaired)},
              'repaired_ratings': len(new), 'recovery_summary': {
                  reason: {'assigned': len(sub), 'valid_stop': sum(r['finish_reason'] == 'stop' and r['rating'] is not None for r in sub),
                           'still_capped': sum(r['finish_reason'] == 'length' for r in sub),
                           'old_prefix_exact': sum(r['old_completion_is_exact_prefix'] for r in sub),
                           'over_original_cap': sum(r['generated_tokens'] > 1024 for r in sub)}
                  for reason in ('length_cap', 'malformed_stop')
                  for sub in [[new[r['uid']] for r in plan['requests'] if r['selection_reason'] == reason]]},
              'arm_summary': analysis.arm_summary(repaired),
              'original_arm_summary': primary['arm_summary'],
              'sensitivity_itt': analysis.clustered_contrasts(repaired, n_boot=BOOTSTRAPS, seed=analysis.BOOT_SEED),
              '179_capped_only_descriptive': point_contrasts(capped),
              'original_primary_unchanged': primary['primary_itt'], 'records': repaired,
              'limitations': plan['limitations']}
    return save(dest, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=PLAN); p.add_argument('--prepare', action='store_true')
    p.add_argument('--gpu-hour-ceiling', type=float, default=20.)
    p.add_argument('--out', type=Path); p.add_argument('--mode', choices=('qualification', 'generation'))
    p.add_argument('--shard-index', type=int, default=0); p.add_argument('--resume-uncommitted', action='store_true')
    p.add_argument('--deadline-epoch', type=float, default=0); p.add_argument('--cpu-preflight', action='store_true')
    p.add_argument('--analyze', type=Path); args = p.parse_args()
    if args.prepare:
        value = prepare(args.plan, args.gpu_hour_ceiling)
        print(json.dumps({'manifest_sha256': value['sha256'], 'ratings': len(value['requests']),
                          'shards': len(value['generation_price']['shards']),
                          'estimated_complete_gpu_hours': value['estimated_complete_gpu_hours']}), flush=True); return
    plan = base.sealed(args.plan); validate(plan)
    if args.cpu_preflight:
        print(json.dumps({'status': 'PASS_CPU_PREFLIGHT', 'manifest_sha256': plan['sha256']})); return
    out = args.out or RUNS / ('mechanism-semantic-cap-recovery-v1-' + plan['sha256'][:16])
    if args.analyze:
        value = analyze(plan, out, args.analyze)
        print(json.dumps({'status': value['status'], 'sha256': value['sha256']}), flush=True); return
    require(args.mode is not None, 'explicit priced mode required')
    run(plan, out, args.mode, args.shard_index, args.resume_uncommitted, args.deadline_epoch)


if __name__ == '__main__':
    main()
