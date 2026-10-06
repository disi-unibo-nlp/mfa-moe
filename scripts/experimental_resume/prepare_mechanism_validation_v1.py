"""Seal accepted, family-disjoint mechanism starts and price the whole 1024-token stage.

The native trace replay is CPU work and this program runs only in a Slurm step.
No failed reader start is replaced outside the three preselected candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import prepare_mechanism_start_frame_v1 as start
import rate_mechanism_start_readers_v2 as rating_v2
import run_boundary_micro_screen as base

REPO = start.REPO
DOC = REPO / 'report/experimental-resume-v1'
FAMILY = DOC / 'family-freeze.json'
DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
REFERENCE_PRICE = DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v4.json'
MANIFEST = DOC / 'MECHANISM_VALIDATION_MANIFEST_v1.json'
PRICE = DOC / 'MECHANISM_VALIDATION_PRICE_v1.json'
QUAL_MANIFEST = DOC / 'MECHANISM_ENGINE_1024_QUAL_MANIFEST_v1.json'
QUAL_RESULT = DOC / 'MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json'
RATINGS = start.FRAME.parent / 'ratings-v2-7497e6aafbaf689c-497b2e8f57646826'
FROZEN_SELECTION_SHA = 'c9859eedced42a41e37577b7ebe64f2a1fd5d24cb200ebcd9fc6b9db4c99d721'
ARMS = ('native', 'native_duplicate', 'target_bias1', 'random_bias1')
SEEDS = (0, 1)
HORIZON = 1024
PROFILE = {'max_num_seqs': 1, 'enforce_eager': True, 'VLLM_BATCH_INVARIANT': 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_accepted_by_family_transition(selection_rows, rating_rows):
    require([r['uid'] for r in rating_rows] ==
            [r['uid'] for r in selection_rows], 'reader rows differ from frozen order')
    positive = {r['uid'] for r in rating_rows if all(
        vote['rating'] == {'start': True} and vote['finish_reason'] == 'stop'
        for vote in r['readers'])}
    grouped = defaultdict(list)
    for row in selection_rows:
        grouped[row['family'], row['transition']].append(row)
    chosen = [next((row for row in candidates if row['uid'] in positive), None)
              for _, candidates in sorted(grouped.items())]
    return [row for row in chosen if row is not None], len(positive)


def accepted_starts(selection, frame, ratings_dir):
    """Audit all 454 assignments before choosing the first accepted start."""
    ratings_dir = Path(ratings_dir)
    require(ratings_dir.resolve() == RATINGS.resolve(),
            'ratings source is not the launched v2 audit directory')
    require(selection['schema'] == 'mechanism-start-selection-v1' and
            selection['sha256'] == FROZEN_SELECTION_SHA and
            selection['families'] == 128 and len(selection['records']) == 454,
            'changed frozen candidate population')
    require(frame['schema'] == 'mechanism-start-frame-v1' and
            frame['selection_sha256'] == selection['sha256'] and
            [r['uid'] for r in frame['records']] ==
            [r['uid'] for r in selection['records']], 'start frame differs')
    return audited_reader_results(selection, frame, ratings_dir)


def audited_reader_results(selection, frame, ratings_dir):
    """Verify every committed v2 reader result and its physical attempt chain."""
    binding = base.sealed(ratings_dir / 'BINDING.json')
    summary = base.sealed(ratings_dir / 'SUMMARY.json')
    rating_price = base.sealed(rating_v2.PRICE)
    require(rating_price['schema'] == 'mechanism-start-reader-price-v2' and
            rating_price['status'] == 'PASS_COMPLETE_20_GPUH' and
            binding['schema'] == 'mechanism-start-reader-binding-v2' and
            binding['frame_sha256'] == frame['sha256'] and
            binding['selection_sha256'] == selection['sha256'] and
            binding['price_sha256'] == rating_price['sha256'] and
            binding['driver_sha256'] == base.file_sha(rating_v2.__file__) and
            binding['message_source_sha256'] == base.file_sha(rating_v2.prior_reader.__file__) and
            binding['rubric_sha256'] == base.file_sha(rating_v2.prior_reader.RUBRIC) and
            binding['model_snapshot'] == str(rating_v2.prior_reader.MODEL) and
            binding['model_revision'] == rating_v2.prior_reader.MODEL.name and
            binding['sampler'] == {'temperature': .2, 'top_p': .95,
                                   'thinking': True, 'reasoning_effort': 'low'} and
            binding['visible_input_allowlist'] == frame['visible_input_allowlist'] and
            binding['batch_size'] == 16 and binding['readers'] == 2 and
            binding['max_tokens_per_rating'] == 1024 and
            summary['schema'] == 'mechanism-start-reader-summary-v2' and
            summary['binding_sha256'] == binding['sha256'] and
            summary['counts']['rows'] == len(selection['records']) and
            summary['counts']['ratings'] == 2 * len(selection['records']),
            'incomplete or rebound two-reader start audit')
    completed, timings, attempts = [], [], []
    committed = set()
    for offset in range(0, len(selection['records']), binding['batch_size']):
        block = selection['records'][offset:offset + binding['batch_size']]
        saved = base.sealed(ratings_dir / 'batches' / f'{offset:06d}.json')
        require(saved['schema'] == 'mechanism-start-rating-batch-v2' and
                saved['binding_sha256'] == binding['sha256'] and
                saved['start'] == offset and
                [r['uid'] for r in saved['records']] == [r['uid'] for r in block],
                'incomplete or reordered start-rating batch')
        reader_results = []
        for reader_index in SEEDS:
            individual = rating_v2.committed_reader(
                ratings_dir, offset, reader_index, block, binding['sha256'])
            require(individual is not None, 'missing committed reader result')
            ledger = rating_v2.attempt_ledger(
                ratings_dir, offset, reader_index, block, binding['sha256'])
            require(all(a['schema'] == 'mechanism-start-reader-attempt-v2' and
                        a['state'] == 'started_before_model_chat' for a in ledger),
                    'attempt ledger state differs')
            attempts.extend(a['sha256'] for a in ledger)
            committed.add(individual['attempt_sha256'])
            for i, row in enumerate(block):
                receipt = base.sealed(ratings_dir / 'assignments' /
                                      f"{row['uid']}-reader{reader_index}-attempt"
                                      f"{individual['attempt_index']:03d}.json")
                require(receipt['schema'] == 'mechanism-start-rating-assignment-v2' and
                        receipt['state'] == 'attempted_before_model_chat' and
                        receipt['binding_sha256'] == binding['sha256'] and
                        receipt['uid'] == row['uid'] and receipt['reader'] == reader_index and
                        receipt['start'] == offset and
                        receipt['attempt_sha256'] == individual['attempt_sha256'] and
                        receipt['attempt_index'] == individual['attempt_index'] and
                        saved['records'][i]['transition'] == row['transition'] and
                        saved['records'][i]['readers'][reader_index] ==
                        individual['records'][i]['result'],
                        'reader result or attempt receipt differs')
            reader_results.append(individual)
        require(saved['reader_timings'] == [r['timing'] for r in reader_results],
                'combined batch timings differ from committed readers')
        completed.extend(saved['records'])
        timings.extend(saved['reader_timings'])
    require(len(completed) == len(selection['records']) and
            [r['uid'] for r in completed] == [r['uid'] for r in selection['records']],
            'start audit lacks full ordered coverage')
    require(summary['reader_timings'] == timings and
            summary['attempt_sha256s'] == attempts and
            summary['counts']['physical_attempts_recorded'] == len(attempts) and
            summary['counts']['uncommitted_attempts_recorded'] ==
            sum(sha not in committed for sha in attempts),
            'attempt ledger differs from completed v2 summary')
    chosen, positive_count = first_accepted_by_family_transition(
        selection['records'], completed)
    require(positive_count == summary['counts']['start_agree_true'],
            'accepted start count differs from sealed summary')
    return chosen, binding, summary


def replay_rows(chosen, frame, attempt_index):
    """Recover native token IDs and verify them against the blind frame."""
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    by_uid = {row['uid']: row for row in frame['records']}
    replay_cache = {}
    rows = []
    for row in chosen:
        attempt = row['attempt_id']
        if attempt not in replay_cache:
            source = attempt_index[attempt]
            trace = start.trace_at(source['source_location'])
            require(trace_digest(TraceRecord(**trace)) == source['trace_sha256'],
                    'native trace digest differs')
            replay_cache[attempt] = (trace['metadata']['token_replay'],
                                     source['trace_sha256'], trace['cot_text'])
        replay, trace_sha, text = replay_cache[attempt]
        ids = replay['completion_token_ids'][:row['prefix_tokens']]
        prompt_ids = replay['prompt_token_ids']
        meta = by_uid[row['uid']]['analysis_meta']
        require(len(ids) == row['prefix_tokens'] and
                len(ids) <= 8192 and
                replay['tokenizer_sha256'] == row['tokenizer_sha256'] ==
                meta['tokenizer_sha256'] and
                base.digest(ids) == meta['prefix_ids_sha256'] and
                base.digest(prompt_ids) == meta['prompt_ids_sha256'] and
                trace_sha == meta['trace_sha256'] and
                hashlib.sha256(text[:row['prefix_end_char']].encode()).hexdigest() ==
                meta['prefix_text_sha256'] and
                text[:row['prefix_end_char']] ==
                by_uid[row['uid']]['reader_input']['emitted_prefix'] and
                replay['completion_offsets'][len(ids) - 1][1] == row['prefix_end_char'],
                'exact native prefix differs from reader input')
        require(all(type(token) is int and token >= 0 for token in prompt_ids + ids),
                'invalid native token ID')
        rows.append({'uid': row['uid'], 'family': row['family'],
                     'transition': row['transition'], 'attempt_id': attempt,
                     'prefix_tokens': len(ids), 'prompt_ids': prompt_ids,
                     'prefix_ids': ids, 'prompt_ids_sha256': meta['prompt_ids_sha256'],
                     'prefix_ids_sha256': meta['prefix_ids_sha256']})
    return rows


def schedule(rows):
    """Fixed random set per family and balanced four-arm positions."""
    families = sorted({r['family'] for r in rows})
    sets = {family: i % 4 for i, family in enumerate(sorted(
        families, key=lambda family: base.digest(['mechanism-random-v1', family])))}
    orders = {}
    offset = 0
    for transition in ('candidate_to_verify', 'approach_to_commit'):
        blocks = [(r['uid'], seed) for r in rows if r['transition'] == transition
                  for seed in SEEDS]
        for i, key in enumerate(sorted(blocks, key=lambda item: base.digest(
                ['mechanism-order-v1', transition, *item]))):
            shift = (offset + i) % 4
            orders[f'{key[0]}|{key[1]}'] = list(ARMS[shift:] + ARMS[:shift])
        offset += len(blocks)
    for transition in ('candidate_to_verify', 'approach_to_commit', 'all'):
        keys = [f"{r['uid']}|{seed}" for r in rows
                if transition == 'all' or r['transition'] == transition for seed in SEEDS]
        if not keys:
            continue
        counts = Counter((name, position) for key in keys
                         for position, name in enumerate(orders[key]))
        require(max(counts.values()) - min(counts.values()) <= 1,
                'four-arm positions are not counterbalanced')
    return sets, orders


def make_manifest(rows, selection, frame, binding, summary, family, dictionary, ratings_dir):
    pool = family['new_parent_pools']['parent_pools']
    require(set(selection['family_pool']) == set(pool['mechanism']) and
            len(set(pool['mechanism'])) == 128 and
            not set(pool['discovery']) & set(pool['mechanism']) and
            not set(pool['utility']) & set(pool['mechanism']) and
            {r['family'] for r in rows} <= set(pool['mechanism']),
            'mechanism enrollment overlaps another parent pool')
    require(dictionary['schema'] == 'routing-discovery-action-dictionary-v1' and
            {t['transition'] for t in dictionary['target_templates']} ==
            {'candidate_to_verify', 'approach_to_commit'},
            'unsupported action dictionary')
    representatives = family['new_parent_pools']['representative_questions']
    rows = [{**row, 'question': representatives[row['family']]} for row in rows]
    random_sets, orders = schedule(rows)
    accepted_families = len({r['family'] for r in rows})
    registered = accepted_families == 128
    requests = len(rows) * len(SEEDS) * len(ARMS)
    prefill = sum(len(r['prompt_ids']) + len(r['prefix_ids']) for r in rows) * 8
    from run_mechanism_validation_v1 import __file__ as runner
    return {'schema': 'mechanism-validation-manifest-v1',
            'selection_sha256': selection['sha256'], 'frame_sha256': frame['sha256'],
            'rating_binding_sha256': binding['sha256'],
            'rating_summary_sha256': summary['sha256'],
            'ratings_dir': str(ratings_dir.resolve()),
            'family_freeze_sha256': family['sha256'],
            'action_dictionary_sha256': dictionary['sha256'],
            'preparation_driver_sha256': base.file_sha(__file__),
            'generation_driver_sha256': base.file_sha(runner),
            'base_driver_sha256': base.file_sha(base.__file__),
            'base_tree_sha256': base.REQUIRED_BASE_TREE,
            'rows': rows, 'arms': list(ARMS), 'seeds': list(SEEDS),
            'frozen_family_pool_size': 128,
            'accepted_family_count': accepted_families,
            'accepted_start_count': len(rows),
            'registered_128_family_feasibility': 'PASS' if registered else 'FAIL',
            'stage_label': ('registered_128_family_mechanism_validation' if registered
                            else 'exploratory_accepted_family_subset'),
            'random_set_by_family': random_sets, 'arm_order_by_uid_seed': orders,
            'horizon': HORIZON, 'pulse_slots': [0], 'pulse_width': 256,
            'template_status': 'one_action_only_no_frozen_two_action_template',
            'expected_requests': requests, 'expected_prefill_tokens': prefill,
            'maximum_decode_tokens': requests * HORIZON,
            'maximum_context_tokens': max(len(r['prompt_ids']) +
                                          len(r['prefix_ids']) + HORIZON for r in rows),
            'analysis_unit': 'family-clustered intention-to-treat; all assigned cap, closure, failure and nonfire outcomes retained',
            'claim_limit': ('One local action tests a causal response at accepted starts. '
                            'If fewer than 128 families were accepted, the registered '
                            '128-family validation is infeasible and this is an exploratory subset. '
                            'No second transition or action-order claim.')}


def validate_qualification(result):
    """Consume the separate sealed 1024-token serial/eager engine qualification."""
    qual_manifest = base.sealed(QUAL_MANIFEST)
    worker = base.sealed(base.WORKER_QUAL)
    require(result['schema'] == 'routing-mechanism-serial-eager-1024-qualification-v1' and
            result['pass'] is True and
            qual_manifest['schema'] ==
            'routing-mechanism-serial-eager-1024-qual-manifest-v1' and
            qual_manifest['bias'] == 1.0 and
            qual_manifest['four_arm_order'] ==
            ['native', 'target', 'random', 'native_duplicate'] and
            result['qualification_manifest_sha256'] == qual_manifest['sha256'] and
            result['qualified_worker_sha256'] == worker['sha256'] and
            result['worker_code_digest'] == worker['worker_code_digest'] ==
            qual_manifest['worker_code_digest'] and
            result['base_tree_sha256'] == base.REQUIRED_BASE_TREE and
            result['engine_profile'] == PROFILE and
            result['max_tokens'] == HORIZON and
            result['pulse_slots'] == [0, 512] and
            result['pulse_length'] == 256 and
            all(result[key] is True for key in (
                'same_prefix_four_arm_pass', 'native_isolation_pass',
                'ordered_pulse_pass', 'preemption_recompute_pass', 'closure_pass',
                'inherited_h14_force_pass')) and
            result['qualification_driver_sha256'] == base.file_sha(
                REPO / 'scripts/experimental_resume/qualify_mechanism_engine_1024_v1.py') and
            result['code_files'] == qual_manifest['code_files'] and
            result['code_files'] and
            all(base.file_sha(path) == sha for path, sha in result['code_files'].items()) and
            result['source_h14_file_sha256'] ==
            qual_manifest['source_h14']['file_sha256'] ==
            base.file_sha(qual_manifest['source_h14']['path']) and
            result['source_ordered_qualification_sha256'] ==
            qual_manifest['source_ordered_qualification']['sha256'] and
            result['source_serial_qualification_sha256'] ==
            qual_manifest['source_serial_qualification']['sha256'] and
            result['requests'] == 12 and len(result['checks']) == 12 and
            all(check['pass'] is True for check in result['checks']) and
            result['job_id'],
            '1024-token serial/eager qualification is missing or differs')
    return result


def price_stage(manifest, reference, max_wall_seconds, qualification=None):
    require(reference['schema'] == 'routing-eligible-micro-serial-price-v4',
            'unexpected measured serial reference')
    require(type(max_wall_seconds) is int and max_wall_seconds >= 3600,
            'live partition wall limit must be supplied in seconds')
    load = reference['cold_load_seconds']
    prefill = manifest['expected_prefill_tokens'] / reference['serial_prefill_stress_tokens_per_second']
    decode = manifest['maximum_decode_tokens'] / reference['serial_decode_stress_tokens_per_second']
    shutdown = reference['shutdown_seconds']
    repeat = reference['repeat_factor']
    work = repeat * (prefill + decode)
    reserve = 900
    usable_per_job = max_wall_seconds - load - shutdown - reserve
    require(usable_per_job > 0, 'measured load and shutdown exceed one job')
    shards = []
    shard_start, shard_work = 0, 0.0
    for i, row in enumerate(manifest['rows']):
        row_work = repeat * (8 * (len(row['prompt_ids']) + len(row['prefix_ids'])) /
                             reference['serial_prefill_stress_tokens_per_second'] +
                             8 * HORIZON / reference['serial_decode_stress_tokens_per_second'])
        require(row_work <= usable_per_job, 'one four-arm, two-seed row exceeds job wall')
        if shard_work and shard_work + row_work > usable_per_job:
            shards.append({'start_row': shard_start, 'end_row': i,
                           'estimated_work_seconds': shard_work})
            shard_start, shard_work = i, 0.0
        shard_work += row_work
    if shard_start < len(manifest['rows']):
        shards.append({'start_row': shard_start, 'end_row': len(manifest['rows']),
                       'estimated_work_seconds': shard_work})
    require(shards and shards[0]['start_row'] == 0 and
            shards[-1]['end_row'] == len(manifest['rows']), 'incomplete stage shards')
    jobs = len(shards)
    recovery_loads = max(1, math.ceil(jobs * (repeat - 1)))
    seconds = (jobs + recovery_loads) * (load + shutdown) + work
    gpu_hours = seconds * reference['gpus'] / 3600
    return {'schema': 'mechanism-validation-price-v1',
            'manifest_sha256': manifest['sha256'], 'reference_sha256': reference['sha256'],
            'qualification_sha256': (qualification['sha256'] if qualification else None),
            'requests': manifest['expected_requests'],
            'prefill_tokens': manifest['expected_prefill_tokens'],
            'maximum_decode_tokens': manifest['maximum_decode_tokens'],
            'shards': shards, 'planned_jobs': jobs,
            'cold_loads': jobs + recovery_loads,
            'shutdowns': jobs + recovery_loads,
            'recovery_load_reserve': recovery_loads,
            'max_wall_seconds_per_job': max_wall_seconds,
            'preemption_reserve_seconds_per_job': reserve,
            'repeat_factor': repeat,
            'estimated_complete_wall_seconds': seconds,
            'estimated_complete_gpu_hours': gpu_hours, 'gpus': reference['gpus'],
            'status': ('PASS_COMPLETE_STAGE' if qualification else
                       'HOLD_FULL_HORIZON_QUALIFICATION'),
            'gate': 'This estimate covers the full accepted-start stage in complete four-arm, two-seed row shards under the supplied live partition wall limit, plus recovery load and retry factor. The separate sealed 1024-token serial qualification is required before submission.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, default=RATINGS)
    parser.add_argument('--max-wall-seconds', type=int, required=True,
                        help='verified live partition wall limit for each GPU job')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('native trace reconstruction requires a CPU Slurm step')
    selection, frame = base.sealed(start.SELECTION), base.sealed(start.FRAME)
    chosen, binding, summary = accepted_starts(selection, frame, args.ratings)
    require(chosen, 'no two-reader accepted mechanism starts')
    import pandas as pd
    table = pd.read_parquet(start.ATTEMPTS,
                            columns=['attempt_id', 'source_location', 'trace_sha256'])
    index = {str(r['attempt_id']): r for r in table.to_dict('records')}
    rows = replay_rows(chosen, frame, index)
    family, dictionary = base.sealed(FAMILY), base.sealed(DICTIONARY)
    qualification = validate_qualification(base.sealed(QUAL_RESULT))
    manifest = start.write_once(MANIFEST, make_manifest(
        rows, selection, frame, binding, summary, family, dictionary, args.ratings))
    price = start.write_once(PRICE, price_stage(
        manifest, base.sealed(REFERENCE_PRICE), args.max_wall_seconds, qualification))
    print(json.dumps({'manifest': str(MANIFEST), 'manifest_sha256': manifest['sha256'],
                      'price': str(PRICE), 'price_sha256': price['sha256'],
                      'accepted_starts': len(rows), 'families': len({r['family'] for r in rows}),
                      'registered_128_family_feasibility':
                      manifest['registered_128_family_feasibility'],
                      'requests': manifest['expected_requests'],
                      'estimated_gpu_hours': price['estimated_complete_gpu_hours'],
                      'status': price['status']}))


if __name__ == '__main__':
    main()
