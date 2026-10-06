"""CPU-only native evidence pipeline for the frozen 220-family extension.

Stages are units, detector, selection, frame and reader price. Full trace work
requires a CPU Slurm step. Nothing here generates intervention outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import audit_transition_detector_mechanism_v1 as prior_audit
import prepare_dense_pool_units as prior_units
import prepare_mechanism_start_frame_v1 as prior_frame
import rate_transition_v22_fullprefix_starts_v2 as reader_messages
import mechanism_extension_strict_veto_v1 as strict_veto
from freeze_mechanism_extension_220_v1 import (
    OUT as FREEZE, SOURCE as SOURCE_FAMILY, derive as derive_family,
    digest, sealed)

REPO = prior_units.REPO
DOC = REPO / 'report/experimental-resume-v1'
ROOT = prior_units.ROOT / 'steering-v1/runs/routing-control-v1/dense-mechanism-extension-220'
UNITS = ROOT / 'MECHANISM_EXTENSION_UNITS_v1.json'
AUDIT = DOC / 'MECHANISM_EXTENSION_DETECTOR_AUDIT_v1.json'
SELECTION = DOC / 'MECHANISM_EXTENSION_START_SELECTION_v1.json'
FRAME = ROOT / 'MECHANISM_EXTENSION_START_FRAME_v1.json'
PRICE = DOC / 'MECHANISM_EXTENSION_START_READER_PRICE_v1.json'
PRIOR_PRICE = DOC / 'MECHANISM_START_READER_PRICE_v2.json'
ACTION_DICTIONARY = DOC / 'CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json'
ATTEMPTS = prior_units.ATTEMPTS
SUPPORTED = ('candidate_to_verify', 'approach_to_commit')
PREFIX_CAP = 8192
MAX_READER_OUTPUT = 1024
MAX_MODEL_LEN = 49152


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, body):
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock')
    with lock.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            require(sealed(path) == value, 'existing extension artifact differs')
        else:
            temporary = path.with_name(path.name + f'.partial-{os.getpid()}-{time.monotonic_ns()}')
            try:
                with temporary.open('x') as output:
                    output.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
    return value


def frozen_extension():
    source, extension = sealed(SOURCE_FAMILY), sealed(FREEZE)
    require(extension == {**derive_family(source),
                          'sha256': digest(derive_family(source))},
            'extension no longer equals full unallocated family suffix')
    return source, extension


def require_cpu_step():
    if (not os.environ.get('SLURM_JOB_ID') or
            not os.environ.get('SLURM_STEP_ID') or
            os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_serial'):
        raise RuntimeError('extension trace work requires lrd_all_serial CPU Slurm step')


def checked_units():
    value = sealed(UNITS)
    require(value['schema'] == 'dense-mechanism-extension-units-v1' and
            value['unit_driver_sha256'] == file_sha(__file__) and
            value['source_window_driver_sha256'] == file_sha(prior_units.__file__),
            'extension unit code differs from sealed producer')
    return value


def checked_audit():
    value = sealed(AUDIT)
    require(value['schema'] == 'mechanism-extension-detector-audit-v1' and
            value['extension_driver_sha256'] == file_sha(__file__) and
            value['source_detector_driver_sha256'] == file_sha(prior_audit.__file__) and
            value['detector_sha256'] == file_sha(
                REPO / 'src/moe_exp/routing_control/transitions_v2.py'),
            'extension detector code differs from sealed producer')
    return value


def checked_selection():
    value = sealed(SELECTION)
    dictionary = sealed(ACTION_DICTIONARY)
    require(value['schema'] == 'mechanism-extension-start-selection-v1' and
            value['selection_driver_sha256'] == file_sha(__file__) and
            value['source_selection_driver_sha256'] == file_sha(prior_frame.__file__) and
            value['action_dictionary_sha256'] == dictionary['sha256'],
            'extension selection code or frozen action dictionary differs')
    return value


def checked_frame():
    value = sealed(FRAME)
    require(value['schema'] == 'mechanism-extension-start-frame-v1' and
            value['driver_sha256'] == file_sha(__file__) and
            value['source_frame_builder_sha256'] == file_sha(prior_frame.__file__),
            'extension frame code differs from sealed producer')
    return value


def build_units(attempt_rows, extension, trace_reader=prior_units.read_line):
    """Apply the existing 40-sentence native window rule to fixed representatives."""
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.dynamics.annotation_plan import contiguous_blocks
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest

    representatives = extension['representative_questions']
    question_to_family = {question: family for family, question in representatives.items()}
    require(len(question_to_family) == 220, 'duplicate canonical representative')
    chosen = {}
    for row in attempt_rows:
        question = str(row['question'])
        if question in question_to_family and (
                question not in chosen or str(row['attempt_id']) < str(chosen[question]['attempt_id'])):
            chosen[question] = row
    require(set(chosen) == set(question_to_family), 'extension native trace coverage incomplete')
    records = []
    for question in sorted(question_to_family):
        source = chosen[question]
        trace = TraceRecord(**trace_reader(source['source_location']))
        require(trace_digest(trace) == source['trace_sha256'],
                'native trace digest differs from attempts table')
        layout = saved_layout(trace)
        units, owners = layout['units'], layout['unit_tokens']
        require(len(units) == len(owners), 'sentence/token owner mismatch')
        selected = {int(unit['index']) for unit in contiguous_blocks(units, width=40)}
        ranges = reasoning_ranges(trace)
        segment = {int(unit['index']): next(i for i, (left, right) in enumerate(ranges)
                   if left <= int(unit['start']) < right) for unit in units}
        messages = trace.generation_messages or []
        problem = '\n\n'.join(str(message['content']) for message in messages
                              if message.get('role') == 'user') or trace.prompt
        for unit in units:
            index = int(unit['index'])
            if index not in selected:
                continue
            owned = owners[index]
            require(bool(owned), 'selected sentence lacks native token ownership')
            previous = (units[index - 1]['text'] if index and
                        segment[index - 1] == segment[index] else '<START OF RESPONSE>')
            following = (units[index + 1]['text'] if index + 1 < len(units) and
                         segment[index + 1] == segment[index] else '<END OF RESPONSE>')
            records.append({'family': question_to_family[question], 'question': question,
                            'attempt_id': str(source['attempt_id']),
                            'trace_sha256': source['trace_sha256'],
                            'sentence_index': index, 'segment': segment[index],
                            'char_start': int(unit['start']), 'char_end': int(unit['end']),
                            'token_start': min(owned), 'token_end': max(owned) + 1,
                            'inputs': {'problem_statement': problem,
                                       'previous_sentence': previous,
                                       'sentence': unit['text'],
                                       'next_sentence': following}})
    require(len({(r['attempt_id'], r['sentence_index']) for r in records}) == len(records),
            'duplicate native sentence identity')
    require({r['family'] for r in records} == set(extension['families']),
            'one or more extension families have no selected native sentence')
    return {'schema': 'dense-mechanism-extension-units-v1',
            'pool': 'mechanism_extension_220',
            'extension_family_freeze_sha256': extension['sha256'],
            'source_family_freeze_sha256': extension['source_family_freeze_sha256'],
            'unit_driver_sha256': file_sha(__file__),
            'source_window_driver_sha256': file_sha(prior_units.__file__),
            'attempts': 220, 'families': 220, 'sentences': len(records),
            'window_rule': ('first, centred, last contiguous 40 sentence blocks per frozen '
                            'representative native trace; overlap deduplicated; no joins '
                            'across unselected gaps or reasoning segments'),
            'records': records}


def audit_detector(units, extension, attempt_index, trace_reader=prior_audit.trace_at):
    """Replay the same frozen v2 detector on contiguous native prefixes."""
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest
    from moe_exp.routing_control.transitions_v2 import StreamingTransitionDetectorV2

    require(units['schema'] == 'dense-mechanism-extension-units-v1' and
            units['extension_family_freeze_sha256'] == extension['sha256'] and
            units['families'] == 220 and
            {r['family'] for r in units['records']} == set(extension['families']),
            'detector source is not the exact extension unit pool')
    grouped = defaultdict(list)
    for unit in units['records']:
        grouped[unit['attempt_id']].append(unit)
    require(len(grouped) == 220, 'extension unit attempt count differs')
    counts, fire_families, eligible = defaultdict(Counter), defaultdict(set), defaultdict(list)
    pairs = 0
    for attempt, member in sorted(grouped.items()):
        source = attempt_index[attempt]
        trace = trace_reader(source['source_location'])
        require(trace_digest(TraceRecord(**trace)) == source['trace_sha256'],
                'native detector trace digest differs')
        replay = trace['metadata']['token_replay']
        ids, offsets, text = (replay['completion_token_ids'],
                              replay['completion_offsets'], trace['cot_text'])
        require(len(ids) == len(offsets) and int(offsets[-1][1]) == len(text),
                'native token/text alignment differs')
        ordered = sorted(member, key=lambda unit: unit['sentence_index'])
        detector = StreamingTransitionDetectorV2()
        for left, right in zip(ordered, ordered[1:]):
            if (right['sentence_index'] != left['sentence_index'] + 1 or
                    right['segment'] != left['segment']):
                continue
            n = left['token_end']
            require(1 <= n <= len(ids), 'detector prefix exceeds native stream')
            end = int(offsets[n - 1][1])
            require(left['char_start'] < left['char_end'] <= end,
                    'native trigger sentence exceeds prefix')
            observed = detector.observe({'problem': left['inputs']['problem_statement'],
                                         'emitted_token_ids': ids[:n],
                                         'emitted_text': text[:end]})
            local = {event.transition: event for event in observed['events']
                     if left['char_start'] <= event.evidence_end <= left['char_end']}
            pairs += 1
            for transition in prior_audit.TRANSITIONS:
                fired = transition in local
                counts[transition]['fire' if fired else 'nonfire'] += 1
                if fired:
                    fire_families[transition].add(left['family'])
                    eligible[transition].append({
                        'family': left['family'], 'attempt_id': attempt,
                        'sentence_index': left['sentence_index'],
                        'segment': left['segment'], 'prefix_tokens': n,
                        'prefix_end_char': end,
                        'event_end_char': int(local[transition].evidence_end),
                        'tokenizer_sha256': replay['tokenizer_sha256']})
    return {'schema': 'mechanism-extension-detector-audit-v1',
            'extension_family_freeze_sha256': extension['sha256'],
            'units_sha256': units['sha256'],
            'detector_sha256': file_sha(REPO / 'src/moe_exp/routing_control/transitions_v2.py'),
            'source_detector_driver_sha256': file_sha(prior_audit.__file__),
            'extension_driver_sha256': file_sha(__file__),
            'families': 220, 'contiguous_pairs': pairs,
            'full_counts': {k: dict(v) for k, v in counts.items()},
            'fire_family_counts': {k: len(v) for k, v in fire_families.items()},
            'eligible_events': {k: v for k, v in eligible.items()},
            'scope': 'Frozen v2 native detector proposals only; no semantic or intervention outcome.'}


def event_key(transition, event):
    return digest(['mechanism-start-hash-order-v1', transition, event['family'],
                   event['attempt_id'], event['sentence_index'], event['segment'],
                   event['prefix_tokens']])


def select_events(units, audit, extension):
    require(units['schema'] == 'dense-mechanism-extension-units-v1' and
            units['extension_family_freeze_sha256'] == extension['sha256'] and
            audit['schema'] == 'mechanism-extension-detector-audit-v1' and
            audit['units_sha256'] == units['sha256'] and
            audit['families'] == 220 and
            audit['extension_family_freeze_sha256'] == extension['sha256'] and
            audit['detector_sha256'] == file_sha(
                REPO / 'src/moe_exp/routing_control/transitions_v2.py'),
            'extension selection sources differ')
    dictionary = sealed(ACTION_DICTIONARY)
    require(dictionary['schema'] == 'routing-discovery-action-dictionary-v1' and
            {item['transition'] for item in dictionary['target_templates']} ==
            set(SUPPORTED) and
            all(len(dictionary['matched_random_control_sets'][t]) == 4
                for t in SUPPORTED), 'same frozen target/random actions are unavailable')
    by_unit = {(u['attempt_id'], u['sentence_index']): u for u in units['records']}
    require(len(by_unit) == len(units['records']) and
            {u['family'] for u in units['records']} == set(extension['families']),
            'duplicate or missing extension source unit')
    selected, coverage, before_cap = [], {}, Counter()
    for family in extension['families']:
        coverage[family] = {transition: 0 for transition in SUPPORTED}
    for transition in SUPPORTED:
        grouped, seen = defaultdict(list), set()
        for event in audit['eligible_events'][transition]:
            key = (event['attempt_id'], event['sentence_index'])
            require(key not in seen, 'duplicate detector fire for transition')
            seen.add(key)
            unit = by_unit.get(key)
            require(unit is not None and event['family'] == unit['family'] and
                    event['segment'] == unit['segment'] and
                    event['prefix_tokens'] == unit['token_end'] and
                    unit['inputs']['problem_statement'] != '',
                    'detector fire differs from sealed native unit')
            if 1 <= event['prefix_tokens'] <= PREFIX_CAP:
                grouped[event['family']].append(event)
                before_cap[transition] += 1
        for family in extension['families']:
            chosen = sorted(grouped[family], key=lambda e: event_key(transition, e))[:3]
            coverage[family][transition] = len(chosen)
            for event in chosen:
                selected.append({'uid': digest(['mechanism-extension-start-uid-v1',
                                                transition, event['family'],
                                                event['attempt_id'], event['sentence_index']]),
                                 'transition': transition, **event})
    selected.sort(key=lambda e: (extension['families'].index(e['family']),
                                 SUPPORTED.index(e['transition']),
                                 event_key(e['transition'], e)))
    require(len({e['uid'] for e in selected}) == len(selected) and
            len(selected) <= 220 * len(SUPPORTED) * 3,
            'duplicate or excessive extension start count')
    return {'schema': 'mechanism-extension-start-selection-v1',
            'extension_family_freeze_sha256': extension['sha256'],
            'units_sha256': units['sha256'],
            'detector_audit_sha256': audit['sha256'],
            'detector_sha256': audit['detector_sha256'],
            'action_dictionary_sha256': dictionary['sha256'],
            'selection_driver_sha256': file_sha(__file__),
            'source_selection_driver_sha256': file_sha(prior_frame.__file__),
            'family_pool': extension['families'], 'families': 220,
            'supported_transitions': list(SUPPORTED),
            'unsupported_transition': 'failed_check_to_revise',
            'prefix_token_cap_native': PREFIX_CAP,
            'selection_rule': ('At most three frozen v2 detector fires per family and '
                               'supported transition at <=8192 native tokens, ordered '
                               'by the same fixed event hash as the registered pool; '
                               'no outcome- or rating-dependent replacement.'),
            'eligible_within_cap': dict(before_cap),
            'selected_counts': dict(Counter(e['transition'] for e in selected)),
            'family_coverage': coverage, 'records': selected,
            'scope': 'Exploratory extension; not registered 128-family validation.'}


def build_frame(selection, units, extension):
    require_cpu_step()
    require(selection['schema'] == 'mechanism-extension-start-selection-v1' and
            selection['extension_family_freeze_sha256'] == extension['sha256'] and
            selection['units_sha256'] == units['sha256'],
            'frame source differs from frozen extension')
    body = prior_frame.build_frame(selection, units)
    body.update({'schema': 'mechanism-extension-start-frame-v1',
                 'families_in_frozen_pool': 220,
                 'extension_family_freeze_sha256': extension['sha256'],
                 'driver_sha256': file_sha(__file__),
                 'source_frame_builder_sha256': file_sha(prior_frame.__file__),
                 'scope': 'Arm-blind native start input for prospective exploratory extension; no intervention outcome.'})
    return body


def price_reader(frame, selection, extension, prior, primary_lengths,
                 veto_lengths, max_wall_seconds):
    require(frame['schema'] == 'mechanism-extension-start-frame-v1' and
            frame['selection_sha256'] == selection['sha256'] and
            frame['extension_family_freeze_sha256'] == extension['sha256'] and
            frame['rows'] == len(selection['records']) == len(primary_lengths) ==
            len(veto_lengths) and frame['rows'] > 0 and
            max(primary_lengths + veto_lengths) + MAX_READER_OUTPUT <= MAX_MODEL_LEN and
            type(max_wall_seconds) is int and max_wall_seconds >= 3600 and
            prior['schema'] == 'mechanism-start-reader-price-v2' and
            prior['status'] == 'PASS_COMPLETE_20_GPUH',
            'reader price lacks exact frame, timing reference or live wall limit')
    requests = 4 * len(primary_lengths)
    primary_prefill, veto_prefill = 2 * sum(primary_lengths), 2 * sum(veto_lengths)
    prefill = primary_prefill + veto_prefill
    decode = requests * MAX_READER_OUTPUT
    repeat = 1.25
    load = prior['components_seconds']['two_cold_loads'] / 2
    shutdown = prior['components_seconds']['two_shutdowns'] / 2
    decode_rate = prior['bounded_decode_tps']
    prefill_rate = prior['bounded_prefill_tps']
    work = repeat * (prefill / prefill_rate + decode / decode_rate)
    usable = max_wall_seconds - load - shutdown - 900
    require(usable > 0, 'reader load and shutdown exceed live wall limit')
    # Keep each 16-start / 32-rating block intact for checkpointing.
    shards, begin, seconds = [], 0, 0.0
    for start_index in range(0, len(primary_lengths), 16):
        primary_block = primary_lengths[start_index:start_index + 16]
        veto_block = veto_lengths[start_index:start_index + 16]
        block_work = repeat * (2 * (sum(primary_block) + sum(veto_block)) /
                               prefill_rate +
                               4 * len(primary_block) * MAX_READER_OUTPUT / decode_rate)
        require(block_work <= usable, 'one complete reader batch exceeds job wall')
        if seconds and seconds + block_work > usable:
            shards.append({'start_row': begin, 'end_row': start_index,
                           'estimated_work_seconds': seconds})
            begin, seconds = start_index, 0.0
        seconds += block_work
    shards.append({'start_row': begin, 'end_row': len(primary_lengths),
                   'estimated_work_seconds': seconds})
    recovery_loads = max(1, math.ceil(len(shards) * .25))
    wall = work + (len(shards) + recovery_loads) * (load + shutdown)
    return {'schema': 'mechanism-extension-start-reader-price-v1',
            'status': 'HOLD_EXTENSION_READER_DRIVER_AND_GPU_QUALIFICATION',
            'extension_family_freeze_sha256': extension['sha256'],
            'frame_sha256': frame['sha256'], 'selection_sha256': selection['sha256'],
            'pricing_driver_sha256': file_sha(__file__),
            'reader_message_source_sha256': file_sha(reader_messages.__file__),
            'reader_rubric_sha256': file_sha(reader_messages.RUBRIC),
            'strict_veto_message_source_sha256': file_sha(strict_veto.__file__),
            'prior_price_sha256': prior['sha256'],
            'rows': len(primary_lengths), 'ratings': requests,
            'primary_start_ratings': 2 * len(primary_lengths),
            'strict_veto_sensitivity_ratings': 2 * len(primary_lengths),
            'prompt_tokens_all_ratings_exact': prefill,
            'primary_prompt_tokens_exact_twice': primary_prefill,
            'strict_veto_prompt_tokens_exact_twice': veto_prefill,
            'prompt_tokens_min': min(primary_lengths + veto_lengths),
            'prompt_tokens_max': max(primary_lengths + veto_lengths),
            'max_decode_tokens': decode, 'max_model_len': MAX_MODEL_LEN,
            'bounded_prefill_tokens_per_second': prefill_rate,
            'bounded_decode_tokens_per_second': decode_rate,
            'retry_factor': repeat, 'cold_loads': len(shards) + recovery_loads,
            'shutdowns': len(shards) + recovery_loads,
            'shards': shards, 'max_wall_seconds_per_job': max_wall_seconds,
            'complete_stage_projected_wall_seconds': wall,
            'complete_stage_projected_GPU_h': 2 * wall / 3600,
            'gate': ('Exact Qwen3.8 chat-template prompt tokens, both primary and '
                     'strict-veto sensitivity readers, all 1024-token caps and complete '
                     '16-start batches are priced. A separate extension '
                     'reader driver, recovery semantics and GPU profile must be sealed '
                     'before any reader submission. This price does not license GPU work.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('units', 'detector', 'selection', 'frame', 'price'))
    parser.add_argument('--max-wall-seconds', type=int,
                        help='verified live GPU partition wall limit for reader pricing')
    args = parser.parse_args()
    source, extension = frozen_extension()
    require_cpu_step()
    if args.stage == 'units':
        import pandas as pd
        table = pd.read_parquet(ATTEMPTS,
                                columns=['attempt_id', 'question', 'source_location',
                                         'trace_sha256'])
        value = write_once(UNITS, build_units(table.to_dict('records'), extension))
    elif args.stage == 'detector':
        import pandas as pd
        units = checked_units()
        table = pd.read_parquet(ATTEMPTS,
                                columns=['attempt_id', 'source_location', 'trace_sha256'])
        index = {str(row['attempt_id']): row for row in table.to_dict('records')}
        value = write_once(AUDIT, audit_detector(units, extension, index))
    elif args.stage == 'selection':
        value = write_once(SELECTION, select_events(checked_units(), checked_audit(), extension))
    elif args.stage == 'frame':
        value = write_once(FRAME, build_frame(checked_selection(), checked_units(), extension))
    else:
        require(args.max_wall_seconds is not None,
                'reader price requires verified live GPU partition wall limit')
        from transformers import AutoTokenizer
        from price_transition_ratings_v3 import count_prompt_tokens
        checked_units(), checked_audit()
        frame, selection = checked_frame(), checked_selection()
        tokenizer = AutoTokenizer.from_pretrained(reader_messages.MODEL,
                                                  local_files_only=True)
        primary_lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
            reader_messages.messages(row), tokenize=True, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort='low')) for row in frame['records']]
        veto_lengths = [count_prompt_tokens(tokenizer.apply_chat_template(
            strict_veto.messages(row), tokenize=True, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort='low')) for row in frame['records']]
        value = write_once(PRICE, price_reader(
            frame, selection, extension, sealed(PRIOR_PRICE), primary_lengths,
            veto_lengths,
            args.max_wall_seconds))
    print(json.dumps({'stage': args.stage, 'path': str({
        'units': UNITS, 'detector': AUDIT, 'selection': SELECTION,
        'frame': FRAME, 'price': PRICE}[args.stage]),
        'sha256': value['sha256'],
        'families': extension['extension_families'],
        'records': len(value.get('records', []))}, sort_keys=True))


if __name__ == '__main__':
    main()
