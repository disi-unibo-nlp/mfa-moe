"""Freeze mechanism start candidates and bind their exact native prefixes.

``--select-only`` is a bounded, outcome-free login-node preflight. The full
frame reads 128 native traces and therefore requires a CPU Slurm allocation.
No answer, correctness field, future sentence or intervention result is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

from audit_transition_detector_mechanism_v1 import DEFAULT_UNITS, ROOT, REPO, digest, sealed, trace_at

AUDIT = REPO / 'report/experimental-resume-v1/MECHANISM_DETECTOR_AUDIT_v1.json'
SELECTION = REPO / 'report/experimental-resume-v1/MECHANISM_START_SELECTION_v1.json'
FRAME = ROOT / 'steering-v1/runs/routing-control-v1/dense-mechanism/MECHANISM_START_FRAME_v1.json'
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
SUPPORTED = ('candidate_to_verify', 'approach_to_commit')
PREFIX_CAP = 8192


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, body):
    path = Path(path)
    value = {**body, 'sha256': digest(body)}
    if path.exists():
        if sealed(path) != value:
            raise ValueError('existing artifact differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    return value


def event_key(transition, event):
    return digest(['mechanism-start-hash-order-v1', transition, event['family'],
                   event['attempt_id'], event['sentence_index'], event['segment'],
                   event['prefix_tokens']])


def select_events(units, audit):
    if (units['schema'] != 'dense-mechanism-units-v1' or units['families'] != 128
            or audit['schema'] != 'transition-detector-mechanism-audit-v1'
            or audit['families'] != 128 or audit['units_sha256'] != units['sha256']
            or audit['detector_sha256'] != file_sha(REPO / 'src/moe_exp/routing_control/transitions_v2.py')):
        raise ValueError('changed sealed mechanism pool or detector')
    by_unit = {(u['attempt_id'], u['sentence_index']): u for u in units['records']}
    families = sorted({u['family'] for u in units['records']})
    if len(by_unit) != len(units['records']) or len(families) != 128:
        raise ValueError('duplicate source sentence or changed family count')
    selected = []
    coverage = {family: {transition: 0 for transition in SUPPORTED} for family in families}
    before_cap = Counter()
    for transition in SUPPORTED:
        grouped = defaultdict(list)
        seen = set()
        for event in audit['eligible_events'][transition]:
            key = (event['attempt_id'], event['sentence_index'])
            if key in seen:
                raise ValueError('duplicate detector fire for one transition')
            seen.add(key)
            unit = by_unit.get(key)
            if (unit is None or event['family'] != unit['family'] or
                    event['segment'] != unit['segment'] or
                    event['prefix_tokens'] != unit['token_end'] or
                    unit['inputs']['problem_statement'] == ''):
                raise ValueError('detector event differs from sealed native unit')
            if 1 <= event['prefix_tokens'] <= PREFIX_CAP:
                grouped[event['family']].append(event)
                before_cap[transition] += 1
        for family in families:
            chosen = sorted(grouped[family], key=lambda e: event_key(transition, e))[:3]
            coverage[family][transition] = len(chosen)
            for event in chosen:
                selected.append({'uid': digest(['mechanism-start-uid-v1', transition,
                                                event['family'], event['attempt_id'],
                                                event['sentence_index']]),
                                 'transition': transition, **event})
    selected.sort(key=lambda e: (e['family'], SUPPORTED.index(e['transition']),
                                 event_key(e['transition'], e)))
    if len({e['uid'] for e in selected}) != len(selected):
        raise ValueError('duplicate start UID')
    body = {
        'schema': 'mechanism-start-selection-v1',
        'units_sha256': units['sha256'], 'detector_audit_sha256': audit['sha256'],
        'detector_sha256': audit['detector_sha256'],
        'selection_driver_sha256': file_sha(__file__),
        'family_pool': families, 'families': 128,
        'supported_transitions': list(SUPPORTED),
        'unsupported_transition': 'failed_check_to_revise',
        'unsupported_reason': 'Only 10 detector-fire families (5 within the 8192-token prefix cap); no validated replacement search.',
        'prefix_token_cap_native': PREFIX_CAP,
        'selection_rule': 'For each frozen family and supported transition, retain at most three detector fires at <=8192 native tokens, ordered by SHA256 of fixed namespace and event identity; no outcome- or rating-dependent replacement.',
        'eligible_within_cap': dict(before_cap),
        'selected_counts': dict(Counter(e['transition'] for e in selected)),
        'family_coverage': coverage, 'records': selected,
    }
    return body


def build_frame(selection, units):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('full native replay requires CPU Slurm step')
    import pandas as pd
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    by_unit = {(u['attempt_id'], u['sentence_index']): u for u in units['records']}
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'source_location', 'trace_sha256'])
    by_attempt = {str(r['attempt_id']): r for r in table.to_dict('records')}
    traces = {}
    frame = []
    for event in selection['records']:
        attempt, index = event['attempt_id'], event['sentence_index']
        unit = by_unit[attempt, index]
        next_unit = by_unit.get((attempt, index + 1))
        if next_unit is None or next_unit['segment'] != unit['segment']:
            raise ValueError('selected event lacks contiguous native successor')
        if attempt not in traces:
            source = by_attempt[attempt]
            trace = trace_at(source['source_location'])
            if trace_digest(TraceRecord(**trace)) != source['trace_sha256']:
                raise ValueError('native trace digest differs')
            traces[attempt] = trace
        trace = traces[attempt]
        replay = trace['metadata']['token_replay']
        ids, offsets, text = (replay['completion_token_ids'],
                              replay['completion_offsets'], trace['cot_text'])
        n = event['prefix_tokens']
        if len(ids) != len(offsets) or n > len(ids) or n != unit['token_end']:
            raise ValueError('selected prefix does not match native replay')
        end = int(offsets[n - 1][1])
        if (not unit['char_start'] < unit['char_end'] <= end <= next_unit['char_start']
                or end != event['prefix_end_char']
                or replay['tokenizer_sha256'] != event['tokenizer_sha256']
                or text[unit['char_start']:unit['char_end']] != unit['inputs']['sentence']):
            raise ValueError('selected sentence or tokenizer differs')
        reasoning = trace['metadata'].get('reasoning_content')
        reasoning_start = text.find(reasoning) if reasoning else -1
        if reasoning_start < 0 or end > reasoning_start + len(reasoning):
            raise ValueError('selected prefix is beyond reasoning closure')
        prefix = text[:end]
        if not prefix.rstrip().endswith(unit['inputs']['sentence'].rstrip()):
            raise ValueError('trigger sentence is not at end of exact prefix')
        frame.append({
            'uid': event['uid'], 'transition': event['transition'],
            'reader_input': {'problem': unit['inputs']['problem_statement'],
                             'emitted_prefix': prefix,
                             'triggering_sentence': unit['inputs']['sentence']},
            'analysis_meta': {'prefix_tokens': n,
                              'prefix_ids_sha256': digest(ids[:n]),
                              'prompt_ids_sha256': digest(replay['prompt_token_ids']),
                              'prefix_text_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
                              'trace_sha256': by_attempt[attempt]['trace_sha256'],
                              'tokenizer_sha256': replay['tokenizer_sha256']},
        })
    body = {'schema': 'mechanism-start-frame-v1',
            'selection_sha256': selection['sha256'],
            'units_sha256': units['sha256'],
            'driver_sha256': file_sha(__file__),
            'families_in_frozen_pool': 128,
            'rows': len(frame),
            'visible_input_allowlist': ['problem', 'emitted_prefix', 'triggering_sentence'],
            'scope': 'arm-blind, preoutcome native starts; no future sentence, gold answer, correctness or intervention output in reader input',
            'records': frame}
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    units, audit = sealed(DEFAULT_UNITS), sealed(AUDIT)
    selection = write_once(SELECTION, select_events(units, audit))
    if args.select_only:
        print(json.dumps({'selection': str(SELECTION), 'sha256': selection['sha256'],
                          'selected_counts': selection['selected_counts'],
                          'families': selection['families']}))
        return
    frame = write_once(FRAME, build_frame(selection, units))
    print(json.dumps({'frame': str(FRAME), 'sha256': frame['sha256'],
                      'rows': frame['rows'], 'families': frame['families_in_frozen_pool']}))


if __name__ == '__main__':
    main()
