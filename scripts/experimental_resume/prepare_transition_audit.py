"""Freeze discovery-only, prefix-exact semantic trigger audit fixtures.

Every candidate firing is retained. A deterministic, family-spread sample of
nonfires permits confusion estimates with the sampling frame disclosed.
Nothing here supplies class labels or future text to the online detector.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
UNITS = R / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'
LABELS = UNITS.parent / 'results-84a85c92-b2633756'
ATTEMPTS = R / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
OUT = UNITS.parent / 'TRANSITION_AUDIT_FIXTURES.json'
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed discovery input seal: ' + str(path))
    return value


def trace_at(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location['path']).open('rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('discovery native trace changed')
    return json.loads(raw)


def label_rows(units):
    binding = sealed(LABELS / 'BINDING.json')
    summary = sealed(LABELS / 'SUMMARY.json')
    if binding['units_sha256'] != units['sha256'] or summary['binding_sha256'] != binding['sha256'] or summary['sentences'] != len(units['records']):
        raise ValueError('dense discovery labels incomplete or rebound')
    result = []
    for start in range(0, len(units['records']), binding['batch_size']):
        part = sealed(LABELS / 'batches' / f'{start:06d}.json')
        if part['binding_sha256'] != binding['sha256'] or part['start'] != start:
            raise ValueError('dense label batch differs')
        result.extend(part['records'])
    if len(result) != len(units['records']):
        raise ValueError('dense label row count differs')
    for unit, label in zip(units['records'], result, strict=True):
        if (unit['family'], unit['attempt_id'], unit['sentence_index']) != (label['family'], label['attempt_id'], label['sentence_index']):
            raise ValueError('dense label/sentence identity differs')
    return result, summary


def select_nonfires(rows, limit=200):
    """Include one per family first, then fill by fixed hash; never replace."""
    ordered = sorted(rows, key=lambda row: digest(['transition-audit-nonfire-v1', row['uid']]))
    selected, seen = [], set()
    for row in ordered:
        family = row['analysis_meta']['family']
        if family not in seen:
            selected.append(row)
            seen.add(family)
    selected_ids = {r['uid'] for r in selected}
    selected.extend(row for row in ordered if row['uid'] not in selected_ids)
    return selected[:limit]


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('semantic audit fixture preparation requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.routing_control.transitions import StreamingTransitionDetector
    from moe_exp.routing_control.analysis import CLASSES
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    units = sealed(UNITS)
    labels, label_summary = label_rows(units)
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id','source_location','trace_sha256'])
    by_attempt = {str(row['attempt_id']): row for row in table.to_dict('records')}
    grouped = defaultdict(list)
    unknown = 0
    for unit, label in zip(units['records'], labels, strict=True):
        if label['label'] not in CLASSES or label['finish_reason'] != 'stop':
            unknown += 1
        grouped[unit['attempt_id']].append((unit, label))
    if len(grouped) != 48:
        raise ValueError('fewer than 48 frozen discovery attempts')
    positives, negatives = defaultdict(list), defaultdict(list)
    frame_counts = Counter()
    pair_count = 0
    for attempt in sorted(grouped):
        source = by_attempt[attempt]
        trace = trace_at(source['source_location'])
        if trace_digest(TraceRecord(**trace)) != source['trace_sha256']:
            raise ValueError('native trace digest differs')
        replay = trace['metadata']['token_replay']
        ids = replay['completion_token_ids']
        offsets = replay['completion_offsets']
        text = trace['cot_text']
        if len(ids) != len(offsets) or int(offsets[-1][1]) != len(text):
            raise ValueError('saved token/text offset alignment differs')
        ordered = sorted(grouped[attempt], key=lambda pair: pair[0]['sentence_index'])
        for (left, left_label), (right, right_label) in zip(ordered, ordered[1:]):
            if right['sentence_index'] != left['sentence_index'] + 1 or right['segment'] != left['segment']:
                continue
            if left['token_end'] > len(ids) or right['token_start'] < left['token_end']:
                raise ValueError('adjacent native sentence token offsets differ')
            prefix_end = int(offsets[left['token_end'] - 1][1])
            if not left['char_start'] < left['char_end'] <= prefix_end:
                raise ValueError('sentence extends beyond emitted token prefix')
            emitted = text[:prefix_end]
            observed = StreamingTransitionDetector().observe({
                'problem': left['inputs']['problem_statement'],
                'emitted_token_ids': ids[:left['token_end']], 'emitted_text': emitted})
            local_events = {e.transition: e for e in observed['events']
                            if left['char_start'] <= e.evidence_end <= left['char_end']}
            pair_count += 1
            for transition in TRANSITIONS:
                # Rate every native opportunity. Filtering by an audit class here
                # would hide detector fires in the wrong starting state and
                # overstate its operational precision.
                fired = transition in local_events
                frame_counts[transition, 'fired' if fired else 'nonfire'] += 1
                uid = digest(['transition-audit-v1', transition, left['family'], attempt,
                              left['sentence_index'], right['sentence_index']])
                row = {'uid': uid, 'transition': transition,
                       'reader_input': {'problem': left['inputs']['problem_statement'],
                                        'previous_sentence': left['inputs']['previous_sentence'],
                                        'triggering_sentence': left['inputs']['sentence'],
                                        'later_sentence': right['inputs']['sentence']},
                       'analysis_meta': {'family': left['family'], 'attempt_id': attempt,
                                         'source_sentence_index': left['sentence_index'],
                                         'later_sentence_index': right['sentence_index'],
                                         'source_class_audit': left_label['label'],
                                         'later_class_audit': right_label['label'],
                                         'detector_fired': fired,
                                         'prefix_tokens': left['token_end'],
                                         'prefix_text_sha256': hashlib.sha256(emitted.encode()).hexdigest(),
                                         'event_end': local_events[transition].evidence_end if fired else None}}
                (positives if fired else negatives)[transition].append(row)
    selected = []
    for transition in TRANSITIONS:
        selected.extend(positives[transition])
        selected.extend(select_nonfires(negatives[transition]))
    selected.sort(key=lambda row: row['uid'])
    if len({row['uid'] for row in selected}) != len(selected):
        raise ValueError('duplicate semantic audit fixture UID')
    body = {'schema': 'transition-audit-fixtures-v1', 'units_sha256': units['sha256'],
            'label_summary_sha256': label_summary['sha256'],
            'detector_sha256': hashlib.sha256((REPO / 'src/moe_exp/routing_control/transitions.py').read_bytes()).hexdigest(),
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'job_id': os.environ['SLURM_JOB_ID'], 'families': len(grouped),
            'contiguous_pairs': pair_count, 'unknown_label_rows': unknown,
            'frame_counts': {t: {kind: frame_counts[t, kind] for kind in ('fired','nonfire')} for t in TRANSITIONS},
            'rated_nonfire_limit_per_transition': 200,
            'selection': 'all fired proposals plus up to 200 deterministic family-spread nonfires per fixed transition',
            'reader_input_allowlist': ['problem','previous_sentence','triggering_sentence','later_sentence'],
            'class_coverage_gate': 'PASS' if unknown / len(units['records']) <= .05 else 'HOLD_OVER_5_PERCENT_UNKNOWN',
            'semantic_rating_status': 'PENDING_ARM_BLIND_LLM_AUDIT_NOT_HUMAN_TRUTH',
            'records': selected}
    result = {**body, 'sha256': digest(body)}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(OUT), 'records': len(selected), 'families': len(grouped),
                      'frame_counts': body['frame_counts'], 'unknown_label_rows': unknown}))


if __name__ == '__main__':
    main()
