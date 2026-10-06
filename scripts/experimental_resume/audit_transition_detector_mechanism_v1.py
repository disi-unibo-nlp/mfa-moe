"""Apply the frozen v2 streaming detector to the 128-family mechanism units.

This is the mechanism-pool counterpart of the discovery detector audit.  It
does not read any outcome, rating or future sentence.  It reconstructs each
contiguous native prefix from the sealed token replay, replays the frozen v2
detector and records the candidate transition events that a later two-reader
start audit will rate.  The detector fire is a proposal, never an accepted
start by itself.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DEFAULT_UNITS = (ROOT / 'steering-v1/runs/routing-control-v1/dense-mechanism/'
                 'MECHANISM_UNITS_v1.json')
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')
sys.path.insert(0, str(REPO / 'src'))
from moe_exp.routing_control.transitions_v2 import StreamingTransitionDetectorV2  # noqa: E402


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({key: item for key, item in value.items()
                                      if key != 'sha256'}):
        raise ValueError('changed sealed input: ' + str(path))
    return value


def trace_at(location):
    if isinstance(location, str):
        location = json.loads(location)
    with Path(location['path']).open('rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('native trace changed')
    return json.loads(raw)


def audit(units_path, out, smoke):
    if not smoke:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('mechanism detector audit requires a Slurm allocation')
        if os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_serial':
            raise RuntimeError('mechanism detector audit must run on lrd_all_serial')
    import pandas as pd
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.spans import trace_digest

    units = sealed(units_path)
    if units.get('schema') != 'dense-mechanism-units-v1':
        raise ValueError('unexpected mechanism unit schema')
    grouped = defaultdict(list)
    for unit in units['records']:
        grouped[unit['attempt_id']].append(unit)
    families = {unit['family'] for unit in units['records']}
    if len(grouped) != units['attempts'] or len(families) != units['families']:
        raise ValueError('mechanism unit family/attempt counts differ')
    if smoke:
        grouped = dict(sorted(grouped.items())[:smoke])
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'source_location', 'trace_sha256'])
    by_attempt = {str(row['attempt_id']): row for row in table.to_dict('records')}
    full_counts = defaultdict(Counter)
    fire_families = defaultdict(set)
    eligible = defaultdict(list)
    pair_count = 0
    for attempt in sorted(grouped):
        row = by_attempt[attempt]
        trace = trace_at(row['source_location'])
        if trace_digest(TraceRecord(**trace)) != row['trace_sha256']:
            raise ValueError('native trace digest differs')
        replay = trace['metadata']['token_replay']
        ids, offsets, text = (replay['completion_token_ids'],
                              replay['completion_offsets'], trace['cot_text'])
        if len(ids) != len(offsets) or int(offsets[-1][1]) != len(text):
            raise ValueError('native token/text alignment differs')
        ordered = sorted(grouped[attempt], key=lambda unit: unit['sentence_index'])
        detector = StreamingTransitionDetectorV2()
        for left, right in zip(ordered, ordered[1:]):
            if (right['sentence_index'] != left['sentence_index'] + 1 or
                    right['segment'] != left['segment']):
                continue
            prefix_tokens = left['token_end']
            if not 1 <= prefix_tokens <= len(ids):
                raise ValueError('mechanism unit prefix exceeds the native stream')
            prefix_end = int(offsets[prefix_tokens - 1][1])
            if not left['char_start'] < left['char_end'] <= prefix_end:
                raise ValueError('source sentence extends beyond available prefix')
            result = detector.observe({'problem': left['inputs']['problem_statement'],
                                       'emitted_token_ids': ids[:prefix_tokens],
                                       'emitted_text': text[:prefix_end]})
            local = {event.transition: event for event in result['events']
                     if left['char_start'] <= event.evidence_end <= left['char_end']}
            pair_count += 1
            for transition in TRANSITIONS:
                fired = transition in local
                full_counts[transition]['fire' if fired else 'nonfire'] += 1
                if fired:
                    fire_families[transition].add(left['family'])
                    eligible[transition].append({
                        'family': left['family'], 'attempt_id': attempt,
                        'sentence_index': left['sentence_index'],
                        'segment': left['segment'],
                        'prefix_tokens': prefix_tokens,
                        'prefix_end_char': prefix_end,
                        'event_end_char': int(local[transition].evidence_end),
                        'tokenizer_sha256': replay['tokenizer_sha256'],
                    })
    body = {
        'schema': 'transition-detector-mechanism-audit-v1',
        'job_id': os.environ.get('SLURM_JOB_ID', 'smoke'),
        'units_sha256': units['sha256'],
        'detector_sha256': hashlib.sha256(
            (REPO / 'src/moe_exp/routing_control/transitions_v2.py').read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'families': len({unit['family'] for unit in units['records']}) if not smoke else len(
            {unit['family'] for attempt in grouped for unit in grouped[attempt]}),
        'contiguous_pairs': pair_count,
        'full_counts': {key: dict(value) for key, value in full_counts.items()},
        'fire_family_counts': {key: len(value) for key, value in fire_families.items()},
        'eligible_events': {key: value for key, value in eligible.items()},
        'interpretation': (
            'Frozen v2 detector proposals on disjoint mechanism families only; '
            'no outcome, rating or future sentence was read. A fire is a candidate '
            'start, not an accepted start; the two-reader start audit is separate.'),
    }
    value = {**body, 'sha256': digest(body)}
    if smoke:
        print(json.dumps({'smoke': True, 'families': body['families'],
                          'contiguous_pairs': pair_count,
                          'full_counts': body['full_counts'],
                          'fire_family_counts': body['fire_family_counts']}))
        return value
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({'out': str(out), 'sha256': value['sha256'],
                      'families': body['families'], 'contiguous_pairs': pair_count,
                      'full_counts': body['full_counts'],
                      'fire_family_counts': body['fire_family_counts']}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units', type=Path, default=DEFAULT_UNITS)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', type=int, default=0,
                        help='process at most this many families on the login node and write nothing')
    args = parser.parse_args()
    if not args.smoke and args.out is None:
        raise SystemExit('--out is required for the full mechanism audit')
    audit(args.units, args.out, args.smoke)


if __name__ == '__main__':
    main()
