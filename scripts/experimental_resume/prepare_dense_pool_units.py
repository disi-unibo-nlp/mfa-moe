"""Freeze contiguous native reasoning windows for a frozen family pool.

This is the pool-general version of ``prepare_dense_discovery.py``.  It keeps
the same window rule and record schema, but can build the 128 mechanism-family
or 96 utility-family unit sets from the existing native attempts table.  It
never mixes families across pools: the caller chooses exactly one frozen pool
and the output records carry that pool's family identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
POOLS = ('discovery', 'mechanism', 'utility')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_line(location):
    if isinstance(location, str):
        location = json.loads(location)
    with open(location['path'], 'rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('native trace source changed')
    return json.loads(raw)


def build(pool, out, smoke):
    if pool not in POOLS:
        raise ValueError('pool must be one of ' + ', '.join(POOLS))
    if not smoke:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('dense pool preparation requires a Slurm allocation')
        if os.environ.get('SLURM_JOB_PARTITION') != 'lrd_all_serial':
            raise RuntimeError('dense pool preparation must run on lrd_all_serial')
    if out is not None and not (str(out).startswith('/leonardo_work/IscrC_MIOSR/lmolfett/') or
                                str(out).startswith('/leonardo/home/userexternal/lmolfett/')):
        raise ValueError('output must remain under the user-owned HOME/WORK trees')
    import pandas as pd
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.dynamics.annotation_plan import contiguous_blocks
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest

    frozen = json.loads(FAMILY.read_text())
    pools = frozen['new_parent_pools']
    representatives = pools['representative_questions']
    families = list(pools['parent_pools'][pool])
    question_to_family = {representatives[family]: family for family in families}
    if len(question_to_family) != len(families):
        raise ValueError('pool representative questions are not unique')
    attempts = pd.read_parquet(ATTEMPTS)
    chosen = {}
    for row in attempts.to_dict('records'):
        question = str(row['question'])
        if question in question_to_family and (
                question not in chosen or str(row['attempt_id']) < str(chosen[question]['attempt_id'])):
            chosen[question] = row
    if set(chosen) != set(question_to_family):
        raise ValueError(f'{pool} native trace coverage incomplete')
    questions = sorted(question_to_family)
    if smoke:
        questions = questions[:smoke]
    records = []
    for question in questions:
        attempt = chosen[question]
        trace = TraceRecord(**read_line(attempt['source_location']))
        layout = saved_layout(trace)
        units, owners = layout['units'], layout['unit_tokens']
        if len(units) != len(owners):
            raise ValueError('sentence/token owner mismatch')
        selected = contiguous_blocks(units, width=40)
        selected_index = {int(unit['index']) for unit in selected}
        ranges = reasoning_ranges(trace)
        segment = {int(unit['index']): next(i for i, (left, right) in enumerate(ranges)
                       if left <= int(unit['start']) < right) for unit in units}
        messages = trace.generation_messages or []
        problem = '\n\n'.join(str(message['content']) for message in messages
                              if message.get('role') == 'user') or trace.prompt
        for unit in units:
            index = int(unit['index'])
            if index not in selected_index:
                continue
            owned = owners[index]
            if not owned:
                raise ValueError('selected sentence has no saved token ownership')
            previous = (units[index - 1]['text'] if index and
                        segment[index - 1] == segment[index] else '<START OF RESPONSE>')
            following = (units[index + 1]['text'] if index + 1 < len(units) and
                         segment[index + 1] == segment[index] else '<END OF RESPONSE>')
            records.append({'family': question_to_family[question], 'question': question,
                            'attempt_id': str(attempt['attempt_id']),
                            'trace_sha256': trace_digest(trace),
                            'sentence_index': index, 'segment': segment[index],
                            'char_start': int(unit['start']), 'char_end': int(unit['end']),
                            'token_start': min(owned), 'token_end': max(owned) + 1,
                            'inputs': {'problem_statement': problem,
                                       'previous_sentence': previous,
                                       'sentence': unit['text'],
                                       'next_sentence': following}})
    if len({(row['attempt_id'], row['sentence_index']) for row in records}) != len(records):
        raise ValueError('duplicate dense sentence identity')
    value = {'schema': f'dense-{pool}-units-v1', 'pool': pool,
             'family_freeze_sha256': frozen['sha256'], 'attempts': len(questions),
             'families': len({row['family'] for row in records}),
             'sentences': len(records),
             'window_rule': ('first, centred, last contiguous 40 sentence blocks per frozen '
                             'representative native trace; overlap deduplicated; no joins '
                             'across unselected gaps or reasoning segments'),
             'records': records}
    value['sha256'] = digest(value)
    if smoke:
        print(json.dumps({'smoke': True, 'pool': pool, 'attempts': len(questions),
                          'families': value['families'], 'sentences': len(records)}))
        return value
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(out), 'sha256': value['sha256'], 'pool': pool,
                      'attempts': len(questions), 'families': value['families'],
                      'sentences': len(records)}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', choices=POOLS, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', type=int, default=0,
                        help='process at most this many families on the login node and write nothing')
    args = parser.parse_args()
    if not args.smoke and args.out is None:
        raise SystemExit('--out is required for a full pool build')
    build(args.pool, args.out, args.smoke)


if __name__ == '__main__':
    main()
