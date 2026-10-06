"""Freeze contiguous native Qwen reasoning windows for 48 discovery families."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
OUT = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'


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


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('dense discovery preparation requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from moe_exp.correlation_pipeline.dynamics.annotation_plan import contiguous_blocks
    from moe_exp.correlation_pipeline.spans import reasoning_ranges, trace_digest

    frozen = json.loads(FAMILY.read_text())
    pools = frozen['new_parent_pools']
    qf = {pools['representative_questions'][f]: f for f in pools['parent_pools']['discovery']}
    attempts = pd.read_parquet(ATTEMPTS)
    chosen = {}
    for row in attempts.to_dict('records'):
        question = str(row['question'])
        if question in qf and (question not in chosen or str(row['attempt_id']) < str(chosen[question]['attempt_id'])):
            chosen[question] = row
    if set(chosen) != set(qf):
        raise ValueError('discovery native trace coverage incomplete')
    records = []
    for question in sorted(qf):
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
        problem = '\n\n'.join(str(m['content']) for m in messages if m.get('role') == 'user') or trace.prompt
        for unit in units:
            index = int(unit['index'])
            if index not in selected_index:
                continue
            owned = owners[index]
            if not owned:
                raise ValueError('selected sentence has no saved token ownership')
            prev = units[index-1]['text'] if index and segment[index-1] == segment[index] else '<START OF RESPONSE>'
            following = units[index+1]['text'] if index+1 < len(units) and segment[index+1] == segment[index] else '<END OF RESPONSE>'
            records.append({'family': qf[question], 'question': question,
                            'attempt_id': str(attempt['attempt_id']), 'trace_sha256': trace_digest(trace),
                            'sentence_index': index, 'segment': segment[index],
                            'char_start': int(unit['start']), 'char_end': int(unit['end']),
                            'token_start': min(owned), 'token_end': max(owned)+1,
                            'inputs': {'problem_statement': problem, 'previous_sentence': prev,
                                       'sentence': unit['text'], 'next_sentence': following}})
    if len({(r['attempt_id'], r['sentence_index']) for r in records}) != len(records):
        raise ValueError('duplicate dense sentence identity')
    value = {'schema': 'dense-discovery-units-v1', 'job_id': os.environ['SLURM_JOB_ID'],
             'family_freeze_sha256': frozen['sha256'], 'attempts': len(chosen), 'sentences': len(records),
             'window_rule': 'first, centred, last contiguous 40 sentence blocks per frozen representative native trace; overlap deduplicated; no joins across unselected gaps or reasoning segments',
             'records': records}
    value['sha256'] = digest(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'], 'sentences': len(records),
                      'families': len({r['family'] for r in records})}))


if __name__ == '__main__':
    main()
