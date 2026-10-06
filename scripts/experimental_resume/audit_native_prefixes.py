"""Audit full native prefixes for supported arithmetic candidates before closure.

This offline audit uses saved text only to measure the prefix parser's coverage.
No gold answer, correctness or future sentence field is passed to the parser.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
OUT = REPO / 'report/experimental-resume-v1/NATIVE_PREFIX_CANDIDATE_AUDIT.json'


def read_line(location):
    if isinstance(location, str):
        location = json.loads(location)
    with open(location['path'], 'rb') as stream:
        stream.seek(int(location['byte_offset']))
        raw = stream.read(int(location['line_bytes']))
    if hashlib.sha256(raw).hexdigest() != location['line_sha256']:
        raise ValueError('native trace location hash mismatch')
    return json.loads(raw)


def recognition_token(offsets, character):
    for index, pair in enumerate(offsets):
        if int(pair[1]) >= character:
            return index + 1
    return None


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('native prefix audit requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.routing_control.prefix import candidates
    frozen = json.loads(FAMILY.read_text())
    pools = frozen['new_parent_pools']
    frames = pd.read_parquet(ATTEMPTS)
    chosen = {}
    for row in frames.to_dict('records'):
        question = str(row['question'])
        if question not in chosen or str(row['attempt_id']) < str(chosen[question]['attempt_id']):
            chosen[question] = row
    stages = {}
    for stage in ('discovery', 'mechanism', 'utility'):
        records = []
        for family in pools['parent_pools'][stage]:
            question = pools['representative_questions'][family]
            attempt = chosen.get(question)
            if attempt is None:
                records.append({'family': family, 'question': question, 'status': 'NO_NATIVE_TRACE'})
                continue
            trace = read_line(attempt['source_location'])
            text = trace['cot_text']
            replay = trace['metadata']['token_replay']
            offsets = replay['completion_offsets']
            ids = replay['completion_token_ids']
            if len(ids) != len(offsets):
                raise ValueError('native token/offset mismatch')
            n_reasoning = int(attempt['n_reasoning'])
            events = []
            for candidate in candidates(text):
                token = recognition_token(offsets, candidate.end)
                if token is None:
                    raise ValueError('candidate recognition offset not in completion tokens')
                events.append({'kind': candidate.kind, 'recognition_token': token,
                               'before_closure': token < n_reasoning,
                               'at_least_256_native_reasoning_tokens_remain': token + 256 <= n_reasoning,
                               'at_least_1024_native_reasoning_tokens_remain': token + 1024 <= n_reasoning})
            records.append({'family': family, 'question': question, 'status': 'AUDITED',
                            'attempt_id': str(attempt['attempt_id']), 'reasoning_tokens': n_reasoning,
                            'candidate_events': events})
        stages[stage] = {'families': len(records), 'with_trace': sum(r['status'] == 'AUDITED' for r in records),
                         'with_preclosure_candidate': sum(any(e['before_closure'] for e in r.get('candidate_events',[])) for r in records),
                         'with_candidate_and_256_native_tokens': sum(any(e['at_least_256_native_reasoning_tokens_remain'] for e in r.get('candidate_events',[])) for r in records),
                         'with_candidate_and_1024_native_tokens': sum(any(e['at_least_1024_native_reasoning_tokens_remain'] for e in r.get('candidate_events',[])) for r in records),
                         'records': records}
    result = {'schema': 'native-prefix-candidate-audit-v1', 'job_id': os.environ['SLURM_JOB_ID'],
              'family_freeze_sha256': frozen['sha256'], 'stages': stages,
              'interpretation': 'Offline arithmetic-candidate coverage only. Native future-length counts describe opportunity, not an online eligibility rule or semantic verification label.'}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({s: {k:v for k,v in d.items() if k != 'records'} for s,d in stages.items()}))


if __name__ == '__main__':
    main()
