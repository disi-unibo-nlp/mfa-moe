"""Freeze 200 discovery-only historical label fixtures for a new judge audit."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
OUT = ROOT / 'steering-v1/runs/routing-control-v1/dense-judge-parity/FIXTURES.json'
PROGRAM = REPO / 'results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json'
PROGRAM_SHA = '467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07'
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('fixture preparation requires CPU Slurm')
    if hashlib.sha256(PROGRAM.read_bytes()).hexdigest() != PROGRAM_SHA:
        raise ValueError('pinned historical judge program changed')
    family = json.loads((REPO / 'report/experimental-resume-v1/family-freeze.json').read_text())
    pools = family['new_parent_pools']
    qf = {pools['representative_questions'][f]: f for f in pools['parent_pools']['discovery']}
    by_class = defaultdict(list)
    for path in sorted((ROOT / 'campaign-v3/labels/qwen36').glob('part-*/annotations.json')):
        for row in json.loads(path.read_text()):
            identity, inputs, label = row['identity'], row['inputs'], row.get('label')
            question = identity['dataset'] + '|' + identity['source_problem_id']
            if question not in qf or label not in CLASSES:
                continue
            item = {'family': qf[question], 'question': question,
                    'identity': {k: identity[k] for k in ('dataset', 'problem_id', 'sample_id', 'sentence_index', 'trace_sha256')},
                    'inputs': {k: inputs[k] for k in ('problem_statement', 'previous_sentence', 'sentence', 'next_sentence')},
                    'historical_label': label}
            by_class[label].append(item)
    # Frozen balanced class selection; no outcome or held-out family enters ranking.
    selected = []
    for label in CLASSES:
        rows = sorted(by_class[label], key=lambda row: digest(['dense-judge-parity-v1', row['identity']]))
        take = 29 if label in CLASSES[:4] else 28
        if len(rows) < take:
            raise ValueError('insufficient discovery labels for balanced parity fixture')
        selected.extend(rows[:take])
    if len(selected) != 200 or len({digest(r['identity']) for r in selected}) != 200:
        raise ValueError('parity fixture count or identity duplication')
    selected.sort(key=lambda row: digest(['execution-order-v1', row['identity']]))
    value = {'schema': 'dense-judge-parity-fixtures-v1', 'job_id': os.environ['SLURM_JOB_ID'],
             'family_freeze_sha256': family['sha256'], 'program_sha256': PROGRAM_SHA,
             'selection': 'discovery representative families only; 29 each for first four classes, 28 each for last three; deterministic identity hash',
             'fixtures': selected}
    value['sha256'] = digest(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(OUT), 'fixtures': len(selected), 'sha256': value['sha256'],
                      'families': len({r['family'] for r in selected})}))


if __name__ == '__main__':
    main()
