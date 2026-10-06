"""Audit seven-class discovery dynamics without joining sparse window gaps."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
UNITS = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed input seal: {path}')
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('dense dynamics summary requires CPU Slurm')
    sys.path.insert(0, str(REPO / 'src'))
    from moe_exp.routing_control.analysis import CLASSES, class_summary
    units = sealed(UNITS)
    code_sha = hashlib.sha256((REPO / 'scripts/experimental_resume/label_dense_discovery.py').read_bytes()).hexdigest()
    directory = UNITS.parent / f"results-{units['sha256'][:8]}-{code_sha[:8]}"
    binding, complete = sealed(directory / 'BINDING.json'), sealed(directory / 'SUMMARY.json')
    if binding['units_sha256'] != units['sha256'] or binding['driver_sha256'] != code_sha:
        raise ValueError('dense label binding changed')
    if complete['binding_sha256'] != binding['sha256'] or complete['sentences'] != len(units['records']):
        raise ValueError('dense labels incomplete')
    rows = []
    for start in range(0, len(units['records']), binding['batch_size']):
        part = sealed(directory / 'batches' / f'{start:06d}.json')
        if part['binding_sha256'] != binding['sha256'] or part['start'] != start:
            raise ValueError('dense label batch binding/order changed')
        rows.extend(part['records'])
    if len(rows) != len(units['records']):
        raise ValueError('dense label record count differs')
    groups = defaultdict(list)
    misses = Counter()
    for unit, label in zip(units['records'], rows, strict=True):
        ident = (unit['family'], unit['attempt_id'], unit['sentence_index'])
        if ident != (label['family'], label['attempt_id'], label['sentence_index']):
            raise ValueError('sentence/label identity differs')
        if unit['token_start'] >= unit['token_end'] or unit['segment'] != 0:
            raise ValueError('invalid discovery sentence token span/segment')
        if label['label'] not in CLASSES or label['finish_reason'] != 'stop':
            misses[unit['family']] += 1
            continue
        groups[unit['family']].append({k: unit[k] for k in ('sentence_index', 'segment', 'token_start', 'token_end')}
                                      | {'label': label['label']})
    overall = Counter()
    transitions = Counter()
    loops = Counter()
    dwell = defaultdict(list)
    reentry = Counter()
    family_summary = {}
    for family in sorted({r['family'] for r in units['records']}):
        selected = sorted(groups[family], key=lambda r: r['sentence_index'])
        result = class_summary(selected)
        class_counts = Counter(r['label'] for r in selected)
        overall.update(class_counts)
        for i, left in enumerate(CLASSES):
            for j, right in enumerate(CLASSES):
                transitions[left, right] += result['transition_counts'][i][j]
        loops.update(result['loop_counts'])
        reentry.update(result['reentries'])
        for label, lengths in result['dwell_sentences_observed_including_censoring'].items():
            dwell[label].extend(lengths)
        family_summary[family] = {'classified_sentences': len(selected),
                                  'unparsed_or_nonstop': misses[family],
                                  'class_counts': dict(class_counts),
                                  'transitions': sum(sum(row) for row in result['transition_counts']),
                                  'explore_to_plan_or_implement': result['transition_counts'][CLASSES.index('Explore')][CLASSES.index('Plan')]
                                    + result['transition_counts'][CLASSES.index('Explore')][CLASSES.index('Implement')],
                                  'verify_to_explore_or_plan': result['transition_counts'][CLASSES.index('Verify')][CLASSES.index('Explore')]
                                    + result['transition_counts'][CLASSES.index('Verify')][CLASSES.index('Plan')]}
    matrix = [[transitions[a,b] for b in CLASSES] for a in CLASSES]
    probabilities = [[n / sum(row) if sum(row) else None for n in row] for row in matrix]
    outcome = {'schema': 'dense-discovery-class-summary-v1', 'job_id': os.environ['SLURM_JOB_ID'],
               'units_sha256': units['sha256'], 'label_summary_sha256': complete['sha256'],
               'families': len(family_summary), 'class_counts': dict(overall),
               'transition_counts': matrix, 'transition_probabilities': probabilities,
               'dwell_sentences_observed_including_censoring': dict(dwell),
               'reentries': dict(reentry), 'loop_counts': dict(loops),
               'family_summary': family_summary,
               'interpretation': ('Exploratory direct-LLM seven-class audit of fixed, contiguous native windows. '
                                  'No joins across unselected indices or failed labels. Class transitions do not '
                                  'establish substantive checks, causal control, or online trigger eligibility.')}
    outcome['sha256'] = digest(outcome)
    path = directory / 'CLASS_DYNAMICS.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(outcome, indent=1) + '\n')
    print(json.dumps({'path': str(path), 'families': len(family_summary),
                      'classified_sentences': sum(overall.values()),
                      'class_transitions': sum(sum(row) for row in matrix)}))


if __name__ == '__main__':
    main()
