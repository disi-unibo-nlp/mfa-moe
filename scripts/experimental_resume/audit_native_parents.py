"""Count existing Qwen native label support for the frozen routing-study parents.

This is an observational feasibility audit. Sparse labels are joined only when
their sentence indices are consecutive within the same attempt and segment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
SENTENCES = ROOT / 'dynamics-routing/results/qwen36/A/sentences.parquet'
ATTEMPTS = ROOT / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
OUT = REPO / 'report/experimental-resume-v1/NATIVE_PARENT_ELIGIBILITY_AUDIT.json'
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('native parent audit requires CPU Slurm')
    import pandas as pd

    frozen = json.loads(FAMILY.read_text())
    pools = frozen['new_parent_pools']
    selected = {pools['representative_questions'][f]: (stage, f)
                for stage in ('discovery', 'mechanism', 'utility')
                for f in pools['parent_pools'][stage]}
    att = pd.read_parquet(ATTEMPTS, columns=['question'])
    sent = pd.read_parquet(SENTENCES, columns=['attempt_id', 'question',
                                              'sentence_index', 'segment', 'cls'])
    sent = sent.loc[sent['question'].isin(selected)].sort_values(
        ['attempt_id', 'sentence_index'], kind='stable')
    counts = defaultdict(Counter)
    for row in sent.itertuples(index=False):
        if not 0 <= int(row.cls) < len(CLASSES):
            raise ValueError('invalid native class')
        counts[row.question][CLASSES[int(row.cls)]] += 1
    pairs = defaultdict(Counter)
    previous = None
    for row in sent.itertuples(index=False):
        if previous is not None and row.attempt_id == previous.attempt_id \
                and int(row.sentence_index) == int(previous.sentence_index) + 1 \
                and int(row.segment) == int(previous.segment) and row.question == previous.question:
            pairs[row.question][CLASSES[int(previous.cls)] + '>' + CLASSES[int(row.cls)]] += 1
        previous = row
    stage_summary = {}
    for stage in ('discovery', 'mechanism', 'utility'):
        families = pools['parent_pools'][stage]
        rows = []
        for family in families:
            question = pools['representative_questions'][family]
            labels, adjacent = counts[question], pairs[question]
            rows.append({'family': family, 'question': question,
                         'native_label_count': sum(labels.values()),
                         'class_counts': dict(labels), 'adjacent_pair_counts': dict(adjacent),
                         'has_verify': labels['Verify'] > 0,
                         'has_explore': labels['Explore'] > 0,
                         'has_plan_or_implement': labels['Plan'] + labels['Implement'] > 0,
                         'has_adjacent_explore_to_commit': adjacent['Explore>Plan'] + adjacent['Explore>Implement'] > 0,
                         'has_adjacent_candidate_to_verify_upper_bound': sum(v for k, v in adjacent.items() if k.endswith('>Verify')) > 0})
        stage_summary[stage] = {
            'families': len(rows),
            'questions_with_native_attempt': sum(r['question'] in set(att['question']) for r in rows),
            'with_any_label': sum(r['native_label_count'] > 0 for r in rows),
            'with_verify': sum(r['has_verify'] for r in rows),
            'with_adjacent_explore_to_commit': sum(r['has_adjacent_explore_to_commit'] for r in rows),
            'with_adjacent_candidate_to_verify_upper_bound': sum(r['has_adjacent_candidate_to_verify_upper_bound'] for r in rows),
            'records': rows,
        }
    result = {'schema': 'native-parent-eligibility-audit-v1',
              'job_id': os.environ['SLURM_JOB_ID'],
              'family_freeze_sha256': frozen['sha256'],
              'input_sha256': {str(path): sha(path) for path in (FAMILY, SENTENCES, ATTEMPTS)},
              'note': 'Historical labels are sparse; adjacent-pair counts are lower bounds and candidate-to-verify is an upper bound without prefix semantics. This does not establish behavioral eligibility.',
              'stages': stage_summary}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({stage: {key: value for key, value in data.items() if key != 'records'}
                      for stage, data in stage_summary.items()}))


if __name__ == '__main__':
    main()
