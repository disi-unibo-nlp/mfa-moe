"""Inspect only discovery-family native sentence labels for trigger-rubric design."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
OUT = REPO / 'report/experimental-resume-v1/DISCOVERY_LABEL_EXAMPLES.json'


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.startswith('login'):
        raise RuntimeError('label inspection requires CPU Slurm')
    frozen = json.loads((REPO / 'report/experimental-resume-v1/family-freeze.json').read_text())
    pools = frozen['new_parent_pools']
    questions = {pools['representative_questions'][family]
                 for family in pools['parent_pools']['discovery']}
    rows = []
    for path in sorted((ROOT / 'campaign-v3/labels/qwen36').glob('part-*/annotations.json')):
        for record in json.loads(path.read_text()):
            identity = record['identity']
            question = identity['dataset'] + '|' + identity['source_problem_id']
            if question not in questions:
                continue
            label = record.get('label')
            if label not in ('Explore', 'Verify', 'Plan', 'Implement'):
                continue
            rows.append({'question': question, 'label': label,
                         'sentence_index': identity['sentence_index'],
                         'sentence': record['unit']['text'],
                         'previous_sentence': record['inputs']['previous_sentence']})
    classes = Counter(r['label'] for r in rows)
    selected = []
    for label in ('Explore', 'Verify', 'Plan', 'Implement'):
        subset = (r for r in rows if r['label'] == label)
        selected.extend(sorted(subset, key=lambda r: hashlib.sha256(
            json.dumps(r, sort_keys=True).encode()).hexdigest())[:20])
    result = {'schema': 'discovery-label-examples-v1', 'job_id': os.environ['SLURM_JOB_ID'],
              'family_freeze_sha256': frozen['sha256'], 'discovery_questions': len(questions),
              'class_counts': dict(classes), 'examples': selected,
              'scope': 'discovery only; historical sparse labels, no contiguous transition inference'}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'class_counts': dict(classes), 'examples': len(selected)}))


if __name__ == '__main__':
    main()
