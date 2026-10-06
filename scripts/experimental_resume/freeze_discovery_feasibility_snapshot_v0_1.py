"""Snapshot independently rated discovery-start support before causal generation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery')
AGREEMENT = REPO / 'report/experimental-resume-v1/FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json'
SELECTED = BASE / 'FULLPREFIX_READER_AGREED_ELIGIBLE_POOL_v0.json'
OUT = REPO / 'report/experimental-resume-v1/CAUSAL_DISCOVERY_FEASIBILITY_SNAPSHOT_v0.1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed source seal')
    return value


def main():
    agreement, selected = sealed(AGREEMENT), sealed(SELECTED)
    fully_agreed = [row for row in agreement['records']
                    if row['v22_fired'] and row['qwen_status'] == 'accepted'
                    and row['native_status'] == 'accepted']
    by_transition = defaultdict(set)
    for row in fully_agreed:
        by_transition[row['transition']].add(row['family'])
    selected_by_transition = defaultdict(set)
    for row in selected['records']:
        selected_by_transition[row['transition']].add(row['family'])
    candidate, approach, failed = ('candidate_to_verify', 'approach_to_commit',
                                   'failed_check_to_revise')
    qwen_only_global = (len(selected_by_transition[candidate]) +
                        len(selected_by_transition[approach] - selected_by_transition[candidate]))
    high_agreement_global = (len(by_transition[candidate]) +
                             len(by_transition[approach] - by_transition[candidate]))
    if (len(by_transition[candidate]), len(by_transition[approach]),
            high_agreement_global) != (8, 6, 13):
        raise ValueError('independent start support differs from prefreeze review')
    body = {'schema': 'routing-discovery-feasibility-snapshot-v0.1',
            'status': 'PRETREATMENT_SUPPORT_ONLY_THIRD_AUDIT_PENDING',
            'fullprefix_agreement_sha256': agreement['sha256'],
            'qwen_selected_pool_sha256': selected['sha256'],
            'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'registered_maximum_discovery_families': 48,
            'qwen_only_selected_counts': {key: len(selected_by_transition[key])
                                          for key in (candidate, approach, failed)},
            'qwen_only_selected_candidate_priority_globally_disjoint': qwen_only_global,
            'qwen_and_native_both_accept_counts': {key: len(by_transition[key])
                                                   for key in (candidate, approach, failed)},
            'qwen_and_native_candidate_priority_globally_disjoint': high_agreement_global,
            'priority_if_both_actions_qualified': [candidate, approach],
            'failed_check_status': 'omitted: zero jointly Qwen/native accepted discovery families',
            'next_gate': 'independent GPT-OSS full-prefix start audit with prospectively fixed rule, then seal exact IDs and distinct-family enrollment before interventions',
            'limits': '13 is pretreatment high-agreement support, not a completed 48-family discovery stage; no mechanism128, utility96, causal efficacy, human truth or validation claim'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('feasibility snapshot changed')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'high_agreement_global_families': high_agreement_global,
                      'qwen_only_selected_global_families': qwen_only_global}))


if __name__ == '__main__':
    main()
