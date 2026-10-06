"""CPU-only prospective layout check for the 21-family full feasibility stage."""
from __future__ import annotations

from collections import Counter
import json

import run_boundary_micro_screen as base
import run_discovery_feasibility_v1 as discovery


def main():
    amendment, dictionary = (base.sealed(p) for p in
                             (discovery.AMENDMENT, discovery.DICTIONARY))
    manifest = {'stage': 'discovery_feasibility21', 'rows': amendment['rows'],
                'random_set_by_transition_family_seed':
                    amendment['random_set_by_transition_family_seed'],
                'seeds': [0, 1], 'max_tokens': 256, 'sha256': 'cpu-layout-only'}
    counts = discovery.workload_counts(manifest)
    manifest.update(counts)
    from moe_steer import manifests as M
    cases = discovery.build_cases(manifest, manifest['rows'],
                                  [{'name': 'active'}, {'name': 'sentinel'}],
                                  M.load_world(),
                                  base.build_policy_table(discovery.action_spec(dictionary)))
    enrolled = [m for _, m in cases if m['analysis_enrolled']]
    technical = [m for _, m in cases if m['technical_filler']]
    by_transition = Counter(m['transition'] for m in enrolled if m['slot'] == 'active')
    if (len(cases) != counts['expected_requests'] or len(cases) != 480 or
            len(enrolled) != 420 or len(technical) != 60 or
            by_transition != {'candidate_to_verify': 120, 'approach_to_commit': 90}):
        raise ValueError('full-stage analysis slots, padding or transitions differ')
    print(json.dumps({'status': 'PASS_DISCOVERY_FULL_LAYOUT_CPU_PREFLIGHT',
                      'amendment_sha256': amendment['sha256'],
                      'requests': len(cases), 'batches': len(cases) // 8,
                      'analysis_slots': len(enrolled), 'technical_fillers': len(technical),
                      'per_transition_active_slots': by_transition,
                      **counts}))


if __name__ == '__main__':
    main()
