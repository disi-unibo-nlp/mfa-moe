"""A changed future/diagnostic field cannot change online routing requests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))

import run_boundary_micro_screen as base
import run_discovery_feasibility_v2 as discovery


def test_forbidden_fields_do_not_change_requests():
    amendment = base.sealed(discovery.AMENDMENT)
    dictionary = base.sealed(discovery.DICTIONARY)
    manifest = {'stage': 'discovery_feasibility21', 'rows': deepcopy(amendment['rows']),
                'random_set_by_transition_family_seed':
                    amendment['random_set_by_transition_family_seed'],
                'seeds': [0, 1], 'max_tokens': 256, 'sha256': 'fixed-test-manifest'}
    manifest.update(discovery.workload_counts(manifest))
    from moe_steer import manifests as M
    world = M.load_world()
    table = base.build_policy_table(discovery.action_spec(dictionary))
    original = discovery.build_cases(manifest, manifest['rows'], [], world, table)
    mutated = deepcopy(manifest)
    for row in mutated['rows']:
        row['question'] = 'FORBIDDEN_FUTURE_TEXT_CHANGED'
        row['gpt_start_diagnostic'] = not row['gpt_start_diagnostic']
        row['native_veto_reader_votes'] = [False, False]
        row['future_native_completion'] = 'FORBIDDEN_FUTURE_TEXT_CHANGED'
        row['gold_answer'] = 'FORBIDDEN'
    alternative = discovery.build_cases(mutated, mutated['rows'], [], world, table)
    assert len(original) == len(alternative) == 480
    for (left, left_meta), (right, right_meta) in zip(original, alternative, strict=True):
        assert left.uid == right.uid
        assert left.prompt == right.prompt
        assert left.sampling == right.sampling
        assert left_meta == right_meta
