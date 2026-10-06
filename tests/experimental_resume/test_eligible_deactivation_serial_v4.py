"""CPU-only checks for counterbalanced negative-action diagnostic."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import run_boundary_micro_screen as base  # noqa: E402
import run_eligible_deactivation_serial_v4 as diagnostic  # noqa: E402


def test_v4_uses_positive_starts_sets_and_slot_balance():
    manifest = base.sealed(diagnostic.MANIFEST)
    positive = base.sealed(diagnostic.POSITIVE)
    rows, _, arms = diagnostic.validate(manifest, base.__file__)
    assert rows == positive['rows']
    assert manifest['random_set_by_family_seed'] == positive['random_set_by_family_seed']
    assert manifest['positive_immediate_manifest_sha256'] == positive['sha256']
    assert manifest['expected_requests'] == 156
    mapping = dict(zip((a['name'] for a in positive['arms']),
                       (a['name'] for a in arms), strict=True))
    for row in rows:
        family = row['family']
        for seed in (0, 1):
            assert manifest['arm_order_by_family_seed'][family][seed] == [
                mapping[name] for name in positive['arm_order_by_family_seed'][family][seed]]
    for transition in (*diagnostic.TRANSITIONS, 'all'):
        families = [row['family'] for row in rows if transition == 'all' or
                    row['transition'] == transition]
        positions = Counter((name, position) for family in families for seed in (0, 1)
                            for position, name in enumerate(
                                manifest['arm_order_by_family_seed'][family][seed]))
        for arm in arms:
            counts = [positions[arm['name'], index] for index in range(6)]
            assert max(counts) - min(counts) <= 1


def test_v4_rejects_assignment_drift():
    manifest = base.sealed(diagnostic.MANIFEST)
    family = manifest['rows'][0]['family']
    changed = copy.deepcopy(manifest)
    changed['random_set_by_family_seed'][family][0] = (
        changed['random_set_by_family_seed'][family][0] + 1) % 4
    with pytest.raises(ValueError, match='random controls'):
        diagnostic.validate(changed, base.__file__)
    changed = copy.deepcopy(manifest)
    changed['arm_order_by_family_seed'][family][0][:2] = list(reversed(
        changed['arm_order_by_family_seed'][family][0][:2]))
    with pytest.raises(ValueError, match='diagnostic arm orders'):
        diagnostic.validate(changed, base.__file__)
