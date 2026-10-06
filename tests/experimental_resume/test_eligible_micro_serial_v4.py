"""CPU-only checks for the sealed counterbalanced eligible pilot."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import run_boundary_micro_screen as base  # noqa: E402
import run_eligible_micro_serial_v4 as screen  # noqa: E402


def _manifest():
    return base.sealed(screen.MANIFEST)


def test_exact_seal_and_counterbalances():
    manifest = _manifest()
    rows, _, arms = screen.validate(manifest, base.__file__)
    assert Counter(row['transition'] for row in rows) == {
        'candidate_to_verify': 8, 'approach_to_commit': 5}
    assert manifest['expected_requests'] == 156
    names = [arm['name'] for arm in arms]
    for transition in (*screen.TRANSITIONS, 'all'):
        families = [row['family'] for row in rows if transition == 'all' or
                    row['transition'] == transition]
        positions = Counter((name, position) for family in families for seed in (0, 1)
                            for position, name in enumerate(
                                manifest['arm_order_by_family_seed'][family][seed]))
        for name in names:
            counts = [positions[name, index] for index in range(6)]
            assert max(counts) - min(counts) <= 1
        if transition == 'all':
            continue
        for seed in (0, 1):
            sets = Counter(manifest['random_set_by_family_seed'][family][seed]
                           for family in families)
            counts = [sets[index] for index in range(4)]
            assert max(counts) - min(counts) <= 1


def test_assignment_tampering_fails_validation():
    manifest = _manifest()
    family = manifest['rows'][0]['family']
    changed = copy.deepcopy(manifest)
    changed['arm_order_by_family_seed'][family][0][0:2] = list(reversed(
        changed['arm_order_by_family_seed'][family][0][0:2]))
    with pytest.raises(ValueError, match='serial arm orders'):
        screen.validate(changed, base.__file__)
    changed = copy.deepcopy(manifest)
    changed['random_set_by_family_seed'][family][0] = (
        changed['random_set_by_family_seed'][family][0] + 1) % 4
    with pytest.raises(ValueError, match='random controls'):
        screen.validate(changed, base.__file__)
