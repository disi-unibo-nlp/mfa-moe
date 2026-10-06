"""The legacy X3 builder must enforce a complete, amended GPU stage price."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import x3_build_v3 as X  # noqa: E402


def frozen():
    report = REPO / 'report/experimental-resume-v1'
    return tuple(X.verified(report / name) for name in (
        'X3_COMPLETE_STAGE_PRICE_v3.json', 'X3_RESOURCE_AMENDMENT_v1.json',
        'X3_ELIGIBILITY_v1.json', 'X3_G3_SELECTION_v2.json'))


def test_complete_price_remains_held_for_new_study_priority():
    price, amendment, eligible, selected = frozen()
    with pytest.raises(ValueError, match='priority'):
        X.assert_complete_stage_price(price, amendment, eligible, selected)
    later = dict(price, new_study_priority_resolved=True)
    assert X.assert_complete_stage_price(later, amendment, eligible, selected) == price['all_in_GPU_h']


def test_missing_component_and_over_ceiling_refused():
    price, amendment, eligible, selected = frozen()
    price = copy.deepcopy(price)
    price['new_study_priority_resolved'] = True
    price['components_GPU_h'].pop('native_nll')
    with pytest.raises(ValueError, match='all seven'):
        X.assert_complete_stage_price(price, amendment, eligible, selected)
    price, amendment, eligible, selected = frozen()
    price = copy.deepcopy(price)
    price['new_study_priority_resolved'] = True
    price['components_GPU_h']['labeling'] += 20
    price['all_in_GPU_h'] += 20
    with pytest.raises(ValueError, match='exceeds'):
        X.assert_complete_stage_price(price, amendment, eligible, selected)
