"""The additive fixture is deterministic and cannot waive unrelated failures."""
import copy
import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/experimental_resume'))
import overnight_routing_pulse_supplement_v4 as supplement
from moe_steer.engine import THINK_END_ID


def source_rows():
    rows = [{'tokens': [1]*1024} for _ in range(28)]
    for i in range(10): rows[i]['tokens'][23] = THINK_END_ID
    return rows


def test_first_fixed_long_original_fixture_without_search():
    assert supplement.choose_fixture(source_rows()) == 2


def test_full_completion_length_does_not_disguise_early_reasoning_closure():
    rows = source_rows(); rows[12]['tokens'][767] = THINK_END_ID
    with pytest.raises(ValueError, match='frozen existing long fixture'):
        supplement.choose_fixture(rows)


def test_earlier_newly_eligible_fixture_does_not_change_the_frozen_choice():
    rows = source_rows()
    for i in range(5): rows[i]['tokens'] = [1]*1024
    with pytest.raises(ValueError, match='frozen existing long fixture'):
        supplement.choose_fixture(rows)


def test_source_preserves_real_sampler_worker_and_only_three_requests():
    text = Path(supplement.__file__).read_text()
    assert "groups=[list(range(3))]" in text
    assert "worker_extension_cls='overnight_routing_worker_v2.OvernightWorkerExtension'" in text
    assert "Q.card_params(plan['horizon'],Q.crn(info,0),extra,presence=0." in text
    assert "common.audit_output_dose = corrected.audit_output_dose" in text
    assert "failed == [(20, ['second complete pulse not exercised'])]" in text


def test_prepared_plan_price_and_source_binding_match_saved_evidence():
    manifest = supplement.prepared()
    assert manifest['maximum_requests'] == 3 and manifest['maximum_decode_tokens'] == 3072
    assert [r['role'] for r in manifest['plan']] == ['native', 'bias', 'native_duplicate']
    assert [r['slots'] for r in manifest['plan']] == [[], [0,512], []]
    assert manifest['original_fixture_index'] == 2
    assert manifest['estimated_wall_seconds'] <= manifest['requested_wall_seconds'] == 3600
    assert str(Path(supplement.__file__).resolve()) in manifest['code_files']
