"""Audit nonfire selection preserves family coverage and fixed limits."""
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/prepare_transition_audit.py'
SPEC = importlib.util.spec_from_file_location('transition_audit_selection_test', SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_nonfire_sample_is_deterministic_and_covers_families_first():
    rows = [{'uid': f'f{family}-{i}', 'analysis_meta': {'family': f'f{family}'}}
            for family in range(4) for i in range(4)]
    first = AUDIT.select_nonfires(rows, limit=4)
    assert len(first) == 4
    assert {row['analysis_meta']['family'] for row in first} == {'f0', 'f1', 'f2', 'f3'}
    assert first == AUDIT.select_nonfires(list(reversed(rows)), limit=4)
