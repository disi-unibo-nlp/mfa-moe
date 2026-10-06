"""Adjudication repairs a checker defect and preserves genuine coverage failures."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/experimental_resume'))
import adjudicate_overnight_operator_qualification_v3 as adjudication
import overnight_routing_dose_audit_v3 as corrected
import overnight_routing_runner_v2 as runner
from test_overnight_routing_dose_audit_v3 import fixture


def case():
    record, manifest = fixture()
    record.update(uid='fixture', closed=False, horizon=1024, operator='force', policy='test')
    route = np.broadcast_to(np.array([9, 189, 1, 2, 3, 4, 5, 6]), (len(record['tokens']), 40, 8)).copy()
    old = {'uid': 'fixture', 'pass': False, 'reasons': [adjudication.CHECKER_REASON]}
    return record, manifest, route, old


def test_checker_only_failure_is_adjudicated_with_exact_raw_force_membership():
    record, manifest, route, old = case()
    result = adjudication.check_one(record, old, manifest, route, runner, corrected)
    assert result['pass'] and result['superseded_checker_reason'] == adjudication.CHECKER_REASON
    assert result['original_pass'] is False


def test_independent_preemption_or_coverage_failure_is_never_removed():
    record, manifest, route, old = case()
    old['reasons'].append('missing force/reweight preemption and recompute')
    result = adjudication.check_one(record, old, manifest, route, runner, corrected)
    assert not result['pass'] and 'missing force/reweight preemption and recompute' in result['reasons']


def test_raw_executed_ids_override_an_optimistic_counter():
    record, manifest, route, old = case(); route[0, 28, 1] = 40
    result = adjudication.check_one(record, old, manifest, route, runner, corrected)
    assert not result['pass'] and any('raw executed force IDs' in r for r in result['reasons'])


def test_incorrect_weight_dose_is_not_hidden_by_combined_original_error():
    record, manifest, route, old = case()
    for rank in ('0', '1'): record['action_dose'][rank]['dose']['test']['28']['weight_l1'] = 49
    result = adjudication.check_one(record, old, manifest, route, runner, corrected)
    assert not result['pass'] and result['superseded_checker_reason'] is None


def test_v3_adapter_binds_sources_and_provenance_without_changing_v2_schema(monkeypatch):
    import overnight_routing_adjudicated_v3 as adapter
    monkeypatch.setattr(adapter, 'INSTALLED', False)
    receipt = {'sha256': 'adjudicated', 'original_result_path': '/unchanged/original',
               'original_result_sha256': 'original-failure'}
    monkeypatch.setattr(adjudication, 'qualified_result', lambda: receipt)
    monkeypatch.setattr(runner, 'qualified_result', runner.qualified_result)
    monkeypatch.setattr(runner, 'validate', lambda manifest, driver: ('validated',))
    monkeypatch.setattr(runner.common, 'audit_output_dose', runner.common.audit_output_dose)
    monkeypatch.setattr(runner.common, 'write_once', lambda path, body: {**body, 'sha256': runner.base.digest(body)})
    adapter.install()
    manifest = runner.common.write_once(Path('/unused'), {'schema': 'overnight-routing-manifest-v2', 'code_files': {}})
    assert manifest['schema'] == 'overnight-routing-manifest-v2'
    assert manifest['cpu_checker_adjudication']['result_sha256'] == 'adjudicated'
    assert manifest['cpu_checker_adjudication']['engine_changed'] is False
    assert runner.validate(manifest, 'unused') == ('validated',)
    manifest['cpu_checker_adjudication']['result_sha256'] = 'other'
    with pytest.raises(ValueError, match='adjudicated checker'):
        runner.validate(manifest, 'unused')
