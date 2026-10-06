"""M9's 16k native reuse and deterministic integer control must charge caps wrong."""
import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experimental_resume/prepare_m9_grading.py'
SPEC = importlib.util.spec_from_file_location('m9_grading_mapping_test', SOURCE)
M9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M9)


def test_native_score_reuse_only_after_natural_stop_by_16k():
    score = {'acc32': 1, 'acc131': 1}
    assert M9.native_16k({'native_endpoint_natural_stop': True}, score) is True
    assert M9.native_16k({'native_endpoint_natural_stop': False}, score) is False


def test_integer_comparator_statuses_do_not_impute_success_from_full_native():
    score = {'acc32': 1, 'acc131': 1}
    base = {'native_endpoint_natural_stop': False}
    for status in ('retain_native_finished', 'fallback_native_no_integer',
                   'failed_source_prefix', 'failed_emission_budget'):
        assert M9.comparator_native_or_fail({**base, 'integer_comparator': {'status': status}}, score) is False
    assert M9.comparator_native_or_fail({**base, 'integer_comparator': {'status': 'integer_emission'}}, score) is None
