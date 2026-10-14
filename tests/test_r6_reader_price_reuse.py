import unittest
from unittest.mock import patch
from pathlib import Path
import importlib.util
from moe_exp.research_campaign_v3_r6_reader_price_r1 import measured_price, resolved_stage, remaining_block

spec = importlib.util.spec_from_file_location('reader_price_entry',
    Path(__file__).resolve().parents[1] / 'scripts/research_campaign_v3_r6_reader_price_r1/run.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


class ReaderPriceReuseContracts(unittest.TestCase):
    def fixture(self):
        a = {'config': {'max_tokens': 8192}, 'revision': 'fixed', 'native_context': 32768, 'engine_changes': []}
        rows = [{'uid': str(i), 'model': 'nemotron'} for i in range(64)]
        values = [{'assignment': row, 'status': 'complete', 'configuration': a['config'],
                   'reader_revision': a['revision'], 'native_context': a['native_context'],
                   'engine_changes': [], 'seconds': 2, 'generated_tokens': 64,
                   'input_tokens': 9726, 'load_seconds': 400} for row in rows]
        coverage = {'maximum_input_covered': True, 'maximum_required_input_tokens': 9726,
                    'original_table_sha256': 'original', 'supplemental_table_sha256': 'measured', 'sha256': 'coverage'}
        original = [{'input_tokens': 9140, 'context_eligible': True}]
        return {'annotator': a}, rows, values, coverage, original

    def test_existing_maximum_input_prices_originals_without_draw(self):
        result = measured_price(*self.fixture())
        self.assertEqual(result['seconds_per_context_upper'], 256)
        self.assertEqual(result['original_price_only_assignments_unexecuted'], 64)
        self.assertEqual(result['new_scientific_draws'], 0)

    def test_uncovered_input_or_missing_timing_rejected(self):
        args = list(self.fixture()); args[-1][0]['input_tokens'] = 10000
        with self.assertRaises(ValueError): measured_price(*args)
        args = list(self.fixture()); args[2][0]['seconds'] = 0
        with self.assertRaises(ValueError): measured_price(*args)

    def test_different_reader_profile_rejected(self):
        args = list(self.fixture()); args[2][0]['reader_revision'] = 'different'
        with self.assertRaises(ValueError): measured_price(*args)

    def test_partial_or_duplicate_coverage_rejected(self):
        args = list(self.fixture()); args[1][1] = args[1][0]
        with self.assertRaises(ValueError): measured_price(*args)
        args = list(self.fixture()); args[2][0]['status'] = 'execution_failed'
        with self.assertRaises(ValueError): measured_price(*args)

    def test_only_exact_resolved_failure_gets_a_scientific_price_view(self):
        failure = {'sha256': 'failed', 'status': 'execution_failed', 'job_id': 'old'}
        resolved = {'resolved_failure_sha256': 'failed', 'sha256': 'resolution'}
        view = resolved_stage(failure, resolved)
        self.assertEqual(failure['status'], 'execution_failed')
        self.assertEqual(view['status'], 'complete')
        self.assertEqual(view['job_id'], 'old')
        other = {**failure, 'sha256': 'other'}
        self.assertIs(resolved_stage(other, resolved), other)

    def test_unrelated_shared_reader_failure_still_blocks(self):
        receipts = {
            'resolved': {'failure': {'sha256': 'failed'}, 'scope': 'shared',
                         'dependent_stages': ['reader_price', 'annotate'], 'model': 'all'},
            'other': {'failure': {'sha256': 'other'}, 'scope': 'shared',
                      'dependent_stages': ['annotate'], 'model': 'all'},
        }
        with patch.object(Path, 'glob', return_value=list(receipts)), \
             patch('moe_exp.research_campaign_v3_r6_reader_price_r1.load', side_effect=receipts.__getitem__):
            resolution = {'resolved_failure_sha256': 'failed'}
            self.assertEqual(remaining_block({'root': '/campaign'}, 'annotate', 'qwen35', resolution), receipts['other'])
            self.assertIsNone(remaining_block({'root': '/campaign'}, 'reader_price', 'all', resolution))

    def test_price_binding_preserves_resources_and_original_assignments(self):
        original = {'stage': 'annotate', 'gpus': 2, 'cpus': 16, 'memory_gib': 120,
                    'shard_index': 2, 'minutes': 60, 'reservation_hours': 16,
                    'cpu_allocation_binding_sha256': 'allocation', 'sha256': 'old', 'comment': 'old'}
        amended = entry.amend_card(original, {'sha256': 'price'})
        for key, value in original.items():
            if key not in ('sha256', 'comment'):
                self.assertEqual(amended[key], value)
        self.assertEqual(amended['reader_price_binding_sha256'], 'price')

    def test_reporting_owner_remains_identical(self):
        value = {'ops_report_only': True, 'sha256': 'protected'}
        self.assertIs(entry.amend_card(value, {'sha256': 'price'}), value)


if __name__ == '__main__': unittest.main()
