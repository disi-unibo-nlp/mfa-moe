import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from moe_exp.research_campaign_v3_r6.contracts import seal
from moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1 import failure_identity, remaining_block

spec = importlib.util.spec_from_file_location('failure_id_entry',
    Path(__file__).resolve().parents[1] / 'scripts/research_campaign_v3_r6_reader_price_failure_id_r1/run.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


class FailureIdentityContracts(unittest.TestCase):
    def test_embedded_unsealed_payload_matches_exact_saved_failure(self):
        payload = {'job_id': 'old', 'stage': 'reader_price', 'status': 'execution_failed',
                   'error': 'original annotator execution price unavailable'}
        failure = seal(payload)
        self.assertEqual(failure_identity(payload), failure['sha256'])
        self.assertEqual(failure_identity(failure), failure['sha256'])

    def test_inconsistent_embedded_seal_rejected(self):
        with self.assertRaises(ValueError): failure_identity({'job_id': 'old', 'sha256': 'wrong'})

    def test_only_exact_payload_is_resolved_even_with_identical_error(self):
        payload = {'job_id': 'old', 'error': 'same assertion'}
        other = {**payload, 'job_id': 'other'}
        receipts = {'resolved': {'failure': payload, 'dependent_stages': ['annotate'],
                                 'scope': 'shared', 'model': 'all'},
                    'other': {'failure': other, 'dependent_stages': ['annotate'],
                              'scope': 'shared', 'model': 'all'}}
        with patch.object(Path, 'glob', return_value=list(receipts)), \
             patch('moe_exp.research_campaign_v3_r6_reader_price_failure_id_r1.load', side_effect=receipts.__getitem__):
            resolution = {'resolved_failure_sha256': failure_identity(payload)}
            self.assertEqual(remaining_block({'root': '/campaign'}, 'annotate', 'nemotron', resolution), receipts['other'])
            self.assertIsNone(remaining_block({'root': '/campaign'}, 'reader_price', 'all', resolution))

    def test_reporting_card_and_scientific_resources_preserved(self):
        report = {'ops_report_only': True, 'sha256': 'protected'}
        self.assertIs(entry.amend_card(report, {'sha256': 'new'}), report)
        original = {'stage': 'annotate', 'gpus': 2, 'cpus': 16, 'memory_gib': 120,
                    'minutes': 60, 'reservation_hours': 16, 'reader_price_binding_sha256': 'old'}
        result = entry.amend_card(original, {'sha256': 'new'})
        for key, value in original.items(): self.assertEqual(result[key], value)
        self.assertEqual(result['reader_price_failure_id_binding_sha256'], 'new')


if __name__ == '__main__': unittest.main()
