import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('lookup_entrypoint',
    Path(__file__).resolve().parents[1] / 'scripts/research_campaign_v3_r6_lookup_r1/run.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


class LookupDispatchContracts(unittest.TestCase):
    def test_amendment_preserves_scientific_resources_and_registered_identity(self):
        from moe_exp.research_campaign_v3_r6.contracts import seal
        value = seal({'stage': 'reader_price', 'partition': 'boost_usr_prod',
                      'model': 'all', 'cpus': 16, 'memory_gib': 120, 'gpus': 2,
                      'minutes': 1440, 'reservation_hours': 384, 'shard_index': 0,
                      'comment': 'old'})
        result = entry.amend_card(value, 'lookup')
        for key, original in value.items():
            if key not in ('sha256', 'comment'):
                self.assertEqual(result[key], original)
        self.assertEqual(result['lookup_binding_sha256'], 'lookup')

    def test_protected_reporting_card_is_identical(self):
        value = {'ops_report_only': True, 'comment': 'protected', 'sha256': 'existing'}
        self.assertIs(entry.amend_card(value, 'lookup'), value)


if __name__ == '__main__':
    unittest.main()
