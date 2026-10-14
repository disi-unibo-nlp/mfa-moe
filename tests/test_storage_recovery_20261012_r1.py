import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_20261012',
    REPO / 'scripts/storage_recovery_20261012_r1.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class CleanupProofs(unittest.TestCase):
    def test_raw_array_id_joins_display_id_and_protects_live_siblings(self):
        with patch.object(recovery.cleanup, 'command', return_value=
                '59342883|59342858_3|COMPLETED|120|2026-09-30T00:00:00|'):
            result = recovery.accounting(['59342883'])
        self.assertEqual(result['59342883']['display_job_id'], '59342858_3')
        self.assertTrue(recovery.is_live('59342883', {'59342858_4'}))
        self.assertFalse(recovery.is_live('59342883', {'59975083'}))

    def test_unlisted_job_cannot_be_queried_for_deletion(self):
        with self.assertRaisesRegex(ValueError, 'outside audited'):
            recovery.accounting(['59975083'])

    def test_all_binding_key_variants_are_protected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'report') as directory:
            root = Path(directory)
            campaign = root / 'report/prior'
            campaign.mkdir(parents=True)
            bound = root / 'bound.py'
            (campaign / 'CAMPAIGN.json').write_text(json.dumps({
                'preserved_scientific_files': {str(bound): 'digest'},
                'preserved_operational_files': {str(root / 'operation.py'): 'digest'}}))
            with patch.object(recovery, 'REPO', root):
                paths = recovery.protected_paths()
            self.assertIn(str(bound), paths)
            self.assertIn(str(root / 'operation.py'), paths)

    def test_apply_still_requires_allocated_step(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'allocated Slurm step'):
                recovery.cleanup.apply(Path('/unused'))


if __name__ == '__main__':
    unittest.main()
