"""Bounded scheduling-adapter tests; scientific calculations stay frozen."""
from pathlib import Path
import sys
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import analyze_routing_first_stage_v2 as adapter
import submit_routing_first_stage_v2 as attach


class AdapterTests(unittest.TestCase):
    def test_helper_and_all_unchanged_sources_are_bound(self):
        files = adapter.code_files()
        helper = str(Path(adapter.__file__).with_name('dispatch_overnight_readers_v1.py'))
        self.assertEqual(files[helper], adapter.HELPER_SHA)
        prior = adapter.source.sealed(adapter.PRIOR)
        for path, sha in prior['code_files'].items():
            if path != helper:
                self.assertEqual(files[path], sha)

    def test_adapter_retains_frozen_functions_and_uses_new_root(self):
        with mock.patch.object(adapter, 'validate_plan', return_value={'sha256': 'p'}), \
             mock.patch.object(adapter.original, 'main') as main, \
             mock.patch.object(adapter.original, 'validate_plan'), \
             mock.patch.object(adapter.original, 'output_path'):
            adapter.main()
            main.assert_called_once_with()
            self.assertIs(adapter.original.validate_plan, adapter.validate_plan)
            self.assertIs(adapter.original.output_path, adapter.output_path)
        path = adapter.output_path({'sha256': 'a'*64}, Path('parent'), {'sha256': 'b'*64})
        self.assertTrue(path.name.startswith('routing-first-stage-v2-'))
        self.assertEqual(attach.WRAPPER.name, 'analyze_routing_first_stage_v2.sbatch')

    def test_changed_dependency_rejected(self):
        actual = adapter.source.file_sha
        def corrupted(path):
            if str(path).endswith('dispatch_overnight_readers_v1.py'):
                return '0' * 64
            return actual(path)
        with mock.patch.object(adapter.source, 'file_sha', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                adapter.code_files()


if __name__ == '__main__':
    unittest.main()
