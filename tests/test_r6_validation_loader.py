"""Reproduce unittest's cached-directory failure and verify independent discovery."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from moe_exp.research_campaign_v3_r6.contracts import REPO, STAGING

spec = importlib.util.spec_from_file_location('r6_loader_candidate',
    REPO / 'scripts/research_campaign_v3_r6_validation_candidate_r6.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class LoaderBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r6-loader-', dir=STAGING)
        self.root = Path(self.temp.name)
        self.directories = []
        for index in (0, 1):
            directory = self.root / ('suite' + str(index)); directory.mkdir()
            (directory / ('test_loader_case_' + str(index) + '.py')).write_text(
                'import unittest\nclass Contract(unittest.TestCase):\n'
                '    def test_contract(self): self.assertTrue(True)\n')
            self.directories.append(directory)

    def tearDown(self):
        import sys
        for index in (0, 1):
            sys.modules.pop('test_loader_case_' + str(index), None)
        self.temp.cleanup()

    def test_shared_loader_reproduces_the_exact_importability_failure(self):
        loader = unittest.TestLoader()
        self.assertEqual(loader.discover(str(self.directories[0])).countTestCases(), 1)
        with self.assertRaisesRegex(ImportError, 'Start directory is not importable'):
            loader.discover(str(self.directories[1]))

    def test_corrected_discovery_handles_both_non_package_suites_and_runs_them(self):
        suites = [candidate.discover_suite(path) for path in self.directories]
        self.assertEqual([suite.countTestCases() for suite in suites], [1, 1])
        result = unittest.TestResult()
        unittest.TestSuite(suites).run(result)
        self.assertEqual(result.testsRun, 2)
        self.assertTrue(result.wasSuccessful())


if __name__ == '__main__':
    unittest.main()
