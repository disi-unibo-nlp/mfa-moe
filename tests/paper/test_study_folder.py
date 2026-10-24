"""The routing study that the paper exporters read is store.STUDY, which scripts/paper/export_paper_tables.sh sets."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import store
from paper import export_routing_results
from paper.cl_paths import REPO

SCRIPT = REPO / 'scripts/paper/export_paper_tables.sh'


class StudyFolder(unittest.TestCase):
    def test_the_exporter_reads_the_receipts_of_the_study_the_store_names(self):
        self.assertEqual(export_routing_results.DRIVER, store.DRIVER)
        self.assertEqual(export_routing_results.MAIN, store.DRIVER / 'analysis-final/EFFECTS.json')
        self.assertEqual(export_routing_results.STRONG, store.DRIVER / 'analysis-strong/EFFECTS.json')

    def test_a_table_cites_a_receipt_inside_the_repository_by_its_relative_path(self):
        self.assertEqual(export_routing_results.shown(REPO / 'report/x/driver/EFFECTS.json'), 'report/x/driver/EFFECTS.json')
        self.assertEqual(export_routing_results.shown(Path('/elsewhere/study/EFFECTS.json')), '/elsewhere/study/EFFECTS.json')

    def test_the_launcher_needs_a_study_folder_that_holds_the_receipts(self):
        run = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('STUDY_DIR', run.stderr)
        with tempfile.TemporaryDirectory(dir=store.WORK / 'tools/tmp') as empty:
            run = subprocess.run(['bash', str(SCRIPT), empty], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('not the routing study', run.stderr)


if __name__ == '__main__':
    unittest.main()
