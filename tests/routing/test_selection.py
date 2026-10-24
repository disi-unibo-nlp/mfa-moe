from pathlib import Path
import tempfile
import unittest

from routing.selection import export
from store import WORK, publish


class ExportTests(unittest.TestCase):
    def test_selection_coverage_table_comes_from_the_stored_receipts(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root = Path(tmp)/'campaign'
            campaign = publish(root/'CAMPAIGN.json', {'name': 'unit'})
            held_out = {'state_balanced': {'minimum_class_coverage': .75, 'minimum_complete': True},
                        'random': {'minimum_class_coverage': .5, 'minimum_complete': False}}
            publish(root/'gpt/SELECTION.json', {'campaign_sha256': campaign['sha256'],
                    'folds': [{'fold': 0, 'test_families': 298, 'held_out': held_out}]})
            export(root, Path(tmp)/'out')
            text = (Path(tmp)/'out/selection_coverage.csv').read_bytes().decode()
            self.assertEqual(text, 'model,fold,arm,minimum_class_coverage,test_families,minimum_complete\r\n'
                                   'gpt,0,random,0.5,298,False\r\ngpt,0,state_balanced,0.75,298,True\r\n')  # stored keys are sorted
            publish(root/'glm/SELECTION.json', {'campaign_sha256': 'someone else', 'folds': []})
            with self.assertRaises(ValueError):
                export(root, Path(tmp)/'again')


if __name__ == '__main__':
    unittest.main()
