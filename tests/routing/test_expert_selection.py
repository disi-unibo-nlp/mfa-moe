from pathlib import Path
import tempfile
import unittest

from routing.expert_selection import export
from store import WORK, publish


class ExportTests(unittest.TestCase):
    def test_selection_coverage_table_comes_from_the_stored_receipts_of_the_campaign_models(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root = Path(tmp)/'campaign'
            campaign = publish(root/'CAMPAIGN.json', {'name': 'unit', 'models': {'gpt': {}}})
            held_out = {'state_balanced': {'minimum_class_coverage': .75, 'minimum_complete': True},
                        'random': {'minimum_class_coverage': .5, 'minimum_complete': False}}
            publish(root/'gpt/SELECTION.json', {'campaign_sha256': campaign['sha256'],
                    'folds': [{'fold': 0, 'test_families': 298, 'held_out': held_out}]})
            export(root, Path(tmp)/'out')
            text = (Path(tmp)/'out/selection_coverage.csv').read_bytes().decode()
            self.assertEqual(text, 'model,fold,arm,minimum_class_coverage,test_families,minimum_complete\r\n'
                                   'gpt,0,random,0.5,298,False\r\ngpt,0,state_balanced,0.75,298,True\r\n')  # stored keys are sorted

    def test_rows_follow_the_study_order_and_every_campaign_model_needs_a_receipt(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root = Path(tmp)/'campaign'
            held_out = {'random': {'minimum_class_coverage': .5, 'minimum_complete': True}}
            campaign = publish(root/'CAMPAIGN.json', {'models': {'glm': {}, 'gpt': {}}})  # sorted, as the file stores them
            for model in ('glm', 'gpt'):
                publish(root/model/'SELECTION.json', {'campaign_sha256': campaign['sha256'],
                        'folds': [{'fold': 0, 'test_families': 1, 'held_out': held_out}]})
            export(root, Path(tmp)/'out')
            lines = (Path(tmp)/'out/selection_coverage.csv').read_text().splitlines()
            self.assertEqual([line.split(',')[0] for line in lines[1:]], ['gpt', 'glm'])  # store.MODELS puts gpt first
            (root/'glm/SELECTION.json').unlink()
            with self.assertRaises(FileNotFoundError):
                export(root, Path(tmp)/'again')

    def test_a_receipt_of_another_campaign_and_a_model_outside_the_study_are_refused(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root = Path(tmp)/'campaign'
            publish(root/'CAMPAIGN.json', {'models': {'glm': {}}})
            publish(root/'glm/SELECTION.json', {'campaign_sha256': 'someone else', 'folds': []})
            with self.assertRaisesRegex(ValueError, 'foreign selection receipt'):
                export(root, Path(tmp)/'out')
            other = Path(tmp)/'other'
            publish(other/'CAMPAIGN.json', {'models': {'not-a-model': {}}})
            with self.assertRaisesRegex(ValueError, 'outside the study'):
                export(other, Path(tmp)/'out2')


if __name__ == '__main__':
    unittest.main()
