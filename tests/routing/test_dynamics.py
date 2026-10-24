from pathlib import Path
import tempfile
import unittest

from routing import dynamics
from store import WORK, publish


class NativeAccuracyTests(unittest.TestCase):
    def test_a_new_out_recomputes_the_native_accuracy_instead_of_reusing_the_campaigns(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root, out = Path(tmp)/'campaign', Path(tmp)/'out'
            for name in ('NATIVE_ESTIMANDS.json', 'NATIVE_ACCURACY.json'):
                publish(root/'gpt'/name, {'exact_attempt_metadata': True})
            campaign = {'root': str(root), 'outcomes': str(Path(tmp)/'missing.parquet')}
            with self.assertRaises(FileNotFoundError):  # it went to the outcomes: nothing of the campaign was read
                dynamics.native_accuracy(campaign, 'gpt', out)
            estimands, _ = dynamics.native_accuracy(campaign, 'gpt', root)  # without --out the campaign folder is the output
            self.assertTrue(estimands['exact_attempt_metadata'])


if __name__ == '__main__':
    unittest.main()
