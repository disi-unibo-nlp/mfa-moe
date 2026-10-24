import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from routing import state_dynamics
from store import WORK, publish


class NativeAccuracyTests(unittest.TestCase):
    def test_a_new_out_recomputes_the_native_accuracy_instead_of_reusing_the_capture_records(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            root, out = Path(tmp)/'campaign', Path(tmp)/'out'
            for name in ('NATIVE_ESTIMANDS.json', 'NATIVE_ACCURACY.json'):
                publish(root/'gpt'/name, {'exact_attempt_metadata': True})
            campaign = {'root': str(root), 'outcomes': str(Path(tmp)/'missing.parquet')}
            with self.assertRaises(FileNotFoundError):  # it went to the outcomes: nothing of the capture record was read
                state_dynamics.native_accuracy(campaign, 'gpt', out)
            estimands, _ = state_dynamics.native_accuracy(campaign, 'gpt', root)  # an `out` that already holds them keeps them
            self.assertTrue(estimands['exact_attempt_metadata'])


class CommandLineTests(unittest.TestCase):
    def run_main(self, tmp, models, *arguments):
        campaign = publish(Path(tmp)/'CAMPAIGN.json', {'models': models})
        with patch.object(state_dynamics, 'derive', return_value={'sha256': 'x'}) as derive, \
                contextlib.redirect_stdout(io.StringIO()):
            state_dynamics.main([tmp, *arguments])
        return campaign, [call.args[1:] for call in derive.call_args_list]

    def test_the_models_are_those_of_the_campaign_unless_named(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            out = str(Path(tmp)/'out')
            _, calls = self.run_main(tmp, {'gpt': {}, 'glm': {}}, '--out', out)
            self.assertEqual(calls, [('glm', Path(out)), ('gpt', Path(out))])  # as the receipt lists them
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp:
            out = str(Path(tmp)/'out')
            _, calls = self.run_main(tmp, {'gpt': {}, 'glm': {}}, 'zai-org/GLM-4.7-Flash', '--out', out)
            self.assertEqual(calls, [('glm', Path(out))])

    def test_out_is_required(self):
        with tempfile.TemporaryDirectory(dir=WORK/'tools/tmp') as tmp, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_main(tmp, {'gpt': {}})


if __name__ == '__main__':
    unittest.main()
