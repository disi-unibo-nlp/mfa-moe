"""Row building blocks of the outcome-steering manifest and the length-limit prefix."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from moe_exp.outcome_steering import manifest
from moe_exp.study.runtime import budget_tokens
from moe_exp.study.common import file_sha


class ManifestPartsTests(unittest.TestCase):
    def test_nudge_actions_group_experts_per_layer(self):
        pairs = [{'layer': 7, 'expert': 3}, {'layer': 2, 'expert': 9}, {'layer': 7, 'expert': 1}]
        actions = manifest.nudge_actions(pairs, .8)
        self.assertEqual([a['layer'] for a in actions], [2, 7])
        self.assertEqual(actions[1]['shift'], [1, 3])
        self.assertTrue(all(a['operator'] == 'score_shift' and a['delta'] == .8 and
                            not a['activate'] and not a['deactivate'] for a in actions))

    def test_reasoning_length_finds_first_closure_or_marks_a_capped_run(self):
        closures = [[5, 6], [9]]
        self.assertEqual(manifest.reasoning_length([1, 2, 5, 6, 3, 9], closures), (2, False))
        self.assertEqual(manifest.reasoning_length([1, 2, 3], closures), (3, True))

    def test_gpt_suffix_ends_with_the_runtime_closure(self):
        # runtime.closure for GPT-OSS matches <|channel|>final<|message|>; the suffix must end with it.
        self.assertEqual(manifest.GPT_TRANSITION[-3:], [200005, 17196, 200008])


class BudgetPrefixTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get('TMPDIR', tempfile.gettempdir()))
        self.folder = tempfile.TemporaryDirectory(dir=root)
        self.output = Path(self.folder.name)/'OUTPUT.json'
        self.output.write_text(json.dumps({'token_ids': [10, 11, 12, 13, 14]}))

    def tearDown(self):
        self.folder.cleanup()

    def test_prefix_is_cut_tokens_plus_suffix(self):
        spec = {'path': str(self.output), 'sha256_file': file_sha(self.output), 'cut': 3, 'suffix': [99, 98]}
        self.assertEqual(budget_tokens(spec), [10, 11, 12, 99, 98])

    def test_changed_source_is_refused(self):
        spec = {'path': str(self.output), 'sha256_file': '0'*64, 'cut': 3, 'suffix': [99]}
        with self.assertRaises(ValueError):
            budget_tokens(spec)

    def test_cut_outside_the_output_is_refused(self):
        spec = {'path': str(self.output), 'sha256_file': file_sha(self.output), 'cut': 9, 'suffix': [99]}
        with self.assertRaises(ValueError):
            budget_tokens(spec)


if __name__ == '__main__':
    unittest.main()
