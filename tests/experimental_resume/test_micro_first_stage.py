"""Routing first-stage counts must respect closure and actual expert identities."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
from analyze_micro_first_stage import rates
from build_micro_blind_frame import THINK_END_ID


class FirstStageTests(unittest.TestCase):
    def test_counts_only_generated_reasoning_before_closure(self):
        routed = np.full((4, 40, 8), 77, dtype=np.int16)
        routed[0, 28, 0] = 9
        routed[1, 28, 0] = 189
        routed[1, 28, 1] = 139
        routed[3, 28, 0] = 9  # Deliberately after reasoning closure.
        result = rates(routed, [100, 101, THINK_END_ID, 102], (139, 120))
        self.assertEqual(result['reasoning_tokens'], 2)
        self.assertEqual(result['target_selected_tokens'], 2)
        self.assertEqual(result['own_set_selected_tokens'], 1)
        self.assertEqual(result['target_expert_token_counts'], {'9': 1, '189': 1})
        self.assertTrue(result['closed_reasoning'])

    def test_shape_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'emitted token count'):
            rates(np.zeros((2, 40, 8), dtype=np.int16), [123], (9, 189))


if __name__ == '__main__':
    unittest.main()
