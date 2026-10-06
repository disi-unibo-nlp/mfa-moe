"""CPU-only checks for the separately bound batch-invariant engine stage."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))
import run_batch_invariant_qual as qualifier

MANIFEST = REPO / 'report/experimental-resume-v1/CAUSAL_BATCH_INVARIANT_QUAL_MANIFEST_v1.json'


class BatchInvariantQualTest(unittest.TestCase):
    def test_real_frozen_pilot_table_and_cases(self):
        with patch.dict(os.environ, {'VLLM_BATCH_INVARIANT': '1'}):
            manifest = qualifier.base.sealed(MANIFEST)
            rows, actions, arms = qualifier.validate_bi(manifest, qualifier.base.__file__)
        self.assertEqual((len(rows), len(actions), len(arms)), (4, 10, 6))
        self.assertEqual(qualifier.base.build_policy_table(actions).hooked_layers(), (28,))
        self.assertEqual(manifest['expected_requests'], 48)

    def test_flag_is_mandatory(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'flag is not enabled'):
                qualifier.preflight()

    def test_frozen_expert_ids_cannot_change(self):
        with patch.dict(os.environ, {'VLLM_BATCH_INVARIANT': '1'}):
            manifest = copy.deepcopy(qualifier.base.sealed(MANIFEST))
            manifest['actions'][0]['experts'][0][1][0] += 1
            with self.assertRaisesRegex(ValueError, 'frozen six-arm pilot field changed'):
                qualifier.validate_bi(manifest, qualifier.base.__file__)


if __name__ == '__main__':
    unittest.main()
