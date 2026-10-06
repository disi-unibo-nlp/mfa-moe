"""Contrast orientation separates Verify-entry from persistence proposals."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experimental_resume'))
from shortlist_boundary_proxy_negative import choose


class BoundaryProxyTests(unittest.TestCase):
    def test_verify_and_persistence_rank_opposite_native_experts(self):
        pairs = []
        for index in range(4):
            positive = np.zeros((40, 256))
            negative = np.zeros((40, 256))
            positive[4, 3] = .5
            negative[5, 4] = .5
            p = {'uid': f'p{index}', 'family': f'f{index}', 'frequency': positive}
            n = {'uid': f'n{index}', 'family': f'f{index}', 'frequency': negative}
            pairs.append((p, n, 0.0))
        exposure = np.full((40, 256), .01)
        verify, _ = choose(pairs, exposure, persistence=False)
        persistent, _ = choose(pairs, exposure, persistence=True)
        self.assertEqual((verify['experts'][0]['layer'], verify['experts'][0]['expert']), (4, 3))
        self.assertEqual((persistent['experts'][0]['layer'], persistent['experts'][0]['expert']), (5, 4))


if __name__ == '__main__':
    unittest.main()
