"""The shortlist must seal NumPy-derived counts without losing exact values."""
from __future__ import annotations

import json
import unittest

import numpy as np

from scripts.experimental_resume.shortlist_boundary_experts import digest, plain


class ShortlistSealTest(unittest.TestCase):
    def test_numpy_scalars_and_arrays_round_trip_through_canonical_seal(self):
        source = {'support': {'positive': np.int64(6), 'distance': np.float32(0.25)},
                  'identities': np.asarray([3, 8], dtype=np.int32),
                  'seen': [np.bool_(True)]}
        value = plain(source)
        text = json.dumps(value, allow_nan=False)
        self.assertEqual(json.loads(text),
                         {'support': {'positive': 6, 'distance': 0.25},
                          'identities': [3, 8], 'seen': [True]})
        self.assertEqual(digest(value), digest(json.loads(text)))


if __name__ == '__main__':
    unittest.main()
