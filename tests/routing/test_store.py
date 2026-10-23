"""store.py and stats.py reproduce the numbers of the modules they replaced (receipts.py and the longer stats.py).

The expected hashes were computed with the removed `receipts.digest` and the removed `bootstrap_panel` options.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import stats
import store

# receipts.digest of these values, before receipts.py was merged into store.py
OLD_DIGESTS = {
    'plain': ({'a': 1, 'b': [1, 2.5, 'x'], 'c': {'d': None, 'e': True}},
              'b49c69dfdb8de1b7097bdf9a8079d3d9d33f9aae72e54e8eac51cc2545c85dda'),
    'tuples and int keys': ({1: (1, 2), 2: [(3, 4)], 'z': ()},
                            '1a26828aa77f74f4189d62d54aa3c6b68ff783320cde52889a4b1e1d64b8f066'),
    'nan and inf': ({'x': float('nan'), 'y': [float('inf'), -float('inf'), 1.5]},
                    'a59302366d6ea9a8e84e38a6ad2cd78ae532f984021a5af79498520a57d78f84'),
    'numpy scalars': ({'i': np.int64(7), 'f': np.float32(0.1), 'g': np.float64(2.5), 'b': np.bool_(True),
                       'u': np.uint64(2**63)},
                      '60e6f835a687b44485c69238a2ecb970f23aac822ad1f322aa4aa3764ffa4572'),
    'numpy nan': ({'x': np.float64('nan'), 'a': np.array([1.0, np.nan, 3.0])},
                  '3e684c34c586cb30782a86e89434c1dd9ef5faa831c1af9739acacbfe06fa7a9'),
    'numpy arrays': ({'a': np.arange(6).reshape(2, 3), 'e': np.zeros((0,))},
                     '3f4ded7d0810444247ec2e2c7337db9c2649272c80f36427d9a573b48cdef72e'),
    'unicode': ({'t': 'café ☃', 'kéy': 'v'},
                '511c4f39fffbf8f9332d3ca85670b91ff1626dd7c12566ae688f03e4a39513bd'),
    'big ints': ({'n': 2**70, 'neg': -2**40}, 'e5578de39fba532dbe199d8dd6e9cc2b3cb134cae196714d72b77007d3926855'),
    'empty containers': ({'a': {}, 'b': [], 'c': [[], [{}]]},
                         'e94e0335c4728eef07cbe339b1a82872406a8a1791245fafc2f2d58bee6fc109'),
    'seed list': ({'v': [20261008, 'bootstrap', 'native:gpt']},
                  '4d662197148a422fb5fc9bfb89098724918573595a8d514c122d00db1cbc4927'),
    'pandas missing values': ({'x': pd.NA, 'y': pd.NaT, 'z': 1},
                              '87c0cc04222f5cec4a6bc67402279a6ea7dc371f766eb482a5b0d32877ed4af6'),
}


def panel_rows():
    rng = np.random.default_rng(5)
    rows = []
    for benchmark in ('aime24', 'math500', 'olympiad'):
        for family in range(12):
            for repeat in range(2):
                row = {'benchmark': benchmark, 'family': f'f{family}', 'attempt_id': f'{benchmark}-{family}-{repeat}',
                       'x': float(rng.normal()), 'y': float(rng.random())}
                if rng.random() < .15:
                    row['x'] = None
                if rng.random() < .1:
                    row['y'] = float('nan')
                rows.append(row)
    rows.append({'benchmark': 'aime24', 'family': 'solo', 'attempt_id': 'solo', 'x': 1.0, 'y': 0.5, 'z': None})
    return rows


class DigestTests(unittest.TestCase):
    def test_digest_matches_the_removed_receipts_module(self):
        for name, (value, expected) in OLD_DIGESTS.items():
            with self.subTest(name):
                self.assertEqual(store.digest(value), expected)

    def test_seal_adds_the_hash_of_the_rest(self):
        sealed = store.seal({'a': (1, 2), 'sha256': 'stale'})
        self.assertEqual(sealed['a'], [1, 2])
        self.assertEqual(sealed['sha256'], store.digest({'a': [1, 2]}))

    def test_clean_says_why_a_value_is_null(self):
        reasons = {}
        row = store.clean({'a': None, 'b': [1.0, float('inf')], 'c': pd.NA, 'd': 3}, reasons)
        self.assertEqual(row, {'a': None, 'b': [1.0, None], 'c': None, 'd': 3})
        self.assertEqual(reasons, {'/a': 'missing_input', '/b/1': 'non_finite_input', '/c': 'missing_input'})

    def test_publish_is_immutable_and_verified_checks_the_hash(self):
        with tempfile.TemporaryDirectory(dir=store.WORK/'tools/tmp') as tmp:
            path = Path(tmp)/'sub/R.json'
            receipt = store.publish(path, {'value': np.float64(1.5), 'items': (1, 2)})
            self.assertEqual(store.publish(path, {'value': 1.5, 'items': [1, 2]}), receipt)
            with self.assertRaises(ValueError):
                store.publish(path, {'value': 2.0})
            self.assertEqual(store.verified(path), receipt)
            path.write_text(path.read_text().replace('1.5', '9.5'))
            with self.assertRaises(ValueError):
                store.verified(path)


class StatsTests(unittest.TestCase):
    def test_bootstrap_panel_matches_the_longer_version_it_replaced(self):
        result = stats.bootstrap_panel(panel_rows(), ['x', 'y', 'z'], panel='unit-test')
        blob = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.assertEqual(hashlib.sha256(blob.encode()).hexdigest(),
                         'b75604f28b181f9828ae8c0b25201cd407a65a309a87da3f6c18c2b0648d4d32')
        self.assertEqual(result['results']['z']['status'], 'insufficient_support_or_degenerate')
        self.assertIsNone(result['results']['z']['estimate'])

    def test_unknown_bounds(self):
        self.assertEqual(stats.unknown_bounds(True, 'stop', True), (1.0, 1.0))
        self.assertEqual(stats.unknown_bounds(False, 'stop', True), (0.0, 1.0))
        self.assertEqual(stats.unknown_bounds(False, 'length', False), (0.0, 0.0))
        self.assertEqual(stats.unknown_bounds(None, 'length', False), (0.0, 1.0))


if __name__ == '__main__':
    unittest.main()
