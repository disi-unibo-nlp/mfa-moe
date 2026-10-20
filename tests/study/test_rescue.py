import unittest
from pathlib import Path

from moe_exp.study.common import BENCHMARKS, load
from moe_exp.study.rescue import DRIVER, RESCUE_COUNT, select_families


class RescuePanel(unittest.TestCase):
    def test_panel_is_deterministic_benchmark_balanced_and_unique(self):
        allocation = load(DRIVER.parent/'ALLOCATION.json')
        first = select_families(allocation, count=RESCUE_COUNT)
        second = select_families(allocation, count=RESCUE_COUNT)
        self.assertEqual(first, second)
        key = {(r['benchmark'], r['family']) for r in first['records']}
        self.assertEqual(len(key), RESCUE_COUNT)
        self.assertEqual(set(first['quotas']), set(BENCHMARKS))
        self.assertTrue(all(values >= 4 for values in first['quotas'].values()))
        self.assertEqual(sum(first['quotas'].values()), RESCUE_COUNT)


if __name__ == '__main__':
    unittest.main()
