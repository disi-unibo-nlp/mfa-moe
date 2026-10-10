"""Full-cap discovery planning must retain recovery and storage bounds."""
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import seal
import verification_study_v3_discovery_continue_r2 as continuation


class DiscoveryContinuationTests(unittest.TestCase):
    def prices(self, generation=.2):
        return seal({"schema": "verification-prices-v3", "components": {
            role: {"status": "MEASURED", "seconds_per_unit_max": value}
            for role, value in {"generator_load": 500, "prefill": 5, "generation": generation,
                                "capture": 20, "closeout": 1}.items()}})

    def test_cap_capture_scaling_and_complete_retry(self):
        work = continuation.work(7, 2048)
        self.assertEqual(work["generation"], 8 * 16384)
        self.assertEqual(work["generator_load"], 2)
        self.assertEqual(work["prefill"], 8)
        self.assertEqual(work["capture"], 64)
        self.assertEqual(work["closeout"], 64)

    def test_maximal_slice_fits_partition_and_minute_rounding(self):
        count, seconds, budget = continuation.sized(self.prices(), 47, 2048)
        self.assertLess(count, 47)
        self.assertLessEqual(seconds, 86400)
        self.assertEqual(seconds % 60, 0)
        self.assertLessEqual(budget["reservation_seconds"], seconds)
        from moe_exp.verification_study.pricing import shard_budget
        with self.assertRaises(ValueError):
            shard_budget(self.prices(), continuation.work(count + 1, 2048), hours=24)

    def test_impossible_single_item_cannot_use_unpriced_fallback(self):
        with self.assertRaisesRegex(ValueError, "exceed partition"):
            continuation.sized(self.prices(10), 47, 2048)

    def test_unmeasured_required_role_stops_dispatch(self):
        prices = self.prices()
        prices["components"]["capture"]["status"] = "UNMEASURED"
        with self.assertRaises(ValueError):
            continuation.sized(seal(prices), 47, 2048)

    def test_exact_project_quota_uses_kib_not_rounded_human_output(self):
        class Output:
            stdout = ""
        project = Output(); project.stdout = "20149688 P /approved/repo\n"
        quota = Output(); quota.stdout = "Disk quotas\n/path\n 966757180 1073741824 0 - 1436174 0 0 -\n"
        with patch.object(continuation.subprocess, "run", side_effect=[project, quota]):
            result = continuation.exact_storage(continuation.REPO)
        self.assertEqual(result["available_bytes"], (1073741824 - 966757180) * 1024)


if __name__ == "__main__":
    unittest.main()
