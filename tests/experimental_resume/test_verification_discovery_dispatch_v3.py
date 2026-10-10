"""Partition QoS limits must defer signals without duplicating scientific work."""
import unittest
from unittest.mock import patch

import verification_study_v3_discovery_dispatch_r1 as dispatch


class QueueTests(unittest.TestCase):
    def output(self, text):
        class Result:
            stdout = text
        return Result()

    def test_full_serial_partition_defers_more_callbacks(self):
        quota = self.output("lrd_all_serial|10|\n")
        queue = self.output("".join(f"{i}|lrd_all_serial\n" for i in range(10)) + "100|boost_usr_prod\n")
        with patch.object(dispatch.subprocess, "run", side_effect=[quota, queue]):
            slots = dispatch.serial_slots()
        self.assertEqual(slots, {"limit": 10, "used": 10, "free": 0})

    def test_one_released_serial_job_opens_exactly_one_slot(self):
        with patch.object(dispatch.subprocess, "run", side_effect=[
                self.output("lrd_all_serial|10|\n"),
                self.output("".join(f"{i}|lrd_all_serial\n" for i in range(9)))]):
            self.assertEqual(dispatch.serial_slots()["free"], 1)

    def test_other_gpu_jobs_do_not_use_serial_callback_slots(self):
        with patch.object(dispatch.subprocess, "run", side_effect=[
                self.output("lrd_all_serial|10|\n"), self.output("99|boost_usr_prod\n")]):
            self.assertEqual(dispatch.serial_slots()["free"], 10)

    def test_blank_limit_is_not_assumed_unlimited(self):
        with patch.object(dispatch.subprocess, "run", return_value=self.output("lrd_all_serial||\n")):
            with self.assertRaisesRegex(ValueError, "entitlement unavailable"):
                dispatch.serial_slots()


if __name__ == "__main__":
    unittest.main()
