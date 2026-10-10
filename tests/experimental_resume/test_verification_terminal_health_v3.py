"""Expired dependency records must never cause repeated scientific generation."""
import unittest
import subprocess
from unittest.mock import patch

from moe_exp.verification_study.contracts import load, save, seal
import verification_study_v3_terminal_health_r1 as terminal
import test_verification_health_v3 as fixtures


class TerminalHealthTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HealthTests(methodName="runTest"); self.fixture.setUp()
        f = self.fixture; self.base = f.base; self.cp = self.base / "CAMPAIGN.json"
        self.campaign = save(self.cp, {**f.campaign, "result_root": str(self.base)}, replace=True)
        save(self.base / "ROSTER.json", {**f.manifest, "stage": "discovery"}, replace=True)
        self.card = save(self.base / "JOB_CARD.json", {**f.card,
            "campaign_sha256": self.campaign["sha256"], "code_files": self.campaign["code_files"],
            "claim": "bounded discovery"}, replace=True)
        self.sp = self.base / "SUBMISSION.json"
        self.submitted = save(self.sp, {"schema": "verification-submission-v3", "job_id": "123",
                                      "job_card_sha256": self.card["sha256"]})
        self.completed = {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 42,
                          "allocated_tres": "billing=16,cpu=16,gres/gpu=2"}

    def tearDown(self):
        self.fixture.tearDown()

    def test_pending_predecessor_cannot_get_terminal_callback(self):
        with self.assertRaisesRegex(ValueError, "terminal predecessor"):
            terminal.prepare(self.cp, self.sp, self.base, {**self.completed, "state": "PENDING"})

    def test_full_serial_slots_do_not_allocate_or_create_intent(self):
        with patch.object(terminal.dispatch, "callback", return_value=None), \
             patch.object(terminal.dispatch, "serial_slots", return_value={"free": 0}), \
             patch.object(terminal, "prepare") as prepare:
            result = terminal.register(self.cp, self.sp, self.base, self.completed, submit=True)
        self.assertEqual(result["status"], "DEFERRED_SERIAL_SUBMIT_LIMIT")
        prepare.assert_not_called()

    def test_registered_callback_is_reused(self):
        with patch.object(terminal.dispatch, "callback", return_value={"job_id": "456"}), \
             patch.object(terminal.health, "command") as submit:
            result = terminal.register(self.cp, self.sp, self.base, self.completed, submit=True)
        self.assertEqual(result["callback_job_id"], "456"); submit.assert_not_called()

    def test_immutable_terminal_proof_cannot_change_to_success(self):
        failed = {**self.completed, "state": "FAILED", "exit_code": "1:0"}
        terminal.prepare(self.cp, self.sp, self.base, failed)
        with self.assertRaisesRegex(ValueError, "immutable artifact"):
            terminal.prepare(self.cp, self.sp, self.base, self.completed)

    def test_unchanged_observer_retains_missing_artifact_status(self):
        card, observer, slot = terminal.prepare(self.cp, self.sp, self.base, self.completed)
        with patch("moe_exp.verification_study.closeout.require_cpu"), \
             patch.object(terminal.health.getpass, "getuser", return_value="lmolfett"), \
             patch.object(terminal.health, "terminal_accounting", return_value=self.completed), \
             patch("moe_exp.verification_study.paper.export", return_value={"sha256": "paper"}), \
             patch.dict("os.environ", {"SLURM_JOB_ID": "456"}):
            result = terminal.health.check(slot / "HEALTH_CARD.json")
        self.assertEqual(result["state"], "INCOMPLETE_ARTIFACTS")
        self.assertEqual(result["automatic_scientific_submissions"], 0)
        self.assertEqual(load(observer["terminal_proof_path"])["scientific_reruns"], 0)

    def test_terminal_registration_has_no_dependency_or_scientific_allocation(self):
        commands = []
        def command(argv):
            commands.append(argv)
            return "456" if argv[0] == "sbatch" else "JobId=456 Account=iscrc_miosr"
        with patch.object(terminal.dispatch, "callback", return_value=None), \
             patch.object(terminal.dispatch, "serial_slots", return_value={"free": 1}), \
             patch.object(terminal.health, "preflight", return_value=seal({"schema": "fixture"})), \
             patch.object(terminal.subprocess, "run") as simulation, \
             patch.object(terminal.health, "command", side_effect=command):
            result = terminal.register(self.cp, self.sp, self.base, self.completed, submit=True)
        self.assertEqual(result["callback_job_id"], "456")
        all_args = commands[0] + simulation.call_args.args[0]
        self.assertFalse(any(a.startswith("--dependency") or a.startswith("--gres") for a in all_args))
        self.assertIn("--partition=lrd_all_serial", all_args)
        self.assertIn("--time=00:10:00", all_args)

    def test_quota_deferral_can_resume_with_a_new_preflight(self):
        rejection = subprocess.CalledProcessError(1, "sbatch", stderr="QOSMaxSubmitJobPerUserLimit")
        with patch.object(terminal.dispatch, "callback", return_value=None), \
             patch.object(terminal.dispatch, "serial_slots", return_value={"free": 1}), \
             patch.object(terminal.health, "preflight", side_effect=[
                 seal({"schema": "fixture", "observed_epoch": 1}),
                 seal({"schema": "fixture", "observed_epoch": 2})]), \
             patch.object(terminal.subprocess, "run", side_effect=[rejection, None]), \
             patch.object(terminal.health, "command", side_effect=["456", "JobId=456 Account=iscrc_miosr"]):
            first = terminal.register(self.cp, self.sp, self.base, self.completed, submit=True)
            self.assertEqual(first["status"], "DEFERRED_SERIAL_SUBMIT_LIMIT")
            self.assertFalse(list((self.base / "health/submissions").glob("*/PREFLIGHT.json")))
            self.assertFalse(list((self.base / "health/submissions").glob("*/INTENT.json")))
            second = terminal.register(self.cp, self.sp, self.base, self.completed, submit=True)
        self.assertEqual(second["callback_job_id"], "456")
        preflight = next((self.base / "health/submissions").glob("*/PREFLIGHT.json"))
        self.assertEqual(load(preflight)["observed_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
