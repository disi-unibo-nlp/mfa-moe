"""Bounded status telemetry contracts, with scheduler calls fully mocked."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest, file_sha, load, save, seal
import verification_study_v3_health_status_r2 as status


class HealthStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cv3-health-status-test-",
            dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.root = Path(self.tmp.name)
        source = self.root / "source.py"; source.write_text("# frozen\n")
        self.files = {str(source): file_sha(source)}
        self.cp = self.root / "CAMPAIGN.json"
        self.campaign = save(self.cp, {"schema": "fixture", "code_files": self.files,
            "result_root": str(self.root), "models": {}})
        self.science = seal({"schema": "verification-job-card-v3", "campaign_sha256": self.campaign["sha256"],
            "campaign_path": str(self.cp), "code_files": self.files, "model": "gpt", "stage": "j1-audit"})
        self.science_slot = self.root / "submissions" / self.science["sha256"]
        save(self.science_slot / "JOB_CARD.json", self.science)
        self.science_receipt = save(self.science_slot / "SUBMISSION.json", {"schema": "verification-submission-v3",
            "job_id": "100", "job_card_sha256": self.science["sha256"]})
        self.work = {"model": "gpt", "stage": "j1-audit", "expected_uids": ["uid"],
            "expected_uids_sha256": digest(["uid"]), "inputs": {}}
        self.health_body = {"schema": "verification-health-card-v3", "campaign_sha256": self.campaign["sha256"],
            "campaign_path": str(self.cp), "result_root": str(self.root), "predecessor_job_id": "100",
            "predecessor_submission_path": str(self.science_slot / "SUBMISSION.json"),
            "predecessor_submission_sha256": self.science_receipt["sha256"],
            "job_card_path": str(self.science_slot / "JOB_CARD.json"), "job_card_sha256": self.science["sha256"],
            "observer_files": self.files, "expected_work": self.work}
        self.callback = seal(self.health_body)
        self.callback_slot = self.root / "health/submissions" / self.callback["sha256"]
        save(self.callback_slot / "HEALTH_CARD.json", self.callback)
        save(self.callback_slot / "SUBMISSION.json", {"schema": "verification-health-submission-v3",
            "job_id": "101", "health_card_sha256": self.callback["sha256"]})
        self.accounting = {"100": {"state": "FAILED", "exit_code": "1:0", "elapsed_seconds": 11},
            "101": {"state": "CANCELLED by 133943", "exit_code": "0:0", "elapsed_seconds": 0}}
        self.queue = {}
        self.scheduler = patch.object(status, "scheduler_snapshot", side_effect=lambda ids: (self.accounting, self.queue))
        self.scheduler.start()

    def tearDown(self):
        self.scheduler.stop(); self.tmp.cleanup()

    def observe(self, write=False):
        return status.observe(self.cp, write=write)

    def reconcile(self, name="HEALTH_RECONCILIATION-r2.json", **extra):
        return save(self.root / name, {"schema": "verification-health-reconciliation-v3", "targets": ["101"], **extra})

    def controller(self, job="59652415", mode="successor", parent="99"):
        card = seal({"schema": "verification-autodriver-card-v3", "campaign_sha256": self.campaign["sha256"],
            "campaign_path": str(self.cp), "result_root": str(self.root), "scientific_code_files": self.files,
            "program_files": self.files, "mode": mode, "parent_job_id": parent, "wait_job_ids": ["100"],
            "dependency": "afterany:100", "operational_version": "autodriver-r2", "fixture_job_identity": job})
        slot = self.root / "automation/submissions" / card["sha256"]
        save(slot / "CARD.json", card)
        receipt = save(slot / "SUBMISSION.json", {"schema": "verification-autodriver-submission-v3",
            "job_id": job, "controller_card_sha256": card["sha256"]})
        self.queue[job] = {"state": "PENDING", "comment": "cv3auto:" + card["sha256"], "account": "iscrc_miosr"}
        return card, receipt, slot

    def event(self, card=None, owner="101", event_state="EXECUTION_FAILED"):
        card = card or self.callback
        return save(self.root / "health/events" / (card["sha256"] + ".json"), {
            "schema": "verification-health-event-v3", "campaign_sha256": self.campaign["sha256"],
            "health_card_sha256": card["sha256"], "job_card_sha256": self.science["sha256"],
            "expected_uids_sha256": self.work["expected_uids_sha256"], "model": "gpt", "stage": "j1-audit",
            "predecessor_job_id": "100", "callback_job_id": owner, "state": event_state,
            "accounting": self.accounting["100"]})

    def test_legacy_target_proof_requires_observed_cancelled_zero_elapsed(self):
        self.reconcile()
        result = self.observe()
        self.assertEqual(result["callbacks"][0]["signal"], "SUPERSEDED_UNSTARTED_OBSERVER")
        self.assertEqual(result["reconciliations"][0]["references"][0]["campaign_binding"], "EXACT_CALLBACK_JOURNAL_JOIN")
        self.assertEqual(result["automatic_submissions"], 0)

    def test_prepared_proof_never_supersedes(self):
        self.reconcile("HEALTH_RECONCILIATION-autodriver-r2-PREPARED.json")
        result = self.observe()
        self.assertEqual(result["callbacks"][0]["signal"], "CALLBACK_FAILED_NO_EVENT")
        self.assertEqual(len(result["skipped_prepared_reconciliations"]), 1)

    def test_r1_preparation_text_requires_completed_accounting(self):
        self.reconcile("HEALTH_RECONCILIATION-autodriver-r1.json", campaign_sha256=self.campaign["sha256"],
            execution_status="EXACT_REPLACEMENT_PREPARED; verify observed cancellation separately")
        self.accounting["101"]["state"] = "PENDING"
        self.queue["101"] = {"state": "PENDING"}
        self.assertEqual(self.observe()["callbacks"][0]["signal"], "SIGNAL_PENDING")
        self.queue.clear(); self.accounting["101"]["state"] = "CANCELLED"
        self.assertEqual(self.observe()["callbacks"][0]["signal"], "SUPERSEDED_UNSTARTED_OBSERVER")

    def test_completed_new_schema_without_targets_works(self):
        save(self.root / "HEALTH_RECONCILIATION-autodriver-r2.json", {
            "schema": "verification-health-observer-reconciliation-v3", "campaign_sha256": self.campaign["sha256"],
            "cancelled_unstarted_observers": [{"job_id": "101", "state": "CANCELLED", "elapsed_seconds": 0}]})
        self.assertEqual(self.observe()["callbacks"][0]["signal"], "SUPERSEDED_UNSTARTED_OBSERVER")

    def test_started_cancelled_callback_is_failure_not_superseded(self):
        self.reconcile(); self.accounting["101"]["elapsed_seconds"] = 3
        self.assertEqual(self.observe()["callbacks"][0]["signal"], "CALLBACK_FAILED_NO_EVENT")

    def test_foreign_reconciliation_campaign_rejected(self):
        self.reconcile(campaign_sha256="foreign")
        with self.assertRaisesRegex(ValueError, "foreign health reconciliation"):
            self.observe()

    def test_unjoined_target_cannot_supersede(self):
        self.reconcile(targets=["999"])
        with self.assertRaisesRegex(ValueError, "exact current callback journal"):
            self.observe()

    def test_source_bound_owned_events_and_three_protected_ids_visible(self):
        self.controller(); self.controller("59652422", "rescue", "59652415")
        self.controller("59652425", "rescue", "59652422")
        owned = seal({**self.health_body, "controller_owned": True})
        save(self.root / "automation/observer-cards" / (owned["sha256"] + ".json"), owned)
        self.event(owned, owner="59652415")
        result = self.observe()
        self.assertEqual({r["job_id"] for r in result["controllers"]}, {"59652415", "59652422", "59652425"})
        self.assertTrue(result["events"][0]["controller_owned"])
        self.assertEqual(result["events"][0]["state"], "EXECUTION_FAILED")
        self.assertEqual(result["errors"], [])

    def test_failed_science_event_retains_failure_and_reconcile_reference(self):
        self.event()
        failure = save(self.root / "failure.json", {"status": "FAIL"})
        save(self.root / "J1_TRANSPORT_RECONCILIATION-r2.json", {
            "schema": "verification-j1-transport-reconciliation-v3", "campaign_sha256": self.campaign["sha256"],
            "scientific_code_files": self.files, "failed_jobs": [{"job_id": "100",
                "job_card_sha256": self.science["sha256"], "submission_sha256": self.science_receipt["sha256"],
                "source_receipts": {str(self.root / "failure.json"): failure["sha256"]}}]})
        event = self.observe()["events"][0]
        self.assertEqual(event["state"], "EXECUTION_FAILED")
        self.assertEqual(len(event["reconciliation_references"]), 1)
        self.assertNotIn("PASS", event["state"])

    def test_bad_intent_comment_is_visible_error(self):
        save(self.science_slot / "INTENT.json", {"job_card_sha256": self.science["sha256"], "comment": "cv3:wrong"})
        result = self.observe()
        self.assertTrue(any("comment differs" in row["message"] for row in result["errors"]))

    def test_foreign_observer_journal_path_is_visible_error(self):
        changed = seal({**self.health_body, "job_card_path": str(self.root / "elsewhere" / "JOB_CARD.json")})
        save(self.root / "automation/observer-cards" / (changed["sha256"] + ".json"), changed)
        self.assertTrue(any("foreign journal path" in row["message"] for row in self.observe()["errors"]))

    def test_foreign_scheduler_comment_rejected(self):
        self.queue["100"] = {"state": "RUNNING", "comment": "cv3:foreign", "account": "iscrc_miosr"}
        with self.assertRaisesRegex(ValueError, "scheduler comment"):
            self.observe()

    def test_owned_event_cannot_use_foreign_observer_source(self):
        self.controller()
        alternate = self.root / "alternate.py"; alternate.write_text("# different observer\n")
        owned = seal({**self.health_body, "controller_owned": True,
            "observer_files": {str(alternate): file_sha(alternate)}})
        save(self.root / "automation/observer-cards" / (owned["sha256"] + ".json"), owned)
        self.event(owned, owner="59652415")
        result = self.observe()
        self.assertEqual(result["events"], [])
        self.assertTrue(any("exact sealed observer source" in r["message"] for r in result["errors"]))

    def test_status_write_only_creates_replaceable_snapshot(self):
        before = {str(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.observe(write=True)
        after = {str(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after - before, {str(self.root / "HEALTH_STATUS.json")})
        self.assertEqual(load(self.root / "HEALTH_STATUS.json"), result)
        self.assertEqual(result["automatic_submissions"], 0)


if __name__ == "__main__":
    unittest.main()
