"""No allocations: an exact unstarted reader replacement is never a blind retry."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from moe_exp.verification_study.contracts import load, save, seal
import verification_study_v3_autostages_r4 as auto


class ReaderCancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv4-reader-cancel-",
            dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.root = Path(self.temp.name); self.cp = self.root / "CAMPAIGN.json"
        self.campaign = save(self.cp, {"code_files": {}})
        self.frame_path = self.root / "AUDIT_FRAME.json"
        self.frame = save(self.frame_path, {"records": [{"uid": "uid"}]})
        self.output = self.root / "ratings"; self.output.mkdir()
        self.executor = {"sha256": "original-reader"}
        self.card = seal({"stage": "rate", "model": "gpt", "full_frame_sha256": self.frame["sha256"],
            "reader_executor": self.executor, "selected_uids": ["uid"], "payload_arguments":
            ["rate", "--frame", str(self.frame_path), "--output", str(self.output), "--role", "eligibility"]})
        self.sub = seal({"job_id": "123", "job_card_sha256": self.card["sha256"]})
        self.jobs = [{"card": self.card, "submission": self.sub}]
        self.states = {"123": {"state": "CANCELLED", "exit_code": "0:0", "elapsed_seconds": 0}}
        self.row = {"job_id": "123", "job_card_sha256": self.card["sha256"],
            "submission_sha256": self.sub["sha256"], "full_frame_sha256": self.frame["sha256"],
            "selected_uids": ["uid"], "role": "eligibility", "output": str(self.output),
            "accounting": {"start": "Unknown", "allocated_cpus": 0},
            "source_receipts": {str(self.frame_path): self.frame["sha256"]}}
        self.body = {"schema": "verification-reader-transport-reconciliation-v3",
            "campaign_sha256": self.campaign["sha256"], "scientific_code_files": {},
            "replacement_transport_executor": {"sha256": "r3-reader"},
            "replacement_allocations_per_uid_max": 1, "cancelled_unstarted_readers": [self.row]}
        self.proof_path = self.root / "READER_TRANSPORT_RECONCILIATION-r4.json"
        save(self.proof_path, self.body)
        self.binding = patch.object(auto.transport, "binding", return_value={"sha256": "r3-reader"})
        self.binding.start()

    def tearDown(self):
        self.binding.stop(); self.temp.cleanup()

    def check(self):
        return auto.reconciled_unstarted_readers(self.cp, self.root, self.jobs, self.states)

    def test_exact_unstarted_cancellation_is_excluded(self):
        self.assertEqual(self.check(), {"123"})

    def test_pending_or_running_cannot_be_excluded(self):
        for state in ("PENDING", "RUNNING", "FAILED"):
            self.states["123"]["state"] = state
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "unstarted"):
                self.check()

    def test_zero_rounded_runtime_does_not_prove_unstarted(self):
        for accounting in ({"start": "2026-10-08T00:00:00", "allocated_cpus": 0},
                           {"start": "Unknown", "allocated_cpus": 16}):
            save(self.proof_path, {**self.body, "cancelled_unstarted_readers":
                [{**self.row, "accounting": accounting}]}, replace=True)
            with self.assertRaisesRegex(ValueError, "unstarted"):
                self.check()

    def test_nonzero_allocation_is_not_excluded(self):
        self.states["123"]["elapsed_seconds"] = 1
        with self.assertRaisesRegex(ValueError, "unstarted"):
            self.check()

    def test_worker_entry_invalidates_unstarted_proof(self):
        save(self.root / "transport/invocations/123-1/INTENT.json", {"job_id": "123"})
        with self.assertRaisesRegex(ValueError, "entered a worker"):
            self.check()

    def test_actual_label_invalidates_unstarted_proof(self):
        save(self.output / "uid.reader-0.json", {"job_id": "123"})
        with self.assertRaisesRegex(ValueError, "produced a draw"):
            self.check()

    def test_foreign_source_or_role_cannot_be_reconciled(self):
        save(self.proof_path, {**self.body, "cancelled_unstarted_readers":
            [{**self.row, "role": "outcome"}]}, replace=True)
        with self.assertRaisesRegex(ValueError, "identity or role"):
            self.check()

    def test_foreign_cancelled_job_still_stops_ratings(self):
        other = {"card": self.card, "submission": seal({"job_id": "124"})}
        self.jobs.append(other)
        self.states["124"] = {"state": "CANCELLED", "exit_code": "0:0", "elapsed_seconds": 0}
        run = object.__new__(auto.Advance)
        run.path = self.cp; run.root = self.root; run.jobs = self.jobs; run.observations = self.states
        run.active_stage = {}; run.price = Mock(); run.dispatch = Mock()
        with patch.object(auto.reader, "deployment", return_value=(self.campaign, {}, self.frame, self.executor)), \
             patch.object(auto.reader, "validate_directory"):
            with self.assertRaisesRegex(ValueError, "exact invocation reconciliation"):
                run.ratings("gpt", self.frame_path, self.output, "eligibility")
        run.price.assert_not_called(); run.dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
