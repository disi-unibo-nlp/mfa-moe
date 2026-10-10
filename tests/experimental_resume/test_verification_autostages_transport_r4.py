"""Scientific gates for the explicitly reconciled pre-initialization recovery."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from moe_exp.verification_study.contracts import file_sha, load, save, seal
import verification_study_v3_autostages_r4 as auto


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv3-transport-gates-",
            dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.root = Path(self.temp.name)
        self.cp = self.root / "CAMPAIGN.json"
        self.campaign = save(self.cp, {"code_files": {}, "result_root": str(self.root)})
        self.invocation = self.root / "J1/audit/invocations/123-1"
        result = save(self.invocation / "RESULT.json", {"status": "FAIL", "loaded": False, "new_verdicts": 0, "error":
            {"type": "PermissionError", "message": "[Errno 13] Permission denied: '/scratch_local'"}})
        self.log = self.root / "error.log"; self.log.write_text("init permission failure\n")
        self.card = seal({"extension": {"selected_uids": ["uid"]}})
        self.receipt = seal({"job_id": "123"})
        self.job = {"card": self.card, "submission": self.receipt}
        self.states = {"123": {"state": "FAILED", "exit_code": "1:0"}}
        self.proof_body = {"schema": "verification-j1-transport-reconciliation-v3",
            "campaign_sha256": self.campaign["sha256"], "scientific_code_files": {},
            "item_draws_in_failed_jobs": 0, "recovery_allocations_per_uid_max": 1,
            "repair_round": 3, "recovery_transport_executor": {"sha256": "test-transport"},
            "eligible_recovery_uids": ["uid"], "failed_jobs": [{"job_id": "123",
                "job_card_sha256": self.card["sha256"], "submission_sha256": self.receipt["sha256"],
                "source_receipts": {str(self.invocation / "RESULT.json"): result["sha256"]},
                "source_logs": {str(self.log): file_sha(self.log)},
                "invocation_result_path": str(self.invocation / "RESULT.json"),
                "failure_kind": "site-default-cache", "expected_error": result["error"]} ]}
        self.proof_path = self.root / "J1_TRANSPORT_RECONCILIATION-r3.json"
        save(self.proof_path, self.proof_body)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, failed=None, states=None):
        with patch.object(auto.transport, "binding", return_value={"sha256": "test-transport"}):
            return auto.validate_j1_reconciliation(self.cp, self.root,
                failed if failed is not None else [self.job], states or self.states)

    def test_only_proven_preinitialization_failure_is_recoverable(self):
        self.assertEqual(self.validate()["eligible_recovery_uids"], ["uid"])

    def test_unseen_failed_recovery_cannot_receive_another_allocation(self):
        extra = {"card": self.card, "submission": seal({"job_id": "124"})}
        with self.assertRaisesRegex(ValueError, "failure set differs"):
            self.validate([self.job, extra])

    def test_running_or_cancelled_predecessor_is_not_recoverable(self):
        for state in ({"state": "RUNNING", "exit_code": "1:0", "live": True},
                      {"state": "CANCELLED", "exit_code": "1:0"}):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "accounting"):
                self.validate(states={"123": state})

    def test_actual_reader_load_invalidates_zero_draw_recovery_proof(self):
        save(self.invocation / "RUNTIME.json", {"load_seconds": 1})
        with self.assertRaisesRegex(ValueError, "before reader initialization"):
            self.validate()

    def test_item_attempt_in_failed_job_requires_new_explicit_reconciliation(self):
        save(self.root / "J1/audit/uid.attempt-1.json", {"job_id": "123"})
        with self.assertRaisesRegex(ValueError, "actually attempted"):
            self.validate()

    def test_changed_failure_log_invalidates_proof(self):
        self.log.write_text("different failure\n")
        with self.assertRaisesRegex(ValueError, "log changed"):
            self.validate()

    def test_no_expansion_beyond_original_failed_slice(self):
        save(self.proof_path, {**self.proof_body, "eligible_recovery_uids": ["uid", "extra"]}, replace=True)
        with self.assertRaisesRegex(ValueError, "roster differs"):
            self.validate()

    def test_full100_artifacts_do_not_bypass_unreconciled_failed_job(self):
        run = object.__new__(auto.Advance)
        run.path = self.cp; run.root = self.root
        run.jobs = [{"card": {"extension": {"kind": "j1-audit-slice"}},
                     "submission": {"job_id": "124"}}]
        run.observations = {"124": {"state": "FAILED", "exit_code": "1:0"}}
        frame = {"records": [{"uid": f"uid{i}"} for i in range(100)]}
        save(self.root / "J1/AUDIT_FRAME.json", frame)
        for row in frame["records"]:
            save(self.root / "J1/audit" / (row["uid"] + ".json"), {"uid": row["uid"]})
        with patch.object(auto.j1, "deployment", return_value=(self.campaign, {}, frame, {})), \
             patch.object(auto.j1, "validate_verdict"), \
             patch.object(run, "cpu") as qualification, \
             patch.object(auto, "validate_j1_reconciliation", side_effect=ValueError("unreconciled failure")):
            with self.assertRaisesRegex(ValueError, "unreconciled failure"):
                run.shared_j1()
            qualification.assert_not_called()

    def recovery_frontier(self, checked=False, failed_previous_check=False):
        run = object.__new__(auto.Advance)
        run.path = self.cp; run.root = self.root
        run.jobs = [{"card": {"extension": {"kind": "j1-audit-slice"}},
                     "submission": {"job_id": "123"}}]
        run.observations = dict(self.states)
        run.request = Mock(return_value=(self.cp, {})); run.dispatch = Mock()
        frame = {"records": [{"uid": f"uid{i}"} for i in range(100)]}
        save(self.root / "J1/AUDIT_FRAME.json", frame)
        save(self.root / "J1/audit/uid12.json", {"retained": True})
        executor = {"sha256": "scientific-executor"}
        prices = {"j1_executor": executor, "components": {"j1_generation": {"max_context_input_tokens": 4}}}
        run.price = Mock(return_value=(self.cp, prices))
        if checked:
            save(self.root / "J1/audit/uid0.json", {"retained": True})
            run.jobs.append({"card": {"transport_executor": {}, "extension":
                {"kind": "j1-audit-slice", "selected_uids": ["uid0"]}},
                "submission": {"job_id": "124"}})
            # A genuine sealed binding is truthy, as in actual transport cards.
            run.jobs[-1]["card"]["transport_executor"] = {"sha256": "transport"}
            run.observations["124"] = {"state": "COMPLETED", "exit_code": "0:0"}
        if failed_previous_check:
            run.jobs.append({"card": {"transport_executor": {"sha256": "old-r2-transport"}, "extension":
                {"kind": "j1-audit-slice", "selected_uids": ["uid0"]}}, "submission": {"job_id": "126"}})
            run.observations["126"] = {"state": "FAILED", "exit_code": "1:0"}
        proof = {"sha256": "reconciliation", "failed_jobs": [{"job_id": "123"}] +
                 ([{"job_id": "126"}] if failed_previous_check else []), "eligible_recovery_uids":
                 [r["uid"] for r in frame["records"] if r["uid"] != "uid12"]}
        with patch.object(auto.j1, "deployment", return_value=(self.campaign, {}, frame, executor)), \
             patch.object(auto.j1, "validate_verdict"), \
             patch.object(auto, "validate_j1_reconciliation", return_value=proof), \
             patch.object(auto.j1, "probe", return_value={"max_input_tokens": 4}), \
             patch.object(auto.transport, "verify_card"), \
             patch.object(auto, "fit_size", side_effect=lambda p, work, n: (n, 3600, {})):
            status = run.shared_j1()
        return run, status

    def test_first_recovery_allocates_only_one_missing_original_item(self):
        run, status = self.recovery_frontier()
        self.assertEqual(status, "DISPATCHED_TRANSPORT_CHECK")
        run.request.assert_called_once()
        self.assertEqual(run.request.call_args.kwargs["count"], 1)
        self.assertEqual(run.request.call_args.kwargs["extension_"]["parameters"]["begin"], 0)
        run.dispatch.assert_called_once()

    def test_remaining_recovery_reuses_both_committed_verdicts(self):
        run, status = self.recovery_frontier(checked=True)
        self.assertEqual(status, "DISPATCHED_RECONCILED_AUDIT")
        slices = [c.kwargs["extension_"]["parameters"] for c in run.request.call_args_list]
        self.assertEqual([(p["begin"], p["count"]) for p in slices], [(1, 11), (13, 87)])
        self.assertEqual(sum(p["count"] for p in slices), 98)


    def test_reconciled_old_failed_check_does_not_skip_new_one_item_gate(self):
        run, status = self.recovery_frontier(failed_previous_check=True)
        self.assertEqual(status, "DISPATCHED_TRANSPORT_CHECK")
        run.request.assert_called_once()
        self.assertEqual(run.request.call_args.kwargs["count"], 1)
        self.assertEqual(run.request.call_args.kwargs["extension_"]["parameters"]["begin"], 0)

    def test_exact_socket_path_failure_is_reconciled_without_relaxing_draw_gate(self):
        error = {"type": "ZMQError", "message": "ipc path original/rpc/uuid is longer than 107 characters"}
        invocation = save(self.invocation / "RESULT.json", {"status": "FAIL", "loaded": False,
            "new_verdicts": 0, "error": error}, replace=True)
        body = dict(self.proof_body)
        row = {**body["failed_jobs"][0], "failure_kind": "unix-socket-path", "expected_error": error,
            "source_receipts": {str(self.invocation / "RESULT.json"): invocation["sha256"]}}
        save(self.proof_path, {**body, "failed_jobs": [row]}, replace=True)
        self.assertEqual(self.validate()["failed_jobs"][0]["failure_kind"], "unix-socket-path")
        save(self.invocation / "RUNTIME.json", {"load_seconds": 1})
        with self.assertRaisesRegex(ValueError, "before reader initialization"):
            self.validate()

    def test_changed_error_does_not_match_an_explicit_socket_reconciliation(self):
        error = {"type": "ZMQError", "message": "different runtime failure"}
        value = save(self.invocation / "RESULT.json", {"status": "FAIL", "loaded": False,
            "new_verdicts": 0, "error": error}, replace=True)
        row = {**self.proof_body["failed_jobs"][0], "source_receipts":
               {str(self.invocation / "RESULT.json"): value["sha256"]}, "failure_kind": "unix-socket-path",
               "expected_error": error}
        save(self.proof_path, {**self.proof_body, "failed_jobs": [row]}, replace=True)
        with self.assertRaisesRegex(ValueError, "before reader initialization"):
            self.validate()

    def test_foreign_repair_transport_cannot_reuse_failure_permission(self):
        save(self.proof_path, {**self.proof_body, "recovery_transport_executor":
            {"sha256": "foreign-transport"}}, replace=True)
        with self.assertRaisesRegex(ValueError, "source or recovery bound"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
