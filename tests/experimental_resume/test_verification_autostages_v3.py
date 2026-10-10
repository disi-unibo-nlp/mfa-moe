"""Readiness and reservation contracts for unattended prospective advancement."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from moe_exp.verification_study.contracts import digest, file_sha, load, save, seal
from moe_exp.verification_study.registry import MODEL_KEYS
import verification_study_v3_autostages_r1 as auto


class AutostagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cv3-autostages-test-",
            dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.base = Path(self.tmp.name)
        self.source_file = self.base / "source.py"; self.source_file.write_text("# fixture\n")
        self.files = {str(self.source_file): file_sha(self.source_file)}
        self.mp = self.base / "MODEL.json"
        self.model = save(self.mp, {"key": "gpt", "expected_routed_layers": 2, "native_k": 4, "experts": 16})
        self.campaign = seal({"schema": "fixture", "code_files": self.files,
            "result_root": str(self.base), "runtime_sha256": "runtime", "reader_sha256": "reader",
            "models": {key: str(self.mp) for key in MODEL_KEYS}})
        self.cp = self.base / "CAMPAIGN.json"; save(self.cp, self.campaign)
        self.run = object.__new__(auto.Advance)
        self.run.path = self.cp; self.run.root = self.base; self.run.campaign = self.campaign
        self.run.home = self.base / "autostages-r1"; self.run.jobs = []; self.run.intents = []
        self.run.actions = []; self.run.errors = []; self.run.limits = []; self.run.active_stage = {}
        self.run.observations = {}; self.run.submit = True

    def tearDown(self):
        self.tmp.cleanup()

    def roster(self, count=48):
        rows = [{"uid": f"{index:064x}", "family": f"f{index}", "cap": 16384,
                 "stage": "discovery", "model_sha256": self.model["sha256"]} for index in range(count)]
        path = self.base / "DISCOVERY.json"
        manifest = save(path, {"schema": "verification-assignments-v3", "model_sha256": self.model["sha256"],
                             "stage": "discovery", "assignments": rows, "assigned": count})
        return path, manifest

    def job(self, manifest_path, source, begin, count, job="123", state="PENDING", exit_code=None):
        card = seal({"schema": "verification-job-card-v3", "model": "gpt", "stage": "run",
                     "campaign_sha256": self.campaign["sha256"], "code_files": self.files,
                     "payload_arguments": ["run", "--manifest", str(manifest_path), "--output", str(source),
                                           "--begin", str(begin), "--count", str(count)],
                     "resources": {"gpus": 2, "billing_core_hours_max": 16}})
        receipt = seal({"schema": "verification-submission-v3", "job_card_sha256": card["sha256"], "job_id": job})
        self.run.jobs.append({"card": card, "submission": receipt, "slot": self.base / "unused"})
        self.run.observations[job] = {"state": state, "exit_code": exit_code}
        return card, receipt

    def generation_receipt(self, source, row, job="123"):
        source.mkdir(parents=True, exist_ok=True)
        archive = source / (row["uid"] + ".npz"); archive.write_bytes(b"fixture archive")
        return save(source / (row["uid"] + ".json"), {"schema": "verification-generation-receipt-v3",
            "assignment_uid": row["uid"], "assignment": row, "model_sha256": self.model["sha256"],
            "runtime_sha256": "runtime", "executor_sha256": digest(self.files), "job_id": job,
            "engineering_only": False, "execution_complete": True,
            "capture": {"array_path": str(archive), "array_bytes": archive.stat().st_size,
                        "array_sha256": file_sha(archive)}})

    def attempt(self, source, row, number):
        return save(source / (row["uid"] + f".attempt-{number}.json"), {
            "schema": "verification-attempt-v3", "uid": row["uid"], "assignment_sha256": digest(row),
            "job_id": "123", "number": number, "status": "FAILED"})

    def prices(self):
        roles = ("generator_load", "prefill", "generation", "capture", "closeout", "controller")
        return seal({"schema": "verification-prices-v3", "components": {
            role: {"status": "MEASURED", "seconds_per_unit_max": .01,
                   "samples": [{"output_tokens": 128}]} for role in roles}})

    def test_live_discovery_blocks_partial_audit(self):
        path, manifest = self.roster(); source = self.base / "discovery"
        self.job(path, source, 0, 48)
        for row in manifest["assignments"][:10]:
            self.generation_receipt(source, row)
        with patch.object(self.run, "native_slices") as submit:
            result = self.run.generation("gpt", path, source, strict_success=True)
        self.assertEqual(result["status"], "WAITING"); submit.assert_not_called()
        self.assertEqual(len(result["covered"]), 48)

    def test_full_receipts_need_terminal_accounting(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1)
        self.generation_receipt(source, manifest["assignments"][0])
        self.assertEqual(self.run.generation("gpt", path, source, strict_success=True)["status"], "WAITING")
        self.run.observations["123"] = {"state": "COMPLETED", "exit_code": "0:0"}
        self.assertEqual(self.run.generation("gpt", path, source, strict_success=True)["status"], "COMPLETE")

    def test_failed_discovery_allocation_requires_reconciliation(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1, state="FAILED", exit_code="1:0")
        self.generation_receipt(source, manifest["assignments"][0])
        with self.assertRaisesRegex(ValueError, "successful completion accounting"):
            self.run.generation("gpt", path, source, strict_success=True)

    def test_corrupt_archive_cannot_qualify_discovery(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1, state="COMPLETED", exit_code="0:0")
        self.generation_receipt(source, manifest["assignments"][0])
        (source / (manifest["assignments"][0]["uid"] + ".npz")).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "archive differs"):
            self.run.generation("gpt", path, source, strict_success=True)

    def test_foreign_receipt_job_cannot_qualify(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1, state="COMPLETED", exit_code="0:0")
        self.generation_receipt(source, manifest["assignments"][0], job="999")
        with self.assertRaisesRegex(ValueError, "slice ownership"):
            self.run.generation("gpt", path, source, strict_success=True)

    def test_overlapping_live_slices_rejected(self):
        path, manifest = self.roster(3); source = self.base / "discovery"
        self.job(path, source, 0, 2); self.job(path, source, 1, 2, job="124")
        with self.assertRaisesRegex(ValueError, "overlapping live"):
            self.run.generation("gpt", path, source)

    def test_two_failed_attempts_stop_discovery_without_partial_frame(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1, state="FAILED", exit_code="1:0")
        self.attempt(source, manifest["assignments"][0], 0); self.attempt(source, manifest["assignments"][0], 1)
        with patch.object(self.run, "native_slices") as submit:
            result = self.run.generation("gpt", path, source, strict_success=True)
        self.assertEqual(result["status"], "LIMIT"); submit.assert_not_called()

    def test_third_identity_attempt_is_rejected(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.job(path, source, 0, 1, state="FAILED", exit_code="1:0")
        for n in range(3):
            self.attempt(source, manifest["assignments"][0], n)
        with self.assertRaisesRegex(ValueError, "third native identity"):
            self.run.generation("gpt", path, source)

    def test_untracked_attempt_is_rejected(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        self.attempt(source, manifest["assignments"][0], 0)
        with self.assertRaisesRegex(ValueError, "untracked native attempt"):
            self.run.generation("gpt", path, source)

    def test_uncovered_blocks_preserve_full_frame_positions(self):
        records = [{"uid": str(i)} for i in range(7)]
        self.assertEqual(auto.uncovered_blocks(records, {"2"}, {"5"}), [(0, 2), (3, 2), (6, 1)])
        with self.assertRaisesRegex(ValueError, "foreign completed"):
            auto.uncovered_blocks(records, {"foreign"}, set())

    def test_native_budget_full_cap_complete_retry_and_two_loads(self):
        work = auto.native_work(3, 16384, 4096, controller=True)
        self.assertEqual(work["generator_load"], 2)
        self.assertEqual(work["generation"], 4 * 16384)
        self.assertEqual(work["controller"], 4 * 16384)
        self.assertEqual(work["capture"], 16)
        self.assertEqual(work["closeout"], 16)

    def test_reader_budget_scales_short_draws_and_both_retry_votes(self):
        prices = {"components": {"eligibility_reader": {"status": "MEASURED", "samples": [{"output_tokens": 128}]}}}
        work = auto.reader_work(prices, "eligibility_reader", 3)
        self.assertEqual(work, {"reader_load": 2, "eligibility_reader": 2 * 4 * 8})

    def test_fit_reservation_rounds_minutes_and_respects_partition(self):
        count, seconds, budget = auto.fit_size(self.prices(), lambda n: auto.native_work(n, 16384, 16384), 5)
        self.assertEqual(count, 5); self.assertEqual(seconds % 60, 0)
        self.assertLessEqual(seconds, 86400); self.assertLessEqual(budget["reservation_seconds"], seconds)

    def test_entitlement_reserves_existing_pending_cost_and_storage(self):
        path, manifest = self.roster(2); source = self.base / "discovery"
        card, receipt = self.job(path, source, 0, 1)
        bound = auto.storage_bound(card, self.campaign)
        snapshot = {"balance": "IscrC_MIOSR a b 1000 c 0"}
        with patch.object(auto.native, "preflight", return_value=snapshot), \
             patch("verification_study_v3_discovery_continue_r2.exact_storage", return_value={"available_bytes": 2 * bound - 1}):
            with self.assertRaisesRegex(ValueError, "outstanding GPU output reservations"):
                self.run.entitlement(card)
        snapshot["balance"] = "IscrC_MIOSR a b 31 c 0"
        with patch.object(auto.native, "preflight", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "pending reservations"):
                self.run.entitlement(card)

    def test_terminal_card_cannot_report_phantom_new_dispatch(self):
        path, manifest = self.roster(1); source = self.base / "discovery"
        card, receipt = self.job(path, source, 0, 1, state="COMPLETED", exit_code="0:0")
        slot = self.base / "submissions" / card["sha256"]; save(slot / "SUBMISSION.json", receipt)
        with patch.object(auto.native, "task_card", return_value=card), patch.object(auto.native, "submit") as submit:
            with self.assertRaisesRegex(ValueError, "existing terminal card"):
                self.run.dispatch("gpt", "discovery", self.base / "request.json", {})
        submit.assert_not_called()

    def test_cpu_resume_records_outputs_without_repeating_function(self):
        output = self.base / "OUTPUT.json"; save(output, {"schema": "fixture-output", "status": "PASS"})
        function = Mock()
        with patch("moe_exp.verification_study.closeout.require_cpu"):
            result = self.run.cpu("gpt", "audit", function, [output], [self.cp])
        self.assertEqual(result["status"], "COMPLETE"); function.assert_not_called()

    def test_partial_terminal_shared_audit_is_an_error(self):
        frame_path = self.base / "J1/AUDIT_FRAME.json"
        frame = save(frame_path, {"schema": "fixture", "records": [{"uid": f"{i:064x}"} for i in range(100)]})
        with patch.object(auto.j1, "deployment", return_value=(self.campaign, {}, frame, {})):
            with self.assertRaisesRegex(ValueError, "terminal shared J1 audit is incomplete"):
                self.run.shared_j1()

    def test_next_actions_is_allocated_cpu_only(self):
        with self.assertRaisesRegex(ValueError, "allocated serial CPU"):
            auto.next_actions(self.cp, self.base, submit=False, observations={})

    def test_no_closeout_before_shared_j1_terminal(self):
        self.run.shared_j1 = Mock(return_value="WAITING")
        self.run.model = Mock(return_value="LIMIT"); self.run.pending = Mock(return_value=[])
        with patch("moe_exp.verification_study.closeout.require_cpu"), \
             patch.object(auto, "Advance", return_value=self.run), \
             patch("moe_exp.verification_study.closeout.assemble") as assemble:
            result = auto.next_actions(self.cp, self.base, submit=False, observations={})
        self.assertEqual(set(result["models"]), set(MODEL_KEYS)); self.assertNotEqual(result["status"], "COMPLETE")
        self.assertFalse(result["ready_cpu_actions"]); assemble.assert_not_called()


if __name__ == "__main__":
    unittest.main()
