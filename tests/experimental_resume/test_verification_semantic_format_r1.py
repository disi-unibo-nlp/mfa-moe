"""Format-only amendment and immutable full-frame interpretation contracts."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.verification_study import semantics, readers, stages, closeout, views
from moe_exp.verification_study.contracts import digest, file_sha, load, save, seal, verify
import verification_study_v3_reader_r1 as finite_reader
import verification_study_v3_semantic_format_r1 as fmt


YES = {"candidate_valid": True, "constraint_bound": True, "already_checked": False, "eligible": True}
CHECKED = {"candidate_valid": True, "constraint_bound": True, "already_checked": True, "eligible": False}


def fenced(value):
    return "<think>bounded reasoning</think>\n\n```json\n" + json.dumps(value) + "\n```"


def bare_rating(record, draw, value=YES, *, finish="stop", role="eligibility"):
    return seal({"schema": "verification-reader-receipt-v3", "uid": record["uid"],
                 "reader": draw, "role": role, "input_sha256": digest(record),
                 "frame_sha256": "filled by caller", "checkpoint": "fixture-checkpoint",
                 "text": fenced(value), "finish_reason": finish})


class FormatTests(unittest.TestCase):
    def test_plain_json_and_single_json_fence_have_identical_decisions(self):
        plain = "<think>reasoning</think>\n" + json.dumps(YES)
        self.assertEqual(fmt.strict_suffix(plain), (json.dumps(YES), "PLAIN_SUFFIX"))
        self.assertEqual(fmt.strict_suffix(fenced(YES)), (json.dumps(YES), "JSON_FENCE"))
        self.assertEqual(fmt.parse_rating(plain, "eligibility"), YES)
        self.assertEqual(fmt.parse_rating(fenced(YES), "eligibility"), YES)
        self.assertIsNone(fmt._ORIGINAL_PARSER(fenced(YES), "eligibility"))

    def test_final_think_suffix_and_crlf_fence(self):
        text = "noise </think>ignored </think>\r\n```json\t\r\n" + json.dumps(YES) + "\r\n```"
        self.assertEqual(fmt.parse_rating(text, "eligibility"), YES)

    def test_surrounding_prose_cannot_be_mined(self):
        for text in ("answer: " + fenced(YES).split("</think>")[-1],
                     fenced(YES) + "\nExplanation", "answer: " + json.dumps(YES)):
            with self.subTest(text=text):
                self.assertIsNone(fmt.parse_rating(text, "eligibility"))

    def test_multiple_nested_and_unclosed_fences_are_rejected(self):
        texts = [fenced(YES) + "\n```json\n" + json.dumps(CHECKED) + "\n```",
                 "```json\n```json\n" + json.dumps(YES) + "\n```\n```",
                 "```json\n" + json.dumps(YES), "```json\n" + json.dumps(YES) + "\n```\n```"]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(fmt.strict_suffix(text)[1], "REJECTED_FENCE")
                self.assertIsNone(fmt.parse_rating(text, "eligibility"))

    def test_bare_uppercase_or_inline_fences_not_added_to_protocol(self):
        for text in ("```\n" + json.dumps(YES) + "\n```", "```JSON\n" + json.dumps(YES) + "\n```",
                     "```json " + json.dumps(YES) + "```"):
            self.assertEqual(fmt.strict_suffix(text)[1], "REJECTED_FENCE")
            self.assertIsNone(fmt.parse_rating(text, "eligibility"))

    def test_schema_bool_and_eligibility_invariants_remain_strict(self):
        invalid = [{**YES, "extra": True}, {k: v for k, v in YES.items() if k != "eligible"},
                   {**YES, "candidate_valid": 1}, {**YES, "eligible": "true"},
                   {**YES, "constraint_bound": False}, {**YES, "already_checked": True}, []]
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(fmt.parse_rating(fenced(value), "eligibility"))
        ambiguous = {**YES, "eligible": None}
        self.assertEqual(fmt.parse_rating(fenced(ambiguous), "eligibility"), ambiguous)

    def test_invalid_text_retains_original_none_behavior(self):
        for text in (None, 9, [], {}):
            self.assertEqual(fmt.strict_suffix(text)[1], "INVALID_TEXT")
            self.assertIsNone(fmt.parse_rating(text, "eligibility"))

    def test_backticks_inside_outcome_evidence_are_preserved(self):
        value = {"substantive_verification": True, "evidence": "checked ``` literal"}
        self.assertEqual(fmt.parse_rating(fenced(value), "outcome"), value)

    def test_install_restores_original_and_leaves_reader_cached_alias_unchanged(self):
        original = semantics.parse_rating; cached = readers.parse_rating
        self.assertIs(cached, original)
        with fmt.installed_parser():
            self.assertIs(semantics.parse_rating, fmt.parse_rating)
            self.assertIs(readers.parse_rating, cached)
            self.assertIsNone(readers.parse_rating(fenced(YES), "eligibility"))
            self.assertEqual(semantics.parse_rating(fenced(YES), "eligibility"), YES)
        self.assertIs(semantics.parse_rating, original)

    def test_install_restores_after_failure_and_refuses_nested_or_foreign_identity(self):
        original = semantics.parse_rating
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with fmt.installed_parser():
                with self.assertRaisesRegex(ValueError, "nested/foreign"):
                    with fmt.installed_parser():
                        self.fail("nested install should not enter")
                self.assertIs(semantics.parse_rating, fmt.parse_rating)
                raise RuntimeError("fixture failure")
        self.assertIs(semantics.parse_rating, original)
        foreign = lambda *_: None
        with patch.object(semantics, "parse_rating", foreign):
            with self.assertRaisesRegex(ValueError, "nested/foreign"):
                with fmt.installed_parser():
                    self.fail("foreign install should not enter")
            self.assertIs(semantics.parse_rating, foreign)

    def test_recognition_precision_coverage_and_cached_stage_alias_use_adapter(self):
        records = []
        for index in range(10):
            for checked in (False, True):
                records.append({"uid": f"{len(records):064x}", "family": f"a{index}", "split": "apparent",
                    "stratum": "unfiltered", "candidate": {"symbolic": True},
                    "proposed": {"eligible": not checked, "already_checked": checked}})
        for checked in (False, True):
            records.append({"uid": f"{len(records):064x}", "family": "held", "split": "held_out",
                "stratum": "unfiltered", "candidate": {},
                "proposed": {"eligible": not checked, "already_checked": checked}})
        frame = seal({"schema": "verification-semantic-frame-v3", "records": records, "model_sha256": "model"})
        ratings = [seal({**bare_rating(r, d, CHECKED if r["proposed"]["already_checked"] else YES),
                         "frame_sha256": frame["sha256"]}) for r in records for d in (0, 1)]
        self.assertEqual(stages.qualify_recognizer(frame, ratings)["status"], "FAIL")
        with fmt.installed_parser():
            result = stages.qualify_recognizer(frame, ratings)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["decisions"]["eligible"]["apparent_family_precision"], 1)
        self.assertTrue(all(r["reader_eligible"] is not None for r in result["unfiltered_coverage_rows"]))

    def test_views_nomination_uses_adapter_and_excludes_length_draw(self):
        import numpy as np
        records = [{"uid": f"{i:064x}", "source_uid": "source", "family": "f", "split": "apparent",
                    "position": 2} for i in (1, 2)]
        frame = seal({"records": records})
        model = seal({"experts": 4})
        ratings = [seal({**bare_rating(r, d, CHECKED if i == 0 else YES), "frame_sha256": frame["sha256"]})
                   for i, r in enumerate(records) for d in (0, 1)]
        captures = {"source": {"L0_native_ids": np.array([[1, 2], [1, 3]]), "L0_positions": np.array([0, 1])}}
        self.assertEqual(views.past_profile_rows(frame, ratings, captures, model), [])
        with fmt.installed_parser():
            rows = views.past_profile_rows(frame, ratings, captures, model)
            self.assertEqual({r["condition"] for r in rows}, {"positive", "comparison"})
            ratings[-1] = seal({**ratings[-1], "finish_reason": "length"})
            rows = views.past_profile_rows(frame, ratings, captures, model)
        self.assertEqual({r["condition"] for r in rows}, {"positive"})

    def test_cached_outcome_aliases_preserve_length_disagreement_and_evidence_rules(self):
        record = {"horizon": 1024, "arm_blinded": True, "window_text": "compute x = 2 and check x + 1 = 3",
                  "execution_complete": True}
        value = {"substantive_verification": True, "evidence": "check x + 1 = 3"}
        ratings = [{"reader": d, "finish_reason": "stop", "input_sha256": digest(record), "text": fenced(value)}
                   for d in (0, 1)]
        self.assertEqual(closeout.outcome(record, ratings)["reason"], "AMBIGUOUS_READER")
        with fmt.installed_parser():
            self.assertTrue(stages.outcome(record, ratings)["success"])
            ratings[0]["finish_reason"] = "length"
            self.assertEqual(closeout.outcome(record, ratings)["reason"], "MISSING_OR_UNBOUND_READER")
            ratings[0]["finish_reason"] = "stop"
            ratings[1]["text"] = fenced({"substantive_verification": False, "evidence": None})
            self.assertEqual(closeout.outcome(record, ratings)["reason"], "READER_DISAGREEMENT")
            ratings[1]["text"] = fenced({"substantive_verification": True, "evidence": "absent span"})
            self.assertEqual(closeout.outcome(record, ratings)["reason"], "INVALID_EVIDENCE_SPAN")
            ratings[0]["text"] = ratings[1]["text"] = fenced({"substantive_verification": None, "evidence": None})
            self.assertEqual(closeout.outcome(record, ratings)["reason"], "AMBIGUOUS_READER")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cv3-semantic-format-test-",
            dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.base = Path(self.tmp.name)
        files = {}
        for index in range(35):
            path = self.base / f"source{index}.py"; path.write_text(f"# fixture {index}\n")
            files[str(path)] = file_sha(path)
        self.files = files
        self.contract_path = self.base / "READER.json"
        self.contract = save(self.contract_path, {"schema": "verification-reader-contract-v3",
            "checkpoint": "fixture-checkpoint", "config_files": {next(iter(files)): next(iter(files.values()))},
            "draws": [0, 1], "output_cap": 1024})
        self.cp = self.base / "CAMPAIGN.json"
        self.campaign = save(self.cp, {"schema": "fixture-campaign", "code_files": files,
            "reader_sha256": self.contract["sha256"], "runtime_sha256": "runtime", "result_root": str(self.base)})
        self.expected = patch.object(fmt, "EXPECTED_CAMPAIGN_SHA256", self.campaign["sha256"])
        self.expected.start(); self.addCleanup(self.expected.stop)
        self.record = {"uid": "1" * 64, "problem": "problem", "prefix_text": "prefix", "candidate": {}}
        self.fp = self.base / "AUDIT_FRAME.json"
        self.frame = save(self.fp, {"schema": "verification-semantic-frame-v3", "records": [self.record],
            "reader_contract_sha256": self.contract["sha256"], "model_sha256": "model"})
        self.out = self.base / "eligibility-ratings"; self.out.mkdir()
        self.executor = finite_reader.binding(self.campaign)
        self.runtime_path = self.out / "invocations/123-1/RUNTIME.json"
        self.runtime = save(self.runtime_path, {"schema": "verification-reader-runtime-v3",
            "campaign_sha256": self.campaign["sha256"], "reader_sha256": self.contract["sha256"],
            "runtime_sha256": "runtime", "frame_sha256": self.frame["sha256"], "job_id": "123",
            "load_seconds": .1, "checkpoint": self.contract["checkpoint"], "role": "eligibility",
            "slice": [0, 1], "selected_uids": [self.record["uid"]],
            "reader_executor_sha256": self.executor["sha256"], "reader_executor": self.executor})
        save(self.out / "RUNTIME.json", self.runtime)
        self.bp = self.base / "SEMANTIC_FORMAT_BINDING-r1.json"
        self.binding = save(self.bp, {"schema": fmt.BINDING_SCHEMA,
            **fmt.binding_requirements(self.cp), "helper_path": str(fmt.SCRIPT)})
        self.write_rating(0, finish="length"); self.write_rating(1)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rating(self, draw, *, finish="stop", text=None, **changes):
        text = fenced(YES) if text is None else text
        path = self.out / (self.record["uid"] + f".reader-{draw}.json")
        return save(path, {"schema": "verification-reader-receipt-v3", "uid": self.record["uid"],
            "reader": draw, "role": "eligibility", "input_sha256": digest(self.record),
            "text": text, "parsed": fmt._ORIGINAL_PARSER(text, "eligibility"), "finish_reason": finish,
            "input_tokens": 20, "output_tokens": 1024 if finish == "length" else 1004, "seconds": .2,
            "checkpoint": self.contract["checkpoint"], "frame_sha256": self.frame["sha256"],
            "reader_contract_sha256": self.contract["sha256"], "runtime_sha256": "runtime",
            "reader_executor_sha256": self.executor["sha256"], "job_id": "123",
            "invocation_runtime_path": str(self.runtime_path), "invocation_runtime_sha256": self.runtime["sha256"],
            **changes}, replace=True)

    def test_binding_requires_exact_campaign_sources_helper_policy_and_deployment(self):
        self.assertEqual(fmt.load_binding(self.cp), self.binding)
        for field, wrong in (("campaign_sha256", "wrong"), ("scientific_code_files", {}),
                             ("helper_file_sha256", "wrong"), ("reader_contract_sha256", "wrong"),
                             ("reader_executor_sha256", "wrong"), ("parser_policy", {}),
                             ("helper_path", str(self.base / "foreign.py"))):
            with self.subTest(field=field):
                save(self.bp, {**self.binding, field: wrong}, replace=True)
                with self.assertRaisesRegex(ValueError, "binding.*differs"):
                    fmt.load_binding(self.cp)
        save(self.bp, self.binding, replace=True)
        with patch.object(fmt, "EXPECTED_CAMPAIGN_SHA256", "foreign"):
            with self.assertRaisesRegex(ValueError, "exact original"):
                fmt.load_binding(self.cp)

    def test_bound_source_change_and_annex_self_closure_rejected(self):
        first = Path(next(iter(self.files))); original = first.read_text(); first.write_text("changed")
        with self.assertRaisesRegex(ValueError, "bound source"):
            fmt.load_binding(self.cp)
        first.write_text(original)
        save(self.bp, {**self.binding, "code_files": {str(self.bp): file_sha(self.bp)}}, replace=True)
        with self.assertRaisesRegex(ValueError, "cannot seal itself"):
            fmt.load_binding(self.cp)

    def test_ledger_is_pure_repeatable_full_frame_and_preserves_unknown_length(self):
        before = {str(p): file_sha(p) for p in self.base.rglob("*") if p.is_file()}
        ledger = fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.assertEqual(verify(ledger, fmt.LEDGER_SCHEMA), ledger)
        self.assertEqual(ledger, fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility"))
        self.assertEqual(before, {str(p): file_sha(p) for p in self.base.rglob("*") if p.is_file()})
        self.assertEqual(ledger["draws"], [0, 1]); self.assertEqual(len(ledger["source_receipts"]), 2)
        first, second = ledger["records"]
        self.assertEqual(first["status"], "LENGTH_UNKNOWN"); self.assertIsNone(first["effective_parsed"])
        self.assertFalse(first["recovered_stop"])
        self.assertEqual(second["status"], "STOP_PARSED"); self.assertEqual(second["effective_parsed"], YES)
        self.assertTrue(second["recovered_stop"]); self.assertIsNone(second["old_parsed"])
        self.assertEqual(second["raw_text_json_sha256"], digest(fenced(YES)))
        self.assertEqual(second["raw_receipt_sha256"], load(second["raw_receipt_path"])["sha256"])

    def test_missing_draw_and_foreign_uid_cannot_make_partial_ledger(self):
        missing = self.out / (self.record["uid"] + ".reader-1.json"); missing.unlink()
        with self.assertRaisesRegex(ValueError, "both fixed draws"):
            fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.write_rating(1)
        foreign = self.out / ("2" * 64 + ".reader-1.json"); foreign.write_text(missing.read_text())
        with self.assertRaisesRegex(ValueError, "foreign reader filename"):
            fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")

    def test_old_diagnostic_tampering_and_changed_runtime_rejected(self):
        self.write_rating(1, parsed=YES)
        with self.assertRaisesRegex(ValueError, "diagnostic differs"):
            fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.write_rating(1)
        save(self.runtime_path, {**self.runtime, "selected_uids": ["2" * 64]}, replace=True)
        with self.assertRaisesRegex(ValueError, "slice identities"):
            fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")

    def test_receipt_draw_input_frame_deployment_and_invocation_are_bound(self):
        for changes in ({"reader": 0}, {"input_sha256": "wrong"}, {"frame_sha256": "wrong"},
                        {"role": "outcome"}, {"checkpoint": "wrong"}, {"runtime_sha256": "wrong"},
                        {"reader_executor_sha256": "wrong"}, {"job_id": "456"}):
            with self.subTest(changes=changes):
                self.write_rating(1, **changes)
                with self.assertRaises(ValueError):
                    fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")

    def test_nonstop_and_malformed_stop_remain_unknown_without_dropping_draws(self):
        self.write_rating(0, finish="abort")
        self.write_rating(1, text="explanation " + fenced(YES).split("</think>")[-1])
        ledger = fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.assertEqual([r["status"] for r in ledger["records"]], ["NONSTOP_UNKNOWN", "STOP_UNPARSEABLE"])
        self.assertTrue(all(r["effective_parsed"] is None for r in ledger["records"]))

    def test_changed_valid_raw_receipt_changes_ledger_provenance(self):
        original = fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.write_rating(1, text=json.dumps(YES))
        changed = fmt.decision_ledger(self.cp, self.fp, self.out, "eligibility")
        self.assertNotEqual(original["sha256"], changed["sha256"])
        self.assertNotEqual(original["source_receipts"], changed["source_receipts"])
        self.assertEqual(original["records"][1]["effective_parsed"], changed["records"][1]["effective_parsed"])


if __name__ == "__main__":
    unittest.main()
