"""Audit continuation requires actual prices and preserves disjoint original items."""
import unittest
import os
from unittest.mock import patch

from moe_exp.verification_study.contracts import load,save,seal
import verification_study_v3_j1_continue_r1 as continuation
import verification_study_v3_j1_r1 as j1
import test_verification_j1_resume_v3 as fixtures


class J1BudgetTests(unittest.TestCase):
    def prices(self,rate=.05):
        return seal({"schema":"verification-prices-v3","components":{
            "j1_load":{"status":"MEASURED","seconds_per_unit_max":500},
            "j1_generation":{"status":"MEASURED","seconds_per_unit_max":rate}}})
    def test_cap_and_complete_item_recovery_control_size(self):
        count,seconds,budget=continuation.priced_size(self.prices(),99)
        self.assertEqual(count,2);self.assertEqual(continuation.work(count)["j1_generation"],3*8192*3)
        self.assertEqual(continuation.work(count)["j1_load"],2)
        self.assertLessEqual(budget["reservation_seconds"],seconds);self.assertLessEqual(seconds,7200)
        self.assertLess(continuation.priced_size(self.prices(.1),99)[0],count)
    def test_slow_single_item_gets_measured_larger_finite_shape(self):
        count,seconds,budget=continuation.priced_size(self.prices(.2),99)
        self.assertEqual(count,1);self.assertGreater(seconds,7200);self.assertLessEqual(seconds,86400)
    def test_impossible_full_item_cannot_become_unpriced_pilot(self):
        with self.assertRaisesRegex(ValueError,"does not fit"):continuation.priced_size(self.prices(3),99)
    def test_completed_and_live_items_split_contiguous_slices(self):
        rows=[{"uid":str(i)} for i in range(8)]
        self.assertEqual(continuation.uncovered_blocks(rows,{"2","7"},{"0","4","5"}),[(1,1),(3,1),(6,1)])
        with self.assertRaisesRegex(ValueError,"foreign"):continuation.uncovered_blocks(rows,{"foreign"},set())


class J1ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.J1ResumeTests(methodName="runTest");self.fixture.setUp()
        self.base=self.fixture.fixture.base;self.cp=self.base/"CAMPAIGN.json"
    def tearDown(self):self.fixture.tearDown()
    def test_committed_verdict_without_health_price_does_not_advance(self):
        self.fixture.run_slice(12,1)
        with patch.object(continuation,"observed_jobs",return_value=([],[])),patch.object(continuation.extension,"submit") as submit:
            result=continuation.continue_audit(self.cp,submit=True)
        self.assertEqual(result["status"],"WAITING_FOR_ACTUAL_J1_PRICE_EVENT");submit.assert_not_called()
        self.assertEqual(fixtures.FakeGrader.loads,1)
    def test_unresolved_intent_cannot_be_bypassed_by_new_price_card(self):
        with patch.object(continuation,"observed_jobs",return_value=([],["immutable-intent"])),patch.object(continuation.extension,"submit") as submit:
            result=continuation.continue_audit(self.cp,submit=True)
        self.assertEqual(result["status"],"UNRESOLVED_SUBMISSION_INTENT");submit.assert_not_called()
    def priced_job(self):
        self.fixture.run_slice(12,1)
        prices_path=self.base/"prices-by-event/price.json"
        prices=j1.collect_prices(self.cp,"gpt",self.base,prices_path)
        item=self.fixture.frame["records"][12];path=self.fixture.output/(item["uid"]+".json")
        event=seal({"state":"STEP_COMPLETE","price_path":str(prices_path),"price_sha256":prices["sha256"],
            "observed_epoch":1,"committed_artifacts":[{"path":str(path),"sha256":load(path)["sha256"]}]})
        return {"card":{"extension":{"kind":"j1-audit-slice","selected_uids":[item["uid"]]}},
            "receipt":{"job_id":"100"},"live_state":None,"accounting":{"state":"COMPLETED","exit_code":"0:0"},"event":event}
    def test_valid_price_plan_excludes_completed_and_live_assignments(self):
        job=self.priced_job();live_uids=[r["uid"] for r in self.fixture.frame["records"][:5]]
        live={"card":{"extension":{"kind":"j1-audit-slice","selected_uids":live_uids}},"receipt":{"job_id":"101"},
            "live_state":"PENDING","accounting":{"state":"PENDING","exit_code":"0:0"},"event":None}
        with patch.object(continuation,"observed_jobs",return_value=([job,live],[])),\
             patch.object(j1,"probe",return_value=seal({"max_input_tokens":100})),patch.object(continuation.extension,"submit") as submit:
            result=continuation.continue_audit(self.cp,target_hours=24)
        submit.assert_not_called();planned=[]
        for card in result["cards"]:
            request=load(card["request_path"]);p=request["extension"]["parameters"]
            planned.extend(r["uid"] for r in self.fixture.frame["records"][p["begin"]:p["begin"]+p["count"]])
            self.assertFalse(request["measurement_pilot"]);self.assertEqual(request["model"],"gpt")
        self.assertEqual(len(planned),94);self.assertEqual(len(set(planned)),94)
        self.assertFalse(set(planned)&set(live_uids));self.assertNotIn(self.fixture.frame["records"][12]["uid"],planned)
        self.assertEqual(fixtures.FakeGrader.loads,1)
    def test_failed_callback_retains_branch_for_reconciliation(self):
        job=self.priced_job();job["event"]={**job["event"],"state":"PRICE_BINDING_FAILURE"}
        with patch.object(continuation,"observed_jobs",return_value=([job],[])),patch.object(continuation.extension,"submit") as submit:
            result=continuation.continue_audit(self.cp,submit=True)
        self.assertEqual(result["status"],"RECONCILE_TERMINAL_EXECUTION_OR_CALLBACK");submit.assert_not_called()
    def test_changed_immutable_price_binding_rejects(self):
        job=self.priced_job();job["event"]={**job["event"],"price_sha256":"changed"}
        with patch.object(continuation,"observed_jobs",return_value=([job],[])),self.assertRaisesRegex(ValueError,"price deployment"):
            continuation.continue_audit(self.cp)
    def test_two_failed_missing_item_attempts_cannot_launch_a_third_job(self):
        job=self.priced_job();fixtures.FakeGrader.fail_after=1
        for job_id in ("101","102"):
            os.environ["SLURM_JOB_ID"]=job_id
            with self.assertRaises(ValueError):self.fixture.run_slice(0,1)
        with patch.object(continuation,"observed_jobs",return_value=([job],[])),patch.object(continuation.extension,"submit") as submit:
            result=continuation.continue_audit(self.cp,submit=True)
        self.assertEqual(result["status"],"RECONCILE_RETRY_BUDGET_OR_UNTRACKED_ITEM_ATTEMPTS")
        submit.assert_not_called();self.assertEqual(len(result["attempt_paths"]),2)


if __name__=="__main__":unittest.main()
