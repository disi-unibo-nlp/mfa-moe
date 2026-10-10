"""Reader completion means this slice and its exact invocation, not neighbors."""
import unittest

from moe_exp.verification_study.contracts import load,save
import verification_study_v3_health_r3 as health
import verification_study_v3_reader_r1 as runner
import test_verification_reader_resume_v3 as fixtures


class ReaderHealthTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReaderResumeTests(methodName="runTest");self.fixture.setUp();f=self.fixture
        self.card_path=f.base/"JOB_CARD.json"
        self.card=save(self.card_path,{"schema":"verification-job-card-v3","stage":"rate","model":"gpt",
            "reader_executor":runner.binding(f.campaign),"full_frame_sha256":f.frame["sha256"],
            "selected_uids":[f.rows[1]["uid"]],"payload_arguments":["rate","--frame",str(f.frame_path),
                "--output",str(f.output),"--role","eligibility","--begin","1","--count","1"]})
        self.work=health.expected_work(self.card,f.base)
        self.observation={"expected_work":self.work,"job_card_path":str(self.card_path),
            "campaign_path":str(f.base/"CAMPAIGN.json"),"predecessor_job_id":"100"}
    def tearDown(self):self.fixture.tearDown()
    def test_completed_slice_excludes_missing_neighbors(self):
        f=self.fixture;f.run_slice(1,1)
        self.assertEqual(self.work["expected_uids"],[f.rows[1]["uid"]])
        result=health.inspect_artifacts(self.observation,f.campaign)
        self.assertEqual(result["missing_artifacts"],[]);self.assertEqual(result["scientific_gate_failures"],[])
    def test_missing_invocation_result_never_reports_complete(self):
        f=self.fixture;f.run_slice(1,1);result_path=next(f.output.glob("invocations/*/RESULT.json"));result_path.unlink()
        result=health.inspect_artifacts(self.observation,f.campaign)
        self.assertIn(str(result_path),result["missing_artifacts"])
    def test_wrong_predecessor_invocation_is_not_credited(self):
        f=self.fixture;f.run_slice(1,1);self.observation["predecessor_job_id"]="999"
        with self.assertRaises(ValueError):health.inspect_artifacts(self.observation,f.campaign)


if __name__=="__main__":unittest.main()
