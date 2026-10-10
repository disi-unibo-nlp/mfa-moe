"""The fixed audit, finite worker and health observer share exact inputs."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest,load,save,seal
import verification_study_v3_extension_r1 as extension
import verification_study_v3_health_r4 as health
import verification_study_v3_j1_r1 as j1
import test_verification_j1_resume_v3 as fixtures


class J1IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.J1ResumeTests(methodName="runTest");self.fixture.setUp();f=self.fixture;self.base=f.fixture.base
        self.p={"frame":str(f.frame_path),"output":str(f.output),"begin":7,"count":1}
        self.request=seal({"schema":"verification-task-request-v3","stage":"j1-audit","model":"gpt",
            "population":"frozen_J1_audit","claim":"qualify shared pinned grader","artifact":"retained audit verdict and costs",
            "reuse":"existing frozen100 audit","missing_cell":"J1 deployment role cost","decision_if_failure":"keep explicit unqualified grading cell",
            "measurement_pilot":True,"finite_assignments":1,"resources":{"partition":"boost_usr_prod","qos":"normal",
                "gpus":2,"cpus":16,"memory_gb":120,"walltime_seconds":3600},
            "arguments":[part for k,v in self.p.items() for part in ("--"+k,str(v))],
            "extension":{"kind":"j1-audit-slice","parameters":self.p}})
        with patch.object(j1,"probe",return_value=seal({"max_input_tokens":100})):
            self.card=extension.compile_card(self.base/"CAMPAIGN.json",self.request)
        self.card_path=self.base/"JOB_CARD.json";save(self.card_path,self.card)
        self.observation={"job_card_path":str(self.card_path),"campaign_path":str(self.base/"CAMPAIGN.json"),
            "predecessor_job_id":"100","expected_work":health.expected_work(self.card,self.base)}
    def tearDown(self):self.fixture.tearDown()
    def test_successful_j1_slice_has_no_missing_neighbors(self):
        f=self.fixture;f.run_slice(7,1)
        result=health.inspect_artifacts(self.observation,f.fixture.campaign)
        self.assertEqual(result["missing_artifacts"],[]);self.assertEqual(result["scientific_gate_failures"],[])
        self.assertEqual(self.observation["expected_work"]["expected_uids"],[f.frame["records"][7]["uid"]])
    def test_missing_j1_invocation_result_is_incomplete(self):
        f=self.fixture;f.run_slice(7,1);path=next(f.output.glob("invocations/*/RESULT.json"));path.unlink()
        result=health.inspect_artifacts(self.observation,f.fixture.campaign)
        self.assertIn(str(path),result["missing_artifacts"])
    def test_observer_cannot_credit_other_predecessor(self):
        f=self.fixture;f.run_slice(7,1);self.observation["predecessor_job_id"]="999"
        result=health.inspect_artifacts(self.observation,f.fixture.campaign)
        self.assertTrue(result["missing_artifacts"])
    def test_worker_and_observer_count_mismatch_rejects(self):
        args=[*self.request["arguments"]];args[args.index("--count")+1]="2"
        request=seal({**self.request,"arguments":args})
        with patch.object(j1,"probe",return_value=seal({"max_input_tokens":100})),self.assertRaisesRegex(ValueError,"observer arguments differ"):
            extension.compile_card(self.base/"CAMPAIGN.json",request)
    def test_final_j1_cannot_be_an_unpriced_measurement(self):
        request=seal({**self.request,"stage":"grade-j1","extension":{"kind":"grade-j1-slice","parameters":self.p}})
        with self.assertRaisesRegex(ValueError,"unpriced final experiment forbidden"):
            extension.compile_card(self.base/"CAMPAIGN.json",request)


class ResidualFreezeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.J1ResumeTests(methodName="runTest");self.fixture.setUp();f=self.fixture.fixture;self.base=f.base
        self.source=self.base/"gpt/final_native";self.grades=self.base/"gpt/final_native-grades"
        model=load(self.base/"MODEL.json");self.assignments=[{"uid":digest(["generation",n]),"stage":"final_native","arm":"native"} for n in range(2)]
        self.manifest_path=self.base/"gpt/FINAL_NATIVE_ASSIGNMENTS.json"
        self.manifest=save(self.manifest_path,{"stage":"final_native","model_sha256":model["sha256"],"assignments":self.assignments})
        self.records=[]
        for row in self.assignments:
            generation=save(self.source/(row["uid"]+".json"),{"schema":"verification-generation-receipt-v3","assignment":row,
                "model_sha256":model["sha256"],"runtime_sha256":f.campaign["runtime_sha256"],"executor_sha256":digest(f.campaign["code_files"])})
            blind=digest(["blind",row["uid"]])
            grade=save(self.grades/(row["uid"]+".json"),{"schema":"verification-grade-v3","assignment_uid":row["uid"],"blind_uid":blind,
                "generation_sha256":generation["sha256"],"manifest_sha256":self.manifest["sha256"],"scorer_contract":3,"accuracy":None})
            self.records.append({"uid":blind,"grade_sha256":grade["sha256"],"problem":"p","gold":"g","candidate":"c"})
        self.full_frame={"schema":"verification-j1-frame-v3","records":self.records,"manifest_sha256":self.manifest["sha256"],
            "reader_contract_sha256":f.campaign["reader_sha256"],"prompt_version":1,"parser_version":2,"votes":3,"output_cap":8192}
        save(self.grades/"J1_FRAME.json",{**self.full_frame,"records":self.records[:1]})
    def tearDown(self):self.fixture.tearDown()
    def freeze(self):
        return j1.freeze_residuals(self.base/"CAMPAIGN.json","gpt",self.manifest_path,self.source,self.grades,self.grades/"J1_FROZEN_FRAME.json")
    def test_rebuilds_full_roster_instead_of_freezing_last_shard(self):
        with patch("moe_exp.verification_study.closeout.require_cpu"),patch("moe_exp.verification_study.closeout.grade_strict",return_value=seal(self.full_frame)) as grader:
            result=self.freeze()
        self.assertEqual(len(result["records"]),2);self.assertTrue(result["full_roster_frozen"])
        self.assertEqual(grader.call_args.kwargs,{"begin":0,"count":None})
        self.assertEqual(len(result["source_grades"]),2)
    def test_pending_generation_cannot_be_frozen_as_missing(self):
        (self.source/(self.assignments[1]["uid"]+".json")).unlink()
        with patch("moe_exp.verification_study.closeout.require_cpu"),self.assertRaisesRegex(ValueError,"unattempted source path remains pending"):
            self.freeze()
        self.assertFalse((self.grades/"J1_FROZEN_FRAME.json").exists())
    def test_incomplete_strict_grading_rejects_before_rebuild(self):
        (self.grades/(self.assignments[1]["uid"]+".json")).unlink()
        with patch("moe_exp.verification_study.closeout.require_cpu"),patch("moe_exp.verification_study.closeout.grade_strict") as grader,\
             self.assertRaisesRegex(ValueError,"strict grading is still incomplete"):
            self.freeze()
        grader.assert_not_called()


if __name__=="__main__":unittest.main()
