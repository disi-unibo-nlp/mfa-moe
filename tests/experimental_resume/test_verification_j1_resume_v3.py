"""Fixed-vote J1 resume, frozen100 audit gates, costs and residual coverage."""
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest,file_sha,load,save,seal
import verification_study_v3_j1_r1 as j1
import test_verification_reader_resume_v3 as fixtures


class FakeGrader:
    loads=0
    graded=[]
    decisions={}
    fail_after=None
    def __init__(self,checkpoint):
        type(self).loads+=1;self.checkpoint=checkpoint;self.load_seconds=4.0
    def grade(self,item):
        if self.fail_after is not None and len(self.graded)>=self.fail_after:raise RuntimeError("interrupted item")
        self.graded.append(item["uid"]);decision=self.decisions[item["uid"]]
        seed=int(digest(["verification-J1",item["uid"]])[:8],16)%2000000000
        return seal({"schema":"verification-j1-verdict-v3","uid":item["uid"],"input_sha256":digest(item),
            "grade_sha256":item.get("grade_sha256"),"votes":[decision]*3,"verdict":decision,
            "draws":[{"vote":n,"seed":seed+n,"finish_reason":"stop","input_tokens":100,"output_tokens":5,
                "seconds":1.0,"text":'{"verdict":"'+decision+'"}'} for n in range(3)],
            "checkpoint":self.checkpoint,"prompt_version":1,"parser_version":2})
    def close(self):pass


class J1ResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReaderResumeTests(methodName="runTest");self.fixture.setUp();f=self.fixture
        self.frame_path=f.base/"J1/AUDIT_FRAME.json";self.frame=save(self.frame_path,j1.audit_body(f.campaign))
        self.output=f.base/"J1/audit";FakeGrader.loads=0;FakeGrader.graded=[];FakeGrader.fail_after=None
        FakeGrader.decisions={uid:label["verdict"] for uid,label in self.frame["labels"].items()}
    def tearDown(self):self.fixture.tearDown()
    def run_slice(self,begin=0,count=1):
        return j1.run_slice(self.fixture.base/"CAMPAIGN.json",self.frame_path,self.output,time.time()+600,begin,count,
            reader_factory=FakeGrader,check_runtime=False)
    def qualify(self):
        with patch("moe_exp.verification_study.closeout.require_cpu"):
            return j1.qualification(self.fixture.base/"CAMPAIGN.json",self.frame_path,self.output,self.fixture.base/"QUALIFIED.json")
    def change_decision(self,item,decision):
        path=self.output/(item["uid"]+".json");value=load(path)
        save(path,{**value,"votes":[decision]*3,"verdict":decision,
            "draws":[{**d,"text":'{"verdict":"'+decision+'"}'} for d in value["draws"]]},replace=True)
    def test_exact_slice_preserves_full_frame_and_three_seeded_votes(self):
        self.run_slice(7,1);item=self.frame["records"][7]
        self.assertEqual(FakeGrader.graded,[item["uid"]]);value=load(self.output/(item["uid"]+".json"))
        self.assertEqual(value["frame_sha256"],self.frame["sha256"]);self.assertEqual(len(value["draws"]),3)
        self.assertFalse((self.output/(self.frame["records"][0]["uid"]+".json")).exists())
    def test_complete_resume_never_reloads_or_resamples(self):
        self.run_slice();files={p:file_sha(p) for p in self.output.glob("*.json")};runtimes=list(self.output.glob("invocations/*/RUNTIME.json"))
        result=self.run_slice();self.assertEqual(result["new_verdicts"],0);self.assertFalse(result["loaded"])
        self.assertEqual(FakeGrader.loads,1);self.assertEqual(len(FakeGrader.graded),1)
        self.assertEqual(files,{p:file_sha(p) for p in self.output.glob("*.json")})
        self.assertEqual(runtimes,list(self.output.glob("invocations/*/RUNTIME.json")))
    def test_partial_failure_keeps_committed_verdict_and_resumes_missing_item(self):
        FakeGrader.fail_after=1
        with self.assertRaises(ValueError):self.run_slice(0,2)
        first=self.output/(self.frame["records"][0]["uid"]+".json");sha=file_sha(first)
        FakeGrader.fail_after=None;os.environ["SLURM_JOB_ID"]="101";result=self.run_slice(0,2)
        self.assertEqual(result["new_verdicts"],1);self.assertEqual(file_sha(first),sha);self.assertEqual(FakeGrader.loads,2)
        self.assertEqual(FakeGrader.graded,[r["uid"] for r in self.frame["records"][:2]])
    def test_invalid_executor_or_parser_or_vote_rejects_before_load(self):
        self.run_slice();path=self.output/(self.frame["records"][0]["uid"]+".json");original=load(path)
        for change in ({"j1_executor_sha256":None},{"parser_version":1},{"grade_sha256":"foreign"},{"votes":["UNCERTAIN"]*3}):
            with self.subTest(change=change):
                save(path,{**original,**change},replace=True)
                with self.assertRaises(ValueError):self.run_slice()
                self.assertEqual(FakeGrader.loads,1);save(path,original,replace=True)
    def test_no_qualification_from_partial_audit(self):
        self.run_slice()
        with self.assertRaisesRegex(ValueError,"audit incomplete"):self.qualify()
        self.assertFalse((self.fixture.base/"QUALIFIED.json").exists())
    def test_original_full100_gate_passes_only_with_complete_correct_votes(self):
        self.run_slice(0,100);qualified=self.qualify()
        self.assertEqual(qualified["status"],"PASS");self.assertEqual(qualified["matches"],100)
        self.assertEqual(qualified["truth_scope"],"frozen Claude/Codex-DeepSeek agent consensus, not independent humans")
    def test_one_clear_wrong_false_equivalent_fails_even_with99_matches(self):
        self.run_slice(0,100)
        item=next(r for r in self.frame["records"] if self.frame["labels"][r["uid"]]["stratum"]=="clear_wrong_control")
        self.change_decision(item,"EQUIVALENT");qualified=self.qualify()
        self.assertEqual(qualified["matches"],99);self.assertEqual(qualified["status"],"FAIL")
        self.assertEqual(len(qualified["false_equivalents"]),1)
    def test_three_errors_in_one_stratum_fail_with97_matches(self):
        self.run_slice(0,100)
        items=[r for r in self.frame["records"] if self.frame["labels"][r["uid"]]["stratum"]=="mv_positive_control"][:3]
        for item in items:self.change_decision(item,"NOT_EQUIVALENT")
        qualified=self.qualify();self.assertEqual(qualified["matches"],97);self.assertEqual(qualified["status"],"FAIL")
        self.assertEqual(qualified["per_stratum_errors"]["mv_positive_control"],3)
    def test_price_counts_each_load_and_all_three_draw_costs_once(self):
        self.run_slice();os.environ["SLURM_JOB_ID"]="101";self.run_slice(1,1);self.run_slice(1,1)
        prices=j1.collect_prices(self.fixture.base/"CAMPAIGN.json","gpt",self.fixture.base,self.fixture.base/"PRICES.json")
        self.assertEqual(len(prices["components"]["j1_load"]["samples"]),2)
        self.assertEqual(len(prices["components"]["j1_generation"]["samples"]),2)
        self.assertEqual(sum(s["seconds"] for s in prices["components"]["j1_generation"]["samples"]),6)
        self.assertEqual(sum(s["units"] for s in prices["components"]["j1_generation"]["samples"]),30)
        self.assertEqual(prices["components"]["grading"]["status"],"UNMEASURED")
    def test_third_failed_item_invocation_rejects_before_load(self):
        FakeGrader.fail_after=0
        for job in ("100","101"):
            os.environ["SLURM_JOB_ID"]=job
            with self.assertRaises(ValueError):self.run_slice()
        os.environ["SLURM_JOB_ID"]="102"
        with self.assertRaisesRegex(ValueError,"retry budget exhausted"):self.run_slice()
        self.assertEqual(FakeGrader.loads,2)
    def test_modified_audit_truth_rejects_before_load(self):
        label_uid=next(iter(self.frame["labels"]));labels={**self.frame["labels"],label_uid:{**self.frame["labels"][label_uid],"verdict":"UNCERTAIN"}}
        save(self.frame_path,{**self.frame,"labels":labels},replace=True)
        with self.assertRaisesRegex(ValueError,"audit frame changed"):self.run_slice()
        self.assertEqual(FakeGrader.loads,0)


if __name__=="__main__":unittest.main()
