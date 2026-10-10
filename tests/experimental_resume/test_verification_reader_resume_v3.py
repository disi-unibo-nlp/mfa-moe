"""Meaningful regressions for the separately sealed reader's resume contract."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest,file_sha,load,save,seal
import verification_study_v3_reader_r1 as runner

STAGING=Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")


class FakeReader:
    loads=0
    draws=[]
    fail_after=None
    def __init__(self,checkpoint):
        type(self).loads+=1;self.checkpoint=checkpoint;self.load_seconds=3.0
    def rate(self,row,role,draw):
        if self.fail_after is not None and len(self.draws)>=self.fail_after:raise RuntimeError("interrupted draw")
        self.draws.append((row["uid"],draw))
        return seal({"schema":"verification-reader-receipt-v3","uid":row["uid"],"reader":draw,"role":role,
            "input_sha256":digest(row),"text":"ambiguous","parsed":None,"finish_reason":"length",
            "input_tokens":100,"output_tokens":1024,"seconds":2.0,"checkpoint":self.checkpoint})
    def close(self):pass


class ReaderResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="cv3-reader-test-",dir=STAGING)
        self.base=Path(self.tmp.name);self.output=self.base/"gpt/eligibility-ratings"
        self.frame_path=self.base/"gpt/AUDIT_FRAME.json"
        config=self.base/"config.json";config.write_text("{}")
        self.contract=save(self.base/"READER.json",{"schema":"verification-reader-contract-v3","checkpoint":"pinned-reader",
            "config_files":{str(config):file_sha(config)}})
        model=save(self.base/"MODEL.json",{"schema":"verification-model-v3","key":"gpt"})
        self.campaign=save(self.base/"CAMPAIGN.json",{"schema":"verification-campaign-v3","code_files":{str(config):file_sha(config)},
            "models":{"gpt":str(self.base/"MODEL.json")},"result_root":str(self.base),
            "runtime_sha256":"installed-runtime","reader_sha256":self.contract["sha256"]})
        self.rows=[{"uid":digest(["row",n]),"problem":"blind","prefix":"candidate"} for n in range(3)]
        self.frame=save(self.frame_path,{"schema":"verification-semantic-frame-v3","records":self.rows,
            "reader_contract_sha256":self.contract["sha256"]})
        FakeReader.loads=0;FakeReader.draws=[];FakeReader.fail_after=None
        self.env=patch.dict(os.environ,{"SLURM_JOB_ID":"100"});self.env.start()
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def run_slice(self,begin=0,count=1):
        return runner.rate_slice(self.base/"CAMPAIGN.json",self.frame_path,self.output,time.time()+600,begin,count,
            "eligibility",reader_factory=FakeReader,check_runtime=False)
    def test_full_frame_hash_and_exact_slice(self):
        self.run_slice(1,1)
        self.assertEqual(FakeReader.draws,[(self.rows[1]["uid"],0),(self.rows[1]["uid"],1)])
        self.assertEqual({load(p)["frame_sha256"] for p in self.output.glob("*.reader-*.json")},{self.frame["sha256"]})
        self.assertFalse((self.output/(self.rows[0]["uid"]+".reader-0.json")).exists())
    def test_complete_resume_never_loads_or_mutates_ratings_or_runtime(self):
        self.run_slice();before={p:file_sha(p) for p in self.output.glob("*.reader-*.json")}
        anchor=file_sha(self.output/"RUNTIME.json");result=self.run_slice()
        self.assertEqual(FakeReader.loads,1);self.assertEqual(result["new_ratings"],0);self.assertFalse(result["loaded"])
        self.assertEqual(before,{p:file_sha(p) for p in self.output.glob("*.reader-*.json")})
        self.assertEqual(anchor,file_sha(self.output/"RUNTIME.json"))
    def test_two_allocations_keep_first_runtime_and_record_both_loads(self):
        self.run_slice();anchor=file_sha(self.output/"RUNTIME.json")
        os.environ["SLURM_JOB_ID"]="101";self.run_slice(1,1)
        self.assertEqual(anchor,file_sha(self.output/"RUNTIME.json"));self.assertEqual(FakeReader.loads,2)
        self.assertEqual(len(list(self.output.glob("invocations/*/RUNTIME.json"))),2)
    def test_partial_failure_resumes_only_missing_draw(self):
        FakeReader.fail_after=1
        with self.assertRaisesRegex(ValueError,"slice failed"):self.run_slice()
        first=self.output/(self.rows[0]["uid"]+".reader-0.json");before=file_sha(first)
        FakeReader.fail_after=None;os.environ["SLURM_JOB_ID"]="101";result=self.run_slice()
        self.assertEqual(result["new_ratings"],1);self.assertEqual(FakeReader.loads,2);self.assertEqual(file_sha(first),before)
        self.assertEqual(FakeReader.draws,[(self.rows[0]["uid"],0),(self.rows[0]["uid"],1)])
    def test_foreign_input_or_draw_rejects_before_load(self):
        for field,value in (("input_sha256","foreign"),("reader",True),("role","outcome"),("frame_sha256","foreign"),
                            ("checkpoint","foreign"),("reader_executor_sha256","foreign")):
            with self.subTest(field=field):
                if not list(self.output.glob("*.reader-*.json")):self.run_slice()
                path=self.output/(self.rows[0]["uid"]+".reader-0.json");original=load(path)
                save(path,{**original,field:value},replace=True);loads=FakeReader.loads
                with self.assertRaises(ValueError):self.run_slice()
                self.assertEqual(FakeReader.loads,loads);save(path,original,replace=True)
    def test_missing_invocation_or_anchor_rejects_before_load(self):
        self.run_slice();path=self.output/(self.rows[0]["uid"]+".reader-0.json");receipt=load(path)
        runtime=Path(receipt["invocation_runtime_path"]);runtime.unlink()
        with self.assertRaises(FileNotFoundError):self.run_slice()
        self.assertEqual(FakeReader.loads,1)
    def test_duplicate_uids_and_invalid_slices_reject_before_load(self):
        with self.assertRaises(ValueError):self.run_slice(-1,1)
        with self.assertRaises(ValueError):self.run_slice(2,2)
        save(self.frame_path,{**self.frame,"records":[self.rows[0],self.rows[0]]},replace=True)
        with self.assertRaises(ValueError):self.run_slice()
        self.assertEqual(FakeReader.loads,0)
    def test_price_counts_each_actual_load_and_rating_once_after_resume(self):
        self.run_slice();os.environ["SLURM_JOB_ID"]="101";self.run_slice(1,1);self.run_slice(1,1)
        prices=runner.collect_reader_prices(self.base/"CAMPAIGN.json","gpt",self.base,self.base/"PRICES.json")
        loads=prices["components"]["reader_load"]["samples"]
        ratings=prices["components"]["eligibility_reader"]["samples"]
        self.assertEqual(len(loads),2);self.assertEqual(sum(v["seconds"] for v in loads),6)
        self.assertEqual(len(ratings),4);self.assertEqual(sum(v["seconds"] for v in ratings),8)
        self.assertEqual(prices["components"]["eligibility_reader"]["max_context_input_tokens"],100)
        self.assertEqual(prices["components"]["outcome_reader"]["status"],"UNMEASURED")
    def test_price_rejects_foreign_invocation_runtime(self):
        self.run_slice();path=next(self.output.glob("invocations/*/RUNTIME.json"));value=load(path)
        save(path,{**value,"runtime_sha256":"foreign"},replace=True)
        with self.assertRaises(ValueError):
            runner.collect_reader_prices(self.base/"CAMPAIGN.json","gpt",self.base,self.base/"PRICES.json")
    def test_missing_executor_provenance_rejects_before_load(self):
        self.run_slice();path=self.output/(self.rows[0]["uid"]+".reader-0.json");value=load(path)
        del value["reader_executor_sha256"];save(path,value,replace=True)
        with self.assertRaisesRegex(ValueError,"executor missing"):self.run_slice()
        self.assertEqual(FakeReader.loads,1)
    def test_changed_invocation_role_or_slice_rejects_before_load(self):
        self.run_slice();path=self.output/(self.rows[0]["uid"]+".reader-0.json");original=load(path)
        runtime_path=Path(original["invocation_runtime_path"]);runtime=load(runtime_path)
        for change in ({"role":"outcome"},{"slice":[1,1],"selected_uids":[self.rows[1]["uid"]]}):
            with self.subTest(change=change):
                changed=save(runtime_path,{**runtime,**change},replace=True)
                save(path,{**original,"invocation_runtime_sha256":changed["sha256"]},replace=True)
                with self.assertRaises(ValueError):self.run_slice()
                self.assertEqual(FakeReader.loads,1)
                save(runtime_path,runtime,replace=True);save(path,original,replace=True)


if __name__=="__main__":unittest.main()
