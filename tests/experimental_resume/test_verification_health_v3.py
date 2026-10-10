"""Failure signals must not become dispatch success or destroy newer evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest, file_sha, load, save
from moe_exp.verification_study.registry import REPO

spec=importlib.util.spec_from_file_location("verification_health",REPO/"scripts/experimental_resume/verification_study_v3_health_r2.py")
health=importlib.util.module_from_spec(spec);spec.loader.exec_module(health)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="cv3-health-test-",dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.base=Path(self.tmp.name)
        self.model=save(self.base/"MODEL.json",{"schema":"verification-model-v3","key":"gpt"})
        self.source=self.base/"source.proof";self.source.write_text("frozen")
        self.campaign=save(self.base/"CAMPAIGN.json",{"schema":"verification-campaign-v3",
            "models":{"gpt":str(self.base/"MODEL.json")},"runtime_sha256":"runtime",
            "code_files":{str(self.source):file_sha(self.source)}})
        self.rows=[{"uid":"a","model_sha256":self.model["sha256"]},{"uid":"b","model_sha256":self.model["sha256"]}]
        self.manifest=save(self.base/"ROSTER.json",{"schema":"verification-assignments-v3","assignments":self.rows})
        self.card=save(self.base/"JOB_CARD.json",{"schema":"verification-job-card-v3","stage":"run","model":"gpt",
            "payload_arguments":["run","--manifest",str(self.base/"ROSTER.json"),"--output",str(self.base/"data"),"--begin","1","--count","1"]})
        self.work=health.expected_work(self.card,self.base)
        self.observation={"expected_work":self.work,"job_card_path":str(self.base/"JOB_CARD.json")}

    def tearDown(self):self.tmp.cleanup()

    def receipt(self,assignment=None):
        assignment=assignment or self.rows[1]
        return {"schema":"verification-generation-receipt-v3","assignment":assignment,"assignment_uid":assignment["uid"],
            "model_sha256":self.model["sha256"],"runtime_sha256":"runtime",
            "executor_sha256":digest(self.campaign["code_files"]),"engineering_only":False,"execution_complete":True}

    def test_exact_slice_does_not_credit_neighbor_receipt(self):
        self.assertEqual(self.work["expected_uids"],["b"])
        save(self.base/"data/b.json",self.receipt(self.rows[0]))
        with self.assertRaises(ValueError):health.inspect_artifacts(self.observation,self.campaign)

    def test_complete_scheduler_without_receipt_emits_incomplete_signal(self):
        predecessor=save(self.base/"SUBMISSION.json",{"schema":"verification-submission-v3","job_id":"123"})
        card=save(self.base/"HEALTH_CARD.json",{"schema":"verification-health-card-v3","result_root":str(self.base),
            "campaign_path":str(self.base/"CAMPAIGN.json"),"campaign_sha256":self.campaign["sha256"],
            "job_card_path":str(self.base/"JOB_CARD.json"),"job_card_sha256":self.card["sha256"],
            "predecessor_submission_path":str(self.base/"SUBMISSION.json"),"predecessor_submission_sha256":predecessor["sha256"],
            "predecessor_job_id":"123","observer_files":{str(self.source):file_sha(self.source)},"expected_work":self.work})
        with patch("moe_exp.verification_study.closeout.require_cpu"),patch.object(health.getpass,"getuser",return_value="lmolfett"),\
             patch.object(health,"terminal_accounting",return_value={"state":"COMPLETED","exit_code":"0:0"}),\
             patch("moe_exp.verification_study.paper.export",return_value={"sha256":"paper"}),\
             patch.dict("os.environ",{"SLURM_JOB_ID":"456"}):
            event=health.check(self.base/"HEALTH_CARD.json")
        self.assertEqual(event["state"],"INCOMPLETE_ARTIFACTS")
        self.assertFalse(event["execution_artifacts_complete"])
        self.assertEqual(event["automatic_scientific_submissions"],0)
        self.assertEqual(load(self.base/"health/events"/(card["sha256"]+".json")),event)

    def test_foreign_executor_receipt_is_not_complete(self):
        receipt=self.receipt();receipt["executor_sha256"]="other"
        save(self.base/"data/b.json",receipt)
        with self.assertRaises(ValueError):health.inspect_artifacts(self.observation,self.campaign)

    def test_old_callback_cannot_overwrite_newer_pointer(self):
        path=self.base/"latest.json"
        health.update_latest(path,{"observed_epoch":20,"state":"STEP_COMPLETE"})
        health.update_latest(path,{"observed_epoch":10,"state":"EXECUTION_FAILED"})
        self.assertEqual(load(path)["state"],"STEP_COMPLETE")

    def test_older_predecessor_observed_later_cannot_replace_newer_work(self):
        path=self.base/"latest.json"
        health.update_latest(path,{"predecessor_job_id":"20","observed_epoch":10,"state":"STEP_COMPLETE"})
        health.update_latest(path,{"predecessor_job_id":"10","observed_epoch":30,"state":"EXECUTION_FAILED"})
        self.assertEqual(load(path)["predecessor_job_id"],"20")

    def test_missing_or_corrupt_capture_is_not_complete(self):
        archive=self.base/"data/b.npz";archive.parent.mkdir(parents=True,exist_ok=True)
        archive.write_bytes(b"captured-native-routing")
        receipt=self.receipt();receipt["capture"]={"array_path":str(archive),"array_sha256":file_sha(archive),"array_bytes":archive.stat().st_size}
        save(self.base/"data/b.json",receipt)
        self.assertEqual(health.inspect_artifacts(self.observation,self.campaign)["missing_artifacts"],[])
        archive.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):health.inspect_artifacts(self.observation,self.campaign)
        archive.unlink()
        self.assertEqual(health.inspect_artifacts(self.observation,self.campaign)["missing_artifacts"],[str(archive)])

    def test_swapped_reader_draw_cannot_fill_other_filename(self):
        config=self.base/"reader.proof";config.write_text("pinned")
        reader=save(self.base/"READER.json",{"config_files":{str(config):file_sha(config)},"checkpoint":"pinned"})
        row={"uid":"u","candidate":"x"}
        frame=save(self.base/"FRAME.json",{"records":[row],"reader_contract_sha256":reader["sha256"]})
        card=save(self.base/"READER_CARD.json",{"stage":"rate","model":"gpt","payload_arguments":["rate","--frame",str(self.base/"FRAME.json"),"--output",str(self.base/"ratings"),"--role","eligibility"]})
        work=health.expected_work(card,self.base)
        observed={"expected_work":work,"job_card_path":str(self.base/"READER_CARD.json"),"campaign_path":str(self.base/"CAMPAIGN.json")}
        rating={"uid":"u","reader":1,"frame_sha256":frame["sha256"],"input_sha256":digest(row),"runtime_sha256":"runtime",
            "reader_contract_sha256":reader["sha256"],"checkpoint":"pinned","role":"eligibility"}
        save(self.base/"ratings/u.reader-0.json",rating)
        with self.assertRaises(ValueError):health.inspect_artifacts(observed,{**self.campaign,"reader_sha256":reader["sha256"]})

    def test_delayed_accounting_reads_do_not_resubmit(self):
        from types import SimpleNamespace
        with patch.object(health.subprocess,"run",side_effect=[SimpleNamespace(stdout=""),SimpleNamespace(stdout="123|COMPLETED|0:0|4|cpu=1|")]),patch.object(health.time,"sleep") as pause:
            result=health.terminal_accounting("123")
        self.assertEqual(result["state"],"COMPLETED")
        pause.assert_called_once_with(1)

    def test_transient_accounting_failure_stays_within_read_retry(self):
        from types import SimpleNamespace
        import subprocess
        with patch.object(health.subprocess,"run",side_effect=[subprocess.CalledProcessError(1,"sacct"),SimpleNamespace(stdout="123|COMPLETED|0:0|4|cpu=1|")]),patch.object(health.time,"sleep"):
            result=health.terminal_accounting("123")
        self.assertEqual(result["state"],"COMPLETED")


if __name__=="__main__":unittest.main()
