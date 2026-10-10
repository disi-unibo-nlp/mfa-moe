"""Terminal observer silence and absent accounting cannot imply science success."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import file_sha,load,save
import verification_study_v3_health_status as health


class HealthStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="cv3-status-test-",dir="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
        self.base=Path(self.tmp.name);source=self.base/"source";source.write_text("frozen")
        self.campaign_path=self.base/"manifests/CAMPAIGN.json"
        self.campaign=save(self.campaign_path,{"schema":"verification-campaign-v3","result_root":str(self.base/"results"),
            "code_files":{str(source):file_sha(source)}})
        card={"schema":"verification-health-card-v3","campaign_sha256":self.campaign["sha256"],
            "predecessor_job_id":"100","expected_work":{"model":"gpt","stage":"run"}}
        self.slot=self.base/"results/health/submissions/one"
        self.card=save(self.slot/"HEALTH_CARD.json",card)
        save(self.slot/"SUBMISSION.json",{"schema":"verification-health-submission-v3","health_card_sha256":self.card["sha256"],"job_id":"101"})
    def tearDown(self):self.tmp.cleanup()
    def observe(self,accounting,queue=""):
        with patch.object(health.subprocess,"run",side_effect=[SimpleNamespace(stdout=accounting),SimpleNamespace(stdout=queue)]):
            return health.observe(self.campaign_path)["callbacks"][0]
    def test_no_accounting_is_explicitly_unavailable(self):
        row=self.observe("")
        self.assertEqual(row["signal"],"ACCOUNTING_OR_SIGNAL_UNAVAILABLE")
    def test_pending_predecessor_keeps_waiting(self):
        row=self.observe("100|PENDING|0:0|0|\n101|PENDING|0:0|0|","100|PENDING|cv3-run-gpt|Priority\n101|PENDING|cv3-health-gpt|Dependency")
        self.assertEqual(row["signal"],"WAITING_FOR_PREDECESSOR")
    def test_terminal_observer_without_event_is_failure(self):
        row=self.observe("100|COMPLETED|0:0|10|\n101|FAILED|1:0|2|")
        self.assertEqual(row["signal"],"CALLBACK_FAILED_NO_EVENT")
        failure=load(row["failure_event_path"])
        self.assertEqual(failure["scientific_success"],"UNKNOWN_FROM_CALLBACK_SILENCE")
    def test_reconciled_unstarted_observer_is_preserved_superseded(self):
        save(self.campaign_path.parent/"HEALTH_RECONCILIATION-r2.json",{"targets":["101"]})
        row=self.observe("100|PENDING|0:0|0|\n101|CANCELLED by 1|0:0|0|")
        self.assertEqual(row["signal"],"SUPERSEDED_UNSTARTED_OBSERVER")


if __name__=="__main__":unittest.main()
