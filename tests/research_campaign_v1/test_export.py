"""Terminal missingness, original-label conflicts and immutable exports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sqlite3

from moe_exp.research_campaign_v1 import contracts as C
from moe_exp.research_campaign_v1.annotation import label_index,context_id
from moe_exp.research_campaign_v1.export import closeout,_response
from moe_exp.research_campaign_v1.scoring import unknown_bounds
from moe_exp.research_campaign_v1 import controller as O,scheduler as S
from moe_exp.research_campaign_v1 import export as E
from test_campaign import FakeScheduler


class TerminalExport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="export-tests-",dir=C.REPO/"report")
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_conflicting_original_contexts_remain_unknown(self):
        inputs=dict(problem_statement="p",previous_sentence="before",sentence="current",next_sentence="after")
        config={"program_sha256":"original","temperature":1.,"base_url":"original-address"}
        annotator={"config":config,"revision":"pinned"}
        folder=self.root/"campaign-v3/labels/gpt/part-00"
        (folder/"checkpoints").mkdir(parents=True)
        (folder/"checkpoints/batch-000000.json").write_text(json.dumps({"binding":{"config":config}}))
        (folder/"annotations.json").write_text(json.dumps([{"inputs":inputs,"label":c} for c in ("Verify","Plan","Verify")]))
        campaign={"root":str(self.root),"annotator":annotator}
        with patch("moe_exp.research_campaign_v1.annotation.BASE",self.root):
            index=label_index(campaign,"gpt")
            self.assertEqual(label_index(campaign,"gpt"),index)
        with sqlite3.connect(index) as db:
            self.assertIsNone(db.execute("select label from labels").fetchone()[0])
            self.assertEqual(db.execute("select count(*) from receipts").fetchone()[0],3)
        self.assertEqual(C.load(self.root/"gpt/LABEL_INDEX.json")["conflicting_original_contexts"],1)
        self.assertNotEqual(context_id(inputs,annotator),context_id({**inputs,"next_sentence":"changed"},annotator))

    def test_unknown_response_and_strict_bounds(self):
        annotations=C.Units(self.root/"annotations")
        values,_=_response({"window":16,"action":{"pulse_tokens":4}},None,None,annotations,{"target":"Verify","windows":[4,8,16]})
        self.assertIsNone(values["first:Verify"])
        self.assertIsNone(values["occupancy:w0:Verify"])
        self.assertIsNone(values["class_switches"])
        self.assertEqual(unknown_bounds(False,"stop",True),(0.,1.))
        self.assertEqual(unknown_bounds(False,"length",True),(0.,0.))
        self.assertEqual(unknown_bounds(False,"stop",False),(0.,0.))
        self.assertEqual(unknown_bounds(True,"stop",True),(1.,1.))
        self.assertEqual(unknown_bounds(None,"stop",False),(0.,1.))

    def test_export_survives_missing_inputs_and_is_idempotent(self):
        campaign=C.seal({"root":str(self.root),"campaign_id":"synthetic-terminal","code_files":{},"authorization_baseline_consumed":25950.,
            "models":{m:{"layers":[0,1,2],"experts":8} for m in C.MODELS},"annotator":{},
            "preserved_scientific_files":{},"preserved_operational_files":{}})
        cohort={"benchmark":"math500","family":"fake-family","question":"fake-question","fold":0}
        C.publish(self.root/"COHORTS.json",{"pruning":[cohort],"causal":[cohort]})
        scheduler=FakeScheduler()
        milestone=S.submit_card(campaign,S.card(campaign,"snapshot",begin="2026-10-15T21:59:59"),scheduler)
        observed={"actual_local_hours":0.,"terminal":{"status":"upstream_unavailable"}}
        # This is a tiny fake campaign: no corpus loading, inference or scientific
        # preprocessing occurs on the login node.
        with patch("moe_exp.research_campaign_v1.controller.status",return_value=observed):
            original_frame=E.frame
            def interrupted_frame(path,records):
                original_frame(path,records)
                if Path(path).name=="cells.parquet":
                    raise RuntimeError("synthetic interrupted export after atomic data commit")
            with patch.object(E,"frame",side_effect=interrupted_frame):
                with self.assertRaisesRegex(RuntimeError,"interrupted export"):
                    closeout(campaign)
            self.assertTrue((self.root/"output/EXPORT_STATE.json").exists())
            self.assertFalse((self.root/"output/REPORT.json").exists())
            # Changing telemetry after interruption must not conflict with the
            # already committed data or the frozen report observation.
            with patch("moe_exp.research_campaign_v1.controller.status",return_value={**observed,"actual_local_hours":1.}):
                result=closeout(campaign)
            again=closeout(campaign)
        self.assertTrue(result["REPORT_COMPLETE"]);self.assertTrue(again["REPORT_COMPLETE"])
        self.assertFalse(C.load(self.root/"SCIENCE_COMPLETE.json")["complete"])
        report=C.load(self.root/"output/REPORT.json")
        self.assertEqual(report["resource_usage"]["actual_local_hours"],0.)
        self.assertEqual(report["cells"],1400)
        self.assertEqual(report["status_counts"],{"upstream_unavailable":1400})
        self.assertGreaterEqual(len(report["figures"]),42)
        self.assertTrue((self.root/"output/cells.parquet").exists())
        self.assertTrue((self.root/"output/missingness.csv").exists())
        owner=O.control(campaign,S.card(campaign,"controller",generation=0),scheduler)
        self.assertIn(milestone["job_id"],owner["milestone_owner"]["wait_ids"])
        scheduler.values[milestone["job_id"]].update(state="COMPLETED",terminal=True,start="2026-10-15")
        C.publish(self.root/"stages"/O.stage_key("snapshot")/"RESULT.json",{"status":"complete"})
        O.control(campaign,S.card(campaign,"controller",generation=1),scheduler)
        self.assertTrue((self.root/"FINAL_ACCOUNTING.json").exists())


if __name__=="__main__":unittest.main()
