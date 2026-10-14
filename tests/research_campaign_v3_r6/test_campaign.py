"""Within-execution and unattended failure contracts; no scientific GPU redraws."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from moe_exp.research_campaign_v3_r6 import contracts as C
from moe_exp.research_campaign_v3_r6 import design as D
from moe_exp.research_campaign_v3_r6 import scheduler as S
from moe_exp.research_campaign_v3_r6 import controller as O
from moe_exp.research_campaign_v3_r6.artifacts import atomic_file
from moe_exp.research_campaign_v3_r6.adapters import SignedCapture,pulse_active,validate_action
from moe_exp.research_campaign_v3_r6.runtime import remaining_allowance
from moe_exp.research_campaign_v3_r6.statistics import bootstrap_panel,joint_tables,log_odds_ratio,trajectories


class FakeScheduler:
    def __init__(self):
        self.values={};self.next=100;self.releases=[];self.lost=False;self.capacity=0
    def submit(self,card,test_only=False):
        self.next+=1
        jid=str(self.next)
        self.values[jid]={"card":card,"state":"PENDING","start":"None","terminal":False}
        for i,child in enumerate(card.get("array_cards",[])):
            self.values[jid+"_"+str(i)]={"card":child,"state":"PENDING","start":"None","terminal":False,"parent":jid}
        if self.lost:
            self.lost=False
            raise TimeoutError("accepted job response lost")
        return {"job_id":jid,"response":jid}
    def matching(self,comment):
        return [jid for jid,r in self.values.items() if r["card"]["comment"]==comment]
    def verify(self,jid,card):
        assert self.values[jid]["card"]["comment"]==card["comment"]
        return {"fields":{"Reason":"JobHeldUser" if card["hold"] and jid not in self.releases else "Dependency"},"raw":"verified"}
    def release(self,jid):
        self.releases.append(jid)
    def observe(self,jid):
        r=self.values[jid]
        return {"job_id":jid,"state":r["state"],"start":r["start"],"elapsed_seconds":0,
            "terminal":r["terminal"],"billing_hours":0.,"exit_code":"0:0","cpus":1,"comment":r["card"]["comment"]}
    def queue(self):
        values=[{"job_id":jid,"partition":r["card"]["partition"],"state":r["state"],"comment":r["card"]["comment"]}
            for jid,r in self.values.items() if not r["terminal"] and not r["card"].get("array_cards")]
        return values+[{"job_id":str(9000+i),"partition":"lrd_all_serial","state":"PENDING","comment":"other"} for i in range(self.capacity)]
    def snapshot(self):
        return {"remaining_local_hours":42050.,"consumed_local_hours":25950.,
            "storage_headroom_bytes":100*1024**3,"jobs":self.queue()}
    def command(self,args):
        if args[0]=="scancel":
            self.values[args[1]].update(state="CANCELLED",terminal=True)
            return ""
        if args[:3]==["scontrol","show","job"]:
            return "Account=other TRES=billing=0 TimeLimit=01:00:00"
        raise AssertionError(args)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="campaign-tests-",dir=C.REPO/"report")
        self.root=Path(self.temp.name)
        self.campaign=C.seal({"root":str(self.root),"campaign_id":self.root.name,"code_files":{},
            "authorization_baseline_consumed":25950.,"models":{m:{"layers":[0,1,2],"experts":8} for m in C.MODELS}})
    def tearDown(self):
        self.temp.cleanup()
    def test_execution_alias_and_prefix_allowance(self):
        row=dict(zip(C.EXECUTION_FIELDS,["gpt","profile",{"prefix_tokens":10},1,"run-000",None,{"bias":-2},90,"history"]))
        self.assertEqual(C.execution_id(row),C.execution_id({**row,"window":100,"threshold_index":8}))
        self.assertNotEqual(C.execution_id(row),C.execution_id({**row,"replicate":"run-001"}))
        self.assertEqual(remaining_allowance({"max_model_len":100,"max_tokens":80},10,60),20)
        self.assertEqual(remaining_allowance({"max_model_len":100,"max_tokens":None},10,60),30)
    def test_atomic_interrupted_commit_and_duplicate_units(self):
        path=self.root/"atomic.json"
        with self.assertRaises(RuntimeError):
            with atomic_file(path) as f:
                f.write(b"unfinished")
                raise RuntimeError("interrupted")
        self.assertFalse(path.exists())
        C.publish(path,{"value":1});C.publish(path,{"value":1})
        with self.assertRaises(ValueError):C.publish(path,{"value":2})
        row={"uid":C.digest("unit"),"source":"original"}
        units=C.Units(self.root/"units")
        self.assertTrue(units.begin(row,"10"));self.assertFalse(units.begin(row,"11"))
        self.assertEqual(units.state(row["uid"])["status"],"started")
        units.commit(row,{"status":"execution_failed","redraw":False})
        self.assertFalse(units.begin(row,"12"))
    def test_lost_submission_and_ambiguous_response(self):
        scheduler=FakeScheduler();scheduler.lost=True
        value=S.card(self.campaign,"measure","gpt")
        with self.assertRaises(TimeoutError):S.submit_card(self.campaign,value,scheduler)
        accepted=S.submit_card(self.campaign,value,scheduler)
        self.assertEqual(accepted["job_id"],"101");self.assertEqual(len(scheduler.values),1)
        self.assertEqual(S.submit_card(self.campaign,value,scheduler),accepted)
        other=S.card(self.campaign,"measure","gpt",shard=1)
        C.publish(self.root/"submissions"/other["sha256"]/"INTENT.json",{"comment":other["comment"]})
        with self.assertRaises(ValueError):S.submit_card(self.campaign,other,scheduler)
        self.assertEqual(len(scheduler.values),1)
    def test_protected_handoff_and_partial_restart(self):
        scheduler=FakeScheduler()
        old=S.card(self.campaign,"controller",generation=0,hold=True,minutes=60)
        S.submit_card(self.campaign,old,scheduler)
        handoff=S.protected_controller(self.campaign,scheduler,generation=0)
        self.assertEqual(len(scheduler.values),3)
        self.assertEqual(scheduler.releases,["103","102","101"])
        self.assertEqual(len(handoff["guards"]),2)
        again=S.protected_controller(self.campaign,scheduler,generation=0,wait_ids=["500","600"],begin="now+5minutes")
        self.assertEqual(again,handoff);self.assertEqual(len(scheduler.values),3)
    def test_array_mapping_and_serial_capacity(self):
        scheduler=FakeScheduler()
        cards=[S.card(self.campaign,"coordination","gpt",i,minutes=180) for i in range(1,4)]
        parent=S.array_card(cards)
        receipt=S.submit_card(self.campaign,parent,scheduler)
        observed=O.jobs(self.campaign,scheduler)
        self.assertEqual([j["card"]["shard_index"] for j in observed],[1,2,3])
        self.assertEqual([j["accepted"]["job_id"] for j in observed],["101_0","101_1","101_2"])
        self.assertEqual(parent["array_concurrency"],1)
        self.assertEqual(len(scheduler.queue()),3)
    def test_one_infrastructure_replacement(self):
        a={"terminal":True,"state":"NODE_FAIL","elapsed_seconds":0,"start":"None"}
        self.assertTrue(S.never_started(a,False,0))
        self.assertFalse(S.never_started(a,False,1))
        self.assertFalse(S.never_started(a,True,0))
        self.assertFalse(S.never_started({**a,"start":"2026-10-08T10:00:00"},False,0))
    def test_budget_storage_and_reservations(self):
        snapshot=FakeScheduler().snapshot()
        args=dict(baseline_consumed=25950.,campaign_actual=0,outstanding=0,residual=1000,output_bytes=1024)
        self.assertTrue(S.admission(snapshot,**args)["admitted"])
        self.assertFalse(S.admission(snapshot,**{**args,"residual":40000})["admitted"])
        self.assertFalse(S.admission(snapshot,**{**args,"output_bytes":95*1024**3})["admitted"])
        self.assertFalse(S.admission(snapshot,**{**args,"outstanding":42000})["admitted"])
        self.assertFalse(S.admission(snapshot,**args,prices_complete=False)["admitted"])
    def test_fold_isolation_weights_and_outcome_blind_sampling(self):
        rows=[{"benchmark":b,"family":f"f{i}","question":f"{b}|{i}","correct":bool(i%2)} for b in ("a","b") for i in range(20)]
        folds={f"f{i}":i%5 for i in range(20)}
        chosen=D.cohorts(rows,folds,4)
        opposite=D.cohorts([{**r,"correct":not r["correct"]} for r in rows],folds,4)
        self.assertEqual([(r["benchmark"],r["question"]) for r in chosen],[(r["benchmark"],r["question"]) for r in opposite])
        self.assertEqual(len(chosen),8);self.assertTrue(all(r["fold"]==folds[r["family"]] for r in chosen))
        repeats=chosen+[chosen[0]]*5
        weights=D.family_weights(repeats)
        self.assertAlmostEqual(weights.sum(),1)
        self.assertAlmostEqual(sum(w for w,r in zip(weights,repeats) if r["benchmark"]=="a"),.5)
        counts=np.ones((40,1,3,2));tokens=np.ones((40,1));train=np.array([folds[r["family"]]!=0 for r in rows])
        a,_=D.inclusion_statistics(counts,tokens,rows,train)
        counts[~train]=9999
        b,_=D.inclusion_statistics(counts,tokens,rows,train)
        np.testing.assert_array_equal(a,b)
    def test_central_plus_one_and_missing_prefix(self):
        folds={f"f{i}":0 for i in range(72)}
        selected=[{"benchmark":"b","family":f,"question":f} for f in folds]
        sentences=[{"benchmark":"b","family":f,"attempt_id":f,"sentence_index":j,"progress":(j+.5)/3} for f in folds for j in range(3)]
        roster=D.prefix_roster(sentences,selected,folds,"gpt")
        self.assertTrue(all(r["configuration_indices"][0]==0 and r["configuration_indices"][1] in range(1,9) for r in roster))
        for third in range(3):
            self.assertEqual(sorted(r["configuration_indices"][1] for r in roster if r["progress_third"]==third),[x for x in range(1,9) for _ in range(3)])
        missing=D.prefix_roster([],selected,folds,"gpt")
        self.assertTrue(all(r["prefix"] is None for r in missing))
    def test_expert_extents_random_strata_and_unequal_storage(self):
        self.assertEqual(D.enriched_set([4,3,2,1,-1],.25),[0])
        self.assertEqual(D.enriched_set([4,3,2,1,-1],.5),[0,1])
        self.assertEqual(D.enriched_set([4,3,2,1,-1],.75),[0,1,2])
        ids=[1,5,9];frequency=np.arange(20)
        control=D.random_control(ids,frequency,"identity")
        self.assertEqual(sorted(i//4 for i in ids),sorted(i//4 for i in control))
        score=np.array([np.arange(8),np.arange(8)*10.])
        mask,size=D.masks(score,100,.5,2,unequal=True,layers=[2,6])
        self.assertEqual(size,800);self.assertGreater(len(mask["6"]),len(mask["2"]))
        from moe_exp.research_campaign_v3_r6.loader import gpt_loader_count
        self.assertEqual(gpt_loader_count("layers.2.mlp.experts.w13_weight",mask,8),len(mask["2"]))
        self.assertEqual(gpt_loader_count("layers.6.mlp.experts.w2_scale",mask,8),len(mask["6"]))
    def test_bootstrap_pairing_missing_and_degeneracy(self):
        rows=[{"benchmark":b,"family":f"f{i}","a":float(i%2),"b":float(i%2),"diff":0.,"missing":None} for b in ("a","b") for i in range(20)]
        panel=bootstrap_panel(rows,["a","b","diff","missing"],panel="test:paired",draws=1000)
        self.assertEqual(panel["results"]["a"],panel["results"]["b"])
        self.assertIsNone(panel["results"]["diff"]["interval"])
        self.assertIsNone(panel["results"]["missing"]["estimate"])
        j,p,q=joint_tables([[0,1],[0,2]],[[1,2],[0,1]],3,3)
        self.assertEqual(j[0,1],1.);self.assertTrue(np.isnan(log_odds_ratio(j,p,q)[0,1]))
    def test_annotation_context_and_no_gap_bridging(self):
        from moe_exp.research_campaign_v3_r6.annotation import context_id
        annotator={"config":{"program_sha256":"program","base_url":"localhost","source_plan_sha256":"a"},"revision":"reader"}
        inputs={"problem_statement":"p","previous_sentence":"a","sentence":"b","next_sentence":"c"}
        a=context_id(inputs,annotator)
        self.assertNotEqual(a,context_id({**inputs,"next_sentence":"d"},annotator))
        self.assertEqual(a,context_id(inputs,{**annotator,"config":{**annotator["config"],"base_url":"elsewhere"}}))
        r=[{"attempt_id":"a","sentence_index":i,"segment":0,"class":c,"n_tokens":3} for i,c in ((0,"Read"),(1,"Analyze"),(3,"Read"))]
        dynamics=trajectories(r)
        self.assertEqual(len(dynamics["transitions"]),1);self.assertEqual(dynamics["reentry"],[])


class SignedRuntime(unittest.TestCase):
    def test_sign_sets_pulse_closure_and_selector_fidelity(self):
        import torch
        class Router:
            top_k=2;global_num_experts=4;eplb_state=None
            def _compute_routing(self,router_logits):
                corrected=router_logits+torch.tensor([.01,.02,.03,.04])
                ids=corrected.topk(2,dim=-1).indices
                weights=torch.softmax(router_logits,dim=-1).gather(-1,ids)*1.7
                return weights,ids
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.router=Router()
            def named_modules(self,*args,**kwargs):
                yield "layers.3.mlp.experts",SimpleNamespace(router=self.router,routed_experts=None)
            def forward(self,input_ids,positions):
                return self.router._compute_routing(torch.tensor([[3.,2.,1.,0.]]*len(input_ids)))
        for sign in (-1,1):
            model=Model();original=model.router._compute_routing
            action={"layer":3,"experts":[0,1] if sign<0 else [2,3],"bias":10.*sign,"start":2,"pulse_tokens":3}
            contract={"expected_routed_layers":1,"native_k":2,"experts":4}
            capture=SignedCapture(model,contract,[9],[8,7],[[6]],action,[3],8,audit=True)
            model(torch.tensor([9,8,7]),torch.tensor([0,1,2]))
            model(torch.tensor([5]),torch.tensor([3]))
            model(torch.tensor([5]),torch.tensor([4]))
            model(torch.tensor([5]),torch.tensor([5]))
            model(torch.tensor([6]),torch.tensor([6]))
            result=capture.finish();rows=result["capture"]["3"]
            self.assertEqual(rows["active"],[True,True,True,False,False])
            self.assertEqual(set(rows["executed_ids"][0]),{2,3})
            self.assertEqual(set(rows["executed_ids"][3]),{0,1})
            self.assertTrue(result["closed"]);self.assertTrue(result["parity"][3]["noop_equal"])
            self.assertEqual(model.router._compute_routing,original)
    def test_prefix_presence_frequency_repetition_history(self):
        from moe_exp.research_campaign_v3_r6.penalties import validate
        params=SimpleNamespace(extra_args={"research_prefix_penalties":{"prefix":[2,2,3],"presence":1.5,"frequency":.5,"repetition":1.}},presence_penalty=0,frequency_penalty=0,repetition_penalty=1)
        self.assertEqual(validate(params)["prefix"],[2,2,3])
        params.presence_penalty=1.5
        with self.assertRaises(ValueError):validate(params)
        import torch
        # vLLM's public penalty implementation uses Triton on CUDA. Test the
        # complete cumulative history and operator ordering on a CPU oracle.
        from moe_exp.research_campaign_v3_r6.penalties import processor_class
        from moe_exp.research_campaign_v3_r6.runtime import LOGITS_PROCESSORS
        from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns
        # The public entry point and Slurm wrapper both add this preserved
        # module directory explicitly; reproduce that qualified import path.
        import sys
        with patch.object(sys,"path",[str(C.REPO/"scripts/experimental_resume"),*sys.path]):
            self.assertEqual(len(_load_logitsprocs_by_fqcns(LOGITS_PROCESSORS)),2)
        def oracle(logits,prompt,presence,frequency,repetition,output):
            for i,hist in enumerate(output):
                for token in set(hist):
                    logits[i,token]-=presence[i]+frequency[i]*hist.count(token)
            return logits
        with patch("vllm.v1.sample.ops.penalties.apply_all_penalties",oracle):
            cls=processor_class();processor=cls(None,None,None)
            live=[3]
            processor.rows[0]=({"prefix":[2,2,3],"presence":1.5,"frequency":.5,"repetition":1.},[9,2,2,3],live)
            logits=processor.apply(torch.zeros((1,12)))
            self.assertEqual(float(logits[0,2]),-2.5);self.assertEqual(float(logits[0,3]),-2.5)
            self.assertEqual(float(logits[0,9]),0.)


class EndToEnd(Contracts):
    def test_synthetic_campaign_to_terminal_with_one_failed_model(self):
        scheduler=FakeScheduler()
        campaign=C.seal({**self.campaign,"results_first_phases":True,"models":{m:{**v,"expert_bytes":1} for m,v in self.campaign["models"].items()}})
        C.publish(self.root/"FOLDS.json",{"folds":{"family":0}})
        for model in C.MODELS:
            uid=C.digest(["synthetic",model]);source={"attempt_id":"a","family":"family","benchmark":"math500","completion_tokens":1}
            C.publish(self.root/model/"MEASURE_TASKS.json",{"records":[{"uid":C.digest(["measurement",model]),"task_index":0,"source":source}]})
            C.publish(self.root/model/"MASKS.json",{"records":[]})
            row={"uid":uid,"task_index":0,"source":{"prefix_tokens":1},"mask":None,"action":{"bias":1},"cap":1,"window":1,"phase":"central","analysis_phases":["central"]}
            records=[row] if model!="gpt" else [{**row,"uid":C.digest(["synthetic",model,i]),"task_index":i} for i in range(3)]
            C.publish(self.root/model/"TASKS.json",{"records":records,"cells":[]})
        C.publish(self.root/"ANNOTATOR_PRICE.json",{"status":"complete","seconds_per_context_upper":1.,"bytes_per_context_upper":1})
        def perform(value):
            stage,model,shard=value["stage"],value["model"],value["shard_index"]
            directory=self.root/"stages"/O.stage_key(stage,model,shard)
            result={"stage":stage,"model":model,"shard_index":shard,"status":"complete"}
            if stage=="derive" and model=="glm":result.update(status="execution_failed",failure_scope="model")
            if stage=="measure":
                r=C.load(self.root/model/"MEASURE_TASKS.json")["records"][0]
                u=C.Units(self.root/model/"measurements");u.begin(r,"fake");u.commit(r,{"status":"complete","seconds":1.,"load_seconds":1.,"artifact":{"bytes":1}})
            if stage=="price":
                C.publish(self.root/model/"PRICES.json",{"status":"complete","records":[{"role":"causal","seconds":1.,"load_seconds":1.,"generated_tokens":1,"output_bytes":1,"scoring_seconds":1.,"context_seconds":1.,"context_executions":1}]})
            if stage=="qualify" and shard==0:
                C.publish(self.root/model/"QUALIFICATION.json",{"status":"complete","failed_shape_keys":[],"cases":[]})
            if stage=="contexts":
                from moe_exp.research_campaign_v3_r6.phases import shard_phase
                C.publish(self.root/model/("ANNOTATION_TASKS-"+shard_phase(stage,shard)+".json"),{"records":[]})
            if stage=="execute":
                records=C.load(self.root/model/"TASKS.json")["records"]
                shards=C.load(self.root/model/"EXECUTION_SHARDS.json")["shards"]
                indices=shards[shard] if shard<len(shards) else C.load(self.root/model/"execution_fragments"/f"{shard:06d}.json")["indices"]
                for i in indices:
                    row=records[i]
                    units=C.Units(self.root/model/"executions")
                    if units.begin(row,"fake"):
                        failed=model=="gpt" and shard==0
                        units.commit(row,{"status":"execution_failed" if failed else "complete","generated_tokens":1})
                        if failed:
                            result["status"]="execution_failed";break
            if stage=="closeout":
                C.publish(self.root/"REPORT_COMPLETE.json",{"complete":True})
                C.publish(self.root/"SCIENCE_COMPLETE.json",{"complete":False,"failed_model":"glm"})
                C.publish(self.root/"PAPER_DELIVERY.json",{"complete":True})
            C.publish(directory/"RESULT.json",result)
        initial=O.launch(campaign,scheduler)
        self.assertEqual(len(initial["initial_jobs"]),5)
        for generation in range(100):
            for jid,r in list(scheduler.values.items()):
                if r["terminal"] or r["card"].get("array_cards") or r["card"]["stage"] in ("controller","snapshot"):continue
                perform(r["card"]);r.update(state="COMPLETED",terminal=True,start="2026-10-08")
            value=S.card(campaign,"controller",generation=generation)
            with patch.dict(os.environ,{"SLURM_JOB_ID":"999999"}),patch("moe_exp.research_campaign_v3_r6.observations.pair_tasks",return_value=[]):
                O.control(campaign,value,scheduler)
            # Complete old primary owners, leaving guards pending for retirement.
            for jid,r in scheduler.values.items():
                if r["card"]["stage"]=="controller" and r["card"]["shard_index"]==0 and r["card"]["generation"]<=generation:
                    r.update(state="COMPLETED",terminal=True,start="2026-10-08")
            if (self.root/"REPORT_COMPLETE.json").exists():break
        self.assertTrue((self.root/"REPORT_COMPLETE.json").exists())
        self.assertFalse(C.load(self.root/"SCIENCE_COMPLETE.json")["complete"])
        self.assertTrue(O._success(self.root,"price","gpt"))
        self.assertTrue(O._failure(self.root,"derive","glm"))
        original=C.load(self.root/"gpt/TASKS.json")["records"]
        units=C.Units(self.root/"gpt/executions")
        self.assertEqual(units.state(original[0]["uid"])["status"],"execution_failed")
        self.assertTrue(all(units.state(r["uid"])["status"]=="complete" for r in original[1:]))
        self.assertEqual([i for p in sorted((self.root/"gpt/execution_fragments").glob("*.json")) for i in C.load(p)["indices"]],[1,2])
        self.assertLess(generation,100)


if __name__=="__main__":unittest.main()
