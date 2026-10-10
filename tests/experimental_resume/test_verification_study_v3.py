"""Scientific boundary and lifecycle regressions; GPU fidelity is a Slurm gate."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from moe_exp.verification_study.contracts import digest, identity, private_path, seal, verify
from moe_exp.verification_study.protocol import assignments, family_boundary
from moe_exp.verification_study.registry import MODEL_KEYS, GEOMETRY
from moe_exp.verification_study.nomination import calibration, nominate, random_expert
from moe_exp.verification_study.adapters import PredictionClock
from moe_exp.verification_study.readers import normalize_prompt_ids
from moe_exp.verification_study.semantics import (candidate_proposals, eligible_proposal, reader_messages,
    parse_rating, outcome, audit_frame, qualify_recognizer, enroll_prefixes)
from moe_exp.verification_study.pricing import price_sheet, shard_budget
from moe_exp.verification_study.analysis import binary_bounds, analyze
from moe_exp.verification_study.policy import action_for
from moe_exp.verification_study.views import reasoning_limit
from moe_exp.verification_study.closeout import engagement, require_cpu
from moe_exp.verification_study.backend import Backend, validate_raw_action
from moe_exp.verification_study.contracts import require_generation_binding


def model(key="qwen36"):
    layers,experts,k=GEOMETRY[key]
    return seal({"schema":"verification-model-v3","key":key,"expected_routed_layers":layers,
                 "experts":experts,"native_k":k})


def boundary():
    return seal({"schema":"verification-families-v3","discovery":[f"d{i}" for i in range(48)],
                 "validation":[f"v{i}" for i in range(32)],"final":[f"f{i}" for i in range(96)]})


def policy(m):
    return seal({"schema":"verification-policy-v3","status":"PASS","model_sha256":m["sha256"],
                 "layer":2,"target":5,"target_bias":.9,"random_biases":{"7":.7,"8":.8}})


class ContractTests(unittest.TestCase):
    def test_invalid_client_action_is_rejected_before_import_or_rpc(self):
        from unittest.mock import Mock
        client=SimpleNamespace(contract=model(),llm=Mock())
        valid={"layer":2,"expert":3,"bias":1.}
        malformed=[{**valid,"expert":256},{**valid,"expert":-1},{**valid,"expert":True},
                   {**valid,"layer":False},{**valid,"layer":-1},{**valid,"bias":True},
                   {**valid,"bias":float("nan")},{**valid,"bias":float("inf")},
                   {**valid,"bias":-1.},{**valid,"start":0},[1,2,3]]
        for action in malformed:
            with self.assertRaises(ValueError):Backend.generate(client,{},"fixture",action=action)
        client.llm.collective_rpc.assert_not_called()
        validate_raw_action(model("nemotron"),{"layer":51,"expert":127,"bias":0.})
        validate_raw_action(model(),None)

    def test_generation_executor_binding_rejects_superseded_source(self):
        m=model();campaign={"runtime_sha256":"runtime","code_files":{"executor.py":"r4"}}
        receipt={"schema":"verification-generation-receipt-v3","model_sha256":m["sha256"],
                 "runtime_sha256":"runtime","executor_sha256":digest(campaign["code_files"])}
        require_generation_binding(receipt,campaign,m)
        for change in ({"executor_sha256":digest({"executor.py":"r3"})},{"runtime_sha256":"other"},
                       {"model_sha256":"other"}):
            with self.assertRaises(ValueError):require_generation_binding({**receipt,**change},campaign,m)

    def test_native_text_multi_axis_prefill_and_decode(self):
        for axes in (3,4):
            clock=PredictionClock([10,11],[],[[999]])
            contexts=clock.observe([10,11],[[0,1] for _ in range(axes)])
            self.assertEqual([r["output_position"] for r in contexts],[-1,0])
            self.assertEqual(clock.observe([21],[[2] for _ in range(axes)])[0]["output_position"],1)

    def test_nontext_position_axes_abstain_before_clock_mutation(self):
        for positions in ([[0,1],[0,2],[0,1]],[[0,1],[0,1]],[[0,1],[0],[0,1]],[[0,1],[0,1],[0,1.0]]):
            clock=PredictionClock([10,11],[],[[999]])
            with self.assertRaises(ValueError):clock.observe([10,11],positions)
            self.assertEqual(clock.last,-1)
            self.assertEqual(clock.contexts,[])

    def test_serial_allocation_requires_job_step_and_entitlement(self):
        with patch.dict("os.environ",{},clear=True):
            with self.assertRaises(ValueError):require_cpu()
        env={"SLURM_JOB_ID":"123","SLURM_STEP_ID":"0","SLURM_JOB_ACCOUNT":"iscrc_miosr","SLURM_JOB_PARTITION":"lrd_all_serial"}
        with patch.dict("os.environ",env,clear=True):require_cpu()
        env["SLURM_JOB_PARTITION"]="boost_usr_prod"
        with patch.dict("os.environ",env,clear=True):
            with self.assertRaises(ValueError):require_cpu()

    def test_observed_dose_aligns_expert_ids_and_excludes_inactive_rows(self):
        import numpy as np
        capture={"L2_active":np.array([True,False]),"L2_native_ids":np.array([[1,2],[1,2]]),
                 "L2_executed_ids":np.array([[1,3],[1,2]]),"L2_native_weights":np.array([[.6,.4],[.6,.4]]),
                 "L2_executed_weights":np.array([[.2,.8],[.6,.4]])}
        receipt={"action":{"layer":2,"expert":3,"bias":1.},"pulse_fired":True}
        with patch("moe_exp.verification_study.storage.capture_arrays",return_value=capture):
            result=engagement(receipt)
        self.assertEqual(result["active_rows"],1)
        self.assertEqual(result["inclusion_increment"],1.)
        self.assertAlmostEqual(result["relative_weight_displacement"],1.6)
        receipt["pulse_fired"]=False
        self.assertEqual(engagement(receipt)["status"],"NO_ACTIVE_PULSE")

    def test_seal_mutation_is_rejected(self):
        x=seal({"a":1});x["a"]=2
        with self.assertRaises(ValueError):verify(x)

    def test_native_layer_keys_survive_json_roundtrip(self):
        receipt=seal({"parity":{0:{"calls":1},2:{"calls":1},10:{"calls":1}}})
        self.assertEqual(verify(json.loads(json.dumps(receipt))),receipt)
        self.assertEqual(set(receipt["parity"]),{"0","2","10"})

    def test_write_scope(self):
        with self.assertRaises(ValueError):private_path("/tmp/science.json")
        with self.assertRaises(ValueError):private_path("/leonardo_work/IscrC_MIOSR/other/results.json")

    def test_family_seal_requires_disjoint176(self):
        old=seal({"new_parent_pools":{"parent_pools":{"discovery":boundary()["discovery"]}}})
        independent=seal({s:[{"family":f} for f in boundary()[s]] for s in ("validation","final")}|{"source_revision":"pinned"})
        self.assertEqual(len(family_boundary(old,independent)["final"]),96)
        independent["final"][0]["family"]="d0";independent=seal(independent)
        with self.assertRaises(ValueError):family_boundary(old,independent)

    def test_all7_scientific_rosters_and_caps(self):
        final=[{"family":f,"question_id":f"q{i}"} for i,f in enumerate(boundary()["final"])]
        total=0
        for key in MODEL_KEYS:
            m=model(key);p=policy(m)
            native=assignments(m,"final_native",final)
            utility=assignments(m,"utility",final,policy=p)
            self.assertEqual(native["assigned"],192);self.assertEqual(utility["assigned"],384)
            self.assertTrue(all(a["cap"]==16384 for a in utility["assignments"]))
            total+=native["assigned"]+utility["assigned"]
        self.assertEqual(total,4032)

    def test_native_seed1_never_enrolls(self):
        m=model();detector=seal({"status":"PASS","model_sha256":m["sha256"]})
        source=[{"uid":f,"family":f,"seed":0,"arm":"native","population":"final","execution_complete":True,
                 "qualified_opportunities":[],"original_prompt_ids_sha256":"p"} for f in boundary()["final"]]
        enrollment=enroll_prefixes(m,source,detector)
        self.assertEqual(enrollment["prefixes"],[])
        source[0]["seed"]=1
        with self.assertRaises(ValueError):enroll_prefixes(m,source,detector)

    def test_conditional_roster_is_enrolled_subset(self):
        m=model();p=policy(m)
        prefix=seal({"schema":"verification-prefix-enrollment-v3","model_sha256":m["sha256"],"source_seed":0,
            "frozen_before_treatment":True,"prefixes":[{"family":"f1"}]})
        rows=[{"family":f,"question_id":f} for f in boundary()["final"]]
        a=assignments(m,"conditional",rows,policy=p,prefixes=prefix)
        self.assertEqual(a["assigned"],6)
        self.assertTrue(all(r["family"]=="f1" and r["cap"]==1024 for r in a["assignments"]))

    def test_random_control_is_fixed_per_family_and_excludes_target(self):
        self.assertEqual(random_expert([3,4],family="f",model="gpt",target=1),random_expert([4,3],family="f",model="gpt",target=1))
        with self.assertRaises(ValueError):random_expert([1,3],family="f",model="gpt",target=1)

    def test_foreign_policy_cannot_dispatch(self):
        m=model();p=policy(m);a=assignments(m,"utility",[{"family":"f","question_id":"q"}],policy=p)["assignments"][0]
        self.assertEqual(action_for(p,a)["expert"],5)
        a["policy_sha256"]="wrong"
        with self.assertRaises(ValueError):action_for(p,a)


class CandidateTests(unittest.TestCase):
    problem="Find x with x*x=4 and x>0."

    def test_intermediate_symbolic_and_complete_forms(self):
        for text in ("We obtain $x=2$.\n",r"We obtain $x=\sqrt{4}$."+"\n","The answer is 2.\n",r"Thus \boxed{2}."+"\n"):
            self.assertIsNotNone(eligible_proposal(self.problem,text),text)

    def test_dangling_math_and_long_unit_abstain(self):
        for text in ("We obtain $x=2.\n","a"*1300+".\n"):
            self.assertIsNone(eligible_proposal(self.problem,text))

    def test_other_value_and_variable_not_checked(self):
        text="We obtain $x=1$.\nWe obtain $y=10$.\nChecking $y=10$, we find $y-10=0$.\n"
        rows=candidate_proposals(self.problem,text)
        self.assertFalse(next(r for r in rows if r["lhs"]=="x")["checked"])
        self.assertTrue(next(r for r in rows if r["lhs"]=="y")["checked"])

    def test_intent_is_not_verification(self):
        text="We obtain $x=2$.\nWe should check $x=2$ against $x^2=4$.\n"
        self.assertFalse(candidate_proposals(self.problem,text)[0]["checked"])

    def test_future_unfinished_unit_never_triggers(self):
        self.assertIsNone(eligible_proposal(self.problem,"We obtain $x=2$.\nWe might"))

    def test_reader_blinding_excludes_gold_arm_and_detector_vote(self):
        record={"problem":self.problem,"prefix_text":"We obtain x=2.","candidate":{"expression":"x=2",
            "originating_sentence":"We obtain x=2.","checked":False,"kind":"complete"},"gold":"2","arm":"target"}
        messages=reader_messages(record,role="eligibility")
        visible=json.loads(messages[1]["content"])
        self.assertEqual(set(visible),{"problem","prefix_text","candidate"})
        self.assertNotIn("checked",visible["candidate"])

    def test_impossible_eligibility_is_invalid(self):
        text=json.dumps({"candidate_valid":False,"constraint_bound":True,"already_checked":False,"eligible":True})
        self.assertIsNone(parse_rating(text,"eligibility"))

    def test_reader_disagreement_and_wrong_input_stay_unknown(self):
        record={"horizon":1024,"arm_blinded":True,"execution_complete":True,"window_text":"Checking x=2 gives4."}
        ratings=[{"reader":i,"finish_reason":"stop","input_sha256":digest(record),
                  "text":json.dumps({"substantive_verification":bool(i),"evidence":"Checking x=2 gives4." if i else None})} for i in (0,1)]
        self.assertIsNone(outcome(record,ratings)["success"])
        ratings[0]["input_sha256"]="wrong"
        self.assertIsNone(outcome(record,ratings)["success"])

    def test_no_event_fully_observed_is_false_not_missing(self):
        record={"horizon":1024,"arm_blinded":True,"execution_complete":True,"window_text":"No check."}
        ratings=[{"reader":i,"finish_reason":"stop","input_sha256":digest(record),
                  "text":'{"substantive_verification":false,"evidence":null}'} for i in (0,1)]
        self.assertIs(outcome(record,ratings)["success"],False)
        record["execution_complete"]=False
        for r in ratings:r["input_sha256"]=digest(record)
        self.assertIsNone(outcome(record,ratings)["success"])

    def test_audit_enrolls_families_before_repeated_prefixes(self):
        rows=[{"uid":identity(f,i),"family":f,"prefix_only":True,"population":"discovery","stratum":"checked"}
              for f in boundary()["discovery"] for i in range(20)]
        frame=audit_frame(rows,boundary(),model())
        selected=[r for r in frame["records"] if r["split"]=="apparent"]
        self.assertEqual(len({r["family"] for r in selected}),12)

    def test_stale_ratings_cannot_qualify(self):
        row={"uid":"u","family":"d0","stratum":"proposal","split":"apparent","proposed":{"eligible":True,"already_checked":False}}
        frame=seal({"schema":"verification-semantic-frame-v3","records":[row],"model_sha256":model()["sha256"]})
        rating=seal({"schema":"verification-reader-receipt-v3","uid":"u","reader":0,"role":"eligibility",
                     "frame_sha256":frame["sha256"],"input_sha256":"stale","checkpoint":"pinned"})
        with self.assertRaises(ValueError):qualify_recognizer(frame,[rating])


class TimingAndInferenceTests(unittest.TestCase):
    def test_late_router_validation_does_not_leave_earlier_patch(self):
        from moe_exp.verification_study.adapters import NativeCapture
        class Router:
            top_k=2;global_num_experts=4;eplb_state=None
            def _compute_routing(self,hidden_states,router_logits,indices_type,*,input_ids=None):return None,None
        class BadRouter(Router):
            def _compute_routing(self,hidden_states,wrong_name,indices_type):return None,None
        routers=[Router(),BadRouter()]
        modules={f"model.layers.{i}.mlp.experts":SimpleNamespace(router=r,routed_experts=SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=False))) for i,r in enumerate(routers)}
        root=SimpleNamespace(named_modules=lambda:modules.items(),get_submodule=lambda name:modules[name])
        contract={"expected_routed_layers":2,"native_k":2,"experts":4}
        before=routers[0]._compute_routing
        with self.assertRaises(ValueError):NativeCapture(root,contract,[1],[],[[2]],capture_all=True)
        self.assertEqual(routers[0]._compute_routing,before)

    def test_hook_registration_failure_restores_every_router(self):
        from moe_exp.verification_study.adapters import NativeCapture
        class Router:
            top_k=2;global_num_experts=4;eplb_state=None
            def _compute_routing(self,hidden_states,router_logits,indices_type,*,input_ids=None):return None,None
        router=Router();name="model.layers.0.mlp.experts"
        module=SimpleNamespace(router=router,routed_experts=SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=False)))
        def fail(*args,**kwargs):raise RuntimeError("fixture hook registration failure")
        root=SimpleNamespace(named_modules=lambda:[(name,module)],get_submodule=lambda key:module,register_forward_pre_hook=fail)
        before=router._compute_routing
        with self.assertRaises(RuntimeError):NativeCapture(root,{"expected_routed_layers":1,"native_k":2,"experts":4},[1],[],[[2]],capture_all=True)
        self.assertEqual(router._compute_routing,before)

    def test_unregistered_stage_and_cap_are_rejected(self):
        from moe_exp.verification_study.protocol import validate_assignment
        m=model();a=assignments(m,"discovery",[{"family":"d0","question_id":"q"}])["assignments"][0]
        a["cap"]=1000
        with self.assertRaises(ValueError):validate_assignment(a)

    def test_prediction_rows_and_split_closure(self):
        clock=PredictionClock([11,12],[21,22],[[31,32]])
        rows=clock.observe([11,12,21,22],[0,1,2,3])
        self.assertEqual([r["output_position"] for r in rows],[-1,0,1,2])
        self.assertTrue(clock.observe([31],[4])[0]["reasoning"])
        self.assertFalse(clock.observe([32],[5])[0]["reasoning"])
        self.assertFalse(clock.observe([99],[6])[0]["reasoning"])

    def test_neighbor_and_frozen_prefix_change_rejected(self):
        clock=PredictionClock([11],[21],[[31]])
        with self.assertRaises(ValueError):clock.observe([12],[0])
        clock=PredictionClock([11],[21],[[31]])
        clock.observe([11],[0])
        with self.assertRaises(ValueError):clock.observe([21],[2])

    def test_reader_batchencoding_normalization(self):
        self.assertEqual(normalize_prompt_ids({"input_ids":[[1,2,3]],"attention_mask":[[1,1,1]]}),[1,2,3])
        with self.assertRaises(ValueError):normalize_prompt_ids([[1],[2]])

    def test_native_closure_limits_primary_window(self):
        self.assertEqual(reasoning_limit({"emitted_token_ids":[1,2,3,4,5],"closure_token_sequences":[[3,4]]}),2)

    def test_dose_fit_and_unreachable_dose(self):
        fit=calibration(lambda bias:min(.5,bias*.1),1.)
        self.assertEqual(fit["status"],"PASS")
        failed=calibration(lambda bias:0.,1.)
        self.assertEqual(failed["status"],"UNQUALIFIED_DOSE")
        self.assertLessEqual(len(failed["points"]),22)

    def test_nomination_never_uses_final_family(self):
        row={"family":"f0","population":"final","information_boundary":"past_only","matched":True,
             "condition":"positive","inclusion_rate":.5,"layer":1,"expert":1}
        with self.assertRaises(ValueError):nominate(model(),[row],boundary())

    def test_unpriced_roles_cannot_get_zero_price(self):
        prices=price_sheet(model(),[],{"code":"hash"})
        with self.assertRaises(ValueError):shard_budget(prices,{"generation":1})

    def test_missing_binary_seed_has_bounds(self):
        self.assertEqual(binary_bounds([True,None]),[.5,1.])
        self.assertEqual(binary_bounds([False,None]),[0.,.5])

    def test_enrolled_family_with_both_missing_arms_is_retained(self):
        from moe_exp.verification_study.analysis import family_values
        values=family_values([],"gpt","semantic","target","native",boundary()["final"],enrolled=["f0"])
        self.assertEqual(values[0].tolist(),[-1.,1.])

    def test_enrolled_family_with_one_missing_arm_gets_bounds(self):
        from moe_exp.verification_study.analysis import family_values
        rows=[{"model":"gpt","experiment":"conditional","family":"f0","seed":i,"arm":"target","semantic":True} for i in (0,1)]
        values=family_values(rows,"gpt","semantic","target","native",boundary()["final"],enrolled=["f0"])
        self.assertEqual(values[0].tolist(),[0.,1.])

    def test_all42_contrasts_keep_missing_models_and_degenerate_uncertainty(self):
        rows=[{"model":"gpt","experiment":"utility","family":f,"seed":seed,"arm":arm,
               "accuracy":True,"total_output_tokens":100,"controller_method":"cheap"}
              for f in boundary()["final"] for seed in (0,1) for arm in ("native","target","matched_random")]
        result=analyze(rows,boundary()["final"],draws=50)
        self.assertEqual(len(result["contrasts"]),42)
        gpt=next(r for r in result["contrasts"] if r["model"]=="gpt" and r["endpoint"]=="accuracy")
        self.assertEqual(gpt["identified_mean_bounds"],[0.,0.])
        self.assertIsNone(gpt["simultaneous_interval95"])
        self.assertEqual(next(r for r in result["contrasts"] if r["model"]=="glm")["status"],"NOT_MEASURED")


if __name__=="__main__":
    unittest.main()
