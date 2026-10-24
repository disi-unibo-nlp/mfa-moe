"""Small CPU-only oracles for study contracts, calibration, causal state and inference."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from moe_exp.routing_study.common import (BENCHMARKS, CHECKPOINT, ENGINE, SAMPLING, digest,
    seal, verify, problem_key, code_hashes)
from moe_exp.routing_study.prepare import split_problems, validate_split
from moe_exp.routing_study.policy import freeze, validate, static_grid
from moe_exp.routing_study.calibration import npmi, fit_steer, fit_rice, fit_identity
from moe_exp.routing_study.temporal import Gate, Features, fires
from moe_exp.routing_study.runtime import TokenHistory, LayerHistory
from moe_exp.routing_study.analysis import bh, nondominated, select_frontier, paired_bootstrap
from moe_exp.routing_study.execution import matching_baseline, planned_rows, repair_interrupted_tail


def policy(action="native", strength=0., **kw):
    return freeze(action,strength,code={},**kw)


def population(n=15):
    return [dict(dataset=b,problem_id=str(i),prompt=f"{b} question {i}")
            for b in BENCHMARKS for i in range(n)]


def test_split_rounding_reproducibility_and_forced_membership():
    problems=population(11)
    forced=[problem_key(p) for p in problems if p["problem_id"]=="0"]
    split=split_problems(problems,pilot=forced[:3],fixtures=forced[3:])
    assert split==split_problems(list(reversed(problems)),pilot=forced[:3],fixtures=forced[3:])
    validate_split(split)
    for b in BENCHMARKS:
        assert split["counts"][b]["group_counts"]=={"calibration":6,"tuning":2,"test":3}
    assert all(split["membership"][k]=="calibration" for k in forced)


def test_duplicate_questions_and_attempts_cannot_leak():
    problems=population()
    problems[1]["prompt"]=problems[0]["prompt"]+"  "
    a,b=problem_key(problems[0]),problem_key(problems[1])
    split=split_problems(problems,pilot=[a])
    assert split["membership"][b]=="calibration"
    damaged=copy.deepcopy(split)
    damaged.pop("sha256")
    damaged["membership"][b]="test"
    with pytest.raises(ValueError,match="leakage"):
        validate_split(seal(damaged))
    with pytest.raises(ValueError):
        split_problems(problems+[problems[0]])


def test_cross_benchmark_duplicate_is_calibration():
    problems=population()
    problems[15]["prompt"]=problems[0]["prompt"]
    split=split_problems(problems)
    assert split["membership"][problem_key(problems[15])]=="calibration"
    assert split["membership"][problem_key(problems[0])]=="calibration"


def test_split_rejects_infeasible_constraints_and_missing_benchmark():
    problems=population(5)
    with pytest.raises(ValueError,match="exceed"):
        split_problems(problems,pilot=[problem_key(p) for p in problems[:4]])
    with pytest.raises(ValueError):
        split_problems(problems[:5])


@pytest.mark.parametrize("targets", [[0,1], [[0,0],[0,0]], [[40,0],[1,0]], [[0,256],[1,0]]])
def test_policy_never_accepts_pooled_or_unsupported_targets(targets):
    with pytest.raises((ValueError,TypeError)):
        policy("rice",2,targets=targets,parameters={"renormalize":True},calibration={"a":1})


def test_unavailable_policy_cannot_execute_and_hash_cannot_change():
    p=policy("rice",2,unavailable="No marker")
    with pytest.raises(ValueError,match="Unavailable"):
        validate(p)
    p["strength"]=4
    with pytest.raises(ValueError,match="hash"):
        validate(p,executable=False)


def test_code_and_checkpoint_are_frozen():
    p=policy()
    with pytest.raises(ValueError,match="code"):
        validate(p,current_code=True)
    p.pop("sha256")
    p["checkpoint"]={**CHECKPOINT,"revision":"nf4"}
    with pytest.raises(ValueError,match="FP8"):
        validate(seal(p))
    assert "moe_exp/routing_study/runtime.py" in code_hashes()


def test_npmi_exact_counts_and_absent_markers():
    assert npmi(10,10,10,100)==pytest.approx(1)
    assert npmi(10,20,50,100)==pytest.approx(0)
    assert npmi(0,20,50,100)==-1
    with pytest.raises(ValueError):
        npmi(0,0,50,100)
    with pytest.raises(ValueError):
        npmi(21,20,50,100)


def samples():
    result=[]
    for correct in (False,True):
        for i in range(10):
            rate=np.zeros(256)
            rate[:8]=10
            rate[0]=20 if correct else 10
            rate[1]=10 if correct else 20
            row=dict(id=f"{correct}-{i}",problem_key=str(i),is_correct=correct,layers={},
                     marker_occurrences={"<think>":1,"</think>":int(i!=0),"Alternatively":2})
            for l in range(40):
                count=np.ones(256)*20
                marker={"<think>":[0]*256,"</think>":[0]*256,"Alternatively":[0]*256}
                marker["<think>"][0]=2 if l==0 else 1
                marker["<think>"][1]=2 if l==1 else 1
                row["layers"][str(l)]=dict(reasoning_counts=rate.tolist(),reasoning_tokens=100,
                    all_counts=count.tolist(),all_tokens=1000,
                    marker_tokens={"<think>":2,"</think>":1,"Alternatively":2},marker_experts=marker)
            result.append(row)
    return result


def test_steer_support_and_within_question_problem_balance():
    data=samples()
    result=fit_steer(data)
    assert result["promote"]["targets"]==[[0,0],[1,0]]
    assert result["suppress"]["targets"]==[[0,1],[1,1]]
    assert result["contrasts"][0]["within_question_problems"]==10
    assert result["contrasts"][0]["difference"]==pytest.approx(.1)
    # Repeating one identical attempt must not increase question weight.
    repeated=fit_steer(data+[copy.deepcopy(data[0]) for _ in range(20)])
    for left,right in zip(repeated["contrasts"],result["contrasts"]):
        assert left["difference"]==pytest.approx(right["difference"],abs=1e-14)
        assert left["within_question_difference"]==pytest.approx(right["within_question_difference"],abs=1e-14)
        assert left["support"]==right["support"]
    assert "unavailable" in fit_steer(data[1:])["promote"]


def test_rice_signed_marker_formula_and_missing_markers():
    data=samples()
    result=fit_rice(data)
    assert result["targets"]==[[0,0],[1,1]]
    assert result["actual_occurrences"]["</think>"]==18
    for s in data:
        for info in s["layers"].values():
            info["marker_tokens"]["Alternatively"]=0
            info["marker_experts"]["Alternatively"]=[0]*256
    assert "unavailable" in fit_rice(data)


def test_identity_problem_balance_and_eight_target_limit():
    data=samples()
    result=fit_identity(data)
    assert result["policy"]["layers"]["0"]["scores"][0]==1
    assert sum(v>0 for v in result["policy"]["layers"]["0"]["scores"])<=8
    repeated=fit_identity(data+[copy.deepcopy(data[0]) for _ in range(20)])
    assert repeated["policy"]["layers"]["0"]["scores"]==pytest.approx(result["policy"]["layers"]["0"]["scores"])


def test_token_markers_align_actual_occurrences_and_close():
    pieces={0:"prompt ",1:"<think>",2:"abc",3:"</th",4:"ink>",5:"Alternatively",6:"answer"}
    history=TokenHistory([0,1],pieces.__getitem__)
    seen=[history.observe(i,t) for i,t in enumerate([0,1,2,5,3,4,6])]
    assert [s["reasoning"] for s in seen]==[None,None,0,1,2,None,None]
    assert history.occurrences=={"<think>":1,"</think>":1,"Alternatively":1}
    assert history.observe(2,2)==seen[2]
    with pytest.raises(ValueError,match="changed"):
        history.observe(2,5)
    with pytest.raises(ValueError,match="Noncontiguous"):
        history.observe(10,2)


def test_capped_trace_never_synthesizes_closing_marker():
    history=TokenHistory([1],{1:"<think>",2:"x"}.__getitem__)
    history.observe(0,1)
    history.observe(1,2)
    assert history.occurrences["</think>"]==0
    assert history.reasoning_count==1


def threshold():
    return {"layers":{"0":{"overlap_q75":.5,"jsd_q25":.1,"turnover_q75":.4,"gap_q10":.1,"weak_q75":.5,
        "controls":{"random_persistent":{"probability":1.,"profile":{}},
                    "position_persistent":{"probability":1.,"profile":{"4":1.,"6":1.}}}}}}


def window():
    return dict(overlap_lag16=.9,mixture_jsd=.01,set_turnover=.8,boundary_gaps=[0.]*256)


@pytest.mark.parametrize("kind",["persistent","unstable","random_persistent","position_persistent"])
def test_gate_preceding_history_next64_cooldown64(kind):
    gate=Gate("question",0,kind,threshold())
    observed=[gate.decide(i,window()) for i in range(512)]
    assert not any(observed[:256])
    assert all(observed[256:320])
    assert not any(observed[320:384])
    assert all(observed[384:448])
    # Recomputed positions retain their original decisions after later history.
    assert gate.decide(300,None)
    assert not gate.decide(350,window())


def test_gate_strict_percentiles_and_native_when_history_missing():
    w=window()
    w["overlap_lag16"]=.5
    assert not fires("persistent",w,threshold()["layers"]["0"])
    assert not fires("unstable",None,threshold()["layers"]["0"])


def test_streaming_matches_independent_batch_and_no_future_effect():
    from moe_exp.routing_study.reducers import jsd
    rng=np.random.default_rng(42)
    ids=np.array([rng.choice(256,8,replace=False) for _ in range(384)])
    weights=rng.uniform(.01,1,(384,8))
    scores=np.zeros((384,256))
    np.put_along_axis(scores,ids,1.,axis=1)
    features=Features(0)
    windows=[]
    for i in range(384):
        windows.extend(features.push(ids[i],weights[i],scores[i]))
        if i==255:
            first=copy.deepcopy(features.last)
    assert windows[0]==first
    mixture=np.zeros((256,256))
    np.put_along_axis(mixture,ids[:256],weights[:256]/weights[:256].sum(1,keepdims=True),axis=1)
    assert first["mixture_jsd"]==pytest.approx(jsd(mixture[:128].mean(0),mixture[128:].mean(0)))
    overlap=np.mean([len(set(ids[i])&set(ids[i-16]))/8 for i in range(16,256)])
    assert first["overlap_lag16"]==pytest.approx(overlap)
    assert features.reducer.retained_tokens<=272


def test_request_interleaving_recompute_and_distinct_native_weights():
    ids=np.arange(8,dtype=np.int32)
    weights=np.arange(1,9,dtype=float)  # Deliberately not router softmax affinities.
    scores=np.r_[np.ones(8),np.zeros(248)]
    states={key:LayerHistory(key,0,policy()) for key in ("a","b")}
    for pos,key in ((0,"a"),(0,"b"),(1,"a"),(1,"b"),(0,"a")):
        layout={"position":pos,"reasoning":pos,"markers":{m:False for m in ("<think>","</think>","Alternatively")}}
        active=states[key].plan(layout,ids,weights,scores)
        states[key].dispatch(layout,ids,weights,ids,weights,active)
    assert states["a"].summary()["diagnostics"]["tokens"]==2
    assert states["a"].summary()["diagnostics"]["recomputations"]==1
    assert states["b"].summary()["diagnostics"]["recomputations"]==0
    assert states["a"].features.reducer.buffers["mixture"][0,7]==pytest.approx(8/36)


def test_dispatch_is_required_and_weight_ids_cannot_change():
    ids=np.arange(8,dtype=np.int32)
    weights=np.ones(8)/8
    scores=np.r_[np.ones(8),np.zeros(248)]
    state=LayerHistory("a",0,policy("temperature",2))
    layout={"position":0,"reasoning":0,"markers":{}}
    active=state.plan(layout,ids,weights,scores)
    with pytest.raises(RuntimeError,match="dispatch"):
        state.summary()
    with pytest.raises(RuntimeError,match="IDs"):
        state.dispatch(layout,ids,weights,ids[::-1],weights,active)


def test_frontier_selection_ties_and_dominance():
    points=[dict(policy_id=p,accuracy=a,tokens=t) for p,a,t in
            (("b",.8,100),("a",.8,100),("c",.85,200),("d",.9,300),("e",.7,150))]
    assert "e" not in [p["policy_id"] for p in nondominated(points)]
    assert select_frontier(points)==["d","a","c"]
    assert bh([.01,.04,.03])==pytest.approx([.03,.04,.04])


def test_bootstrap_is_paired_stratified_and_repeated_attempts_not_units():
    arrays={b:np.array([[[0.,10.],[1.,20.]],[[1.,20.],[2.,30.]]]) for b in BENCHMARKS}
    result=paired_bootstrap(arrays,replicates=1000)
    assert result["equal_benchmark_mean"].shape==(1000,2,2)
    assert np.allclose(result["equal_benchmark_mean"][:,1]-result["equal_benchmark_mean"][:,0],[1.,10.])
    with pytest.raises(ValueError):
        paired_bootstrap({"math500":arrays["math500"]})


def test_final_test_requires_frozen_selection():
    study={"sha256":"study","attempts":{"test":[1]}}
    p=policy()
    with pytest.raises(ValueError,match="sealed"):
        planned_rows(study,"test",policy=p)
    selection=seal(dict(study_sha256="study",phase="tuning",stage="final",allowed_policy_hashes=[p["sha256"]]))
    assert planned_rows(study,"test",selection,p)==[1]


def test_incompatible_baseline_and_partial_json_tail(tmp_path):
    baseline=dict(status="complete",policy={"action":"native"},attempts_sha256="hash",dispatch_verified=True)
    target=copy.deepcopy(baseline)
    assert matching_baseline(baseline,target)
    target["checkpoint"]={"revision":"other"}
    assert not matching_baseline(baseline,target)
    path=tmp_path/"attempts.jsonl"
    path.write_bytes(b'{"id":1}\n{"id":')
    repair_interrupted_tail(path)
    assert path.read_bytes()==b'{"id":1}\n'
    path.write_bytes(b'broken\n')
    with pytest.raises(ValueError):
        repair_interrupted_tail(path)


def test_exact_gap_histogram_quantiles_match_unique_token_batch():
    from collections import Counter
    from moe_exp.routing_study.temporal import histogram_quantile,weak_fraction
    values=[0.,0.,.1,.2,.2,.2,.8,1.]
    histogram=sorted(Counter(values).items())
    for q in (0,.1,.25,.75,1):
        assert histogram_quantile([histogram],q)==pytest.approx(np.quantile(values,q))
    assert weak_fraction({"gap_histogram":[[0.,64],[1.,192]]},0.)==.25
    with pytest.raises(ValueError):
        histogram_quantile([[[0.,-1]]],.1)


def test_prompt_quoted_close_does_not_disable_assistant_reasoning():
    pieces={0:"Quoted </think>",1:"<think>",2:"reason"}
    history=TokenHistory([0,1],pieces.__getitem__)
    history.observe(0,0)
    history.observe(1,1)
    assert history.observe(2,2)["reasoning"]==0


def test_static_grid_never_substitutes_unavailable_calibration():
    missing={"unavailable":"Awaiting exact FP8 calibration"}
    calibration=seal(dict(split_sha256="split",attempts=[],identity=missing,
        steer={"promote":missing,"suppress":missing},rice=missing,margin=missing))
    grid=static_grid(calibration,code={})
    assert len(grid)==16
    assert sum(p["status"]=="unavailable" for p in grid)==12
    assert [p["strength"] for p in grid if p["action"]=="identity"]==[0.,-1.,-.5,.5,1.]


@pytest.mark.parametrize("answer",[None,""])
def test_missing_answer_with_known_gold_is_incorrect(answer):
    from moe_exp.routing_study.execution import score_attempt
    scorer=lambda *args,**kw:(answer,True,"fixture")
    assert score_attempt({"gold_answer":"3"},"capped reasoning",scorer)[1] is False
    with pytest.raises(ValueError):
        score_attempt({"gold_answer":None},"text",scorer)


def test_compact_calibration_pipeline_and_non_fp8_rejection():
    from moe_exp.routing_study.calibration import calibrate
    problems=population(10)
    split=split_problems(problems)
    mapping={problem_key(p):p for p in problems}
    keys=[k for k,s in split["membership"].items() if s=="calibration"]
    sources={digest([k,correct]):{"problem_key":k} for k in keys for correct in (False,True)}
    study=seal(dict(split=split,problems=mapping,calibration_sources=sources,code={},backend={"mock":True}))
    templates={}
    for correct in (False,True):
        counts=[0]*256
        counts[:8]=[100]*8
        counts[0 if not correct else 1]=50
        counts[8]=50
        all_counts=counts.copy()
        marker_ids={"<think>":[0,2,3,4,5,6,7,8],"</think>":[1,2,3,4,5,6,7,8],"Alternatively":list(range(8))}
        marker_experts={m:[int(e in selected) for e in range(256)] for m,selected in marker_ids.items()}
        for marker in ("<think>","</think>"):
            all_counts=[a+b for a,b in zip(all_counts,marker_experts[marker])]
        templates[correct]={str(l):dict(reasoning_tokens=100,reasoning_counts=counts,
            all_tokens=102,all_counts=all_counts,marker_tokens={m:1 for m in marker_ids},
            marker_experts=marker_experts,gap_histogram=[[1.,100]],windows=[],
            mean_probability_gap=.001 if correct else .01) for l in range(40)}
    data=[seal(dict(id=digest([key,correct]),problem_key=key,is_correct=correct,
        study_sha256=study["sha256"],checkpoint=CHECKPOINT,code={},backend={"mock":True},
        capture="verified_modular_dispatch",infrastructure_failure=False,layers=templates[correct],
        marker_occurrences={"<think>":1,"</think>":1,"Alternatively":1})) for key in keys for correct in (False,True)]
    result=calibrate(data,study)
    assert "unavailable" in result["timing"]
    grid=static_grid(result,code={})
    assert len(grid)==16 and all(p["status"]=="frozen" for p in grid)
    for p in grid: validate(p)
    bad=copy.deepcopy(data[0])
    bad.pop("sha256")
    bad["checkpoint"]={**CHECKPOINT,"quantization":"nf4"}
    with pytest.raises(ValueError,match="FP8"):
        calibrate([seal(bad),*data[1:]],study)


def test_random_controls_need_native_history_and_low_support_contrasts_stay_descriptive():
    gate=Gate("q",0,"random_persistent",threshold())
    assert not gate.decide(256,None)
    data=[s for s in samples() if int(s["problem_key"])<5]
    result=fit_steer(data)
    assert "unavailable" in result["promote"]
    assert result["within_question"]["problems"]==5
    assert result["within_question"]["differences"][0][0]==pytest.approx(.1)


def test_source_attempt_ids_group_without_losing_pilot_and_fixture_identity(tmp_path):
    import hashlib
    from moe_exp.routing_study.common import MODEL
    from moe_exp.routing_study.prepare import collect, resolve_pilot
    source = []
    for sample in range(2):
        source.append(dict(dataset="aime24", problem_id=f"question__sample_{sample:02d}",
            source_problem_id="question", sample_id=sample, model_id=MODEL,
            prompt="same question", gold_answer="1", metadata={
                "generation_config": {**SAMPLING, "seed": 42 + sample},
                "token_replay": {"prompt_token_ids": [1, 2], "completion_token_ids": [3]}}))
    path = tmp_path / "sources.jsonl"
    def save():
        path.write_text("".join(json.dumps(r) + "\n" for r in source))
    save()
    problems, attempts = collect([path])
    assert len(problems) == 1 and len(attempts) == 2
    key = problem_key(source[0])
    assert problems[key]["problem_id"] == "question"
    assert {a["original_problem_id"] for a in attempts.values()} == {r["problem_id"] for r in source}
    identity = [MODEL, "aime24", source[1]["problem_id"], 1]
    legacy = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    pilot = {"traces": [{"identity": identity}], "excluded_fixtures": [legacy]}
    assert resolve_pilot(problems, attempts, pilot) == ({key}, {key})
    for field, value in (("prompt", "changed question"), ("gold_answer", "2")):
        previous = source[1][field]
        source[1][field] = value
        save()
        with pytest.raises(ValueError, match=field):
            collect([path])
        source[1][field] = previous
    source[1]["metadata"]["token_replay"]["prompt_token_ids"] = [9, 2]
    save()
    with pytest.raises(ValueError, match="prompt_token_ids"):
        collect([path])


def test_steer_rejects_disjoint_outcome_question_arms():
    data=samples()
    for sample in data:
        sample['problem_key'] += '-right' if sample['is_correct'] else '-wrong'
    result=fit_steer(data)
    assert result['within_question']['problems']==0
    assert 'unavailable' in result['promote']
    assert not result['contrasts']


def test_steer_unpaired_questions_cannot_change_rankings():
    data=samples()
    expected=fit_steer(data)
    distractors=[]
    for i in range(30):
        row=copy.deepcopy(data[0]);row['id']=f'distractor-{i}';row['problem_key']=row['id']
        row['is_correct']=True
        for entry in row['layers'].values():
            entry['reasoning_counts'][0]=0
            entry['reasoning_counts'][1]=100
        distractors.append(row)
    actual=fit_steer(data+distractors)
    assert actual['promote']==expected['promote']
    assert actual['suppress']==expected['suppress']
    assert actual['contrasts']==expected['contrasts']
    assert actual['method_status']=='equal_question_paired_outcome_rate_difference'


def test_scoring_extractor_is_bound_by_study_code_hash():
    hashes=code_hashes()
    assert 'moe_exp/utils.py' in hashes
    assert 'moe_exp/correlation_pipeline/scoring.py' in hashes
