"""Independent numerical and contract tests; synthetic tensors require no model."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from moe_exp.schemas import TraceRecord
from moe_exp.correlation_pipeline.spans import sentence_spans, trace_digest
from moe_exp.correlation_pipeline.dynamics.classes import class_dynamics, sentence_table, summaries
from moe_exp.correlation_pipeline.dynamics.common import CLASSES, keys
from moe_exp.correlation_pipeline.dynamics.routing import RoutingReducer, entropy, probabilities
from moe_exp.correlation_pipeline.dynamics.evaluation import (
    _design, _thresholds, bh_adjust, evaluate, grouped_folds, prediction_rows,
)
from moe_exp.correlation_pipeline.dynamics.pilot import select


def trace(labels=("Read", "Implement", "Verify"), *, finish="stop", question="q", sample=0):
    t = TraceRecord(dataset="math500", problem_id=question + str(sample),
        source_problem_id=question, sample_id=sample, prompt="Question", gold_answer="1",
        model_id="model", model_answer="1", is_correct=True,
        cot_text=" ".join(f"Sentence {i}." for i in range(len(labels))),
        metadata={"finish_reason": finish})
    units = sentence_spans(t)
    annotation = dict(schema_version=1, trace_sha256=trace_digest(t),
        dataset=t.dataset, problem_id=t.problem_id, sample_id=t.sample_id,
        source_model=t.model_id, units=[{**u, "label": label} for u, label in zip(units, labels) if label],
        sentence_selection=None)
    return t, annotation


def reducer(n=8, k=2, config=None):
    return RoutingReducer(n, k, model="m", layer=3, semantics={"deterministic_topk": True},
                          config={"windows": [4, 8], "stride": 4, "lags": [1, 4], **(config or {})})


def test_stationary_allocation_and_native_weights():
    r = reducer()
    ids = np.tile([0, 1], (12, 1))
    p = np.tile([.4, .3, .05, .05, .05, .05, .05, .05], (12, 1))
    weights = np.tile([.1, .9], (12, 1))
    out = r.push(ids, weights=weights, full_probabilities=p, selection_scores=np.log(p))
    for row in out:
        assert row["overlap_lag1"] == row["overlap_corrected_lag1"] == 1
        assert row["full_entropy_difference"] == pytest.approx(0, abs=1e-12)
        assert row["mixture_local_entropy"] == pytest.approx(-.1*np.log(.1)-.9*np.log(.9))
        assert row["mixture_effective_experts"] == pytest.approx(np.exp(row["mixture_local_entropy"]))
        assert row["full_jsd"] == pytest.approx(0)
        assert row["boundary_gap_mean"] == pytest.approx(np.log(.3/.05))
        assert row["weak_boundary_fraction"] is None
    assert r.retained_tokens <= 12


def test_changing_sets_rank_swaps_and_ties():
    r = reducer(4, 2)
    ids = np.array([[0,1],[1,0],[0,1],[1,0],[2,3],[3,2],[2,3],[3,2]])
    rows = r.push(ids, weights=np.ones((8,2)), full_probabilities=np.full((8,4),.25),
                  selection_scores=np.ones((8,4)))
    row = rows[-1]
    assert row["overlap_lag1"] == pytest.approx(6/7)
    assert row["boundary_gap_mean"] == 0
    assert row["full_local_entropy"] == pytest.approx(np.log(4))
    assert row["full_jsd"] == pytest.approx(0)
    assert row["mixture_jsd"] == pytest.approx(np.log(2))
    assert row["set_turnover"] == pytest.approx(1/7)


def test_shuffle_expectation_independent_exhaustive_oracle():
    ids = np.array([[0],[0],[1],[2]])
    row = reducer(3,1).push(ids)[0]
    brute = []
    for order in itertools.permutations(range(4)):
        shuffled = ids[list(order), 0]
        brute.append(np.mean(shuffled[1:] == shuffled[:-1]))
    assert row["overlap_shuffled_lag1"] == pytest.approx(np.mean(brute))
    assert row["mixture_jsd"] is None
    assert row["boundary_gaps"] is None


@pytest.mark.parametrize("chunk", [1,3,4,7,32])
def test_chunking_prefix_causality_and_bounded_memory(chunk):
    rng = np.random.default_rng(32)
    scores = rng.normal(size=(40,8))
    ids = np.argsort(scores, axis=1)[:,-2:]
    p = probabilities(scores)
    def run(n, step):
        r, rows = reducer(), []
        for start in range(0,n,step):
            stop=min(n,start+step)
            rows.extend(r.push(ids[start:stop], full_probabilities=p[start:stop],
                               selection_scores=scores[start:stop]))
            assert r.retained_tokens <= r.limit
        return rows
    assert run(40, chunk) == run(40,40)
    assert run(20,chunk) == [r for r in run(40,chunk) if r["end_token"] <= 20]


def test_batch_entropy_and_jsd_oracle():
    rng=np.random.default_rng(3)
    p=rng.dirichlet(np.ones(8),size=8)
    ids=np.argsort(p,axis=1)[:,-2:]
    row=reducer().push(ids,full_probabilities=p)[-1]
    local=float(np.mean([-sum(v*np.log(v)) for v in p]))
    a,b=p[:4].mean(0),p[4:].mean(0)
    m=(a+b)/2
    j=.5*sum(a*np.log(a/m))+.5*sum(b*np.log(b/m))
    assert row["full_local_entropy"] == pytest.approx(local)
    assert row["full_jsd"] == pytest.approx(j)


@pytest.mark.parametrize("invalid", ["duplicate","range","score","weights","probability"])
def test_signal_validation(invalid):
    r=reducer()
    ids=np.tile([0,1],(4,1))
    kwargs={}
    if invalid=="duplicate": ids[:]=0
    elif invalid=="range": ids[:]=9
    elif invalid=="score": kwargs["selection_scores"]=np.tile(np.arange(8),(4,1))
    elif invalid=="weights": kwargs["weights"]=np.zeros((4,2))
    else: kwargs["full_probabilities"]=np.ones((4,8))
    with pytest.raises(ValueError): r.push(ids,**kwargs)


def test_incompatible_native_router_does_not_claim_boundary():
    r=reducer()
    r.semantics={"deterministic_topk":False}
    row=r.push(np.tile([0,1],(4,1)),selection_scores=np.ones((4,8)))[0]
    assert row["selection_score_status"]=="unavailable"


def test_gaps_unknowns_censored_durations_and_motifs():
    t,a=trace(["Read","Read",None,"Read","Read","Implement","Read"],finish="length")
    rows=sentence_table(t,a)
    tables=class_dynamics(rows)
    assert len(tables["transitions"])==4
    runs=tables["runs"]
    assert [r["entry_observed"] for r in runs]==[True,False,True,True]
    assert runs[0]["exit_event"]=="gap"
    assert runs[-1]["exit_event"]=="generation_cap"
    assert rows[3]["run_age_sentences"] is None
    assert [r["pattern"] for r in tables["motifs"] if len(r["pattern"])==3]==[
        ["Read","Read","Implement"],["Read","Implement","Read"]]
    assert tables["returns"][0]["return_sentences"]==3
    hazards=summaries(tables)["exit_hazards"]
    assert all(r["at_risk"]>0 for r in hazards)
    assert all(r["destination"] not in ("gap","generation_cap") for r in hazards)
    assert all(r["status"]=="insufficient_support" for r in summaries(tables)["transition_rates"])


def test_natural_termination_not_eighth_class():
    t,a=trace()
    tables=class_dynamics(sentence_table(t,a))
    assert tables["runs"][-1]["exit_event"]=="natural_termination"
    assert not tables["runs"][-1]["right_censored"]
    assert len(summaries(tables)["transition_rates"])==49
    assert all(r["destination"] in CLASSES for r in tables["transitions"])


def test_absent_class_rate_undefined():
    t,a=trace(["Read","Read"])
    rows=summaries(class_dynamics(sentence_table(t,a)))["transition_rates"]
    assert all(r["rate"] is None for r in rows if r["source"]=="Explore")


def test_annotation_identity_offsets_and_schema():
    t,a=trace()
    for field,value in [("trace_sha256","bad"),("source_model","bad"),("schema_version",2)]:
        b=copy.deepcopy(a);b[field]=value
        with pytest.raises(ValueError):sentence_table(t,b)
    b=copy.deepcopy(a);b["units"][0]["text"]="wrong"
    with pytest.raises(ValueError):sentence_table(t,b)


def test_repeated_attempt_grouping():
    t,_=trace(question="one",sample=0)
    other,_=trace(question="one",sample=1)
    assert keys(t)["trace_id"]!=keys(other)["trace_id"]
    assert keys(t)["question_id"]==keys(other)["question_id"]
    groups=[str(i//3) for i in range(36)]
    for train,test in grouped_folds(groups,4):
        assert not {groups[i] for i in train}&{groups[i] for i in test}
    assert [b.tolist() for _,b in grouped_folds(groups,4)]==[b.tolist() for _,b in grouped_folds(groups,4)]


def test_threshold_scaling_and_screening_train_only():
    rows=[]
    for i in range(12):
        rows.append(dict(trace_id=str(i),question_id=str(i),base={"position":i},
            static={"expert_L0_W64_0":i},temporal={"turnover":i/12},shuffled={},
            gaps={"L0_W64_":[(0,float(i))]}))
    train,test=rows[:8],rows[8:]
    a,b,audit=_design(train,test,2,10)
    assert not any(c.startswith("expert_") for c in audit["columns"])
    changed=copy.deepcopy(test)
    for row in changed:row["gaps"]["L0_W64_"]=[(0,1e9)]
    a2,b2,audit2=_design(train,changed,2,10)
    np.testing.assert_equal(a,a2)
    assert audit==audit2
    assert _thresholds(train)["L0"]==pytest.approx(.7)


def test_bh():
    assert bh_adjust([.01,.04,.03,None])==pytest.approx([.03,.04,.04,None])


def test_prediction_refuses_small_support():
    rows=[dict(question_id=str(i),target=i%2) for i in range(4)]
    assert evaluate(rows)["status"]=="unsupported"


def test_nested_evaluation_on_known_signal():
    rows=[dict(trace_id=str(i),question_id=str(i),target=i%2,
               base={"x":i/24},static={"signal":i%2},temporal={},shuffled={},gaps={})
          for i in range(24)]
    result=evaluate(rows,bootstrap=20)
    assert result["models"]["static_routing"]["log_loss"]<result["models"]["history_text"]["log_loss"]
    assert len(result["outer_splits"])==4
    assert len(result["predictions"])==24


def test_pilot_selection_independent_of_correctness():
    traces=[]
    for i in range(80):
        t,_=trace(question=str(i))
        t.metadata["reasoning_content"]=t.cot_text
        t.metadata["token_replay"]={"completion_offsets":[[0,len(t.cot_text)]]*(i+1),
                                  "completion_token_ids":[1]*(i+1)}
        traces.append(t)
    a,bins=select(traces)
    for t in traces:t.is_correct=not t.is_correct
    b,_=select(traces)
    assert [(q,n,t.problem_id) for q,n,t in a]==[(q,n,t.problem_id) for q,n,t in b]
    assert [q["selected"] for q in bins]==[16]*4


def test_class_cli_resume_relocation_and_invalidation(tmp_path):
    from moe_exp.correlation_pipeline.dynamics.cli import analyze
    t,a=trace()
    source=tmp_path/"traces.jsonl";source.write_text(t.model_dump_json()+"\n")
    annotation=tmp_path/"annotations.jsonl";annotation.write_text(json.dumps(a)+"\n")
    result=analyze([source],tmp_path/"out",annotations=[annotation],mode="class-only")
    assert result["summary"]["adjacent_labeled_pairs"]==2
    assert analyze([source],tmp_path/"out",annotations=[annotation],mode="class-only")==result
    moved=tmp_path/"moved.jsonl";moved.write_bytes(source.read_bytes())
    assert analyze([moved],tmp_path/"out",annotations=[annotation],mode="class-only")==result
    with pytest.raises(ValueError):
        analyze([source],tmp_path/"out",annotations=[annotation],mode="class-only",config={"seed":1})
    (tmp_path/"out"/"sentences.jsonl").write_text("corrupt")
    with pytest.raises(ValueError):
        analyze([source],tmp_path/"out",annotations=[annotation],mode="class-only")

def test_prospective_prefix_features_ignore_future_text_labels_and_length():
    t,a=trace()
    t.cot_text="x"*320
    t.metadata["token_replay"]={"completion_token_ids":[1]*320,
        "completion_offsets":[[i,i+1] for i in range(320)]}
    first=prediction_rows(t,[],[],landmarks=(256,))[0]
    t.cot_text+="future"*1000
    t.metadata["token_replay"]["completion_offsets"] += [[320+i,321+i] for i in range(6000)]
    t.metadata["token_replay"]["completion_token_ids"] += [2]*6000
    t.metadata["finish_reason"]="length"
    second=prediction_rows(t,[{"label":None,"next_class":None}],[],landmarks=(256,))[0]
    assert first["base"]==second["base"]
    assert first["routing_windows"]==second["routing_windows"]
    assert not {"termination","eventual_length","label"} & set(first["base"])


def test_expert_associations_keep_identical_ids_in_different_layers_separate():
    from moe_exp.correlation_pipeline.dynamics.experts import sentence_experts, associations
    t,a=trace(["Read","Verify"])
    rows=sentence_table(t,a)
    rows[0].update(token_indices=[0,1],tokens=2)
    rows[1].update(token_indices=[2,3],tokens=2)
    ids=np.array([[[0],[0],[1],[1]],[[1],[1],[0],[0]]])
    sparse=sentence_experts(rows,ids,[4,17])
    result=associations(rows,sparse,min_problems=10,bootstrap=10)
    deltas={(r["layer"],r["expert"]):r["rate_difference"] for r in result
            if r["source_class"] is None and r["target_class"]=="Read"}
    assert deltas[(4,0)]==1
    assert deltas[(17,0)]==-1
    assert all(r["q_bh"] is None for r in result)


def test_dynamics_callback_receives_native_weights_and_resume_is_bound(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from moe_exp.models.routing_extraction import process_file
    from moe_exp.correlation_pipeline.dynamics.routing import reduce_trace
    t,_=trace()
    t.cot_text="abcdefgh"
    t.metadata.update(reasoning_content=t.cot_text,token_replay={
        "completion_token_ids":[1]*8,"completion_offsets":[[i,i+1] for i in range(8)]})
    source=tmp_path/"input.jsonl";source.write_text(t.model_dump_json()+"\n")
    output=tmp_path/"out/traces_with_routing.jsonl"
    scores=torch.tensor([[[3.,2.,1.,0.]]*8])
    native=torch.tensor([[[0,1]]*8])
    weights=torch.tensor([[[.1,.9]]*8])
    semantics=dict(native_selection=True,deterministic_topk=True,selection_scores="router_signal")
    monkeypatch.setattr("moe_exp.models.router_adapters.available_router_layers",lambda model:[4])
    def fake(**kwargs):
        kwargs["routing_details"].update(selected_experts=native,expert_weights=weights,
                                        router_distribution="softmax",semantics=semantics)
        return scores
    monkeypatch.setattr("moe_exp.models.routing_extraction.extract_logs_single_pass",fake)
    def callback(*args):
        return reduce_trace(*args,config={"windows":[4,8],"stride":4,"lags":[1,4]})
    opts=dict(input_path=source,model_id="replay",output_path=output,
        model=SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=2)),
        tokenizer=object(),layer_indices=[4],save_expert_weights=False,save_raw_tensors=False,
        feature_reducer=lambda *args:{},feature_schema_version=1,forward_provenance={"revision":"a"},
        dynamics_reducer=callback,dynamics_config={"schema":1})
    assert process_file(**opts)["forwards"]==1
    row=json.loads(output.read_text())
    assert row["model_logs"]["expert_weights"] is None
    assert row["metadata"]["routing_dynamics"]["windows"][0]["mixture_local_entropy"]==pytest.approx(-.1*np.log(.1)-.9*np.log(.9))
    assert process_file(**opts)["forwards"]==0
    opts["dynamics_config"]={"schema":2}
    assert process_file(**opts)["forwards"]==1

    from moe_exp.correlation_pipeline.dynamics.cli import analyze
    config={"windows":[4,8],"stride":4,"lags":[1,4]}
    combined=analyze([output],tmp_path/"combined",config=config)
    routing=analyze([output],tmp_path/"routing",mode="routing-only",config=config)
    assert combined["summary"]["routing_windows"]==routing["summary"]["routing_windows"]==3
    assert routing["summary"]["sentences"]==0
