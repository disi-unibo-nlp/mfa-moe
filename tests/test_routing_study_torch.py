"""Small CPU tensor checks of the actual actions and dispatch adapter."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moe_exp.routing_study.policy import freeze
from moe_exp.routing_study.actions import logits_action,weights_action
from moe_exp.routing_study.runtime import RequestHistory
from moe_exp.routing_study.vllm_backend import DispatchHook,validate_runner


def policy(action,strength=0,**kw):
    return freeze(action,strength,code={},**kw)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
def test_zero_identity_bit_exact_with_ties(dtype):
    logits=torch.zeros(3,256,dtype=dtype)
    assert logits_action(logits,policy("identity",0),0) is logits


@pytest.mark.parametrize("tau",[.5,2])
def test_native_weights_not_affinities_and_fixed_ids(tau):
    ids=torch.arange(8).reshape(1,8)
    weights=torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.]])
    before=ids.clone()
    result=weights_action(weights,ids,policy("temperature",tau),0)
    expected=weights.pow(1/tau)/weights.pow(1/tau).sum()
    assert torch.allclose(result,expected)
    assert torch.equal(ids,before)
    assert result.sum().item()==pytest.approx(1)


@pytest.mark.parametrize("normalize",[False,True])
def test_rice_only_targets_already_selected(normalize):
    ids=torch.arange(8).reshape(1,8)
    weights=torch.ones(1,8)/8
    p=policy("rice",2,targets=[(0,1),(1,9)],parameters={"renormalize":normalize},calibration={"x":1})
    actual=weights_action(weights,ids,p,0)
    expected=weights.clone()
    expected[0,1]*=2
    if normalize: expected/=expected.sum()
    assert torch.equal(actual,expected)
    assert weights_action(weights,ids,p,2) is weights


@pytest.mark.parametrize("action",["steer_promote","steer_suppress"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
def test_steer_strict_order_when_epsilon_rounds_away(action,dtype):
    logits=torch.linspace(-200,0,256,dtype=dtype).reshape(1,256)
    p=policy(action,1,targets=[(0,3),(0,4)],parameters={"epsilon":.01},calibration={"x":1})
    result=logits_action(logits,p,0)
    other=torch.cat((result[:,:3],result[:,5:]),dim=1)
    if action=="steer_promote":
        assert result[0,3]>other.max()
    else:
        assert result[0,3]<other.min()
    assert torch.isfinite(result).all()


def test_existing_identity_and_margin_formulas_reused():
    logits=torch.arange(256,dtype=torch.float32).reshape(1,256)/256
    scores=[0.]*256
    scores[0]=1.
    p=policy("identity",-.5,parameters={"layers":{"0":{"scores":scores}}},calibration={"x":1})
    assert logits_action(logits,p,0)[0,0].item()==-.5
    p=policy("margin",1,parameters={"layers":{"0":{"lower":0.,"upper":0.}}},calibration={"x":1})
    result=logits_action(logits,p,0)
    assert torch.isfinite(result).all()
    assert result.shape==logits.shape


def test_dispatch_boundary_padding_shared_expert_and_native_ids():
    p=policy("temperature",2)
    state=RequestHistory({"id":"a","prompt_token_ids":[0]},p,{0:"<think>",1:"reason"}.__getitem__)
    state.tokens.observe(0,0)
    layout=state.tokens.observe(1,1)
    owner=SimpleNamespace(policy=p,frame=[(state,layout),None])
    shared=object()
    calls=[]
    class Router:
        def select_experts(self,hidden_states,router_logits,topk_indices_dtype=None,input_ids=None):
            ids=torch.argsort(router_logits,descending=True,stable=True)[:,:8].int()
            # Native weights intentionally differ from any normalized router estimate.
            return torch.arange(1,9,dtype=torch.float32).repeat(2,1),ids
    class Experts:
        def forward_modular(self,**kw):
            assert kw["shared_experts"] is shared
            calls.append(kw)
            return "executed"
    runner=SimpleNamespace(router=Router(),routed_experts=Experts())
    hook=DispatchHook(0,owner,runner)
    logits=torch.arange(256,dtype=torch.float32).repeat(2,1)
    hidden=torch.zeros(2,4)
    weights,ids=runner.router.select_experts(hidden,logits)
    assert torch.equal(ids[0],torch.arange(255,247,-1,dtype=torch.int32))
    assert torch.equal(weights[1],torch.arange(1,9,dtype=torch.float32))  # padding unchanged
    result=runner.routed_experts.forward_modular(x=hidden,topk_weights=weights,topk_ids=ids,shared_experts=shared)
    assert result=="executed" and hook.calls==hook.dispatches==1
    assert state.layers[0].summary()["diagnostics"]["tokens"]==1
    assert state.layers[0].summary()["diagnostics"]["changed_sets"]==0
    assert state.layers[0].summary()["diagnostics"]["changed_weights"]==1
    with pytest.raises(RuntimeError,match="Unverified"):
        runner.routed_experts.forward_modular(x=hidden,topk_weights=weights,topk_ids=ids)


def test_monolithic_backend_is_rejected():
    with pytest.raises(RuntimeError,match="Unsupported"):
        validate_runner(SimpleNamespace(experts=SimpleNamespace()))


def test_rice_does_not_renormalize_rows_without_a_selected_target():
    ids=torch.arange(8).reshape(1,8)
    weights=torch.tensor([[.1,.2,.1,.05,.03,.02,.1,.3999999]])
    p=policy("rice",2,targets=[(0,9),(1,9)],parameters={"renormalize":True},calibration={"x":1})
    assert torch.equal(weights_action(weights,ids,p,0),weights)


def test_compact_identity_estimator_matches_colleague_tensor_fit():
    from moe_exp.moe_identity_guiding.calibration import fit as legacy_fit
    from moe_exp.routing_study.calibration import fit_identity
    from moe_exp.routing_study.common import MODEL
    tensors={}
    for correct in (False,True):
        ids=torch.arange(8).reshape(1,1,8).repeat(40,100,1)
        ids[:,50:,1 if correct else 0]=8
        tensors[correct]=ids
    legacy=[]
    compact=[]
    for question in range(10):
        for correct in (False,True):
            tensor=tensors[correct]
            legacy.append(dict(dataset="math500",problem_id=str(question),is_correct=correct,
                               selected_experts=tensor,layer_indices=list(range(40)),model_id=MODEL))
            compact.append(dict(problem_key=str(question),is_correct=correct,layers={
                str(l):dict(reasoning_tokens=100,reasoning_counts=torch.bincount(tensor[l].flatten(),minlength=256).tolist())
                for l in range(40)}))
    original=legacy_fit(legacy,model=MODEL,num_experts=256,top_k=8)
    actual=fit_identity(compact)["policy"]
    for l in map(str,range(40)):
        assert actual["layers"][l]["scores"]==pytest.approx(original["layers"][l]["scores"])
