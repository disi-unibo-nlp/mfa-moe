from types import SimpleNamespace
import numpy as np
import pytest
import torch
from measurement_study.native_gaps import glm_reducer


def test_native_glm_gap_uses_bias_and_keeps_affinities(monkeypatch):
    from moe_exp.models import router_adapters
    from moe_exp.correlation_pipeline.dynamics import classes
    from moe_exp.correlation_pipeline import spans,provenance
    gate=SimpleNamespace(e_score_correction_bias=torch.tensor([-.6,0.,0.]))
    model=SimpleNamespace(config=SimpleNamespace(model_type='glm4_moe_lite',n_group=1,topk_group=1))
    monkeypatch.setattr(router_adapters,'decoder_layers',lambda m:[None,SimpleNamespace(mlp=SimpleNamespace(gate=gate))])
    monkeypatch.setattr(router_adapters,'available_router_layers',lambda m:[1])
    monkeypatch.setattr(classes,'saved_layout',lambda t:dict(token_count=64,reasoning_tokens=list(range(64))))
    monkeypatch.setattr(spans,'trace_digest',lambda t:'fixture')
    monkeypatch.setattr(provenance,'code_provenance',lambda:{})
    trace=SimpleNamespace(model_id='glm',metadata={})
    router=torch.tensor([.8,.7,.3]).log().repeat(1,64,1);before=router.clone()
    ids=torch.ones((1,64,1),dtype=torch.int64);weights=torch.ones((1,64,1))*1.8
    reduce=glm_reducer(model);result=reduce(trace,router,ids,weights,[1],{'native_selection':True})
    assert result['windows'][0]['boundary_gap_mean']==pytest.approx(.4,abs=1e-6)
    assert result['semantics']['selection_scores']=='native_sigmoid_plus_bias'
    assert torch.equal(router,before)
    with pytest.raises(ValueError,match='incompatible'):
        reduce(trace,router,torch.zeros_like(ids),weights,[1],{'native_selection':True})


def test_group_restricted_glm_is_rejected():
    model=SimpleNamespace(config=SimpleNamespace(model_type='glm4_moe_lite',n_group=2,topk_group=1))
    with pytest.raises(ValueError,match='single-group'):glm_reducer(model)
